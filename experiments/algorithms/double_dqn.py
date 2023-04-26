import os
import time
import re

from typing import List, Dict, Any
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl

from common.typing import HParams
from common.environment import make_env
from common.utils import linear_schedule
from common.networks import CNNDQN, SwinDQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_latest_checkpoint_file(files: List[str]) -> any:
    """
    Return the most recent checkpoint file from the passed list of files.

    If multiple files with same datetime are passed, only the first is returned

    :param files: list of file names containing a formatted datetime (=> %d-%m-%Y_%H:%M:%S)
    :return: the file with the most recent date time or ``None`` if no files were found (because of the lack of correctly formatted date in the file name)
    """
    datetime_regex = r"\d{2}-\d{2}-\d{4}_\d{2}:\d{2}:\d{2}"

    latest_file = None
    latest_datetime = datetime.min
    for file in files:
        match = re.search(datetime_regex, file)
        if not match: continue  # Go to next element in list if no match is found

        file_datetime = datetime.strptime(match.group(), "%d-%m-%Y_%H:%M:%S")
        if file_datetime > latest_datetime:
            latest_datetime = file_datetime
            latest_file = file

    return latest_file


class DQNAgent:
    def __init__(self, name, env_id, hparams: HParams, network_type: str = "CNN"):
        self.name = name
        self.env = make_env(env_id, seed=hparams["env_seed"])

        self.start_datetime = None
        self.start_time = None

        self.MAX_TIMESTEPS = hparams["max_timesteps"]  # Maximum number of total steps
        self.TARGET_UPDATE_INTERVAL = hparams[
            "target_update_interval"]  # Number of steps between the synchronisation of q and target network
        self.LEARNING_STARTS = hparams[
            "learning_starts"]  # The number of steps to wait before we start the training, so the agent can explore and store its experience in the replay buffer

        self.TRAIN_FREQUENCY = hparams["train_frequency"]  # Training is done each 4 steps

        self.CHECKPOINT_INTERVAL_EPISODE = 1000  # Checkpoint saving interval per episode (a checkpoint will be saved each X episodes)

        self.REPLAY_SIZE = hparams["replay_buffer_size"]
        self.BATCH_SIZE = hparams["batch_size"]

        self.GAMMA = hparams["discount_rate"]  # Discount rate

        self.EXPLORATION_FRACTION = hparams[
            "exploration_fraction"]  # The fraction of 'TOTAL_TIMESTEPS' it takes from 'EPSILON_START' to 'EPSILON_END'.
        self.EPSILON_INITIAL = 1.0
        self.EPSILON_FINAL = hparams["exploration_final"]

        self.epsilon = self.EPSILON_INITIAL  # Exploration probability

        self.memory = ReplayBuffer(
            buffer_size=self.REPLAY_SIZE,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            device=device,
            optimize_memory_usage=True,
            handle_timeout_termination=False
        )

        self.timesteps = 0

        if network_type == "Swin":
            self.policy_network = SwinDQN(4, self.env.action_space.n).to(device)
            self.target_network = SwinDQN(4, self.env.action_space.n).to(device)
        else:
            self.policy_network = CNNDQN(4, self.env.action_space.n).to(device)
            self.target_network = CNNDQN(4, self.env.action_space.n).to(device)

        self.target_network.load_state_dict(self.policy_network.state_dict())

        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=hparams["learning_rate"])
        self.loss_fn = hparams["loss"]

        # Metrics/Logs
        self.PATH = "runs"
        if not os.path.exists(self.PATH):
            os.makedirs(self.PATH)

        self.CHECKPOINTS_PATH = f"{self.PATH}/checkpoints"
        self.LOGS_PATH = f"{self.PATH}/logs"
        self.VIDEO_PATH = f"{self.PATH}/videos"

        self.is_loaded_from_checkpoint = False

        self.writer = None

    def remember(self, observation, next_observation, action, reward, done, infos):
        self.memory.add(observation, next_observation, action, reward, done, infos)

    def act(self, state):
        # Reduce epsilon when learning started
        if self.timesteps >= self.LEARNING_STARTS:
            # Minus LEARNING_STARTS to takes into account that learning only started after LEARNING_STARTS,
            # and so we want to start reducing epsilon only when learning start
            self.epsilon = linear_schedule(
                self.EPSILON_INITIAL,
                self.EPSILON_FINAL,
                int(self.EXPLORATION_FRACTION * self.MAX_TIMESTEPS),
                self.timesteps - self.LEARNING_STARTS
            )

        if self.timesteps < self.LEARNING_STARTS or np.random.rand() < self.epsilon:
            # Random action
            return np.array(self.env.action_space.sample())
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(np.array(state), device=device).unsqueeze(0)
                q_values = self.policy_network(state_tensor)
                return q_values.argmax(dim=1)[0].cpu().numpy()

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def optimize_model(self):
        minibatch = self.memory.sample(self.BATCH_SIZE)

        # Calculate Q values for current states
        # For each q_values, get the action according to the minibatch
        q_values = self.policy_network(minibatch.observations).gather(1, minibatch.actions)

        # Then, calculate the best actions for the next states, and return its indices
        with torch.no_grad():
            best_next_actions = self.policy_network(minibatch.next_observations).argmax(1).unsqueeze(1)

        # Calculate the Q values for the next states using the target network, and return the action according to the best next action returned by the q network
        target_next_q_values = self.target_network(minibatch.next_observations).gather(1, best_next_actions)

        # Calculate the target Q values using Double DQN
        target_q_values = minibatch.rewards + (1 - minibatch.dones) * self.GAMMA * target_next_q_values

        # Compute the loss
        loss = self.loss_fn(q_values, target_q_values)

        # Compute metrics for loss
        if self.timesteps % 100 == 0:
            self.writer.add_scalar("train/loss", loss, self.timesteps)
            self.writer.add_scalar("train/q_values", q_values.squeeze().mean().item(), self.timesteps)
            steps_per_second = int(self.timesteps / (time.time() - self.start_time))
            # print("Steps per second: ", steps_per_second)
            self.writer.add_scalar("train/steps_per_second", steps_per_second, self.timesteps)

        # Optimise Q network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_checkpoint(self):
        if self.start_datetime is None:
            print("SAVE_CHECKPOINT_ERROR: Training need to have started to save a checkpoint.")
            return

        print("Saving checkpoint...")
        current_datetime_str = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        start_datetime_str = self.start_datetime.strftime("%d-%m-%Y_%H:%M:%S")

        save_parent_directory = f"{self.CHECKPOINTS_PATH}/{self.name}_{start_datetime_str}"
        save_path = save_parent_directory + "/chkpt_" + current_datetime_str + ".tar"
        replay_buffer_path = save_parent_directory + "/replay_buffer_" + current_datetime_str

        if not os.path.exists(save_parent_directory):
            os.makedirs(save_parent_directory)

        checkpoint = {
            "env": self.env,
            "timesteps": self.timesteps,
            "start_datetime": self.start_datetime,
            "epsilon": self.epsilon,
            "policy_network": self.policy_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        torch.save(checkpoint, save_path)
        # Saving the replay buffer will takes time! But it is needed to properly resume training
        save_to_pkl(replay_buffer_path, self.memory, verbose=1)

        print(f"Checkpoint saved into {save_parent_directory}")

    def load_last_checkpoint(self, path):
        """
        Load the last saved checkpoint found in the given ``path``

        :param path: the path to the directory containing the checkpoint(s)
        """
        print(f"Loading most recent checkpoint from {path}")
        self.is_loaded_from_checkpoint = True

        # Using list comprehension to filter directories and only get the files
        files = [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]

        checkpoint_files = [chkpt_file for chkpt_file in files if "chkpt" in chkpt_file]
        replay_buffer_files = [chkpt_file for chkpt_file in files if "replay_buffer" in chkpt_file]

        checkpoint_file = get_latest_checkpoint_file(checkpoint_files)
        replay_buffer_file = get_latest_checkpoint_file(replay_buffer_files)

        checkpoint: Dict[str, any] = torch.load(path + "/" + checkpoint_file)

        self.env = checkpoint["env"]
        self.timesteps = checkpoint["timesteps"]
        self.start_datetime: datetime = checkpoint["start_datetime"]
        self.start_time = self.start_datetime.timestamp()

        self.epsilon = checkpoint["epsilon"]

        self.policy_network.load_state_dict(checkpoint["policy_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.memory: ReplayBuffer = load_from_pkl(path + "/" + replay_buffer_file)
        print("Checkpoint successfully loaded, you can resume the training now.")

    def run(self):
        # Either create a new SummaryWriter or resume from previous one
        if not self.is_loaded_from_checkpoint:
            current_datetime = datetime.now()
            self.start_datetime = current_datetime
            self.start_time = current_datetime.timestamp()

        start_datetime_str = self.start_datetime.strftime("%d-%m-%Y_%H:%M:%S")
        self.writer = SummaryWriter(f"{self.LOGS_PATH}/{self.name}_{start_datetime_str}")

        video_folder_path = f"{self.VIDEO_PATH}/{self.name}_{start_datetime_str}"
        if not os.path.exists(video_folder_path):
            os.makedirs(video_folder_path)
        self.env.video_folder = video_folder_path

        while self.timesteps < self.MAX_TIMESTEPS:
            state, _ = self.env.reset()
            done = False

            while not done:
                self.timesteps += 1

                action = self.act(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                self.remember(state, next_state, action, reward, terminated, info)

                if self.timesteps >= self.LEARNING_STARTS and self.timesteps % self.TRAIN_FREQUENCY == 0:
                    self.optimize_model()

                state = next_state

                if done:
                    mean_reward = np.mean(self.env.return_queue)
                    mean_length = np.mean(self.env.length_queue)

                    # Get episode statistics from info ("episode" key only exist when episode is done)
                    episode_reward = info["episode"]["r"]
                    self.writer.add_scalar("rollout/episodic_return", episode_reward, self.timesteps)
                    self.writer.add_scalar("rollout/episodic_length", info["episode"]["l"], self.timesteps)

                    self.writer.add_scalar("rollout/ep_rew_mean", mean_reward, self.timesteps)
                    self.writer.add_scalar("rollout/ep_len_mean", mean_length, self.timesteps)

                    self.writer.add_scalar("rollout/exploration_rate", self.epsilon, self.timesteps)

                    print(
                        f"Episode {self.env.episode_count} finished (timesteps: {self.timesteps}/{self.MAX_TIMESTEPS})\n"
                        f"Epsilon: {self.epsilon:.2f}, Episode reward: {episode_reward.item()}, Mean reward: {mean_reward:.2f}")

                    if self.env.episode_count % self.CHECKPOINT_INTERVAL_EPISODE == 0:
                        self.save_checkpoint()
                    print("***************************")

                if self.timesteps >= self.LEARNING_STARTS and self.timesteps % self.TARGET_UPDATE_INTERVAL == 0:
                    self.update_target_network()
                    # print("Target model updated.")

        self.save_checkpoint()  # Save last checkpoint at the end of training

        self.writer.flush()
        self.writer.close()
