from typing import Tuple
from app.utils.typing import EnvConfig, AgentConfig

import os

from tkinter import *
from tkinter import ttk

from PIL import Image, ImageTk

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack, RecordEpisodeStatistics

from app.networks import SwinMicrosoftDQN, ConvDQN, SwinHuggingFaceDQN

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EnvironmentDisplayFrame(ttk.LabelFrame):
    def __init__(self, master: Misc, *args, **kwargs):
        super(EnvironmentDisplayFrame, self).__init__(master, *args, **kwargs)

        self.IMAGE_SIZE = (160 * 5, 240 * 5)

        self.padding = int(str(self.cget("padding")[0]))
        self.is_playing = False

        self.columnconfigure(0, weight=1, minsize=self.IMAGE_SIZE[0])
        self.rowconfigure(0, weight=1, minsize=self.IMAGE_SIZE[1])

        self.agent = None
        self.image = None  # Keep reference to the image during showcase_app lifetime. Required by the Label widget
        self.render_window = ttk.Label(self, image=self.image, anchor="center")

        self.render_window.grid(column=0, row=0, sticky="nwes")

    def setup_agent(self, env_config: EnvConfig, agent_config: AgentConfig):
        self.agent = Agent(env_config["env_id"], env_config, agent_config)
        self.update_image()

    def update_image(self):
        self.image = self.agent.render_as_tk_image(self.IMAGE_SIZE)
        self.render_window["image"] = self.image

    def toggle_playing(self, playing=True):
        self.is_playing = playing
        self.render_env()

    def render_env(self):
        if self.is_playing:
            self.agent.step()
            self.update_image()
            self.after(1, self.render_env)


def make_env(env_name="ALE/Pong-v5", seed=42):
    env = gym.make(env_name, render_mode="rgb_array", autoreset=True, full_action_space=False, frameskip=1)
    env = AtariPreprocessing(env)
    env = FrameStack(env, 4)
    env = RecordEpisodeStatistics(env)

    env.observation_space.seed(seed)
    env.action_space.seed(seed)

    env.metadata['render_fps'] = 30

    return env


class Agent:
    def __init__(self, env_name: str, env_config: EnvConfig, agent_config: AgentConfig):
        self.network_type = agent_config["type"]
        self.writer = SummaryWriter(f"logs/{env_config['env_name']}")
        self.timesteps = 0

        assert self.network_type in ["MicrosoftSwin", "HuggingFaceSwin", "StableBaselines3Swin", "Conv", "Random"], \
            f"{self.network_type} does not exist. Network type can only be 'Conv', 'MicrosoftSwin', " \
            f"'StableBaselines3Swin' or 'HuggingFaceSwin', 'Random'."

        if self.network_type == "StableBaselines3Swin":
            self.env = make_atari_env(env_name, n_envs=1, seed=42,
                                      env_kwargs={"full_action_space": False, "frameskip": 1})
            self.env = VecFrameStack(self.env, n_stack=4)
            self.model = DQN.load(os.path.join(env_config["path"], agent_config["path"]))
            self.model.verbose = 0
            self.model.set_env(self.env)
            self.model.policy.eval()

            self.obs = self.reset()
        elif self.network_type == "Random":
            self.env = make_env(env_name, 42)
            self.obs = self.reset()
        else:
            self.env = make_env(env_name, 42)
            if self.network_type == "MicrosoftSwin":
                self.network = SwinMicrosoftDQN(4, self.env.action_space.n).to(device)
            elif self.network_type == "HuggingFaceSwin":
                self.network = SwinHuggingFaceDQN(4, self.env.action_space.n).to(device)
            else:
                self.network = ConvDQN(4, self.env.action_space.n).to(device)

            state_dict = torch.load(os.path.join(env_config["path"], agent_config["path"]))
            self.network.load_state_dict(state_dict)
            self.network.eval()

            self.obs = self.reset()

    def render_as_tk_image(self, img_size: Tuple[int, int] = None):
        env_img = self.env.render()
        pillow_img = Image.fromarray(env_img)

        if img_size is not None:
            pillow_img = pillow_img.resize(img_size)

        return ImageTk.PhotoImage(pillow_img)

    def reset(self):
        if self.network_type == "StableBaselines3Swin":
            self.obs = self.model.env.reset()
        else:
            self.obs, _ = self.env.reset()

        return self.obs

    def __log_scalar(self, episode_reward, episode_length):
        self.writer.add_scalar("rollout/episodic_return", episode_reward, self.timesteps)
        self.writer.add_scalar("rollout/episodic_length", episode_length, self.timesteps)

    def step(self):
        self.timesteps += 1

        if self.network_type == "StableBaselines3Swin":
            action, _states = self.model.predict(self.obs, deterministic=True)
            self.obs, reward, done, info = self.model.env.step(action)
            #done = terminated or truncated
            # if done:
            #     episode_reward = info["episode"]["r"]
            #     episode_length = info["episode"]["l"]
            #     self.__log_scalar(episode_reward, episode_length)
        elif self.network_type == "Random":
            self.obs, reward, terminated, truncated, info = self.env.step(self.env.action_space.sample())
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(np.array(self.obs), device=device).unsqueeze(0)
                q_values = self.network(obs_tensor)
                action = q_values.argmax(dim=1)[0].cpu().numpy()

            self.obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            # if done:
            #     episode_reward = info["episode"]["r"]
            #     episode_length = info["episode"]["l"]
            #     self.__log_scalar(episode_reward, episode_length)

        return self.obs
