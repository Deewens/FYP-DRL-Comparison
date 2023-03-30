import datetime
import pickle
import json
import os

import tensorflow as tf

import numpy as np

from environment import make_env
from replay_buffer import ReplayBuffer
from networks import create_cnn_model, create_swin_model


def linear_schedule(start_epsilon: float, end_epsilon: float, duration: int, timestep: int):
    slope = (end_epsilon - start_epsilon) / duration
    return max(slope * timestep + start_epsilon, end_epsilon)


class DoubleDQNAgent:
    def __init__(self, env):
        self.env = env

        # Constants
        self.NUM_ACTIONS = self.env.action_space.n
        self.DISCOUNT_FACTOR = 0.99  # Discount factor gamma used in the Q-learning update
        self.REPLAY_START_SIZE = 80000  # The agent is run for this number of steps before the training start. The resulting experience is used to populate the replay memory
        # self.REPLAY_START_SIZE = 50 # Small value used for quick debugging
        self.FINAL_EXPLORATION_STEP = 1000000  # Number of frames over which the initial value of epsilon is linearly annealed to its final value.

        self.TOTAL_TIMESTEPS = 10000000  # Number of frames over which the initial value of epsilon is linearly annealed to its final value.

        self.EXPLORATION_FRACTION = 0.10  # The fraction of 'TOTAL_TIMESTEPS' it takes from 'EPSILON_START' to 'EPSILON_END'.
        self.INITIAL_EXPLORATION = 1.0  # Initial value of epsilon in Epsilon-Greedy exploration
        self.FINAL_EXPLORATION = 0.01  # Final value of epsilon in Epsilon-Greedy exploration

        self.REPLAY_MEMORY_SIZE = 10000
        self.MINIBATCH_SIZE = 32
        self.TARGET_NETWORK_UPDATE_FREQUENCY = 1000  # The frequency with which the tqrget netzork is updqted (measured in the number of parameter updates)
        self.LEARNING_RATE = 1e-04
        self.UPDATE_FREQUENCY = 4

        # self.buffer = ReplayMemory(self.REPLAY_MEMORY_SIZE)
        self.replay_buffer = ReplayBuffer(
            self.REPLAY_MEMORY_SIZE,
            env.observation_space,
            env.action_space,
            optimize_memory_usage=True,
            handle_timeout_termination=False)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE)
        self.loss_function = tf.keras.losses.MeanSquaredError()

        self.model = create_swin_model(self.NUM_ACTIONS)
        self.target_model = create_swin_model(self.NUM_ACTIONS)
        self.target_model.set_weights(self.model.get_weights())

        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, network=self.model,
                                              target_network=self.target_model)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, "checkpoints", max_to_keep=5)

        self.saved_epsilon = None
        self.saved_episode_idx = None
        self.saved_timestep = None
        self.saved_episode_reward_history = None

    def train(self, from_checkpoint=False):
        if from_checkpoint:
            epsilon = self.saved_epsilon
            episode_idx = self.saved_episode_idx
            timestep = self.saved_timestep
            episode_reward_history = self.saved_episode_reward_history
        else:
            epsilon = self.INITIAL_EXPLORATION
            episode_idx = 0
            timestep = 0
            episode_reward_history = []

        while timestep < self.TOTAL_TIMESTEPS:  # Run until max number of step has been reached
            episode_idx += 1
            episode_reward = 0
            done = False

            state = np.array(self.env.reset()[0])

            while not done:
                timestep += 1
                action = self.choose_action(state, epsilon)

                # Reduce epsilon only if training started
                if timestep > self.REPLAY_START_SIZE:
                    epsilon = linear_schedule(self.INITIAL_EXPLORATION, self.FINAL_EXPLORATION,
                                              int(self.EXPLORATION_FRACTION * self.TOTAL_TIMESTEPS),
                                              timestep)

                next_state, reward, terminated, truncated, info = self.env.step(action)
                next_state = np.array(next_state)
                done = terminated or truncated

                episode_reward += reward
                # self.buffer.append(Experience(state, action, reward, done, next_state))
                self.replay_buffer.add(state, next_state, action, reward, done, info)
                state = next_state

                if timestep % 1000 == 0:
                    print("Timestep: {}, Epsilon: {}, Current episode reward: {}".format(timestep, epsilon, episode_reward))

                # Only train if done observing (buffer has been filled enough)
                if timestep % self.UPDATE_FREQUENCY == 0 and timestep > self.REPLAY_START_SIZE:
                    # states_sample, actions_sample, rewards_sample, dones_sample, next_states_sample = self.buffer.sample(
                    #    self.MINIBATCH_SIZE)

                    # states_sample_tensor = tf.convert_to_tensor(states_sample)
                    # next_states_sample_tensor = tf.convert_to_tensor(next_states_sample)
                    # actions_sample_tensor = tf.convert_to_tensor(actions_sample)

                    observations_sample, actions_sample, next_observations_sample, dones_sample, rewards_sample = self.replay_buffer.sample(
                        self.MINIBATCH_SIZE)
                    # Flatten actions, dones and rewards
                    actions_sample = tf.reshape(actions_sample, [-1])
                    dones_sample = tf.reshape(dones_sample, [-1])
                    rewards_sample = tf.reshape(rewards_sample, [-1])

                    # Perform experience replay
                    # Predict the target q value from the next sample sates
                    target_q_values = self.test_step(next_observations_sample)

                    # print(f"target_q_values: {target_q_values}")
                    # print(target_q_values.shape)

                    # Calculate the target q values by discounting the discount rate from the Q Value predicted by
                    # the target model (1 - minibatch.done) will be 0 if this is the terminated state, and thus,
                    # won't update the q_learning target (because 0 * x = 0) reduce_max get the maximum Q_value for
                    # each list of q_values returned by the target model (because we gave a batch of 32 states to the
                    # model)
                    target_q_values = rewards_sample + (1 - dones_sample) * self.DISCOUNT_FACTOR * tf.reduce_max(
                        target_q_values, axis=1)

                    # print(f"target_q_values after: {target_q_values}")
                    # print(target_q_values.shape)

                    # Create a mask on the action stores in the sampled minibatch This allows us to only calculate
                    # the loss on the updated Q-values WHAT IS A ONE_HOT Tensor? A one hot tensor is a matrix
                    # representation of a categorical variable, where the matrix has a single 1 in the column
                    # corresponding to the category and all other entries are 0. In other words, a one-hot tensor is
                    # a vector of length equal to the number of categories with a single 1 in the position
                    # corresponding to the category and all other values as 0.
                    masks = tf.one_hot(actions_sample, self.NUM_ACTIONS)

                    self.train_step(observations_sample, target_q_values, masks)

                    # Now, we need to calculate the gardient descent, using GradientTape to record the operation made
                    # during the training of the Q Function network (main model) As stated above, GradientTape just
                    # record the operation made inside it, such as model training or calculation
                    # with tf.GradientTape() as tape:
                    #     # We train the main network and record the training into the tape
                    #     q_values = self.model(observations_sample, training=True)
                    #
                    #     # Apply the masks to the Q-values to get the Q-value only for taken action from the minibatch
                    #     masked_q_values = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    #     loss = self.loss_function(target_q_values, masked_q_values)
                    #
                    # # We can then performe the back propagation on te taped operation made while training the network
                    # # Backpropagation
                    # gradients = tape.gradient(loss, self.model.trainable_variables)
                    # self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                if timestep % self.TARGET_NETWORK_UPDATE_FREQUENCY == 0:
                    # update the target network with weights from the main network
                    self.target_model.set_weights(self.model.get_weights())
                    print("Target network updated")

                # Save model every 100,000 iterations
                if timestep % 100000 == 0:
                    print("Saving checkpoint...")
                    self.save_checkpoint(epsilon, episode_idx, timestep, episode_reward_history)
                    print("Checkpoint saved!")

            # Update running reward to check condition for solving
            episode_reward_history.append(episode_reward)
            if len(episode_reward_history) > 100:
                del episode_reward_history[:1]
            mean_reward = np.mean(episode_reward_history)

            print("Episode {} finished!".format(episode_idx))
            print("Episode reward: {}, Mean reward: {}".format(episode_reward, mean_reward))
            print("******************")

    def choose_action(self, state, epsilon):
        # Use epsilon greedy policy to select actions.
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            # Predict action Q-values
            state_tensor = tf.convert_to_tensor(
                state)  # Convert the state numpy array to a tensor array because tensorflow only accept tensor
            state_tensor = tf.expand_dims(state_tensor,
                                          0)  # Add one dimension to the state because this is a batch of only one state
            q_values = self.model(state_tensor,
                                  training=False)  # Call the model to predict the Q-Value according to the passed state

            # Take best action from the returned q_values
            # tf.argmax return the index with the largest q_values
            action = tf.argmax(q_values[0]).numpy()  # convert it back to np because it returns a Tensor

        return action

    @tf.function
    def train_step(self, observations, target_q_values, masks):
        # Now, we need to calculate the gardient descent, using GradientTape to record the operation made
        # during the training of the Q Function network (main model) As stated above, GradientTape just
        # record the operation made inside it, such as model training or calculation
        with tf.GradientTape() as tape:
            # We train the main network and record the training into the tape
            q_values = self.model(observations, training=True)

            # Apply the masks to the Q-values to get the Q-value only for taken action from the minibatch
            masked_q_values = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            loss = self.loss_function(target_q_values, masked_q_values)

        # We can then performe the back propagation on te taped operation made while training the network
        # Backpropagation
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    @tf.function
    def test_step(self, next_observations):
        # Perform experience replay
        # Predict the target q value from the next sample sates
        target_q_values = self.target_model(next_observations, training=False)
        return target_q_values

    def save_checkpoint(self, epsilon, episode_idx, timestep, episode_reward_history):
        save_datetime = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
        checkpoint_dir = 'checkpoints/data/' + save_datetime
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.checkpoint_manager.save()

        # Save replay memory into a binary file using Pickle
        with open(checkpoint_dir + '/replay_memory_' + save_datetime, 'wb') as file:
            pickle.dump(self.replay_buffer, file)

        # Save hyperparameters used to train the model into a JSON file:
        data = {
            'datetime': save_datetime,
            'epsilon': epsilon,
            'episode_idx': episode_idx,
            'timestep': timestep,
            'episode_reward_history': episode_reward_history
        }

        with open(checkpoint_dir + '/data_' + save_datetime + '.json', 'w') as file:
            json.dump(data, file, indent=4)

    def load_checkpoint(self, directory):
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)

        with open(directory + '/replay_memory', 'rb') as file:
            self.replay_buffer = pickle.load(file)

        with open(directory + '/data.json', 'r') as file:
            data = json.load(file)
            self.saved_epsilon = data['epsilon']
            self.saved_timestep = data['timestep']
            self.saved_episode_idx = data['episode_idx']
            self.saved_episode_reward_history = data['episode_reward_history']
