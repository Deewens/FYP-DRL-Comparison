import random
from collections import namedtuple
import numpy as np

import tensorflow as tf

import gymnasium as gym

env = gym.make('FrozenLake-v1', render_mode='ansi')

num_episodes = 100_000
TOTAL_TIMESTEPS = 10000000  # Number of frames over which the initial value of epsilon is linearly annealed to its final value.

LEARNING_RATE = 1e-04
DISCOUNT_RATE = 0.99  # Gamma. We choose 0.99 as an arbitrary value

MAX_EXPLORATION_RATE = 1
MIN_EXPLORATION_RATE = 0.01
EXPLORATION_FRACTION = 0.10  # The fraction of 'TOTAL_TIMESTEPS' it takes from 'EPSILON_START' to 'EPSILON_END'.

exploration_rate_decay = 0.01  # We choose 0.001 to decay the exploration rate, again it is arbitrary.

episode_reward_history = []

MINIBATCH_SIZE = 32
REPLAY_START_SIZE = 80000
REPLAY_MEMORY_SIZE = 10000
TARGET_NETWORK_UPDATE_FREQUENCY = 1000

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss_function = tf.keras.losses.MeanSquaredError()

replay_buffer = namedtuple()


def linear_schedule(start_epsilon: float, end_epsilon: float, duration: int, timestep: int):
    slope = (end_epsilon - start_epsilon) / duration
    return max(slope * timestep + start_epsilon, end_epsilon)


def create_q_model(input_shape, num_actions):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,)),
        tf.keras.layers.Dense(24, activation="relu"),
        tf.keras.layers.Dense(12, activation="relu"),
        tf.keras.layers.Dense(num_actions, activation="linear")
    ])

    return model


model = create_q_model(env.observation_space, env.action_space.n)
target_model = create_q_model(env.observation_space, env.action_space.n)
target_model.set_weights(model.get_weights())


def choose_action(state, epsilon):
    exploration_rate_threshold = random.uniform(0, 1)  # Generate a random number between 0 and 1 (0 and 1 are included)

    if exploration_rate_threshold > epsilon:
        state = tf.expand_dims(state, 0)  # Add one dimension to the state because this is a batch of only one state
        q_values = model(state, training=False)  # Call the model to predict the Q-Value according to the passed state
        action = tf.argmax(q_values[0]).numpy()
    else:
        action = env.action_space.sample()

    return action


def training_loop():
    exploration_rate = 1
    step = 0

    for episode in range(num_episodes):
        done = False
        state, _ = env.reset()
        reward_current_episode = 0

        while not done:
            step += 1

            action = choose_action(state, exploration_rate)

            if step > REPLAY_START_SIZE:
                print(env.render())
                exploration_rate = linear_schedule(MAX_EXPLORATION_RATE, MIN_EXPLORATION_RATE,
                                          int(EXPLORATION_FRACTION * TOTAL_TIMESTEPS), step)

            # We can now take the action
            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            reward_current_episode += reward
            replay_buffer.add(state, new_state, action, reward, done, info)

            if step > REPLAY_START_SIZE:
                observations_sample, actions_sample, next_observations_sample, dones_sample, rewards_sample = replay_buffer.sample(
                    MINIBATCH_SIZE)

                # Flatten actions, dones and rewards
                actions_sample = tf.reshape(actions_sample, [-1])
                dones_sample = tf.reshape(dones_sample, [-1])
                rewards_sample = tf.reshape(rewards_sample, [-1])

                target_q_values = target_model(next_observations_sample, training=False)
                target_q_values = rewards_sample + (1 - dones_sample) * DISCOUNT_RATE * tf.reduce_max(target_q_values,
                                                                                                      axis=1)

                masks = tf.one_hot(actions_sample, env.action_space.n)

                with tf.GradientTape() as tape:
                    # We train the main network and record the training into the tape
                    q_values = model(observations_sample, training=True)

                    # Apply the masks to the Q-values to get the Q-value only for taken action from the minibatch
                    masked_q_values = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    loss = loss_function(target_q_values, masked_q_values)

                # We can then performe the back propagation on te taped operation made while training the network
                # Backpropagation
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                if step % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
                    target_model.set_weights(model.get_weights())

            state = new_state  # The new state is now the current state

        episode_reward_history.append(reward_current_episode)
        if len(episode_reward_history) > 100:
            del episode_reward_history[:1]
        mean_reward = np.mean(episode_reward_history)

        print(f"Episode {episode} finished (step: {step}). Reward: {reward_current_episode} - Mean reward: {mean_reward} - Eps: {exploration_rate}")
        print("******************")

training_loop()