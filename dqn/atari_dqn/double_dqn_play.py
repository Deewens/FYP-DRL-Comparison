import tensorflow as tf

import numpy as np

from double_dqn import make_env, create_q_model


class DQNAgentPlayer:
    def __init__(self):
        self.env = make_env()

        self.NUM_ACTIONS = self.env.action_space.n

        self.model = create_q_model(self.NUM_ACTIONS)

        self.checkpoint = tf.train.Checkpoint(network=self.model)

    def play(self):
        observation, info = self.env.reset(seed=42)
        observation = np.array(observation)
        for _ in range(100000):
            action = self.choose_action(observation)
            observation, reward, terminated, truncated, info = self.env.step(action)
            observation = np.array(observation)

            if terminated or truncated:
                observation, info = self.env.reset()
                observation = np.array(observation)

    def choose_action(self, state):
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

    def load_checkpoint(self):
        self.checkpoint.restore(tf.train.latest_checkpoint('checkpoints'))


agent = DQNAgentPlayer()
agent.load_checkpoint()
agent.play()
