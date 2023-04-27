# Double DQN

```python
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(64, input_dim=self.state_dim, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.action_dim, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def replay(self, batch_size, memory):
        minibatch = memory.sample(batch_size)
        states, actions, rewards, next_states, dones = minibatch

        q_values_next = self.target_model.predict(next_states)
        max_actions = tf.argmax(q_values_next, axis=1)

        q_values = self.model.predict(states)
        target_q_values = q_values.copy()
        targets = rewards + (1 - dones) * self.discount_factor * q_values_next[np.arange(batch_size), max_actions]
        target_q_values[np.arange(batch_size), actions] = targets

        self.model.fit(states, target_q_values, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
```

Slighly better implementation with better comments and variable names (chatGPT)

```python
import tensorflow as tf
import numpy as np

class DoubleDQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Main network and target network
        self.main_network = self._build_network()
        self.target_network = self._build_network()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def _build_network(self):
        # Define the neural network architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_dim=self.state_dim, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation='linear')
        ])
        model.compile(loss='mse', optimizer=self.optimizer)
        return model

    def update_target_network(self):
        # Update target network weights with main network weights
        self.target_network.set_weights(self.main_network.get_weights())

    def select_action(self, state):
        # Choose action with epsilon-greedy policy
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        q_values = self.main_network.predict(state)
        return np.argmax(q_values[0])

    def replay(self, batch_size, memory):
        # Sample a minibatch from memory
        states, actions, rewards, next_states, dones = memory.sample(batch_size)

        # Compute target Q-values using the target network
        q_values_next = self.target_network.predict(next_states)
        max_actions = np.argmax(q_values_next, axis=1)
        q_values_target = np.zeros((batch_size, self.action_dim))
        q_values_target[np.arange(batch_size), max_actions] = 1
        target_q_values = rewards + (1 - dones) * self.discount_factor * np.sum(q_values_next * q_values_target, axis=1)

        # Compute current Q-values using the main network
        q_values = self.main_network.predict(states)

        # Update Q-values using Double DQN update rule
        q_values_target = q_values.copy()
        q_values_target[np.arange(batch_size), actions] = target_q_values
        self.main_network.fit(states, q_values_target, epochs=1, verbose=0)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
```

# Why do we have two “implementations” of Double DQN?

We have two implementation of double DQN, one predicting the q_values using the target network on the “next_states” and one predicting the q_values on the “next_states” using both the target network and the main network.

Response from ChatGPT about that:

> Yes, using only one set of predictions from both the main network and the target network is sufficient for implementing the Double DQN algorithm. The main network is used to predict the Q-values for the current state and the next state, while the target network is used to predict the maximum Q-value for the next state.
The second implementation that you mentioned, which predicts Q-values three times, is simply an alternative way of implementing the Double DQN algorithm that uses the main network to predict the Q-values for the next state as well as the current state. The first approach may provide some benefits in terms of computational efficiency, as it avoids the need to make additional calls to the target network.
Both implementations are valid and should produce similar results, as long as the Double DQN update rule is correctly applied. The choice between the two approaches may depend on factors such as the size of the network and the computational resources available for training.
> 

So, basically, the other implementation is jsut a slighly different way of implementing Double DQN

Below, implementation of Double DQN using main network two times (from ChatGPT)

```python
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(64, input_dim=self.state_dim, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.action_dim, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def replay(self, batch_size, memory):
        minibatch = memory.sample(batch_size)
        states, actions, rewards, next_states, dones = minibatch

        target = self.model.predict(states)
        target_next = self.model.predict(next_states)
        target_val = self.target_model.predict(next_states)

        max_actions = tf.argmax(target_next, axis=1)

        targets = rewards + (1 - dones) * self.discount_factor * target_val[np.arange(batch_size), max_actions]
        target[np.arange(batch_size), actions] = targets

        self.model.fit(states, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
```

Same implementaiton with better comments and variable name (from chatgpt)

```python
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class DoubleDQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Create main and target networks
        self.model = self._build_model()
        self.target_model = self._build_model()

        # Set up optimizer and loss function
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(64, input_dim=self.state_dim, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.action_dim, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        # Update target network with weights from main network
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        # Select action using epsilon-greedy policy
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def replay(self, batch_size, memory):
        # Sample a minibatch of experiences from memory
        minibatch = memory.sample(batch_size)
        states, actions, rewards, next_states, dones = minibatch

        # Predict Q-values for next states using main network
        main_q_values_next = self.model.predict(next_states)

        # Predict Q-values for next states using target network
        target_q_values_next = self.target_model.predict(next_states)

        # Select the action with the highest Q-value from the main network for each next state
        main_max_actions = np.argmax(main_q_values_next, axis=1)

        # Use the target network to calculate the maximum Q-value for each next state and action pair
        target_max_q_values_next = target_q_values_next[np.arange(batch_size), main_max_actions]

        # Predict Q-values for current states using main network
        main_q_values = self.model.predict(states)

        # Create a copy of the main network's Q-values for the current states
        target_q_values = main_q_values.copy()

        # Calculate targets for each experience in the minibatch using the Double DQN update rule
        targets = rewards + (1 - dones) * self.discount_factor * target_max_q_values_next

        # Update the target Q-values for the actions taken in each experience in the minibatch
        target_q_values[np.arange(batch_size), actions] = targets

        # Train the main network on the minibatch of experiences
        self.model.fit(states, target_q_values, epochs=1, verbose=0)

        # Update epsilon for epsilon-greedy policy
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
```