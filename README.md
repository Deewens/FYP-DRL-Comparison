# FYP: Comparison of Deep RL neural networks

The purpose of this project is to compare the traditional Convolutional Neural Network against Swin Transformer in the
context of Deep Reinforcement Learning.

## Experiment

In this expriment, I implemented the Double Deep Q Network, and compared the Convolutional Neural Network architecture
against the Swin Transformer architecture to solves the Atari game environment Pong and Breakout. Please read the
project report located in docs/ to have detailed information on the purpose of this experiment.

## Project structure

- **experiments/**: contains the final clean code that have been used to generate the result of the experiment
    - **algorithms/**:  implementation of the Double DQN algorithm and environment preprocessing
    - **prototyping/**: contains some of the code that I used to experiment and prototype. Not everything is there
      because a lot of the code has been scrapped too. It also contains some experiment I have made before
      implementing them to the final algorithm implementation.
- **showcase_app/**: small tkinter application developed only for the purposes of the Industry Showcase Day at the
  college. Showing trained agents playing Breakout and Pong
- **docs/**: contains the documentation for the project including the _Research Report_, the _Technical Design Document_
  and _Software Requirements Specification_.
- **learning/**: I wrote some of the code that I gathered from some of the tutorials and guides I followed at the start
  of the project of whenever I needed to learn a feature of a library to do something

## Experiments

### Hyperparameters

One step in the training process is equivalent to 4 frames in the environment.

| Hyperparameter         | Value      | Description                                                                            |
|------------------------|------------|----------------------------------------------------------------------------------------|
| Optimiser              | Adam       |                                                                                        |
| Learning Rate          | 0.0001     |                                                                                        |
| Loss                   | Smooth L1  |                                                                                        |
| Max timesteps          | 10,000,000 | Agent trained on this number of steps                                                  |
| Target update interval | 1,000      | Number of steps between the synchronisation of the policy and target networks          |
| Learning starts        | 100,000    | Number of steps to wait before we start to optimise the model                          |
| Train frequency        | 4          | Network optimisation is done every `Train frequency` steps                             |
| Replay buffer size     | 10,000     | Size of the experience replay buffer                                                   |
| Batch size             | 32         | Size of minibatch to be used to calculate the expected Q Values and optimise the model |
| Discount rate          | 0.99       | Gamma                                                                                  |
| Exploration fraction   | 0.1        | Fraction of the training steps used for epsilon greedy decay                           |
| Epsilon final          | 0.01       | Minimum exploration value                                                              |
| Frame Stack            | 4          | Number of frames stacked together                                                      |

README will be updated later with more information, to include a summary of the research result. As well as a guide on
how to start the experiments and a list of the technology used. (For now, you can check the Technical Design Document)