# FYP: Comparison of Deep RL neural networks

The purpose of this project is to compare the traditional Convolutional Neural Network against Swin Transformer in the
context of Deep Reinforcement Learning.

## Overview

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

## Documents
I exported some of the documentation from my Notion, a note taking-app I used throughout the project.

- [Notion Link](https://nifty-garlic-a2b.notion.site/Fourth-Year-Project-709413d18af747268ef9964444851f5c)
- [Thesis, Software Requirements Specification, Technical Design Document](docs)
- [Guides](docs/Guides): technical guides I wrote
- [Notes and Knowledge](docs/Notes%20and%20Knowledges): contains the courses and thoughts I wrote to myself during the project

## Requirements and Usages

### System requirements

- **Linux** - Ubuntu-based system
- Python 3.8
- NVIDIA GPU

**Required for TensorFlow 2.12**

- NVIDIA GPU drivers version 450.80.02 or higher
- CUDA Toolkit 11.8
- cuDNN SDK 8.6.0

See my installation guides that I made to install CUDA in `docs/guides`. Only needed for a TensorFlow
installation. Because I switched my code to PyTorch, it is not needed any more (except if you want to try and start my
TensorFlow code)

- [How to install CUDA 11.8 (TensorFlow 2.12)](docs/guides/How%20to%20install%20CUDA%2011.8.md)
- [How to install CUDA 11.2 (TensorFlow 2.11)](docs/guides/How%20to%20install%20CUDA%2011.2.md)

> For **PyTorch 2.0**, you **do not need** to install the above requirements yourself, as they are already included in
> the PyTorch binaries.

### Python Packages

You should ideally uses a conda environment to install these package. PyTorch can be installed using `conda`, the other
should be installed with `pip`

| Package           | Version | Description                                                                                                   |
|-------------------|---------|---------------------------------------------------------------------------------------------------------------|
| PyTorch           | 2.0     | Machine learning library ([Installation guide](https://pytorch.org/get-started/locally/#linux-prerequisites)) |
| Gymnasium         | 0.28.1  | Provides a standard API for RL, and the Atari environments                                                    |
| Stable Baselines3 | 1.8.0   | Set of implementation or RL algorithms, including DQN                                                         |
| Transformers      | 4.28.1  | Collection of Transformers NN, including Swin Transformer                                                     |
| Moviepy           | -       | To record video from environment                                                                              |

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