# Supervisor Meeting Summary

Quick recap’ of meeting with supervisor when required.

# 22/11/2022

- Have the question to be more “specific”.
    - On what context am I going to compare CNN and ViT exactly?
    - On what specific part of a game am I going to do the comparison? (Detecting bug in the image? Take a decision for an NPC? Detect danger for an NPC in an image? Things like that). I need to have more specific things on what I am going to research
- On what exactly am I going to compare the neural networks?
- Backpropagation and reinforcement learning are two different ways of learning for a NN. These two things are used for the AI to take decisions.
    - Backpropagation uses static data already known
    - Reinforcement learning generates these data while learning on the game environment at runtime
- How am I going to compare?
    - Compare regular ANN against CNN and against ViT using back propagation
    - Compare the same but with reinforcement learning instead
- Start a tutorial on [https://gymnasium.farama.org/environments/atari/](https://gymnasium.farama.org/environments/atari/) to understand how it works
- Do a comparison using gymnasium stuff between ANN and CNN/ViT using back propagation

# 21/03/2023

- Summarise a “story” for my reasearch, from what I started, step in the middle and conclusion
    - Reinforcement Learning Implementation
    - Deep Q-Learning (using a simple FeedForward and CNN)
    - Implentation of the Vision Transformer (⇒ Not working, not learning)
    - Looking at more complex Vision Transformer (Swin Transformer, etc). Mix of CNN with Transformer Architecture
    So goal would be to explain why Vision Transformer did not work compared to a combination of both
    - Implentation of one of this Vision Transformer algorithm on Q-Learning Algorithm
- Understand and summarise how a Transformer is working, explain the relationship and how we came from simple Transformer to Vision Transformer
- Then, from Vision Transformer, explain the architecture of the simple one, explain why it does not work well with RL in Atari env
- Explain the architecture for a Vision Tranformer mixed with CNN (or something else) and their relationship with the basic Vision Transformer