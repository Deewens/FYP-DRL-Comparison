from typing import TypedDict
from torch import nn


class HParams(TypedDict):
    env_seed: int
    learning_rate: float
    loss: nn.Module
    max_timesteps: int
    target_update_interval: int
    learning_starts: int
    train_frequency: int
    replay_buffer_size: int
    batch_size: int
    discount_rate: float
    exploration_fraction: float
    exploration_final: float
