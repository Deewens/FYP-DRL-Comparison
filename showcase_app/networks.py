import torch
import torch.nn as nn
import torch.nn.functional as F

from swin_transformer.models.swin_transformer import SwinTransformer
from transformers import SwinModel, SwinConfig

from stable_baselines3.dqn.policies import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import is_image_space

from gymnasium import spaces


class ConvDQN(nn.Module):
    def __init__(self, num_channels, num_actions):
        super(ConvDQN, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, 32, stride=4, kernel_size=8)
        self.conv2 = nn.Conv2d(32, 64, stride=2, kernel_size=4)
        self.conv3 = nn.Conv2d(64, 64, stride=1, kernel_size=3)
        self.flatten = nn.Flatten()

        # 64 * 7 * 7 is the output of last conv layer, for an input of 84*84
        self.linear = nn.Linear(64 * 7 * 7, 512)
        self.head = nn.Linear(512, num_actions)  # Head layer

    def forward(self, x):
        x = x.float() / 255  # Rescale input from [0, 255] to [0, 1]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.linear(x))
        return self.head(x)


class SwinMicrosoftDQN(nn.Module):
    def __init__(self, num_channels, num_actions):
        super(SwinMicrosoftDQN, self).__init__()

        self.swin = SwinTransformer(
            img_size=84,
            in_chans=num_channels,
            num_classes=num_actions,
            depths=[2, 3, 2],
            num_heads=[3, 3, 6],
            patch_size=3,
            window_size=7,
            embed_dim=96,
            mlp_ratio=4,
            drop_path_rate=0.1
        )

    def forward(self, x):
        x = x.float() / 255  # Rescale input from [0, 255] to [0, 1]
        return self.swin(x)


swin_config = SwinConfig(
    image_size=84,
    patch_size=3,
    num_channels=4,
    embed_dim=96,
    depths=[2, 3, 2],
    num_heads=[3, 3, 6],
    window_size=7,
    mlp_ratio=4.0,
    drop_path_rate=0.1,
)


class SwinHuggingFaceDQN(nn.Module):
    def __init__(self, num_channels, num_actions):
        super(SwinHuggingFaceDQN, self).__init__()

        config = SwinConfig(
            image_size=84,
            patch_size=3,
            num_channels=num_channels,
            embed_dim=96,
            depths=[2, 3, 2],
            num_heads=[3, 3, 6],
            window_size=7,
            mlp_ratio=4.0,
            drop_path_rate=0.1,
        )
        self.swin = SwinModel(config)
        self.head = nn.Linear(384, num_actions)

    def forward(self, x):
        x = x.float() / 255  # Rescale input from [0, 255] to [0, 1]
        x = self.swin(x).pooler_output
        return self.head(x)


class SB3SwinFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 384, normalized_image: bool = False, ):
        assert isinstance(observation_space, spaces.Box), (
            "SwinDQN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image), (
            "You should use SwinDQN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using `VecNormalize` or already normalized channel-first images "
            "you should pass `normalize_images=False`: \n"
            "https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html"
        )

        self.swin = SwinModel(swin_config)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.swin(observations).pooler_output
