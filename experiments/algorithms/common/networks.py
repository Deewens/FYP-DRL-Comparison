import torch.nn as nn
import torch.nn.functional as F

from transformers import SwinModel, SwinConfig


class CNNDQN(nn.Module):
    def __init__(self, num_channels, num_actions):
        super(CNNDQN, self).__init__()

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


class SwinDQN(nn.Module):
    def __init__(self, num_channels, num_actions):
        super(SwinDQN, self).__init__()

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
