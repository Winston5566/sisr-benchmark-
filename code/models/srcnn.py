"""
SRCNN — Learning a Deep Convolutional Network for Image Super-Resolution
Dong et al., ECCV 2014  |  https://arxiv.org/abs/1501.00092

Input:  bicubic-upsampled LR image  [B, C, H*s, W*s]
Output: HR image                     [B, C, H*s, W*s]
Params: ~57 K
"""
import torch.nn as nn


class SRCNN(nn.Module):
    def __init__(self, num_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_channels, kernel_size=5, padding=2),
        )

    def forward(self, x):
        return self.net(x)
