"""
ESPCN — Real-Time Single Image and Video Super-Resolution Using an
        Efficient Sub-Pixel Convolutional Neural Network
Shi et al., CVPR 2016  |  https://arxiv.org/abs/1609.05158

Input:  LR image   [B, C, H, W]        (no pre-upsampling)
Output: HR image   [B, C, H*s, W*s]
Params: ~20 K

Key design choice:
  - PixelShuffle (sub-pixel convolution) avoids checkerboard artefacts
    produced by transposed convolutions and enables real-time video SR.
"""
import torch.nn as nn


class ESPCN(nn.Module):
    def __init__(self, scale: int = 4, num_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            # Output r²·C feature maps, then rearrange into HR space
            nn.Conv2d(32, num_channels * scale * scale, kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
        )

    def forward(self, x):
        return self.net(x)
