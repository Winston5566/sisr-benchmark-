"""
VDSR — Accurate Image Super-Resolution Using Very Deep Convolutional Networks
Kim et al., CVPR 2016  |  https://arxiv.org/abs/1511.04587

Input:  bicubic-upsampled LR image  [B, C, H*s, W*s]
Output: HR image                     [B, C, H*s, W*s]
Params: ~665 K (20 layers, 64 channels)

Key design choices:
  - Global residual: network predicts the *difference* (HR - bicubic LR)
  - Very high learning rate (0.1) stabilised by gradient clipping (clip=0.4)
  - Multi-scale training: same model handles ×2, ×3, ×4
"""
import torch.nn as nn


class VDSR(nn.Module):
    def __init__(self, num_channels: int = 1, num_layers: int = 20):
        super().__init__()
        layers = [nn.Conv2d(num_channels, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 2):
            layers += [nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(64, num_channels, kernel_size=3, padding=1))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        # Global residual: add bicubic input back to the predicted residual
        return x + self.body(x)
