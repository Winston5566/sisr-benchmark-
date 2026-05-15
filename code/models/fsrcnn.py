"""
FSRCNN — Accelerating the Super-Resolution Convolutional Neural Network
Dong et al., ECCV 2016  |  https://arxiv.org/abs/1608.00367

Input:  LR image   [B, C, H, W]        (no pre-upsampling)
Output: HR image   [B, C, H*s, W*s]
Params: ~12 K  (d=56, s=12, m=4)
"""
import torch.nn as nn


class FSRCNN(nn.Module):
    def __init__(self, scale: int = 4, num_channels: int = 1,
                 d: int = 56, s: int = 12, m: int = 4):
        super().__init__()
        self.feature_extract = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=2),
            nn.PReLU(d),
        )
        self.shrink = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU(s),
        )
        mapping = []
        for _ in range(m):
            mapping += [nn.Conv2d(s, s, kernel_size=3, padding=1), nn.PReLU(s)]
        self.map = nn.Sequential(*mapping)
        self.expand = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU(d),
        )
        # Transposed conv for sub-pixel upsampling
        self.deconv = nn.ConvTranspose2d(
            d, num_channels,
            kernel_size=9, stride=scale,
            padding=4, output_padding=scale - 1,
        )

    def forward(self, x):
        x = self.feature_extract(x)
        x = self.shrink(x)
        x = self.map(x)
        x = self.expand(x)
        return self.deconv(x)
