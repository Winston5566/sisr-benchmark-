"""
EDSR — Enhanced Deep Residual Networks for Single Image Super-Resolution
Lim et al., CVPR Workshops 2017  |  https://arxiv.org/abs/1707.02921
NTIRE 2017 Super-Resolution Challenge winner.

Input:  LR image   [B, C, H, W]
Output: HR image   [B, C, H*s, W*s]

Configurations:
  EDSR-baseline : n_resblocks=16, n_feats=64   (~1.5 M params)
  EDSR-large    : n_resblocks=32, n_feats=256  (~43 M params)

Key design choice:
  - Batch Normalisation is *removed* from residual blocks; BN normalises
    feature statistics that carry spatial detail needed for reconstruction.
  - Residual scaling (res_scale=0.1) stabilises training of wide networks.
"""
import torch.nn as nn


class ResBlock(nn.Module):
    """Basic residual block without BatchNorm."""
    def __init__(self, n_feats: int, res_scale: float = 0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
        )
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.body(x) * self.res_scale


class EDSR(nn.Module):
    def __init__(self, scale: int = 4, num_channels: int = 1,
                 n_resblocks: int = 16, n_feats: int = 64):
        super().__init__()
        # Shallow feature extraction
        self.head = nn.Conv2d(num_channels, n_feats, kernel_size=3, padding=1)
        # Deep feature extraction
        self.body = nn.Sequential(*[ResBlock(n_feats) for _ in range(n_resblocks)])
        self.body_tail = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)
        # Upsampling + reconstruction
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * scale * scale, kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(n_feats, num_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        head = self.head(x)
        body = self.body_tail(self.body(head))
        return self.tail(head + body)   # long skip connection
