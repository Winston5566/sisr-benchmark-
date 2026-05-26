"""
RCAN — Image Super-Resolution Using Very Deep Residual Channel Attention Networks
Zhang et al., ECCV 2018  |  https://arxiv.org/abs/1807.02758

Input:  LR image   [B, C, H, W]
Output: HR image   [B, C, H*s, W*s]
Params: ~15.6 M  (n_groups=10, n_rcab=20, n_feats=64)

Architecture: Residual-in-Residual (RIR)
  Head → [ResidualGroup × G] → long skip → Tail
  Each ResidualGroup: [RCAB × B] → short skip
  Each RCAB: Conv-ReLU-Conv + ChannelAttention → short skip

Key design choice:
  Channel Attention (CA) uses global average pooling to squeeze spatial
  information, then a bottleneck FC pair to produce per-channel rescaling
  weights, enabling the network to focus on high-frequency features.
"""
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, n_feats: int, reduction: int = 16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(n_feats, n_feats // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats // reduction, n_feats, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.gap(x))


class RCAB(nn.Module):
    """Residual Channel Attention Block."""
    def __init__(self, n_feats: int, reduction: int = 16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            ChannelAttention(n_feats, reduction),
        )

    def forward(self, x):
        return x + self.body(x)


class ResidualGroup(nn.Module):
    """G residual groups, each with B RCABs and a short skip connection."""
    def __init__(self, n_feats: int, n_rcab: int, reduction: int = 16):
        super().__init__()
        blocks = [RCAB(n_feats, reduction) for _ in range(n_rcab)]
        blocks.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1))
        self.body = nn.Sequential(*blocks)

    def forward(self, x):
        return x + self.body(x)


class RCAN(nn.Module):
    def __init__(self, scale: int = 4, num_channels: int = 1,
                 n_groups: int = 10, n_rcab: int = 20,
                 n_feats: int = 64, reduction: int = 16):
        super().__init__()
        self.head = nn.Conv2d(num_channels, n_feats, kernel_size=3, padding=1)

        groups = [ResidualGroup(n_feats, n_rcab, reduction) for _ in range(n_groups)]
        groups.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1))
        self.body = nn.Sequential(*groups)

        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * scale * scale, kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(n_feats, num_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        head = self.head(x)
        return self.tail(head + self.body(head))   # long skip over all groups
