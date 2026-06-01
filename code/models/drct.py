"""
DRCT — Saving Image Super-Resolution away from Information Bottleneck
Hsu et al., CVPR Workshops 2024  |  https://arxiv.org/abs/2404.00722

Input:  LR image   [B, C, H, W]
Output: HR image   [B, C, H*s, W*s]
Params: ~14.1 M

Architecture: Head → [DenseRSTB × N] → norm → conv → long skip → Upsample
  DenseRSTB (Dense Residual Swin Transformer Block):
    [SwinTransformerLayer × depth] with dense connections → conv → skip

Key design choice:
  In standard SwinIR/HAT stacks, feature-map intensities decay toward the
  network tail, creating an information bottleneck. DRCT counters this by
  adding DenseNet-style connections: each STL's output is summed into ALL
  subsequent STL inputs within the block, preserving high-frequency spatial
  detail at every depth.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .swinir import SwinTransformerLayer


# ── Dense RSTB ────────────────────────────────────────────────────────────────

class DenseRSTB(nn.Module):
    """Residual Swin Transformer Block with dense skip connections.

    For depth STLs, the output of STL_i is added to the input of every
    subsequent STL_{i+1}, …, STL_{depth-1} (DenseNet-style within block).
    A final conv + outer residual mirrors the standard RSTB structure.
    """

    def __init__(self, dim: int, num_heads: int, window_size: int = 8,
                 depth: int = 6, mlp_ratio: float = 4.):
        super().__init__()
        self.layers = nn.ModuleList([
            SwinTransformerLayer(
                dim, num_heads, window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                mlp_ratio=mlp_ratio,
            )
            for i in range(depth)
        ])
        # Per-layer projection to mix accumulated dense features back to dim
        self.dense_projs = nn.ModuleList([
            nn.Linear(dim, dim) if i > 0 else nn.Identity()
            for i in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        feat = x.flatten(2).transpose(1, 2)   # [B, H*W, C]

        # Dense connections: accumulate previous outputs into current input
        accumulated = feat
        for i, (layer, proj) in enumerate(zip(self.layers, self.dense_projs)):
            feat = layer(accumulated, H, W)
            # Add current output to the running accumulation for all future layers
            accumulated = accumulated + proj(feat)

        feat = accumulated.transpose(1, 2).view(B, C, H, W)
        return self.conv(feat) + x   # outer short skip


# ── DRCT ──────────────────────────────────────────────────────────────────────

class DRCT(nn.Module):
    """Dense-Residual-Connected Transformer for image super-resolution."""

    def __init__(self, scale: int = 4, num_channels: int = 1,
                 embed_dim: int = 180,
                 depths: tuple = (6, 6, 6, 6, 6, 6),
                 num_heads: tuple = (6, 6, 6, 6, 6, 6),
                 window_size: int = 8,
                 mlp_ratio: float = 2.):
        super().__init__()
        self.scale       = scale
        self.window_size = window_size

        self.conv_first = nn.Conv2d(num_channels, embed_dim, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([
            DenseRSTB(embed_dim, num_heads[i], window_size, depths[i], mlp_ratio)
            for i in range(len(depths))
        ])
        self.norm            = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * scale * scale, kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(embed_dim, num_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        ph = (self.window_size - H % self.window_size) % self.window_size
        pw = (self.window_size - W % self.window_size) % self.window_size
        x  = F.pad(x, (0, pw, 0, ph), mode='reflect')

        feat = self.conv_first(x)
        deep = feat
        for block in self.blocks:
            deep = block(deep)

        B, C, Hp, Wp = deep.shape
        deep = self.norm(deep.flatten(2).transpose(1, 2)).transpose(1, 2).view(B, C, Hp, Wp)
        deep = self.conv_after_body(deep) + feat

        out = self.upsample(deep)
        return out[:, :, :H * self.scale, :W * self.scale]
