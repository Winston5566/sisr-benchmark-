"""
HAT — Activating More Pixels in Image Super-Resolution Transformer
Chen et al., CVPR 2023  |  https://arxiv.org/abs/2205.04437

Input:  LR image   [B, C, H, W]
Output: HR image   [B, C, H*s, W*s]
Params: ~20.8 M

Architecture: Head → [RHAG × N] → norm → conv → long skip → Upsample
  RHAG (Residual Hybrid Attention Group):
    [HAB × depth] → OCAB → conv → short skip
  HAB (Hybrid Attention Block):
    Window-MSA + Channel Attention + MLP
  OCAB (Overlapping Cross-Attention Block):
    Cross-attention with overlap_ratio-enlarged key/value windows

Key design choice:
  Attribution analysis showed SwinIR only activates pixels within a small
  local region (comparable to CNNs). HAT adds channel attention inside each
  block and cross-window communication via OCAB, activating a much wider
  pixel region and achieving state-of-the-art PSNR at CVPR 2023.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .swinir import window_partition, window_reverse, WindowAttention


# ── Channel Attention ─────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    def __init__(self, dim: int, reduction: int = 16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W]"""
        return x * self.fc(self.gap(x))


# ── HAB ───────────────────────────────────────────────────────────────────────

class HAB(nn.Module):
    """Hybrid Attention Block: W-MSA → Channel-Attn → MLP (all with residual)."""

    def __init__(self, dim: int, num_heads: int, window_size: int = 8,
                 shift_size: int = 0, mlp_ratio: float = 4., reduction: int = 16):
        super().__init__()
        self.dim         = dim
        self.window_size = window_size
        self.shift_size  = shift_size

        self.norm1  = nn.LayerNorm(dim)
        self.w_attn = WindowAttention(dim, window_size, num_heads)
        self.norm2  = nn.LayerNorm(dim)
        self.c_attn = ChannelAttention(dim, reduction)
        self.norm3  = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def _mask(self, H: int, W: int, device) -> torch.Tensor | None:
        if self.shift_size == 0:
            return None
        img = torch.zeros(1, H, W, 1, device=device)
        ws, ss = self.window_size, self.shift_size
        for hi, hs in enumerate([slice(0,-ws), slice(-ws,-ss), slice(-ss,None)]):
            for wi, ws_ in enumerate([slice(0,-ws), slice(-ws,-ss), slice(-ss,None)]):
                img[:, hs, ws_, :] = hi * 3 + wi
        wins = window_partition(img, ws).view(-1, ws * ws)
        m = wins.unsqueeze(1) - wins.unsqueeze(2)
        return m.masked_fill(m != 0, -100.0).masked_fill(m == 0, 0.0)

    def _window_attn(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Apply windowed self-attention on [B, H*W, C], return same shape."""
        B, _, C = x.shape
        x2d = x.view(B, H, W, C)

        pb = (self.window_size - H % self.window_size) % self.window_size
        pr = (self.window_size - W % self.window_size) % self.window_size
        if pb or pr:
            x2d = F.pad(x2d, (0, 0, 0, pr, 0, pb))
        _, Hp, Wp, _ = x2d.shape

        if self.shift_size > 0:
            x2d = torch.roll(x2d, (-self.shift_size, -self.shift_size), dims=(1, 2))

        mask = self._mask(Hp, Wp, x2d.device)
        ws   = self.window_size
        wins = window_partition(x2d, ws).view(-1, ws * ws, C)
        wins = self.w_attn(wins, mask=mask)
        x2d  = window_reverse(wins.view(-1, ws, ws, C), ws, Hp, Wp)

        if self.shift_size > 0:
            x2d = torch.roll(x2d, (self.shift_size, self.shift_size), dims=(1, 2))
        if pb or pr:
            x2d = x2d[:, :H, :W, :].contiguous()
        return x2d.view(B, H * W, C)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, _, C = x.shape

        # 1. Window self-attention
        x = x + self._window_attn(self.norm1(x), H, W)

        # 2. Channel attention (operates on 2-D feature map)
        x_img = x.transpose(1, 2).view(B, C, H, W)
        n2    = self.norm2(x).transpose(1, 2).view(B, C, H, W)
        x     = x + (self.c_attn(n2) - n2).flatten(2).transpose(1, 2)

        # 3. MLP
        x = x + self.mlp(self.norm3(x))
        return x


# ── OCAB ──────────────────────────────────────────────────────────────────────

class OCAB(nn.Module):
    """Overlapping Cross-Attention Block.

    Each query window attends to an enlarged (overlap_ratio × size) key/value
    window centred at the same location, enabling cross-window communication.
    """

    def __init__(self, dim: int, num_heads: int, window_size: int = 8,
                 overlap_ratio: float = 0.5, mlp_ratio: float = 4.):
        super().__init__()
        self.window_size  = window_size
        self.overlap_size = int(window_size * (1 + overlap_ratio))
        self.num_heads    = num_heads
        head_dim          = dim // num_heads
        self.scale        = head_dim ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.q     = nn.Linear(dim, dim)
        self.kv    = nn.Linear(dim, dim * 2)
        self.proj  = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

        # Unfold for extracting overlapping patches
        self.unfold = nn.Unfold(
            kernel_size=(self.overlap_size, self.overlap_size),
            stride=window_size,
            padding=(self.overlap_size - window_size) // 2,
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, _, C = x.shape
        shortcut = x
        x_norm = self.norm1(x)

        # Queries from non-overlapping windows
        x2d = x_norm.view(B, H, W, C)
        pb  = (self.window_size - H % self.window_size) % self.window_size
        pr  = (self.window_size - W % self.window_size) % self.window_size
        if pb or pr:
            x2d = F.pad(x2d, (0, 0, 0, pr, 0, pb))
        _, Hp, Wp, _ = x2d.shape

        ws  = self.window_size
        nWh = Hp // ws
        nWw = Wp // ws
        nW  = nWh * nWw

        q_wins = window_partition(x2d, ws).view(B * nW, ws * ws, C)
        q = self.q(q_wins)  # [B*nW, ws², C]

        # Keys / values from overlapping windows via unfold
        x_img = x2d.permute(0, 3, 1, 2)  # [B, C, Hp, Wp]
        kv_unf = self.unfold(x_img)       # [B, C*os², nW]
        os  = self.overlap_size
        kv_unf = kv_unf.view(B, C, os * os, nW).permute(0, 3, 2, 1)  # [B, nW, os², C]
        kv_unf = kv_unf.reshape(B * nW, os * os, C)
        kv = self.kv(kv_unf).reshape(B * nW, os * os, 2, self.num_heads, C // self.num_heads)
        k, v = kv.permute(2, 0, 3, 1, 4).unbind(0)  # each [B*nW, h, os², C//h]

        q = q.reshape(B * nW, ws * ws, self.num_heads, C // self.num_heads).transpose(1, 2)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        out  = (attn @ v).transpose(1, 2).reshape(B * nW, ws * ws, C)
        out  = self.proj(out)

        # Reconstruct spatial feature map
        out_2d = (out.view(B, nW, ws * ws, C)
                     .view(B, nWh, nWw, ws, ws, C)
                     .permute(0, 1, 3, 2, 4, 5)
                     .reshape(B, Hp, Wp, C))
        if pb or pr:
            out_2d = out_2d[:, :H, :W, :].contiguous()

        x = shortcut + out_2d.view(B, H * W, C)
        x = x + self.mlp(self.norm2(x))
        return x


# ── RHAG ──────────────────────────────────────────────────────────────────────

class RHAG(nn.Module):
    """Residual Hybrid Attention Group: depth HABs + 1 OCAB + conv + skip."""

    def __init__(self, dim: int, num_heads: int, window_size: int = 8,
                 depth: int = 6, mlp_ratio: float = 4.):
        super().__init__()
        self.habs = nn.ModuleList([
            HAB(dim, num_heads, window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                mlp_ratio=mlp_ratio)
            for i in range(depth)
        ])
        self.ocab = OCAB(dim, num_heads, window_size, mlp_ratio=mlp_ratio)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        feat = x.flatten(2).transpose(1, 2)
        for hab in self.habs:
            feat = hab(feat, H, W)
        feat = self.ocab(feat, H, W)
        feat = feat.transpose(1, 2).view(B, C, H, W)
        return self.conv(feat) + x


# ── HAT ───────────────────────────────────────────────────────────────────────

class HAT(nn.Module):
    """Hybrid Attention Transformer for classical image super-resolution."""

    def __init__(self, scale: int = 4, num_channels: int = 1,
                 embed_dim: int = 180,
                 depths: tuple = (6, 6, 6, 6, 6, 6),
                 num_heads: tuple = (6, 6, 6, 6, 6, 6),
                 window_size: int = 8,
                 mlp_ratio: float = 2.):
        super().__init__()
        self.scale       = scale
        self.window_size = window_size

        self.conv_first  = nn.Conv2d(num_channels, embed_dim, kernel_size=3, padding=1)
        self.rhags       = nn.ModuleList([
            RHAG(embed_dim, num_heads[i], window_size, depths[i], mlp_ratio)
            for i in range(len(depths))
        ])
        self.norm             = nn.LayerNorm(embed_dim)
        self.conv_after_body  = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
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
        for rhag in self.rhags:
            deep = rhag(deep)

        B, C, Hp, Wp = deep.shape
        deep = self.norm(deep.flatten(2).transpose(1, 2)).transpose(1, 2).view(B, C, Hp, Wp)
        deep = self.conv_after_body(deep) + feat

        out = self.upsample(deep)
        return out[:, :, :H * self.scale, :W * self.scale]
