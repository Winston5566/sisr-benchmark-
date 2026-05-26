"""
SwinIR — Image Restoration Using Swin Transformer
Liang et al., ICCV Workshops 2021  |  https://arxiv.org/abs/2108.10257

Input:  LR image   [B, C, H, W]
Output: HR image   [B, C, H*s, W*s]
Params (light): ~0.9 M   (embed_dim=60, depths=[6,6,6,6], num_heads=[6,6,6,6])
Params (full) : ~11.8 M  (embed_dim=180, depths=[6,6,6,6,6,6])

Architecture:
  Shallow extraction → [RSTB × N] → norm → conv → long skip → Upsample
  RSTB: [SwinTransformerLayer × depth] → conv → short skip
  SwinTransformerLayer: W-MSA (or SW-MSA) → LayerNorm → MLP

Key design choice:
  Shifted-window attention captures long-range dependencies at O(n) cost
  (vs. O(n²) for global attention), while relative position bias encodes
  spatial structure within each local window.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Window utilities ──────────────────────────────────────────────────────────

def window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    """[B, H, W, C] → [B*nW, ws, ws, C]"""
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)


def window_reverse(windows: torch.Tensor, ws: int, H: int, W: int) -> torch.Tensor:
    """[B*nW, ws, ws, C] → [B, H, W, C]"""
    B = int(windows.shape[0] / (H * W / ws / ws))
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


# ── Window Attention ──────────────────────────────────────────────────────────

class WindowAttention(nn.Module):
    """Multi-head self-attention within non-overlapping windows,
    with learnable relative position bias."""

    def __init__(self, dim: int, window_size: int, num_heads: int,
                 qkv_bias: bool = True, attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5

        # Relative position bias table: [(2W-1)², num_heads]
        table_size = (2 * window_size - 1) ** 2
        self.rel_pos_bias = nn.Parameter(torch.zeros(table_size, num_heads))
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

        # Precompute relative position index
        coords = torch.stack(
            torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing='ij')
        )  # [2, W, W]
        coords_flat = coords.flatten(1)  # [2, W²]
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # [2, W², W²]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer('rel_pos_index', rel.sum(-1))  # [W², W²]

        self.qkv      = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        # Add relative position bias
        rpb = self.rel_pos_bias[self.rel_pos_index.view(-1)]
        rpb = rpb.view(self.window_size**2, self.window_size**2, self.num_heads)
        attn = attn + rpb.permute(2, 0, 1).unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


# ── Swin Transformer Layer ────────────────────────────────────────────────────

class SwinTransformerLayer(nn.Module):
    """One Swin Transformer layer (W-MSA or SW-MSA) + LayerNorm + MLP."""

    def __init__(self, dim: int, num_heads: int, window_size: int = 8,
                 shift_size: int = 0, mlp_ratio: float = 4.):
        super().__init__()
        self.shift_size  = shift_size
        self.window_size = window_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn  = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def _attn_mask(self, H: int, W: int, device) -> torch.Tensor | None:
        if self.shift_size == 0:
            return None
        mask = torch.zeros(1, H, W, 1, device=device)
        ws, ss = self.window_size, self.shift_size
        for hi, hs in enumerate([slice(0, -ws), slice(-ws, -ss), slice(-ss, None)]):
            for wi, ws_ in enumerate([slice(0, -ws), slice(-ws, -ss), slice(-ss, None)]):
                mask[:, hs, ws_, :] = hi * 3 + wi
        wins = window_partition(mask, self.window_size).view(-1, self.window_size**2)
        attn_mask = wins.unsqueeze(1) - wins.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, _, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        # Pad to window multiple
        pb = (self.window_size - H % self.window_size) % self.window_size
        pr = (self.window_size - W % self.window_size) % self.window_size
        if pb or pr:
            x = F.pad(x, (0, 0, 0, pr, 0, pb))
        _, Hp, Wp, _ = x.shape

        # Cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, (-self.shift_size, -self.shift_size), dims=(1, 2))

        # Windowed attention
        mask = self._attn_mask(Hp, Wp, x.device)
        wins = window_partition(x, self.window_size).view(-1, self.window_size**2, C)
        wins = self.attn(wins, mask=mask)
        x = window_reverse(wins.view(-1, self.window_size, self.window_size, C), self.window_size, Hp, Wp)

        # Reverse shift and unpad
        if self.shift_size > 0:
            x = torch.roll(x, (self.shift_size, self.shift_size), dims=(1, 2))
        if pb or pr:
            x = x[:, :H, :W, :].contiguous()

        x = shortcut + x.view(B, H * W, C)
        return x + self.mlp(self.norm2(x))


# ── RSTB ─────────────────────────────────────────────────────────────────────

class RSTB(nn.Module):
    """Residual Swin Transformer Block: depth STLs + conv + short skip."""

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
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        feat = x.flatten(2).transpose(1, 2)          # [B, H*W, C]
        for layer in self.layers:
            feat = layer(feat, H, W)
        feat = feat.transpose(1, 2).view(B, C, H, W) # [B, C, H, W]
        return self.conv(feat) + x


# ── SwinIR ────────────────────────────────────────────────────────────────────

class SwinIR(nn.Module):
    """SwinIR for classical image super-resolution.

    Light config  : embed_dim=60,  depths=(6,6,6,6),       num_heads=(6,6,6,6)
    Default config: embed_dim=180, depths=(6,6,6,6,6,6),   num_heads=(6,6,6,6,6,6)
    """

    def __init__(self, scale: int = 4, num_channels: int = 1,
                 embed_dim: int = 60,
                 depths: tuple = (6, 6, 6, 6),
                 num_heads: tuple = (6, 6, 6, 6),
                 window_size: int = 8,
                 mlp_ratio: float = 2.):
        super().__init__()
        self.scale       = scale
        self.window_size = window_size

        # Shallow feature extraction
        self.conv_first = nn.Conv2d(num_channels, embed_dim, kernel_size=3, padding=1)

        # Deep feature extraction: N RSTBs
        self.rstbs = nn.ModuleList([
            RSTB(embed_dim, num_heads[i], window_size, depths[i], mlp_ratio)
            for i in range(len(depths))
        ])
        self.norm             = nn.LayerNorm(embed_dim)
        self.conv_after_body  = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)

        # Upsampling reconstruction
        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * scale * scale, kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(embed_dim, num_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape

        # Pad so H, W are multiples of window_size
        ph = (self.window_size - H % self.window_size) % self.window_size
        pw = (self.window_size - W % self.window_size) % self.window_size
        x  = F.pad(x, (0, pw, 0, ph), mode='reflect')

        # Shallow features
        feat = self.conv_first(x)

        # Deep features
        deep = feat
        for rstb in self.rstbs:
            deep = rstb(deep)

        # Normalise in sequence form, then reshape back
        B, C, Hp, Wp = deep.shape
        deep = self.norm(deep.flatten(2).transpose(1, 2)).transpose(1, 2).view(B, C, Hp, Wp)

        # Long skip connection + upsampling
        out = self.upsample(self.conv_after_body(deep) + feat)

        # Crop padding artefacts from the output
        return out[:, :, :H * self.scale, :W * self.scale]
