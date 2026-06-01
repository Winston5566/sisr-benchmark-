"""
IPT — Pre-Trained Image Processing Transformer
Chen et al., CVPR 2021  |  https://arxiv.org/abs/2012.00364

Input:  LR image   [B, C, H, W]
Output: HR image   [B, C, H*s, W*s]
Params: ~115.5 M

Architecture (single-task SR variant):
  Head (task-specific conv) → PatchEmbed → Transformer encoder (ViT body)
  → PatchUnembed → Tail (task-specific conv + PixelShuffle)

Original IPT trains on ImageNet with multiple tasks (SR ×2/×3/×4,
denoising, deraining) sharing one transformer body and using separate
head/tail pairs per task. This implementation provides the SR-only variant
matching the paper's architecture.

Note on usage:
  IPT's 115M parameters make training from scratch impractical on a single
  GPU. The recommended workflow is to load official pre-trained weights:
    model = IPT(scale=4)
    state = torch.load('ipt_pretrain.pt', map_location='cpu')
    model.load_state_dict(state, strict=False)
"""
import torch
import torch.nn as nn


# ── Transformer components ────────────────────────────────────────────────────

class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden     = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


# ── Patch utilities ───────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """Split feature map into non-overlapping patches → token sequence."""
    def __init__(self, patch_size: int = 4, dim: int = 64, embed_dim: int = 1024):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(dim, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        x = self.proj(x)                          # [B, E, H/p, W/p]
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)          # [B, N, E]
        return self.norm(x), Hp, Wp


class PatchUnembed(nn.Module):
    """Token sequence → feature map."""
    def __init__(self, patch_size: int = 4, embed_dim: int = 1024, dim: int = 64):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.ConvTranspose2d(embed_dim, dim,
                                       kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor, Hp: int, Wp: int) -> torch.Tensor:
        B, N, E = x.shape
        x = x.transpose(1, 2).view(B, E, Hp, Wp)
        return self.proj(x)


# ── Positional Encoding ───────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 10000):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embed[:, :x.shape[1], :]


# ── IPT ───────────────────────────────────────────────────────────────────────

class IPT(nn.Module):
    """Image Processing Transformer — SR-only variant.

    Recommended configurations (from paper):
      patch_size=4, embed_dim=64, transformer_dim=1024,
      num_heads=8, num_layers=12  → ~115 M params

    For inference only, load official pre-trained weights:
      torch.load('ipt_sr_x4.pt')
    """

    def __init__(self, scale: int = 4, num_channels: int = 1,
                 patch_size: int = 4,
                 feat_dim: int = 64,
                 embed_dim: int = 1024,
                 num_heads: int = 8,
                 num_layers: int = 12,
                 mlp_ratio: float = 4.):
        super().__init__()
        self.scale      = scale
        self.patch_size = patch_size

        # Task-specific head
        self.head = nn.Sequential(
            nn.Conv2d(num_channels, feat_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1),
        )

        # Shared transformer body
        self.patch_embed = PatchEmbed(patch_size, feat_dim, embed_dim)
        self.pos_enc     = PositionalEncoding(embed_dim)
        self.transformer = nn.Sequential(*[
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])
        self.patch_unembed = PatchUnembed(patch_size, embed_dim, feat_dim)

        # Task-specific tail
        self.tail = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim * scale * scale, kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(feat_dim, num_channels, kernel_size=3, padding=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape

        # Pad to patch_size multiple
        ph = (self.patch_size - H % self.patch_size) % self.patch_size
        pw = (self.patch_size - W % self.patch_size) % self.patch_size
        x  = F.pad(x, (0, pw, 0, ph), mode='reflect') if (ph or pw) else x

        import torch.nn.functional as F  # local import avoids circular issue
        feat = self.head(x)

        tokens, Hp, Wp = self.patch_embed(feat)
        tokens = self.pos_enc(tokens)
        for layer in self.transformer:
            tokens = layer(tokens)
        feat = self.patch_unembed(tokens, Hp, Wp)

        out = self.tail(feat)
        return out[:, :, :H * self.scale, :W * self.scale]
