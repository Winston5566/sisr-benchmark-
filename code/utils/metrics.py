"""
PSNR and SSIM on the Y channel (YCbCr), following the standard SR evaluation
convention used by SRCNN, EDSR, SwinIR, etc.

Boundary pixels equal to the upscaling factor are excluded before scoring.
"""
import torch
import torch.nn.functional as F


# ── Colour conversion ─────────────────────────────────────────────────────────

def _rgb_to_y(img: torch.Tensor) -> torch.Tensor:
    """Convert an RGB tensor [..., 3, H, W] → Y channel [..., 1, H, W].
    Coefficients follow the ITU-R BT.601 standard (values in [0, 1]).
    """
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    return 16.0 / 255.0 + (65.481 / 255.0) * r \
                         + (128.553 / 255.0) * g \
                         + (24.966 / 255.0) * b


# ── Public metrics ────────────────────────────────────────────────────────────

def psnr(sr: torch.Tensor, hr: torch.Tensor, scale: int = 4) -> float:
    """Peak Signal-to-Noise Ratio (dB) on the Y channel.

    Args:
        sr, hr : float tensors [C, H, W] in [0, 1]
        scale  : upscaling factor (used as border crop width)
    """
    with torch.no_grad():
        sr = sr.unsqueeze(0).clamp(0.0, 1.0)
        hr = hr.unsqueeze(0).clamp(0.0, 1.0)
        y_sr = _rgb_to_y(sr)
        y_hr = _rgb_to_y(hr)
        # Crop border
        y_sr = y_sr[..., scale:-scale, scale:-scale]
        y_hr = y_hr[..., scale:-scale, scale:-scale]
        mse = F.mse_loss(y_sr, y_hr)
        if mse == 0:
            return float('inf')
        return (-10.0 * torch.log10(mse)).item()


def ssim(sr: torch.Tensor, hr: torch.Tensor, scale: int = 4,
         window_size: int = 11, sigma: float = 1.5) -> float:
    """Structural Similarity Index on the Y channel.

    Args:
        sr, hr       : float tensors [C, H, W] in [0, 1]
        scale        : upscaling factor (border crop)
        window_size  : Gaussian window side length
        sigma        : Gaussian standard deviation
    """
    with torch.no_grad():
        sr = sr.unsqueeze(0).clamp(0.0, 1.0)
        hr = hr.unsqueeze(0).clamp(0.0, 1.0)
        y_sr = _rgb_to_y(sr)[..., scale:-scale, scale:-scale]
        y_hr = _rgb_to_y(hr)[..., scale:-scale, scale:-scale]
        return _ssim(y_sr, y_hr, window_size=window_size, sigma=sigma)


# ── Internal SSIM helper ──────────────────────────────────────────────────────

def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    kernel = g.outer(g)
    return (kernel / kernel.sum()).unsqueeze(0).unsqueeze(0)  # [1,1,K,K]


def _ssim(x: torch.Tensor, y: torch.Tensor,
          window_size: int, sigma: float,
          C1: float = (0.01) ** 2,
          C2: float = (0.03) ** 2) -> float:
    kernel = _gaussian_kernel(window_size, sigma).to(x.device)
    C = x.shape[1]
    kernel = kernel.expand(C, 1, window_size, window_size)
    pad = window_size // 2

    mu_x  = F.conv2d(x, kernel, padding=pad, groups=C)
    mu_y  = F.conv2d(y, kernel, padding=pad, groups=C)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sig_x2  = F.conv2d(x * x, kernel, padding=pad, groups=C) - mu_x2
    sig_y2  = F.conv2d(y * y, kernel, padding=pad, groups=C) - mu_y2
    sig_xy  = F.conv2d(x * y, kernel, padding=pad, groups=C) - mu_xy

    num = (2.0 * mu_xy + C1) * (2.0 * sig_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2)
    return (num / den).mean().item()
