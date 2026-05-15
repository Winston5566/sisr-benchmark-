"""
Dataset classes for SISR training and evaluation.

Training : DIV2KDataset  — random patch crop + bicubic degradation + augmentation
Evaluation: SRBenchmark  — full-image bicubic degradation, no augmentation
"""
import random
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class DIV2KDataset(Dataset):
    """DIV2K HR images with on-the-fly bicubic LR generation.

    Args:
        hr_dir    : path to folder containing HR PNG images
        scale     : downscaling factor (2, 3, or 4)
        patch_size: HR patch size (must be divisible by scale)
        augment   : whether to apply random flip / 90° rotation
    """

    def __init__(self, hr_dir: str, scale: int = 4,
                 patch_size: int = 96, augment: bool = True):
        self.scale = scale
        self.patch_size = patch_size
        self.augment = augment
        self.paths = sorted(Path(hr_dir).glob('*.png'))
        if not self.paths:
            raise FileNotFoundError(f"No PNG images found in {hr_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        hr = Image.open(self.paths[idx]).convert('RGB')
        w, h = hr.size

        # Random HR patch
        x0 = random.randint(0, w - self.patch_size)
        y0 = random.randint(0, h - self.patch_size)
        hr = hr.crop((x0, y0, x0 + self.patch_size, y0 + self.patch_size))

        # Bicubic downscale to LR
        lr_size = self.patch_size // self.scale
        lr = hr.resize((lr_size, lr_size), Image.BICUBIC)

        if self.augment:
            if random.random() > 0.5:
                hr, lr = TF.hflip(hr), TF.hflip(lr)
            if random.random() > 0.5:
                hr, lr = TF.vflip(hr), TF.vflip(lr)
            k = random.randint(0, 3)
            if k:
                hr = TF.rotate(hr, 90 * k)
                lr = TF.rotate(lr, 90 * k)

        return TF.to_tensor(lr), TF.to_tensor(hr)


class SRBenchmark(Dataset):
    """Standard SR benchmark datasets (Set5, Set14, BSD100, Urban100).

    Images are loaded as HR, cropped to be divisible by scale, then
    bicubic-downsampled to produce LR. No augmentation is applied.

    Args:
        hr_dir : path to folder of HR images (.png or .bmp)
        scale  : downscaling factor
    """

    def __init__(self, hr_dir: str, scale: int = 4):
        self.scale = scale
        paths = sorted(Path(hr_dir).glob('*.png'))
        if not paths:
            paths = sorted(Path(hr_dir).glob('*.bmp'))
        if not paths:
            raise FileNotFoundError(f"No images found in {hr_dir}")
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        hr = Image.open(self.paths[idx]).convert('RGB')
        w, h = hr.size
        # Crop so dimensions are exact multiples of scale
        w = (w // self.scale) * self.scale
        h = (h // self.scale) * self.scale
        hr = hr.crop((0, 0, w, h))
        lr = hr.resize((w // self.scale, h // self.scale), Image.BICUBIC)
        return TF.to_tensor(lr), TF.to_tensor(hr), self.paths[idx].name
