"""
Evaluate a trained SISR model on standard benchmarks.

Usage:
    python evaluate.py --model srcnn --checkpoint checkpoints/srcnn/final.pth \
                       --test-dir data/benchmarks --scale 4
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import get_model
from data import SRBenchmark
from utils import psnr, ssim

PRE_UPSAMPLE = {'srcnn', 'vdsr'}
BENCHMARK_NAMES = ['Set5', 'Set14', 'BSD100', 'Urban100']


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(args.model, scale=args.scale).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    test_root = Path(args.test_dir)
    datasets = {name: test_root / name
                for name in BENCHMARK_NAMES
                if (test_root / name).exists()}

    if not datasets:
        raise FileNotFoundError(f"No benchmark folders found under {test_root}")

    header = f"{'Model':<10} {'Dataset':<12} {'PSNR (dB)':>10} {'SSIM':>8} {'ms/img':>8}"
    print('\n' + header)
    print('-' * len(header))

    for name, path in datasets.items():
        ds = SRBenchmark(str(path), scale=args.scale)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

        total_psnr = total_ssim = total_ms = 0.0

        with torch.no_grad():
            for lr_img, hr_img, _ in loader:
                lr_img = lr_img.to(device)
                hr_img = hr_img.to(device)

                if args.model in PRE_UPSAMPLE:
                    lr_img = F.interpolate(lr_img, scale_factor=args.scale,
                                           mode='bicubic', align_corners=False)

                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                sr_img = model(lr_img)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                total_ms += (time.perf_counter() - t0) * 1000.0

                total_psnr += psnr(sr_img.squeeze(0), hr_img.squeeze(0), scale=args.scale)
                total_ssim += ssim(sr_img.squeeze(0), hr_img.squeeze(0), scale=args.scale)

        n = len(ds)
        print(f"{args.model.upper():<10} {name:<12} "
              f"{total_psnr/n:>10.2f} {total_ssim/n:>8.4f} {total_ms/n:>8.1f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate SISR models on standard benchmarks')
    parser.add_argument('--model',      required=True, choices=['srcnn','fsrcnn','vdsr','espcn','edsr'])
    parser.add_argument('--scale',      type=int,   default=4)
    parser.add_argument('--checkpoint', required=True, help='Path to .pth checkpoint file')
    parser.add_argument('--test-dir',   required=True, help='Root dir containing Set5/, Set14/, etc.')
    args = parser.parse_args()
    evaluate(args)
