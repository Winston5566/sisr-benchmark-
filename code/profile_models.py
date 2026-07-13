"""
Profile all SISR models: parameter count and inference speed.

Outputs a formatted table to stdout and saves results/efficiency.csv.

Usage:
    python profile_models.py --scale 4 --input-size 64 64 --repeats 50
"""
import argparse
import csv
import time
from pathlib import Path

import torch

from models import get_model

MODELS = ['srcnn', 'fsrcnn', 'vdsr', 'espcn', 'edsr', 'rcan', 'swinir', 'hat', 'drct', 'ipt']
PRE_UPSAMPLE = {'srcnn', 'vdsr'}


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_time(model: torch.nn.Module, inp: torch.Tensor,
                 device: torch.device, repeats: int) -> float:
    """Return mean inference time in milliseconds."""
    model.eval()
    # Warm-up
    with torch.no_grad():
        for _ in range(5):
            _ = model(inp)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(repeats):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(inp)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    return sum(times) / len(times)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  Input LR size: {args.input_size}  |  Scale: ×{args.scale}\n")

    H, W   = args.input_size
    lr_inp = torch.randn(1, 3, H, W).to(device)
    bi_inp = torch.nn.functional.interpolate(
        lr_inp, scale_factor=args.scale, mode='bicubic', align_corners=False
    )

    rows = []
    header = f"{'Model':<10} {'Params (M)':>12} {'Time (ms)':>12} {'Input type':<16}"
    print(header)
    print('-' * len(header))

    for name in MODELS:
        try:
            model = get_model(name, scale=args.scale, num_channels=3).to(device)
            n_params = count_params(model)
            inp = bi_inp if name in PRE_UPSAMPLE else lr_inp
            ms  = measure_time(model, inp, device, args.repeats)
            input_type = 'bicubic up' if name in PRE_UPSAMPLE else 'LR direct'
            print(f"{name.upper():<10} {n_params/1e6:>12.2f} {ms:>12.1f} {input_type:<16}")
            rows.append({
                'model': name.upper(),
                'params_M': f'{n_params/1e6:.2f}',
                'time_ms': f'{ms:.1f}',
                'input_type': input_type,
            })
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"{name.upper():<10} {'ERROR':>12}  {e}")

    # Save CSV
    out = Path('results')
    out.mkdir(exist_ok=True)
    csv_path = out / 'efficiency.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'params_M', 'time_ms', 'input_type'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved → {csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Profile SISR model efficiency')
    parser.add_argument('--scale',      type=int,   default=4)
    parser.add_argument('--input-size', type=int,   nargs=2, default=[64, 64],
                        metavar=('H', 'W'), help='LR input spatial size')
    parser.add_argument('--repeats',    type=int,   default=50,
                        help='Number of forward passes for timing average')
    args = parser.parse_args()
    main(args)
