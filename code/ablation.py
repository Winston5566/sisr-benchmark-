"""
Ablation studies for SISR architecture design choices.

Studies implemented:
  1. edsr_depth   — vary number of residual blocks (8 / 16 / 32)
  2. edsr_width   — vary feature channel width (32 / 64 / 128 / 256)
  3. swinir_window — vary window size (4 / 8 / 16)
  4. rcan_groups  — vary number of residual groups (4 / 6 / 8 / 10)

Each ablation trains the variant for --epochs epochs on DIV2K and evaluates
PSNR/SSIM on Set5. Results are saved to results/ablation_<study>.csv.

Usage:
    python ablation.py --study edsr_depth   --data-dir data/DIV2K/train_HR \
                       --test-dir data/benchmarks --scale 4 --epochs 100

    python ablation.py --study edsr_width   --data-dir ... --epochs 100
    python ablation.py --study swinir_window --data-dir ... --epochs 50
    python ablation.py --study rcan_groups  --data-dir ... --epochs 100
"""
import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.edsr   import EDSR
from models.swinir import SwinIR
from models.rcan   import RCAN
from data          import DIV2KDataset, SRBenchmark
from utils         import psnr, ssim

PRE_UPSAMPLE = {'srcnn', 'vdsr'}


# ── Ablation configs ──────────────────────────────────────────────────────────

STUDIES = {
    'edsr_depth': [
        {'label': 'EDSR-8',  'model': EDSR, 'kwargs': {'n_resblocks':  8, 'n_feats': 64}},
        {'label': 'EDSR-16', 'model': EDSR, 'kwargs': {'n_resblocks': 16, 'n_feats': 64}},
        {'label': 'EDSR-32', 'model': EDSR, 'kwargs': {'n_resblocks': 32, 'n_feats': 64}},
    ],
    'edsr_width': [
        {'label': 'EDSR-32ch',  'model': EDSR, 'kwargs': {'n_resblocks': 16, 'n_feats':  32}},
        {'label': 'EDSR-64ch',  'model': EDSR, 'kwargs': {'n_resblocks': 16, 'n_feats':  64}},
        {'label': 'EDSR-128ch', 'model': EDSR, 'kwargs': {'n_resblocks': 16, 'n_feats': 128}},
        {'label': 'EDSR-256ch', 'model': EDSR, 'kwargs': {'n_resblocks': 16, 'n_feats': 256}},
    ],
    'swinir_window': [
        {'label': 'SwinIR-W4',  'model': SwinIR, 'kwargs': {'window_size':  4}},
        {'label': 'SwinIR-W8',  'model': SwinIR, 'kwargs': {'window_size':  8}},
        {'label': 'SwinIR-W16', 'model': SwinIR, 'kwargs': {'window_size': 16}},
    ],
    'rcan_groups': [
        {'label': 'RCAN-G4',  'model': RCAN, 'kwargs': {'n_groups':  4}},
        {'label': 'RCAN-G6',  'model': RCAN, 'kwargs': {'n_groups':  6}},
        {'label': 'RCAN-G8',  'model': RCAN, 'kwargs': {'n_groups':  8}},
        {'label': 'RCAN-G10', 'model': RCAN, 'kwargs': {'n_groups': 10}},
    ],
}


# ── Training / eval loop ──────────────────────────────────────────────────────

def run_variant(cfg: dict, args) -> dict:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = cfg['model'](scale=args.scale, num_channels=3, **cfg['kwargs']).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    train_set = DIV2KDataset(args.data_dir, scale=args.scale, patch_size=48)
    loader    = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                           num_workers=4, pin_memory=True, drop_last=True)

    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = MultiStepLR(optimizer,
                            milestones=[args.epochs // 2, args.epochs * 3 // 4],
                            gamma=0.5)
    criterion = nn.L1Loss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for lr_imgs, hr_imgs in tqdm(loader, desc=f"{cfg['label']} ep{epoch}/{args.epochs}",
                                     leave=False):
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            optimizer.zero_grad()
            criterion(model(lr_imgs), hr_imgs).backward()
            optimizer.step()
        scheduler.step()

    # Evaluate on Set5
    test_dir = Path(args.test_dir) / 'Set5'
    test_set = SRBenchmark(str(test_dir), scale=args.scale)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    total_psnr = total_ssim = 0.0
    model.eval()
    with torch.no_grad():
        for lr_img, hr_img, _ in test_loader:
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            sr = model(lr_img)
            total_psnr += psnr(sr.squeeze(0), hr_img.squeeze(0), scale=args.scale)
            total_ssim += ssim(sr.squeeze(0), hr_img.squeeze(0), scale=args.scale)
    n = len(test_set)
    return {
        'label':    cfg['label'],
        'params_M': f'{n_params/1e6:.2f}',
        'psnr':     f'{total_psnr/n:.2f}',
        'ssim':     f'{total_ssim/n:.4f}',
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    if args.study not in STUDIES:
        raise ValueError(f"Unknown study '{args.study}'. Choose from {list(STUDIES)}")

    configs = STUDIES[args.study]
    rows    = []

    print(f"\nAblation: {args.study}  ({len(configs)} variants × {args.epochs} epochs)\n")
    print(f"{'Variant':<16} {'Params (M)':>12} {'PSNR Set5':>12} {'SSIM Set5':>10}")
    print('-' * 54)

    for cfg in configs:
        result = run_variant(cfg, args)
        rows.append(result)
        print(f"{result['label']:<16} {result['params_M']:>12} "
              f"{result['psnr']:>12} {result['ssim']:>10}")

    out = Path('results')
    out.mkdir(exist_ok=True)
    csv_path = out / f'ablation_{args.study}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['label', 'params_M', 'psnr', 'ssim'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved → {csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SISR ablation studies')
    parser.add_argument('--study',      required=True,
                        choices=list(STUDIES), help='Which ablation to run')
    parser.add_argument('--scale',      type=int,   default=4)
    parser.add_argument('--data-dir',   required=True)
    parser.add_argument('--test-dir',   required=True)
    parser.add_argument('--epochs',     type=int,   default=100)
    parser.add_argument('--batch-size', type=int,   default=16)
    args = parser.parse_args()
    main(args)
