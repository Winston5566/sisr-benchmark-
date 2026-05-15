"""
Train a SISR model on DIV2K.

Usage:
    python train.py --model srcnn  --data-dir data/DIV2K/train_HR --scale 4
    python train.py --model fsrcnn --data-dir data/DIV2K/train_HR --scale 4 --epochs 300
    python train.py --model edsr   --data-dir data/DIV2K/train_HR --scale 4 --epochs 300 --batch-size 16
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from models import get_model
from data import DIV2KDataset

# Models that need bicubic-upsampled input (pre-upsampling architectures)
PRE_UPSAMPLE = {'srcnn', 'vdsr'}


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[train] model={args.model.upper()}  scale=×{args.scale}  device={device}")

    dataset = DIV2KDataset(args.data_dir, scale=args.scale, patch_size=args.patch_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)

    model = get_model(args.model, scale=args.scale).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] trainable parameters: {n_params:,}")

    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    milestones = [args.epochs // 2, args.epochs * 3 // 4]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    criterion = nn.L1Loss()

    ckpt_dir = Path('checkpoints') / args.model
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for lr_imgs, hr_imgs in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            if args.model in PRE_UPSAMPLE:
                lr_imgs = F.interpolate(lr_imgs, scale_factor=args.scale,
                                        mode='bicubic', align_corners=False)

            optimizer.zero_grad()
            sr_imgs = model(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)
            loss.backward()

            # Gradient clipping stabilises VDSR training (high learning rate)
            if args.model == 'vdsr':
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.4)

            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        if epoch % args.log_every == 0:
            avg = running_loss / len(loader)
            lr_now = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:4d}/{args.epochs}  loss={avg:.4f}  lr={lr_now:.2e}")

        if epoch % args.save_every == 0:
            path = ckpt_dir / f'epoch_{epoch:04d}.pth'
            torch.save(model.state_dict(), path)

    final_path = ckpt_dir / 'final.pth'
    torch.save(model.state_dict(), final_path)
    print(f"[train] saved → {final_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CNN-based SISR models on DIV2K')
    parser.add_argument('--model',      required=True, choices=['srcnn','fsrcnn','vdsr','espcn','edsr'])
    parser.add_argument('--scale',      type=int,   default=4)
    parser.add_argument('--data-dir',   required=True, help='Path to DIV2K HR training images')
    parser.add_argument('--epochs',     type=int,   default=300)
    parser.add_argument('--batch-size', type=int,   default=16)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--patch-size', type=int,   default=96)
    parser.add_argument('--log-every',  type=int,   default=10)
    parser.add_argument('--save-every', type=int,   default=50)
    args = parser.parse_args()
    train(args)
