"""
Generate visual comparison images across all SISR models.

Outputs per input image:
  1. A side-by-side grid PNG  (all models in one figure)
  2. A zoomed patch PNG       (cropped region highlighting texture detail)
  3. An interactive HTML slider (before = Bicubic / after = chosen model)

Usage:
    # Run inference from checkpoints
    python visualize.py --input data/benchmarks/Set5/butterfly.png \
                        --ckpt-dir checkpoints --scale 4 \
                        --models srcnn fsrcnn vdsr espcn edsr rcan swinir

    # Use pre-saved SR images (skip inference)
    python visualize.py --input data/benchmarks/Set5/butterfly.png \
                        --sr-dir results/Set5 --scale 4

    # Also generate HTML slider comparing Bicubic vs EDSR
    python visualize.py ... --slider-models bicubic edsr
"""
import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torchvision.transforms.functional as TF

from models import get_model

PRE_UPSAMPLE = {'srcnn', 'vdsr'}

# ── Inference helpers ─────────────────────────────────────────────────────────

def bicubic_sr(lr: Image.Image, scale: int) -> Image.Image:
    w, h = lr.size
    return lr.resize((w * scale, h * scale), Image.BICUBIC)


def model_sr(lr_tensor: torch.Tensor, model_name: str,
             ckpt_path: str, scale: int, device: torch.device) -> Image.Image:
    model = get_model(model_name, scale=scale).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    with torch.no_grad():
        inp = lr_tensor.unsqueeze(0).to(device)
        if model_name in PRE_UPSAMPLE:
            inp = F.interpolate(inp, scale_factor=scale, mode='bicubic', align_corners=False)
        sr = model(inp).squeeze(0).clamp(0, 1).cpu()
    return TF.to_pil_image(sr)


# ── Grid visualisation ────────────────────────────────────────────────────────

def label_image(img: Image.Image, text: str, font_size: int = 18) -> Image.Image:
    """Add a white label banner at the bottom of an image."""
    banner_h = font_size + 8
    out = Image.new('RGB', (img.width, img.height + banner_h), (30, 30, 30))
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', font_size)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((img.width - tw) // 2, img.height + 3), text, fill=(255, 255, 255), font=font)
    return out


def make_grid(images: list[tuple[str, Image.Image]], patch_box=None) -> Image.Image:
    """Arrange labelled images in a horizontal grid.
    If patch_box=(x, y, w, h), also show zoomed patch below each image.
    """
    labelled = [label_image(img, name) for name, img in images]
    w = labelled[0].width
    h = labelled[0].height
    grid = Image.new('RGB', (w * len(labelled), h), (20, 20, 20))
    for i, img in enumerate(labelled):
        grid.paste(img, (i * w, 0))

    if patch_box is not None:
        x, y, pw, ph = patch_box
        patches = []
        for name, img in images:
            patch = img.crop((x, y, x + pw, y + ph)).resize((w, w), Image.NEAREST)
            patches.append(label_image(patch, f'{name} [patch]'))
        patch_row = Image.new('RGB', (w * len(patches), patches[0].height), (20, 20, 20))
        for i, p in enumerate(patches):
            patch_row.paste(p, (i * w, 0))
        combined = Image.new('RGB', (grid.width, grid.height + patch_row.height), (20, 20, 20))
        combined.paste(grid, (0, 0))
        combined.paste(patch_row, (0, grid.height))
        return combined
    return grid


# ── HTML slider ───────────────────────────────────────────────────────────────

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SR Comparison: {before_name} vs {after_name}</title>
<style>
  body {{ margin: 0; background: #111; display: flex; flex-direction: column;
          align-items: center; padding: 20px; font-family: sans-serif; color: #eee; }}
  h2   {{ margin-bottom: 8px; }}
  .img-comp-container {{
    position: relative; width: {width}px; height: {height}px;
    overflow: hidden; cursor: ew-resize; border: 2px solid #444;
  }}
  .img-comp-img {{ position: absolute; top: 0; left: 0;
                   width: {width}px; height: {height}px; }}
  .img-comp-img img {{ display: block; width: {width}px; height: {height}px;
                       object-fit: cover; }}
  .img-comp-overlay {{ width: 50%; overflow: hidden; }}
  .divider {{
    position: absolute; top: 0; left: 50%; width: 2px; height: 100%;
    background: #fff; cursor: ew-resize; z-index: 10;
  }}
  .divider::after {{
    content: "◄►"; position: absolute; top: 50%; left: -12px;
    background: #fff; color: #000; padding: 4px 6px; border-radius: 4px;
    font-size: 12px; transform: translateY(-50%);
  }}
  .labels {{ display: flex; justify-content: space-between;
             width: {width}px; margin-top: 6px; font-size: 13px; color: #aaa; }}
  input[type=range] {{
    width: {width}px; margin-top: 10px; accent-color: #fff;
  }}
</style>
</head>
<body>
<h2>Super-Resolution Comparison</h2>
<div class="img-comp-container" id="container">
  <div class="img-comp-img">
    <img src="{after_img}" alt="{after_name}">
  </div>
  <div class="img-comp-img img-comp-overlay" id="overlay">
    <img src="{before_img}" alt="{before_name}">
  </div>
  <div class="divider" id="divider"></div>
</div>
<div class="labels"><span>{before_name}</span><span>{after_name}</span></div>
<input type="range" min="0" max="100" value="50" id="slider">
<script>
  const overlay  = document.getElementById('overlay');
  const divider  = document.getElementById('divider');
  const slider   = document.getElementById('slider');
  const container = document.getElementById('container');
  const W = container.offsetWidth;

  function setPos(pct) {{
    const px = W * pct / 100;
    overlay.style.width  = px + 'px';
    divider.style.left   = px + 'px';
  }}
  setPos(50);
  slider.addEventListener('input', () => setPos(+slider.value));
  container.addEventListener('mousemove', e => {{
    const rect = container.getBoundingClientRect();
    const pct  = Math.min(100, Math.max(0, (e.clientX - rect.left) / rect.width * 100));
    slider.value = pct;
    setPos(pct);
  }});
</script>
</body>
</html>
"""


def make_html_slider(before_path: str, after_path: str,
                     before_name: str, after_name: str,
                     out_path: str, width: int = 600, height: int = 400):
    html = HTML_TEMPLATE.format(
        before_img=os.path.relpath(before_path, os.path.dirname(out_path)),
        after_img=os.path.relpath(after_path, os.path.dirname(out_path)),
        before_name=before_name, after_name=after_name,
        width=width, height=height,
    )
    with open(out_path, 'w') as f:
        f.write(html)
    print(f"[visualize] HTML slider → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load HR and generate LR
    hr = Image.open(args.input).convert('RGB')
    w, h = hr.size
    w = (w // args.scale) * args.scale
    h = (h // args.scale) * args.scale
    hr = hr.crop((0, 0, w, h))
    lr = hr.resize((w // args.scale, h // args.scale), Image.BICUBIC)
    lr_tensor = TF.to_tensor(lr)

    stem = Path(args.input).stem
    results: list[tuple[str, Image.Image]] = [('HR (GT)', hr), ('Bicubic', bicubic_sr(lr, args.scale))]

    # Run inference or load pre-saved SR images
    for model_name in args.models:
        if args.sr_dir:
            sr_path = Path(args.sr_dir) / f'{stem}_{model_name}_x{args.scale}.png'
            if sr_path.exists():
                sr = Image.open(sr_path).convert('RGB')
            else:
                print(f"[warn] {sr_path} not found, skipping {model_name}")
                continue
        else:
            ckpt = Path(args.ckpt_dir) / model_name / 'final.pth'
            if not ckpt.exists():
                print(f"[warn] checkpoint {ckpt} not found, skipping {model_name}")
                continue
            sr = model_sr(lr_tensor, model_name, str(ckpt), args.scale, device)
        results.append((model_name.upper(), sr))
        # Save individual SR image
        sr.save(out_dir / f'{stem}_{model_name}_x{args.scale}.png')

    # Comparison grid (optionally with zoomed patch)
    patch_box = tuple(map(int, args.patch.split(','))) if args.patch else None
    grid = make_grid(results, patch_box)
    grid_path = out_dir / f'{stem}_comparison_x{args.scale}.png'
    grid.save(grid_path)
    print(f"[visualize] grid → {grid_path}")

    # HTML slider
    if len(args.slider_models) == 2:
        m1, m2 = args.slider_models
        p1 = out_dir / f'{stem}_{m1}_x{args.scale}.png' if m1 != 'bicubic' else None
        p2 = out_dir / f'{stem}_{m2}_x{args.scale}.png' if m2 != 'bicubic' else None
        bic_path = out_dir / f'{stem}_bicubic_x{args.scale}.png'
        bicubic_sr(lr, args.scale).save(bic_path)
        before = str(bic_path) if m1 == 'bicubic' else str(p1)
        after  = str(bic_path) if m2 == 'bicubic' else str(p2)
        make_html_slider(before, after, m1.upper(), m2.upper(),
                         str(out_dir / f'{stem}_slider_{m1}_vs_{m2}.html'),
                         width=w, height=h)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualise SR model comparisons')
    parser.add_argument('--input',   required=True, help='Path to HR input image')
    parser.add_argument('--scale',   type=int, default=4)
    parser.add_argument('--models',  nargs='+',
                        default=['srcnn','fsrcnn','vdsr','espcn','edsr','rcan','swinir','hat','drct'],
                        help='Models to compare (must have checkpoints or --sr-dir)')
    parser.add_argument('--ckpt-dir', default='checkpoints')
    parser.add_argument('--sr-dir',   default=None, help='Folder of pre-saved SR PNGs')
    parser.add_argument('--out-dir',  default='results/visual')
    parser.add_argument('--patch',    default=None,
                        help='Zoom patch as x,y,w,h  e.g. "100,80,64,64"')
    parser.add_argument('--slider-models', nargs=2, default=['bicubic','edsr'],
                        metavar=('BEFORE', 'AFTER'),
                        help='Two models for the HTML slider comparison')
    args = parser.parse_args()
    main(args)
