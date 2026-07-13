#!/usr/bin/env bash
# Download DIV2K + SR benchmarks using HuggingFace mirror (for China servers).
# Data is saved to /root/autodl-tmp/data/ (fast data disk on AutoDL).
#
# Usage:  bash scripts/download_data_cn.sh

set -e

DATA_DIR="/root/autodl-tmp/data"
mkdir -p "$DATA_DIR"

echo "=== Installing dependencies (Tsinghua PyPI mirror) ==="
# datasets>=3.0 dropped support for script-based dataset repos (e.g. eugenesiow/Div2k's
# Div2k.py loader); pin to a pre-3.0 release and pass trust_remote_code=True below.
pip install "datasets<3.0" pillow -q -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "=== Downloading DIV2K training set (800 HR images) ==="
HF_ENDPOINT=https://hf-mirror.com python3 - <<'PYEOF'
import os
from datasets import load_dataset

out = "/root/autodl-tmp/data/DIV2K_train_HR"
os.makedirs(out, exist_ok=True)

ds = load_dataset("eugenesiow/Div2k", "bicubic_x4",
                  split="train", cache_dir="/root/autodl-tmp/hf_cache",
                  trust_remote_code=True)
for i, item in enumerate(ds):
    item["hr"].save(f"{out}/{i+1:04d}.png")
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/800 saved")
print(f"DIV2K done -> {out}")
PYEOF

echo "=== Downloading test benchmarks ==="
HF_ENDPOINT=https://hf-mirror.com python3 - <<'PYEOF'
import os
from datasets import load_dataset

BASE = "/root/autodl-tmp/data/benchmarks"
SETS = {
    "Set5":     ("eugenesiow/Set5",     "bicubic_x4", "validation"),
    "Set14":    ("eugenesiow/Set14",    "bicubic_x4", "validation"),
    "BSD100":   ("eugenesiow/BSD100",   "bicubic_x4", "validation"),
    "Urban100": ("eugenesiow/Urban100", "bicubic_x4", "validation"),
}

for name, (repo, config, split) in SETS.items():
    out = f"{BASE}/{name}"
    os.makedirs(out, exist_ok=True)
    print(f"  Downloading {name}...")
    ds = load_dataset(repo, config, split=split, cache_dir="/root/autodl-tmp/hf_cache",
                      trust_remote_code=True)
    for i, item in enumerate(ds):
        item["hr"].save(f"{out}/{i+1:04d}.png")
    print(f"  {name} done ({len(ds)} images) -> {out}")

print("All benchmarks done.")
PYEOF

echo ""
echo "=== All data ready ==="
echo "  Training : /root/autodl-tmp/data/DIV2K_train_HR"
echo "  Benchmarks: /root/autodl-tmp/data/benchmarks/{Set5,Set14,BSD100,Urban100}"
