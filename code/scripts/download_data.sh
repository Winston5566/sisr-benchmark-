#!/usr/bin/env bash
# Download DIV2K training images and standard SR test benchmarks.
# Usage:  bash scripts/download_data.sh [TARGET_DIR]
# Default target: ./data

set -euo pipefail

DATA_DIR="${1:-./data}"
mkdir -p "$DATA_DIR"

# ── DIV2K training set ────────────────────────────────────────────────────────
echo "Downloading DIV2K HR training images..."
wget -q --show-progress \
    -O "$DATA_DIR/DIV2K_train_HR.zip" \
    "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
unzip -q "$DATA_DIR/DIV2K_train_HR.zip" -d "$DATA_DIR"
rm  "$DATA_DIR/DIV2K_train_HR.zip"
echo "  -> $DATA_DIR/DIV2K_train_HR/"

# ── Test benchmarks (via BasicSR mirror) ─────────────────────────────────────
# Set5, Set14, BSD100, Urban100 are redistributed in the BasicSR test dataset pack.
echo "Downloading test benchmarks..."
wget -q --show-progress \
    -O "$DATA_DIR/benchmarks.zip" \
    "https://github.com/XPixelGroup/BasicSR/releases/download/TestDatasets/TestDatasets.zip" || {
        echo "  [warn] automatic download failed — please download test sets manually."
        echo "         Set5/Set14/BSD100/Urban100 are available in the BasicSR repo releases."
        exit 1
    }
unzip -q "$DATA_DIR/benchmarks.zip" -d "$DATA_DIR/benchmarks"
rm  "$DATA_DIR/benchmarks.zip"
echo "  -> $DATA_DIR/benchmarks/"

echo "Done."
