"""Download MNIST, build the train, val, and test digit pools, print their statistics, and save a sprite sheet."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from peekaboo.data.mnist_pool import SPLITS, DigitPool, build_digit_pool, pool_summary
from peekaboo.paths import FIGURES_DIR, MNIST_DIR


def sprite_sheet(pool: DigitPool, per_label: int = 10, scale: int = 4) -> np.ndarray:
    """Tile the first sprites of each label (one row per label) into one image, each centered in its box."""
    box = pool.box_size
    sheet = np.zeros((10 * box, per_label * box), dtype=np.uint8)
    for label in range(10):
        members = np.flatnonzero(pool.labels == label)[:per_label]
        for col, i in enumerate(members):
            sprite = pool.sprite(int(i))
            h, w = sprite.shape
            top = label * box + (box - h) // 2
            left = col * box + (box - w) // 2
            sheet[top:top + h, left:left + w] = sprite
    return np.kron(sheet, np.ones((scale, scale), dtype=np.uint8))  # nearest neighbor upscaling


def main() -> int:
    """Build the pools, check that train and val do not overlap, and write the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--box-size", type=int, default=20)
    parser.add_argument("--ink-threshold", type=float, default=0.1)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--holdout-seed", type=int, default=0)
    parser.add_argument("--root", type=Path, default=MNIST_DIR)
    parser.add_argument("--no-download", action="store_true", help="fail instead of downloading")
    parser.add_argument("--sheet", type=Path, default=FIGURES_DIR / "mnist_pool.png")
    args = parser.parse_args()

    pools = {}
    for split in SPLITS:
        pools[split] = build_digit_pool(split, args.box_size, args.ink_threshold, args.val_size,
                                        args.holdout_seed, args.root, download=not args.no_download)
        print(json.dumps(pool_summary(pools[split])))

    # train and val both come from the MNIST train file and must not share a digit
    overlap = np.intersect1d(pools["train"].mnist_indices, pools["val"].mnist_indices).size
    covered = len(pools["train"]) + len(pools["val"])
    print(f"train/val overlap: {overlap} digits, train + val = {covered}")

    args.sheet.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sprite_sheet(pools["val"])).save(args.sheet)
    print(f"sprite sheet (val, one row per label): {args.sheet}")
    return 0 if overlap == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
