"""Check the data pipeline on real digits: loader speed and determinism, then a stored dataset written and read back."""

import argparse
import sys
import time
from pathlib import Path

import torch

from peekaboo.config import load_config
from peekaboo.data.conditions import GeneratorSettings
from peekaboo.data.dataset import OnTheFlyDataset, make_loader
from peekaboo.data.mnist_pool import build_digit_pool
from peekaboo.data.render import RenderSettings
from peekaboo.data.splicing import SURPRISES, sample_surprise_tuple, surprise_speeds
from peekaboo.data.store import StoredDataset, write_dataset
from peekaboo.paths import CONFIGS_DIR, DATASETS_DIR


def batches_equal(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    """True if two batches have the same keys and identical tensors, NaNs included."""
    if a.keys() != b.keys():
        return False
    return all(torch.equal(a[k], b[k]) or (a[k].is_floating_point() and torch.allclose(a[k], b[k], equal_nan=True))
               for k in a)


def folder_size_mb(folder: Path) -> float:
    """Total size of the files in a folder, in megabytes."""
    return sum(f.stat().st_size for f in folder.iterdir()) / 1e6


def main() -> int:
    """Time the loader, check it against a single process loader, and round trip a small stored dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "data" / "base.yaml")
    parser.add_argument("--split", default="val")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DATASETS_DIR / "smoke")
    args = parser.parse_args()

    config = load_config(args.config)
    settings = GeneratorSettings.from_config(config)
    render_settings = RenderSettings.from_config(config)
    digits = config["digits"]
    pool = build_digit_pool(args.split, digits["box_size"], digits["ink_threshold"], digits["val_size"],
                            digits["holdout_seed"], download=False)
    dataset = OnTheFlyDataset(pool, settings, render_settings, "check", args.seed)

    # 1. loader speed (the first batch includes worker start up, so it is timed apart)
    count = args.batch_size * args.batches
    loader = make_loader(dataset, args.batch_size, start=0, count=count, num_workers=args.workers)
    start = time.perf_counter()
    batches = []
    for batch in loader:
        batches.append(batch)
        if len(batches) == 1:
            first = time.perf_counter() - start
            start = time.perf_counter()
    rest = time.perf_counter() - start
    per_batch = rest / max(len(batches) - 1, 1)
    print(f"{args.workers} workers, batch {args.batch_size}: first batch {first:.2f} s, then {per_batch * 1000:.0f} ms "
          f"per batch ({args.batch_size / per_batch:.0f} sequences/s)")
    print("batch shapes: " + ", ".join(f"{k} {tuple(v.shape)}" for k, v in batches[0].items()
                                       if k in ("frames", "center", "state")))

    # 2. determinism: the same indices in a single process give the same batches
    serial = list(make_loader(dataset, args.batch_size, start=0, count=2 * args.batch_size, num_workers=0))
    same = all(batches_equal(a, b) for a, b in zip(serial, batches[:2]))
    print(f"same batches with {args.workers} workers and in a single process: {'yes' if same else 'NO'}")

    # 3. stored dataset: training mix sequences and one tuple of each surprise, written then read back
    specs = [dataset.spec(i) for i in range(64)]
    for kind in SURPRISES:
        specs += sample_surprise_tuple(pool, settings, "check", 0, args.seed, kind, 6,
                                       surprise_speeds(kind, settings)[0]).specs()
    start = time.perf_counter()
    folder = write_dataset(args.out, specs, pool, render_settings, settings.crossing.thresholds,
                           info={"name": args.out.name, "split": args.split, "base_seed": args.seed,
                                 "generator_config": config}, overwrite=True)
    written = time.perf_counter() - start
    stored = StoredDataset(folder)
    verified = all(stored.verify(i) for i in range(len(stored)))
    matches = all(batches_equal(stored[i], dataset[i]) for i in range(8))
    print(f"wrote {len(specs)} sequences to {folder} in {written:.1f} s ({folder_size_mb(folder):.2f} MB)")
    print(f"checksums verified: {'yes' if verified else 'NO'}, stored samples equal the stream: "
          f"{'yes' if matches else 'NO'}")

    ok = same and verified and matches
    print("all checks passed" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
