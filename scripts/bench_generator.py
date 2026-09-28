"""Time the data generator stage by stage, and the loader throughput for several numbers of workers."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from peekaboo.config import load_config
from peekaboo.data.conditions import CONDITIONS, GeneratorSettings, sample_spec
from peekaboo.data.dataset import OnTheFlyDataset, make_loader
from peekaboo.data.mnist_pool import build_digit_pool
from peekaboo.data.render import RenderSettings, compose_observed, render_amodal
from peekaboo.data.splicing import SURPRISES, sample_surprise_tuple, surprise_speeds
from peekaboo.data.truth import compute_truth
from peekaboo.paths import CONFIGS_DIR


def main() -> int:
    """Print milliseconds per sequence for each stage and condition, then sequences per second for each loader."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "data" / "base.yaml")
    parser.add_argument("--split", default="val")
    parser.add_argument("--n", type=int, default=400, help="sequences per timed stage")
    parser.add_argument("--tuples", type=int, default=20, help="tuples per surprise type")
    parser.add_argument("--workers", default="0,2,4", help="comma separated worker counts for the loader")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batches", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    settings = GeneratorSettings.from_config(config)
    render_settings = RenderSettings.from_config(config)
    digits = config["digits"]
    pool = build_digit_pool(args.split, digits["box_size"], digits["ink_threshold"], digits["val_size"],
                            digits["holdout_seed"], download=False)
    thresholds = settings.crossing.thresholds

    # 1. the stages of one training sequence, single process
    times = {"spec": [], "amodal": [], "observed": [], "truth": []}
    for i in range(args.n):
        t0 = time.perf_counter()
        spec = sample_spec(pool, settings, "bench", i, args.seed)
        t1 = time.perf_counter()
        sprite = pool.sprite(spec.digit_index)
        amodal = render_amodal(spec, sprite)
        t2 = time.perf_counter()
        compose_observed(amodal, spec, render_settings)
        t3 = time.perf_counter()
        compute_truth(spec, sprite, thresholds)
        t4 = time.perf_counter()
        for key, value in zip(times, (t1 - t0, t2 - t1, t3 - t2, t4 - t3)):
            times[key].append(value)
    total = sum(np.mean(v) for v in times.values())
    print(f"training mix, {args.n} sequences, single process: {total * 1000:.2f} ms per sequence")
    for key, values in times.items():
        print(f"  {key:>9}: {np.mean(values) * 1000:6.2f} ms (median {np.median(values) * 1000:.2f}, "
              f"max {np.max(values) * 1000:.1f})")

    # 2. spec generation by condition and surprise (placement cost varies)
    print("\nspec generation by condition (ms per sequence)")
    for condition in CONDITIONS:
        start = time.perf_counter()
        n = max(args.n // 4, 1)
        for i in range(n):
            sample_spec(pool, settings, f"bench_{condition}", i, args.seed, condition=condition)
        print(f"  {condition:>14}: {(time.perf_counter() - start) / n * 1000:6.2f}")
    print("tuple generation by surprise (ms per tuple of four sequences, k = 6)")
    for kind in SURPRISES:
        allowed = surprise_speeds(kind, settings)
        speed = 3 if 3 in allowed else allowed[0]
        start = time.perf_counter()
        for i in range(args.tuples):
            sample_surprise_tuple(pool, settings, f"bench_{kind}", i, args.seed, kind, 6, speed)
        print(f"  {kind:>14}: {(time.perf_counter() - start) / args.tuples * 1000:6.1f}")

    # 3. loader throughput, the first batch (worker start up) left out
    print(f"\nloader throughput, batch {args.batch_size}, {args.batches} batches")
    dataset = OnTheFlyDataset(pool, settings, render_settings, "bench_loader", args.seed)
    for workers in (int(w) for w in args.workers.split(",")):
        loader = make_loader(dataset, args.batch_size, start=0, count=args.batch_size * args.batches,
                             num_workers=workers)
        iterator = iter(loader)
        next(iterator)
        start = time.perf_counter()
        n_batches = sum(1 for _ in iterator)
        elapsed = time.perf_counter() - start
        print(f"  {workers} workers: {elapsed / n_batches * 1000:6.1f} ms per batch, "
              f"{n_batches * args.batch_size / elapsed:6.0f} sequences/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
