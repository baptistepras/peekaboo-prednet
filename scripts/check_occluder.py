"""Place occluders for every (k, speed) cell on real digits, report success rates and biases, and draw space time diagrams."""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only, no window
import matplotlib.pyplot as plt
import numpy as np

from peekaboo.config import load_config
from peekaboo.data.mnist_pool import build_digit_pool
from peekaboo.data.occluder import CrossingSettings, plan_crossing
from peekaboo.paths import CONFIGS_DIR, FIGURES_DIR
from peekaboo.seeding import make_rng
from peekaboo.viz.space_time import draw_space_time


def main() -> int:
    """Run the placement solver on every cell and print one report line per cell."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "data" / "base.yaml")
    parser.add_argument("--split", default="val")
    parser.add_argument("--n", type=int, default=200, help="digits per cell")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES_DIR / "occluder_examples.png")
    args = parser.parse_args()

    config = load_config(args.config)
    settings = CrossingSettings.from_config(config)
    digits = config["digits"]
    pool = build_digit_pool(args.split, digits["box_size"], digits["ink_threshold"], digits["val_size"],
                            digits["holdout_seed"], download=False)
    print(f"frame {settings.frame_height}x{settings.frame_width}, {settings.seq_len} frames, "
          f"pool '{args.split}' mean ink width {pool.widths.mean():.1f}")
    print(f"{'k':>3} {'v':>2} {'placed':>7} {'attempts':>9} {'width placed':>13} {'bar width':>14} "
          f"{'bar left':>9} {'onset':>8} {'occluded':>9}")

    examples = []
    all_ok = True
    for k, speeds in settings.k_speeds.items():
        for speed in speeds:
            placed, attempts, widths, bar_widths, bar_lefts, onsets = 0, [], [], [], [], []
            for i in range(args.n):
                rng = make_rng(args.seed, "check_occluder", k, speed, i)
                d = pool.sample(rng)
                try:
                    plan = plan_crossing(rng, pool.sprite(d), k, settings, speeds=[speed])
                except RuntimeError:
                    continue
                assert plan.event.k == k  # the solver only returns exact placements
                placed += 1
                attempts.append(plan.attempts)
                widths.append(int(pool.widths[d]))
                bar_widths.append(plan.bar_width)
                bar_lefts.append(plan.bar_left)
                onsets.append(plan.event.onset_frame)
                if i == 0 and (speed == 3 or (k == 12 and speed == 2)):
                    examples.append((plan, int(pool.widths[d]), f"k={k}, v={speed}, bar {plan.bar_width} px"))
            all_ok &= placed == args.n
            if placed == 0:
                print(f"{k:>3} {speed:>2} {'0':>7}  no placement found")
                continue
            print(f"{k:>3} {speed:>2} {placed / args.n:>6.0%} {np.mean(attempts):>9.1f} "
                  f"{np.mean(widths):>13.1f} {min(bar_widths):>4}-{int(np.median(bar_widths)):>3}-{max(bar_widths):<4} "
                  f"{min(bar_lefts):>3}-{max(bar_lefts):<4} {min(onsets):>3}-{max(onsets):<4} "
                  f"{k / settings.seq_len:>8.0%}")

    cols = 3
    rows = int(np.ceil(len(examples) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows), squeeze=False)
    for ax, (plan, width, title) in zip(axes.flat, examples):
        e = plan.event
        marks = {"entry": e.entry_frame, "onset": e.onset_frame, "reappear": e.reappear_frame, "exit": e.exit_frame}
        draw_space_time(ax, plan.trajectory.positions[:, 1], width, plan.fractions, plan.bar_left, plan.bar_width,
                        settings.frame_width, settings.thresholds, marks, title)
    for ax in axes.flat[len(examples):]:
        ax.axis("off")
    fig.suptitle("Ink extent over time: green visible, orange partial, red occluded, gray band = bar", fontsize=9)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"saved {args.out}")
    print("all digits placed" if all_ok else "some digits could not be placed (see the 'placed' column)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
