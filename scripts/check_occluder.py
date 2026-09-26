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
from peekaboo.data.occluder import (STATE_OCCLUDED, STATE_PARTIAL, STATE_VISIBLE, CrossingSettings, OcclusionPlan,
                                    plan_crossing, visibility_states)
from peekaboo.paths import CONFIGS_DIR, FIGURES_DIR
from peekaboo.seeding import make_rng

STATE_COLORS = {STATE_VISIBLE: "tab:green", STATE_PARTIAL: "tab:orange", STATE_OCCLUDED: "tab:red"}


def draw_space_time(ax: plt.Axes, plan: OcclusionPlan, width: int, settings: CrossingSettings, title: str) -> None:
    """Draw x (horizontal) against time (downward): the bar as a gray band, the ink extent colored by visibility state."""
    seq_len = settings.seq_len
    ax.axvspan(plan.bar_left, plan.bar_left + plan.bar_width, color="0.8", zorder=0)
    states = visibility_states(plan.fractions, settings.thresholds)
    for t in range(seq_len):
        x = plan.trajectory.positions[t, 1]
        ax.plot([x, x + width], [t, t], color=STATE_COLORS[int(states[t])], lw=2.5, solid_capstyle="butt")
    e = plan.event
    for frame, name in ((e.entry_frame, "entry"), (e.onset_frame, "onset"), (e.reappear_frame, "reappear"),
                        (e.exit_frame, "exit")):
        ax.axhline(frame, color="black", lw=0.5, ls=":")
        ax.text(settings.frame_width + 1, frame, name, fontsize=6, va="center")
    ax.set_xlim(0, settings.frame_width)
    ax.set_ylim(seq_len - 0.5, -0.5)
    ax.set_xlabel("x (px)", fontsize=7)
    ax.set_ylabel("frame", fontsize=7)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)


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
                if i == 0 and speed == 3 or (i == 0 and k == 12 and speed == 2):
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
        draw_space_time(ax, plan, width, settings, title)
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
