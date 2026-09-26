"""Draw random trajectories built from anchor frames, marking bounces, to check the motion model by eye."""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only, no window
import matplotlib.pyplot as plt
import numpy as np

from peekaboo.config import load_config
from peekaboo.data.trajectory import Bounds, build_trajectory, sample_velocity
from peekaboo.paths import CONFIGS_DIR, FIGURES_DIR
from peekaboo.seeding import make_rng


def main() -> int:
    """Sample trajectories with random anchors and save a grid of their paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "data" / "base.yaml")
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES_DIR / "trajectories.png")
    args = parser.parse_args()

    config = load_config(args.config)
    fh, fw = config["frame_height"], config["frame_width"]
    seq_len, motion = config["seq_len"], config["motion"]
    cols = 4
    rows = int(np.ceil(args.n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.6 * rows), squeeze=False)

    for i, ax in enumerate(axes.flat):
        if i >= args.n:
            ax.axis("off")
            continue
        rng = make_rng(args.seed, "plot_trajectories", i)
        height, width = int(rng.integers(11, 17)), int(rng.integers(4, 17))  # typical ink sizes
        bounds = Bounds.for_sprite(fh, fw, height, width)
        anchor_frame = int(rng.integers(0, seq_len))
        anchor_pos = (int(rng.integers(bounds.y_min + 1, bounds.y_max)),
                      int(rng.integers(bounds.x_min + 1, bounds.x_max)))
        vel = sample_velocity(rng, motion["x_speeds"], motion["y_velocities"])
        traj = build_trajectory(anchor_frame, anchor_pos, vel, seq_len, bounds)

        # plot the ink center, not the top left corner
        cy = traj.positions[:, 0] + height / 2
        cx = traj.positions[:, 1] + width / 2
        ax.plot(cx, cy, "-", color="0.6", lw=1)
        ax.scatter(cx, cy, c=np.arange(seq_len), cmap="viridis", s=12, zorder=2)
        hits = traj.bounced.any(axis=1)
        ax.scatter(cx[hits], cy[hits], marker="x", color="red", s=40, zorder=3, label="bounce")
        ax.scatter(cx[anchor_frame], cy[anchor_frame], marker="*", color="black", s=90, zorder=4, label="anchor")
        ax.add_patch(plt.Rectangle((0, 0), fw, fh, fill=False, color="black"))
        ax.set_xlim(-2, fw + 2)
        ax.set_ylim(fh + 2, -2)  # image convention: y grows downward
        ax.set_aspect("equal")
        ax.set_title(f"v=({vel[0]},{vel[1]}) anchor t={anchor_frame}", fontsize=8)
        ax.tick_params(labelsize=6)
    axes.flat[0].legend(fontsize=6, loc="lower left")
    fig.suptitle("Ink center over time (dark = early, yellow = late)", fontsize=10)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
