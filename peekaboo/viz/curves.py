"""Training curves of a run, read from its CSV logs. Drawn on a bare matplotlib Figure, so no display is needed."""

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from peekaboo.train.metrics import read_csv


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Mean over the last `window` values at every point (fewer at the start)."""
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    starts = np.maximum(np.arange(1, len(values) + 1) - window, 0)
    return (cumulative[1:] - cumulative[starts]) / (np.arange(1, len(values) + 1) - starts)


def plot_training(run_dir: str | Path, title: str = "") -> Path:
    """Save curves.png: the training loss (raw and smoothed), and the validation L1 error against copying the last
    frame."""
    run_dir = Path(run_dir)
    train = read_csv(run_dir / "train_metrics.csv")
    val = read_csv(run_dir / "val_metrics.csv")
    fig = Figure(figsize=(10, 3.6))
    left, right = fig.subplots(1, 2)
    if train:
        steps = np.array([int(r["step"]) for r in train])
        loss = np.array([float(r["loss"]) for r in train])
        left.plot(steps, loss, lw=0.5, color="0.7", label="every step")
        window = max(10, len(loss) // 50)
        left.plot(steps, moving_average(loss, window), lw=1.2, color="C0", label=f"mean of the last {window} steps")
        left.set_yscale("log")
        left.legend(fontsize=8)
    left.set_xlabel("step")
    left.set_ylabel("training loss")
    if val:
        steps = [int(r["step"]) for r in val]
        right.plot(steps, [float(r["val_l1"]) for r in val], "o-", ms=3, color="C0", label="model")
        right.plot(steps, [float(r["copy_l1"]) for r in val], "--", color="0.4", label="copy last frame")
        best = int(val[-1]["best_step"])
        if best >= 0:
            right.axvline(best, color="C2", lw=0.8, label=f"best, step {best}")
        right.set_yscale("log")
        right.legend(fontsize=8)
    right.set_xlabel("step")
    right.set_ylabel("validation L1 error per pixel")
    fig.suptitle(title or run_dir.name, fontsize=9)
    fig.tight_layout()
    path = run_dir / "curves.png"
    fig.savefig(path, dpi=110)
    return path
