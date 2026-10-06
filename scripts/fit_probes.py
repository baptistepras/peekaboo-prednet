"""Fit the position probes (option B) on a trained model and on its random initialization, and evaluate them on a
stored set next to the programmed trackers.

1. Features: the pooled internal states R of the frozen model, on the frames of the analysis windows of new sequences
   from the training stream (split "probe", training digits), about 20k frames by default.
2. Probes: a ridge regression of the centroid (y, x) and a logistic regression on 8 position bins per axis, penalties
   chosen on held out training sequences. The same is done on the same architecture at random initialization, the
   control that shows what is readable without training (decision D15).
3. Evaluation on the stored set (held out digits): the mean absolute error on y, the main axis while hidden
   (critique C4), and on x, per condition and state, for the two probes, the last seen position, the Kalman filters,
   and the bar center (x only).

Saves in <run>/probes/ the two probes (position_model.npz, position_random.npz), and in <run>/eval/ the position of
every method on every frame (probe_<set>.parquet), the summaries (probe_<set>.json), and the errors against the
frames since the onset of the occlusion (probe_<set>.png). Draw the probe on the predictions with
scripts/show_predictions.py --probe.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure

from peekaboo.data.conditions import GeneratorSettings
from peekaboo.data.dataset import OnTheFlyDataset
from peekaboo.data.mnist_pool import build_digit_pool
from peekaboo.data.render import RenderSettings
from peekaboo.data.store import StoredDataset
from peekaboo.device import describe_device, get_device
from peekaboo.models import build_model
from peekaboo.paths import DATASETS_DIR
from peekaboo.probes.evaluate import (BASELINES, add_errors, baseline_positions, position_table, probe_positions,
                                      summarize_errors)
from peekaboo.probes.features import LAYER_LIMIT, collect_frames
from peekaboo.probes.position import fit_position_probe
from peekaboo.train.checkpoint import find_checkpoint, load_model

PROBES = ("probe", "random_probe")
METHODS = PROBES + BASELINES
COLORS = {"probe": "gold", "random_probe": "0.6", "last_seen": "C0", "kalman": "C1", "kalman_walls": "C2"}
SINCE_ONSET = range(-3, 16)
CONDITIONS = ("occlusion", "hidden_bounce", "control")


def show(table: pd.DataFrame, title: str) -> None:
    """Print a summary table with a title."""
    print(f"\n{title}")
    print(table.to_string(float_format=lambda v: f"{v:.2f}"))


def plot_curves(table: pd.DataFrame, path: Path, title: str) -> None:
    """Mean absolute error on y (top) and x (bottom) against the frames since the onset, per method."""
    fig = Figure(figsize=(10, 6.4))
    axes = fig.subplots(2, 2, sharex=True)
    for column, condition in enumerate(("occlusion", "hidden_bounce")):
        rows = table[(table["condition"] == condition) & table["in_window"]
                     & table["since_onset"].isin(list(SINCE_ONSET))]
        for row, axis in enumerate(("y", "x")):
            ax = axes[row, column]
            names = METHODS + (("bar_center",) if axis == "x" else ())
            for name in names:
                curve = rows.groupby("since_onset")[f"{name}_err_{axis}"].mean()
                ax.plot(curve.index, curve.values, "o-" if name in PROBES else ".--", ms=3, label=name,
                        color=COLORS.get(name, "C4"), lw=2 if name == "probe" else 1)
            ax.axvline(0, color="0.6", lw=0.8)
            ax.set_ylim(bottom=0)
            ax.set_ylabel(f"mean |error| on {axis} (px)")
            if row == 0:
                ax.set_title(condition.replace("_", " "), fontsize=9)
            else:
                ax.set_xlabel("frames since the onset of the occlusion")
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main() -> int:
    """Fit both probes, evaluate them with the baselines, print the summaries, and save everything."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="run folder, for example runs/pilot_prednet5l_w10/seed0")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint file name inside the run folder (default: best.pt, last.pt, or model.pt)")
    parser.add_argument("--train-sequences", type=int, default=800,
                        help="sequences of the training stream for the probes (their analysis windows, about 25 "
                             "frames each)")
    parser.add_argument("--data", default="val_v1", help="evaluation set: a set name in data/datasets/ or a folder")
    parser.add_argument("--n", type=int, default=None, help="first sequences of the evaluation set (default: all)")
    parser.add_argument("--limit", type=int, default=LAYER_LIMIT, help="pooled features per layer")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0, help="probe stream, random initialization, validation split")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    started = time.perf_counter()
    device = get_device(args.device)
    path = find_checkpoint(args.run, args.checkpoint)
    model, checkpoint = load_model(path, device)
    config = checkpoint["config"]
    torch.manual_seed(args.seed)
    random_model = build_model(config["model"]).to(device).eval()
    data_config = config["data"]
    digits = data_config["digits"]
    render_settings = RenderSettings.from_config(data_config)
    pool = build_digit_pool("train", digits["box_size"], digits["ink_threshold"], digits["val_size"],
                            digits["holdout_seed"], download=False)
    stream = OnTheFlyDataset(pool, GeneratorSettings.from_config(data_config), render_settings, "probe", args.seed)
    data_dir = Path(args.data) if Path(args.data).is_dir() else DATASETS_DIR / args.data
    evaluation = StoredDataset(data_dir, render_settings=render_settings)  # drawn with the training colors
    count = min(args.n or len(evaluation), len(evaluation))
    frame_size = (evaluation.specs[0].frame_height, evaluation.specs[0].frame_width)
    options = {"batch_size": args.batch_size, "limit": args.limit}
    print(f"{args.run} ({path.name}, step {checkpoint['step']}) on {describe_device(device)}: probes on "
          f"{args.train_sequences} training stream sequences, evaluation on {count} sequences of {data_dir.name}")

    probe_rows, scores = {}, {}
    for name, network in (("model", model), ("random", random_model)):
        frames = collect_frames(network, stream, args.train_sequences, device, window_only=True,
                                workers=args.workers, **options)
        probe, validation = fit_position_probe(frames.features, frames.rows[["y", "x"]].to_numpy(),
                                               frames.rows["sequence"].to_numpy(), frame_size, args.limit,
                                               seed=args.seed)
        probe.save(args.run / "probes" / f"position_{name}.npz")
        scores[name] = {"frames": len(frames.rows), "features": int(frames.features.shape[1]), "alpha": probe.alpha,
                        "c": probe.c, "validation": {k: {str(a): v for a, v in s.items()}
                                                     for k, s in validation.items()}}
        print(f"{name}: {len(frames.rows)} frames x {frames.features.shape[1]} features, ridge alpha {probe.alpha:g}, "
              f"logistic C {probe.c:g} ({time.perf_counter() - started:.0f} s)")
        del frames
        probe_rows["probe" if name == "model" else "random_probe"] = probe_positions(network, probe, evaluation,
                                                                                      count, device, **options)

    table = add_errors(position_table(probe_rows, baseline_positions(evaluation, count)), METHODS)
    window = table[table["in_window"] & table["condition"].isin(CONDITIONS)]
    summaries = {}
    for axis in ("y", "x"):
        summary = summarize_errors(window, METHODS, ["condition", "state_name"], axis)
        summaries[f"error_{axis}"] = summary
        show(summary, f"mean |error| on {axis} in px, analysis window of {data_dir.name}")
    bins = window.groupby(["condition", "state_name"])[[f"{p}_p_bin_{a}" for p in PROBES for a in ("y", "x")]].mean()
    summaries["p_true_bin"] = bins
    show(bins, "mean probability of the true bin (8 bins per axis; 0.125 is chance)")

    out = args.run / "eval"
    out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / f"probe_{data_dir.name}.parquet", index=False)
    report = {"run": str(args.run), "checkpoint": path.name, "step": checkpoint["step"], "data": data_dir.name,
              "sequences": count, "probes": scores,
              **{key: {" / ".join(map(str, index)): row.to_dict() for index, row in value.iterrows()}
                 for key, value in summaries.items()}}
    (out / f"probe_{data_dir.name}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plot_curves(table, out / f"probe_{data_dir.name}.png",
                f"{args.run.parent.name}/{args.run.name}, step {checkpoint['step']}: position read in the model's "
                f"state (probe) and by the baselines, {data_dir.name}")
    print(f"\nsaved in {args.run}: probes/position_model.npz, probes/position_random.npz, "
          f"eval/probe_{data_dir.name}.parquet, .json, .png ({time.perf_counter() - started:.0f} s)")
    print(f"draw the probe on the predictions with: python -m scripts.show_predictions --run {args.run} --probe")
    return 0


if __name__ == "__main__":
    sys.exit(main())
