"""Fit the amodal decoder (option C) on a trained model and on its random initialization, and evaluate it on a stored
set next to template baselines.

1. Training: the decoder reads the frozen model's states R (each layer through a 1 x 1 convolution, upsampled to the
   frame and summed) and is trained with Adam to output the amodal frame, the whole digit as if there were no bar,
   on the analysis windows of new sequences of the training stream (split "decoder", training digits). The same is
   done on the same architecture at random initialization, the control of decision D15.
2. Evaluation on the stored set (held out digits): per frame, the correlation and the ink IoU between the decoded and
   the true amodal digit, per condition and state, for both decoders and for two templates that copy the last fully
   visible digit and move it with the constant velocity Kalman filter or with the filter with walls (exact).

Saves in <run>/probes/ the two decoders (amodal_model.npz, amodal_random.npz), and in <run>/eval/ the scores of every
frame (amodal_<set>.parquet), the summaries (amodal_<set>.json), and the IoU against the frames since the onset of the
occlusion (amodal_<set>.png). Draw the decoded digit on the predictions with scripts/show_predictions.py --decoder.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from matplotlib.figure import Figure

from peekaboo.data.conditions import GeneratorSettings
from peekaboo.data.dataset import OnTheFlyDataset
from peekaboo.data.mnist_pool import build_digit_pool
from peekaboo.data.render import RenderSettings
from peekaboo.data.store import StoredDataset
from peekaboo.data.truth import STATE_NAMES
from peekaboo.device import describe_device, get_device
from peekaboo.models import build_model
from peekaboo.paths import DATASETS_DIR
from peekaboo.probes.amodal import TEMPLATES, evaluate_decoders, fit_decoder
from peekaboo.train.checkpoint import find_checkpoint, load_model

DECODERS = ("decoder", "random_decoder")
METHODS = DECODERS + tuple(TEMPLATES)
COLORS = {"decoder": "magenta", "random_decoder": "0.6", "template_kalman": "C1", "template_walls": "C2"}
SINCE_ONSET = range(-3, 16)
CONDITIONS = ("occlusion", "hidden_bounce", "control")


def plot_curves(table: pd.DataFrame, path: Path, title: str) -> None:
    """Mean ink IoU (top) and correlation (bottom) against the frames since the onset, per method."""
    fig = Figure(figsize=(10, 6.4))
    axes = fig.subplots(2, 2, sharex=True)
    for column, condition in enumerate(("occlusion", "hidden_bounce")):
        rows = table[(table["condition"] == condition) & table["in_window"]
                     & table["since_onset"].isin(list(SINCE_ONSET))]
        for row, score in enumerate(("iou", "corr")):
            ax = axes[row, column]
            for name in METHODS:
                curve = rows.groupby("since_onset")[f"{name}_{score}"].mean()
                ax.plot(curve.index, curve.values, "o-" if name in DECODERS else ".--", ms=3, label=name,
                        color=COLORS[name], lw=2 if name == "decoder" else 1)
            ax.axvline(0, color="0.6", lw=0.8)
            ax.set_ylim(0, 1)
            ax.set_ylabel("ink IoU" if score == "iou" else "correlation")
            if row == 0:
                ax.set_title(condition.replace("_", " "), fontsize=9)
            else:
                ax.set_xlabel("frames since the onset of the occlusion")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main() -> int:
    """Fit both decoders, evaluate them with the templates, print the summaries, and save everything."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="run folder, for example runs/pilot_prednet5l_w10/seed0")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint file name inside the run folder (default: best.pt, last.pt, or model.pt)")
    parser.add_argument("--steps", type=int, default=400, help="training batches of the decoder")
    parser.add_argument("--batch-size", type=int, default=8, help="sequences per training batch")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--kernel", type=int, default=1, help="kernel of the per layer convolutions (1: minimal)")
    parser.add_argument("--data", default="val_v1", help="evaluation set: a set name in data/datasets/ or a folder")
    parser.add_argument("--n", type=int, default=None, help="first sequences of the evaluation set (default: all)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0, help="decoder stream and random initialization")
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
    stream = OnTheFlyDataset(pool, GeneratorSettings.from_config(data_config), render_settings, "decoder", args.seed,
                             include_amodal=True)
    data_dir = Path(args.data) if Path(args.data).is_dir() else DATASETS_DIR / args.data
    evaluation = StoredDataset(data_dir, render_settings=render_settings, include_amodal=True)
    count = min(args.n or len(evaluation), len(evaluation))
    print(f"{args.run} ({path.name}, step {checkpoint['step']}) on {describe_device(device)}: decoders trained on "
          f"{args.steps} batches of {args.batch_size} training stream sequences, evaluation on {count} sequences of "
          f"{data_dir.name}")

    decoders, training = {}, {}
    for name, network in (("model", model), ("random", random_model)):
        print(f"{name}:")
        decoder, losses = fit_decoder(network, stream, args.steps, device, args.batch_size, args.lr,
                                      kernel_size=args.kernel, workers=args.workers)
        decoder.save(args.run / "probes" / f"amodal_{name}.npz")
        training[name] = {"parameters": sum(p.numel() for p in decoder.parameters()),
                          "loss_first": losses[0], "loss_last_50": sum(losses[-50:]) / len(losses[-50:])}
        decoders["decoder" if name == "model" else "random_decoder"] = (network, decoder)
        print(f"  {training[name]['parameters']} parameters ({time.perf_counter() - started:.0f} s)")

    table = evaluate_decoders(decoders, evaluation, count, device)
    table["state_name"] = [STATE_NAMES[s] for s in table["state"]]
    window = table[table["in_window"] & table["condition"].isin(CONDITIONS)]
    summaries = {}
    for score, title in (("iou", "ink IoU"), ("corr", "correlation")):
        summary = window.groupby(["condition", "state_name"])[[f"{m}_{score}" for m in METHODS]].mean()
        summary.columns = list(METHODS)
        summary.insert(0, "frames", window.groupby(["condition", "state_name"]).size())
        summaries[score] = summary
        print(f"\nmean {title} between the decoded and the true amodal digit, analysis window of {data_dir.name}")
        print(summary.to_string(float_format=lambda v: f"{v:.3f}"))

    out = args.run / "eval"
    out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / f"amodal_{data_dir.name}.parquet", index=False)
    report = {"run": str(args.run), "checkpoint": path.name, "step": checkpoint["step"], "data": data_dir.name,
              "sequences": count, "kernel": args.kernel, "training": training,
              **{key: {" / ".join(map(str, index)): row.to_dict() for index, row in value.iterrows()}
                 for key, value in summaries.items()}}
    (out / f"amodal_{data_dir.name}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plot_curves(table, out / f"amodal_{data_dir.name}.png",
                f"{args.run.parent.name}/{args.run.name}, step {checkpoint['step']}: digit read in the model's state "
                f"(decoder) and templates, {data_dir.name}")
    print(f"\nsaved in {args.run}: probes/amodal_model.npz, probes/amodal_random.npz, "
          f"eval/amodal_{data_dir.name}.parquet, .json, .png ({time.perf_counter() - started:.0f} s)")
    print(f"draw the decoded digit on the predictions with: python -m scripts.show_predictions --run {args.run} "
          f"--decoder --probe")
    return 0


if __name__ == "__main__":
    sys.exit(main())
