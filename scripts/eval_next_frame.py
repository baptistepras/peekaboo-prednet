"""Evaluate a model's next frame predictions on a stored set against two baselines, and apply the gate D16.

The baselines are copying the last frame, and the true frame with the digit erased (what a model that never draws
digits would predict at best). Prints MSE and SSIM on every frame, then on the visible moving frames (the digit fully
visible in the frame and the one before, from frame 2 on), overall and per condition, speed, and k, then the gate: the
model's MSE there must be at least 30% below the better baseline. Saves in <run>/eval/ the scores of every frame,
MAE included (next_frame_<set>.csv), and the summaries with the gate (next_frame_<set>.json).
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from peekaboo.data.store import StoredDataset
from peekaboo.device import describe_device, get_device
from peekaboo.eval.next_frame import GATE_REDUCTION, evaluate, gate, summarize, tables_to_dict
from peekaboo.paths import DATASETS_DIR
from peekaboo.train.checkpoint import find_checkpoint, load_model

SHOWN = ["frames", "model_mse", "copy_mse", "blank_mse", "reduction_vs_copy", "reduction_vs_blank", "model_ssim",
         "copy_ssim", "blank_ssim"]


def show(summary: pd.DataFrame, title: str) -> None:
    """Print a summary table with a title."""
    print(f"\n{title}")
    print(summary[SHOWN].to_string(float_format=lambda v: f"{v:.5f}"))


def main() -> int:
    """Score every frame, print the summaries and the gate, and save them next to the run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="run folder, for example runs/pilot_prednet5l/seed0")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint file name inside the run folder (default: best.pt, last.pt, or model.pt, "
                             "the first that exists)")
    parser.add_argument("--data", default="val_v1", help="a set name in data/datasets/ or a folder")
    parser.add_argument("--n", type=int, default=None, help="first sequences of the set (default: all)")
    parser.add_argument("--seq-len", type=int, default=None, help="frames per sequence (default: all)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--reduction", type=float, default=GATE_REDUCTION, help="MSE reduction the gate requires")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    path = find_checkpoint(args.run, args.checkpoint)
    model, checkpoint = load_model(path, device)
    data_dir = Path(args.data) if Path(args.data).is_dir() else DATASETS_DIR / args.data
    dataset = StoredDataset(data_dir)
    print(f"{args.run.parent.name}/{args.run.name} ({path.name}, step {checkpoint['step']}) on "
          f"{describe_device(device)}: {args.n or len(dataset)} sequences of {data_dir.name}")

    table = evaluate(model, dataset, device, args.batch_size, args.seq_len, args.n)
    visible = table[table["visible_moving"]]
    summaries = {"all_frames": summarize(table), "visible_moving": summarize(visible),
                 "visible_moving_by_condition": summarize(visible, "condition"),
                 "visible_moving_by_speed": summarize(visible, "speed"),
                 "visible_moving_by_k": summarize(visible, "k")}
    show(summaries["all_frames"], "every frame from 1 on (as in the training validation)")
    show(summaries["visible_moving"], "visible moving frames (the gate)")
    show(summaries["visible_moving_by_condition"], "visible moving frames, per condition")
    show(summaries["visible_moving_by_speed"], "visible moving frames, per speed in px/frame (does the model fall "
                                               "back to copying at some speed?)")
    show(summaries["visible_moving_by_k"], "visible moving frames, per k of the sequence")

    result = gate(table, args.reduction)
    verdict = "PASSED" if result["passed"] else "FAILED"
    print(f"\ngate D16 {verdict}: on {result['frames']} visible moving frames, MSE {result['model_mse']:.5f} against "
          f"{result['copy_mse']:.5f} for the copy ({100 * result['reduction_vs_copy']:+.1f}%) and "
          f"{result['blank_mse']:.5f} for the blank frame ({100 * result['reduction_vs_blank']:+.1f}%); the better "
          f"baseline is {result['baseline']}, and at least {100 * result['required_reduction']:.0f}% below it is "
          f"required")

    out = args.run / "eval"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"next_frame_{data_dir.name}.csv", index=False)
    report = {"run": str(args.run), "checkpoint": path.name, "step": checkpoint["step"], "data": data_dir.name,
              "gate": result, **{name: tables_to_dict(summary) for name, summary in summaries.items()}}
    (out / f"next_frame_{data_dir.name}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"saved next_frame_{data_dir.name}.csv and .json in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
