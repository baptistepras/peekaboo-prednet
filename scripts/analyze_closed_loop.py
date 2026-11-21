"""Blackout and closed loop (step 3.6): reads the closed loop passes of every sweep model on the blackout and the
occlusion cells of a set (scripts/extract.py --closed-loop), and the baselines of the set, and writes the tables and
figures of peekaboo/eval/closed_loop.py. Nothing runs a model.

Saved in runs/analysis/<set>/closed_loop/:
- summary.csv: per mode (blackout, occlusion), frames (the blackout or hidden frames, and the reappearance frame and
  the next two), k (all and each), model, and loop (open, closed), the probe's error on y, the decoder's correlation,
  the drawn digit's error on y and the share of frames where it is drawn (not behind the bar of the occlusion cells,
  where the digit is hidden), and the MSE around the digit, with 95%
  bootstrap intervals, and the trackers next to the position measures; paired.csv: closed minus open loop on the same
  sequences;
- curves.csv: the same measures against the frames since the onset, per mode, k, model, and loop;
- the figures blackout.png and occlusion.png.

Every choice is fixed on dev_v1; test_v1 is analyzed once, with the same choices.
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from peekaboo.eval.closed_loop import (FRAMES, K_SHOWN, MEASURES, closed_loop, loop_runs, measures_of, plot_loop,
                                       read_loops)
from peekaboo.eval.extract import LOOP_MODES, LOOPS
from peekaboo.eval.tables import baselines_folder
from peekaboo.paths import RUNS_DIR


def report(tables: dict[str, pd.DataFrame]) -> None:
    """Print, per mode and frames, every measure of each model in open and closed loop with the difference, then the
    open loop readouts of the blackout frames next to the hidden frames of the occlusion cells, per k."""
    summary, paired = tables["summary"], tables["paired"]
    every = summary[summary["k"] == "all"]
    for mode in LOOP_MODES:
        for frames in FRAMES:
            measures = measures_of(mode, frames)
            rows = every[(every["mode"] == mode) & (every["frames"] == frames)]
            if not len(rows):
                continue
            where = {"hidden": "blackout frames" if mode == "blackout" else "hidden frames",
                     "return": "the reappearance frame and the next two"}[frames]
            print(f"\n{mode} cells, {where}: open loop, closed loop, closed - open [95% interval]")
            for measure in measures:
                own = rows[rows["measure"] == measure]
                difference = paired[(paired["mode"] == mode) & (paired["frames"] == frames) & (paired["k"] == "all")
                                    & (paired["measure"] == measure)].set_index("who")
                lines = []
                for model in own.loc[own["kind"] == "model", "who"].unique():
                    value = own[own["who"] == model].set_index("loop")["mean"]
                    d = difference.loc[model]
                    lines.append({"who": model, **{loop: value.get(loop) for loop in LOOPS},
                                  "closed - open": f"{d['mean']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}]"})
                for name in own.loc[own["kind"] == "reference", "who"].unique():
                    lines.append({"who": name, "open": own[own["who"] == name]["mean"].iloc[0]})
                print(f"{MEASURES[measure][0]}")
                print(pd.DataFrame(lines).set_index("who").to_string(
                    float_format=lambda v: f"{v:.4f}" if measure == "mse_digit" else f"{v:.2f}", na_rep=""))

    rows = summary[(summary["frames"] == "hidden") & (summary["k"] != "all") & (summary["measure"] == "probe_y")
                   & (summary["loop"].isin(["open", ""]))]
    shown = rows.pivot_table(index="who", columns=["k", "mode"], values="mean", sort=False)
    print("\nopen loop, probe |error| on y (px) on the blackout frames and on the hidden frames of the occlusion "
          "cells, per k")
    print(shown.to_string(float_format=lambda v: f"{v:.2f}", na_rep=""))


def main() -> int:
    """Read every available model on the set, write the tables and figures, and print the main numbers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="dev_v1", help="a set with complete closed loop passes (dev_v1 or test_v1)")
    parser.add_argument("--runs", type=Path, default=RUNS_DIR, help="folder of the runs")
    parser.add_argument("--out", type=Path, default=None,
                        help="output folder (default runs/analysis/<set>/closed_loop)")
    args = parser.parse_args()

    started = time.perf_counter()
    runs = {mode: loop_runs(args.runs, args.data, mode) for mode in LOOP_MODES}
    runs = {mode: own for mode, own in runs.items() if own}
    baselines = baselines_folder(args.runs, args.data)
    if not runs or not (baselines / "info.json").exists():
        print(f"no complete closed loop pass on {args.data}: python -m scripts.extract --status --closed-loop "
              f"blackout --data {args.data}")
        return 1
    out = args.out or args.runs / "analysis" / args.data / "closed_loop"
    out.mkdir(parents=True, exist_ok=True)
    for mode, own in runs.items():
        print(f"{args.data}, {mode}: {', '.join(f'{run.label} (seed {run.seed})' for run in own)}")
    table = read_loops(runs, baselines)
    print(f"{len(table)} rows read")

    tables = closed_loop(table)
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    figures = []
    for mode in runs:
        what = "blackout" if mode == "blackout" else "hidden"
        path = plot_loop(tables["curves"], mode, out / f"{mode}.png",
                         f"{mode} k = {K_SHOWN}: open loop (dotted) and closed loop on the {what} frames (solid, "
                         f"shaded frames), against the frame without the blackout (bands: standard errors; "
                         f"{args.data})")
        figures.append(path.name)
    report(tables)
    print(f"\nsaved in {out}: {', '.join(f'{n}.csv' for n in tables)}, {', '.join(figures)} "
          f"({(time.perf_counter() - started) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
