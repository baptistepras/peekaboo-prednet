"""Compare the models of the training sweep (step 2.10) from the evaluation outputs of their runs, once every run is
complete (python -m scripts.sweep says so). Nothing is recomputed: it reads each run's validation log and the reports
of the gate D16, the position probes, and the amodal decoder.

Prints the main numbers of every model, and saves in runs/sweep_summary/ the table (summary.csv), the validation MSE
of every run against the training step (validation.png, the failed runs dotted), the readouts against the frames
since the onset of the occlusion (since_onset.png), and a copy of the prediction GIFs of the reference model (PredNet 5
layers, shown in the README).
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

from peekaboo.eval.compare import plot_since_onset, plot_validation, summary_table
from peekaboo.paths import RUNS_DIR
from peekaboo.train.sweep import DATA, FAILED, FIGURES, REFERENCE, SWEEP, find_run


def main() -> int:
    """Read every run of the sweep, print the comparison, and save the table and the figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=RUNS_DIR, help="folder of the runs")
    parser.add_argument("--out", type=Path, default=RUNS_DIR / "sweep_summary", help="output folder")
    args = parser.parse_args()

    missing = [f"{run.key}/seed{run.seed}" for run in SWEEP
               if not all((run.run_dir(args.runs) / "eval" / f"{name}_{DATA}.parquet").exists()
                          for name in ("probe", "amodal"))]
    if missing:
        print(f"not evaluated yet: {', '.join(missing)}; run python -m scripts.sweep for the next command")
        return 1

    table = summary_table(list(SWEEP) + list(FAILED), args.runs)
    shown = table.set_index("model").T.map(lambda v: f"{v:.4g}" if isinstance(v, float) else str(v))
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(f"training sweep on {DATA} (errors in px; gate: MSE reduction against the blank frame on visible moving "
              f"frames)\n")
        print(shown.to_string())

    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "summary.csv", index=False)
    plot_validation({run.label: run.run_dir(args.runs) for run in SWEEP}, args.out / "validation.png",
                    failed={run.label: run.run_dir(args.runs) for run in FAILED
                            if (run.run_dir(args.runs) / "val_metrics.csv").exists()})
    tables = {}
    for run in SWEEP:
        evaluation = run.run_dir(args.runs) / "eval"
        tables[run.label] = (pd.read_parquet(evaluation / f"probe_{DATA}.parquet"),
                             pd.read_parquet(evaluation / f"amodal_{DATA}.parquet"))
    plot_since_onset(tables, args.out / "since_onset.png")
    reference = find_run(REFERENCE)
    gifs = [f"{reference.key}_{condition}.gif" for condition in FIGURES]
    for condition, name in zip(FIGURES, gifs):
        shutil.copyfile(reference.run_dir(args.runs) / "predictions_probe" / f"{condition}.gif", args.out / name)
    print(f"\nsaved in {args.out}: summary.csv, validation.png, since_onset.png, {', '.join(gifs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
