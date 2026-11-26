"""Correction after the reappearance (RQ2, step 3.3): reads the extraction tables of every model of a model set (the
sweep by default) with a complete pass on a set (scripts/extract.py) and the baselines of the set, and writes the tables
and figures of peekaboo/eval/correction.py. Nothing runs a model.

Saved in runs/analysis/<set>/correction/ (runs/analysis/<set>_blackout/correction/ with --models blackout):
- taus.csv: the threshold of every model and reference for each readout;
- curves.csv: every readout against j, the frames since the reappearance, per event and role;
- latency.csv: per event, role, k and speed, readout, and line, the shares found, censored, and short, and the
  restricted mean latency and mean error over the first frames, with 95% bootstrap intervals;
- latency_paired.csv: the paired differences of each surprising member against its control (AB - B, BA - A), and
  of each model against the reference model;
- the figures correction.png and latency.png.

Every choice is fixed on dev_v1; test_v1 is analyzed once, with the same choices.
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from peekaboo.eval.correction import (COLUMNS, CONDITIONS, HORIZON, M, PANELS, ROWS, SURPRISING, correction,
                                      latency_readouts, plot_curves, plot_latency)
from peekaboo.eval.tables import analysis_folder, available_runs, baselines_folder, read_runs
from peekaboo.paths import RUNS_DIR
from peekaboo.train.sweep import MODEL_SETS, REFERENCE, find_run


def interval(row: pd.Series, prefix: str = "", digits: int = 2) -> str:
    """A mean and its 95% interval."""
    return f"{row[prefix + 'mean']:.{digits}f} [{row[prefix + 'low']:.{digits}f}, {row[prefix + 'high']:.{digits}f}]"


def report(tables: dict[str, pd.DataFrame]) -> None:
    """Print the thresholds, then per readout the latency of every line on every event and the paired differences
    of the surprising members against their controls."""
    taus = tables["taus"]
    print("\ntau of every line (the 90% quantile of its error on visible frames, with the floor)")
    print(taus.pivot(index="who", columns="readout", values="tau").reindex(taus["who"].unique()).to_string(
        float_format=lambda v: f"{v:.3g}"))
    summary, paired = tables["latency"], tables["latency_paired"]
    summary, paired = summary[summary["scope"] == "all"], paired[paired["scope"] == "all"]
    for readout in latency_readouts():
        own = summary[summary["readout"] == readout.name]
        print(f"\n{readout.name}: latency in frames (restricted mean, censored at {HORIZON - M + 1}; censored "
              f"share in parentheses)")
        lines = []
        for who in own["who"].unique():
            row = {"line": who}
            for event, role, control in PANELS:
                for member in (role, control):
                    match = own[(own["event"] == event) & (own["role"] == member) & (own["who"] == who)]
                    if member is None or match.empty:
                        continue
                    value = match.iloc[0]
                    censored = f" ({100 * value['censored']:.0f}%)" if value["censored"] > 0 else ""
                    label = event if member == "none" else f"{event} {member}"
                    row[label] = f"{value['latency_mean']:.2f}{censored}"
            lines.append(row)
        print(pd.DataFrame(lines).fillna("").to_string(index=False))

        for measure, what, digits in (("latency", "latency", 2), ("error", f"mean {readout.name} error", 5 if
                                                                       readout.name == "digit_mse" else 2)):
            diff = paired[(paired["readout"] == readout.name) & (paired["measure"] == measure)
                          & (paired["against"].isin(SURPRISING.values()))]
            print(f"\n{readout.name}: {what} of the surprising member minus its control, frames 0 to "
                  f"{HORIZON - 1} (95% intervals)")
            lines = []
            for who in diff["who"].unique():
                row = {"line": who}
                for event, role, control in PANELS:
                    match = diff[(diff["event"] == event) & (diff["role"] == role) & (diff["who"] == who)]
                    if len(match):
                        row[f"{event} {role} - {control}"] = interval(match.iloc[0], digits=digits)
                lines.append(row)
            print(pd.DataFrame(lines).fillna("").to_string(index=False))


def main() -> int:
    """Read every available model on the set, write the tables and figures, and print the main numbers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="dev_v1", help="a set with complete extractions (dev_v1 or test_v1)")
    parser.add_argument("--runs", type=Path, default=RUNS_DIR, help="folder of the runs")
    parser.add_argument("--models", choices=MODEL_SETS, default="sweep",
                        help="the model set: the sweep (Phase 3), or blackout (step 4.1: PredNet 5 layers and the "
                             "same model trained with blackouts, in runs/analysis/<set>_blackout/)")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default runs/analysis/<set>/correction)")
    args = parser.parse_args()

    started = time.perf_counter()
    runs = available_runs(args.runs, args.data, args.models)
    baselines = baselines_folder(args.runs, args.data)
    if not runs or not (baselines / "info.json").exists():
        print(f"no complete extraction on {args.data}: python -m scripts.extract --status --data {args.data} "
              f"--models {args.models}")
        return 1
    out = args.out or analysis_folder(args.runs, args.data, args.models) / "correction"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{args.data}: {', '.join(f'{run.label} (seed {run.seed})' for run in runs)}")
    table = read_runs(runs, baselines, list(CONDITIONS), COLUMNS, rows=ROWS)
    reference = find_run(REFERENCE).label
    print(f"{len(table)} rows read")

    tables = correction(table, reference)
    counts = tables["latency"]
    counts = counts[(counts["scope"] == "all") & (counts["who"] == reference) & (counts["readout"] == "probe")]
    print("sequences: " + ", ".join(f"{e} {r} {n}" for e, r, n in counts[["event", "role", "sequences"]].itertuples(
        index=False)))
    short = counts.set_index(["event", "role"])["short"]
    if (short > 0).any():
        print("sequences whose window ends before the latency could be read: "
              + ", ".join(f"{e} {r} {100 * v:.1f}%" for (e, r), v in short[short > 0].items()))
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)

    plot_curves(tables["curves"], out / "correction.png",
                f"readouts against the frames since the reappearance (mean and standard error over sequences; "
                f"{args.data})")
    plot_latency(tables["latency"], out / "latency.png",
                 f"frames to be right again after the reappearance (95% intervals; open: the matched control; "
                 f"{args.data})")
    report(tables)
    print(f"\nsaved in {out}: {', '.join(f'{n}.csv' for n in tables)}, correction.png, latency.png "
          f"({(time.perf_counter() - started) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
