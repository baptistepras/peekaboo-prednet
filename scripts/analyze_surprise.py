"""Surprise (RQ3, step 3.4): reads the extraction tables of every model of a model set (the sweep by default) with a
complete pass on a set (scripts/extract.py) and the baselines of the set, and writes the tables and figures of
peekaboo/eval/surprise.py. Nothing runs a model.

Saved in runs/analysis/<set>/surprise/ (runs/analysis/<set>_blackout/surprise/ with --models blackout):
- index.csv: per surprise kind, pair (AB - B, BA - A), scope (all, per k and speed for the MSE), signal, and line,
  the surprise index over the window with its 95% bootstrap interval, the AUROC with its paired bootstrap interval,
  and the share of tuples where the impossible member is higher;
- contrast.csv: the key contrast (C13), hidden bounce against occlusion and `direction` AB against hidden bounce;
- curves.csv: every signal against the frames since the window start, per kind and role;
- the figures surprise.png, layers.png, and curves.png.

Every choice is fixed on dev_v1; test_v1 is analyzed once, with the same choices.
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from peekaboo.eval.surprise import (COLUMNS, CONDITIONS, CONTRASTS, KINDS, MAIN, PAIRS, ROWS, SIGNALS, WINDOW,
                                    plot_auroc, plot_curves, plot_layers, surprise)
from peekaboo.eval.tables import analysis_folder, available_runs, baselines_folder, read_runs
from peekaboo.paths import RUNS_DIR
from peekaboo.train.sweep import MODEL_SETS


def index_text(row: pd.Series) -> str:
    """A surprise index and its 95% interval."""
    return f"{row['index_mean']:.3g} [{row['index_low']:.3g}, {row['index_high']:.3g}]"


def auroc_text(row: pd.Series) -> str:
    """An AUROC and its 95% interval."""
    return f"{row['auroc']:.2f} [{row['auroc_low']:.2f}, {row['auroc_high']:.2f}]"


def report(tables: dict[str, pd.DataFrame]) -> None:
    """Print, for the main signals, the AUROC and the surprise index of every line on every kind; then the key
    contrast; then the AUROC of the error units of every layer."""
    index = tables["index"]
    index = index[index["scope"] == "all"]
    for signal in MAIN:
        own = index[index["signal"] == signal]
        for what, text in (("AUROC (95% interval)", auroc_text),
                           ("surprise index, mean over the window (95% interval)", index_text),
                           ("share of tuples where the impossible member is higher",
                            lambda row: f"{row['higher']:.2f}")):
            print(f"\n{SIGNALS[signal][0]}: {what}")
            lines = []
            for who in own["who"].unique():
                row = {"line": who}
                for kind in KINDS:
                    for role, control in PAIRS.items():
                        match = own[(own["event"] == kind) & (own["pair"] == f"{role} - {control}")
                                    & (own["who"] == who)]
                        if len(match):
                            row[f"{kind} {role} - {control}"] = text(match.iloc[0])
                lines.append(row)
            print(pd.DataFrame(lines).fillna("").to_string(index=False))

    contrast = tables["contrast"]
    contrast = contrast[contrast["signal"].isin(MAIN)]
    print("\nkey contrast (C13): AUROC of the window signal, unpaired (95% interval), with the mean of each side")
    lines = []
    for (signal, who), own in contrast.groupby(["signal", "who"], sort=False):
        row = {"signal": signal, "line": who}
        for positive, negative in CONTRASTS:
            match = own[(own["positive"] == positive) & (own["negative"] == negative)]
            if len(match):
                value = match.iloc[0]
                row[f"{positive} vs {negative}"] = (f"{auroc_text(value)} ({value['positive_mean']:.3g} vs "
                                                    f"{value['negative_mean']:.3g})")
        lines.append(row)
    print(pd.DataFrame(lines).fillna("").to_string(index=False))

    layers = index[index["signal"].str.match(r"e\d") & (index["pair"] == "AB - B") & (index["kind"] == "model")]
    if len(layers):
        for what, column in (("share of tuples where AB is higher (paired)", "higher"),
                             ("AUROC (also falls when the layer varies between tuples)", "auroc")):
            print(f"\nerror units per layer, AB against B: {what}; whole frame, then bar")
            lines = []
            for (who, signal), own in layers.groupby(["who", "signal"], sort=False):
                row = {"line": who, "signal": signal}
                row.update({kind: f"{own.loc[own['event'] == kind, column].iloc[0]:.2f}" for kind in KINDS
                            if (own["event"] == kind).any()})
                lines.append(row)
            print(pd.DataFrame(lines).to_string(index=False))


def main() -> int:
    """Read every available model on the set, write the tables and figures, and print the main numbers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="dev_v1", help="a set with complete extractions (dev_v1 or test_v1)")
    parser.add_argument("--runs", type=Path, default=RUNS_DIR, help="folder of the runs")
    parser.add_argument("--models", choices=MODEL_SETS, default="sweep",
                        help="the model set: the sweep (Phase 3), or blackout (step 4.1: PredNet 5 layers and the "
                             "same model trained with blackouts, in runs/analysis/<set>_blackout/)")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default runs/analysis/<set>/surprise)")
    args = parser.parse_args()

    started = time.perf_counter()
    runs = available_runs(args.runs, args.data, args.models)
    baselines = baselines_folder(args.runs, args.data)
    if not runs or not (baselines / "info.json").exists():
        print(f"no complete extraction on {args.data}: python -m scripts.extract --status --data {args.data} "
              f"--models {args.models}")
        return 1
    out = args.out or analysis_folder(args.runs, args.data, args.models) / "surprise"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{args.data}: {', '.join(f'{run.label} (seed {run.seed})' for run in runs)}")
    table = read_runs(runs, baselines, list(CONDITIONS), COLUMNS, rows=ROWS)
    print(f"{len(table)} rows read; window of {WINDOW} frames")

    tables = surprise(table)
    counts = tables["index"]
    counts = counts[(counts["scope"] == "all") & (counts["signal"] == "mse") & (counts["who"] == "copy")]
    print("tuples: " + ", ".join(f"{e} {p} {n}" for e, p, n in counts[["event", "pair", "tuples"]].itertuples(
        index=False)))
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)

    plot_auroc(tables["index"], tables["contrast"], out / "surprise.png",
               f"AUROC of the window signal, impossible against possible member (95% paired bootstrap intervals; "
               f"{args.data})")
    plot_layers(tables["index"], out / "layers.png",
                f"error units of every layer, AB against B (AUROC with 95% intervals; {args.data})")
    plot_curves(tables["curves"], out / "curves.png",
                f"signals against the frames since the window start (gray band: the window; {args.data})")
    report(tables)
    print(f"\nsaved in {out}: {', '.join(f'{n}.csv' for n in tables)}, surprise.png, layers.png, curves.png "
          f"({(time.perf_counter() - started) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
