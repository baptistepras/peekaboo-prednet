"""Inside the state (step 3.5): reads the extraction tables and the layer passes of every model of a model set (the
sweep by default) with both complete on a set (scripts/extract.py, with and without --layers), and the baselines of the
set, and writes the tables and figures of peekaboo/eval/layers.py. Nothing runs a model.

Saved in runs/analysis/<set>/layers/ (runs/analysis/<set>_blackout/layers/ with --models blackout):
- readouts.csv: per cell condition, frames (hidden, visible before the onset), model, and readout (all the layers,
  all fitted on visible frames, every layer alone, every layer but one), the probe's error on y and the decoder's
  correlation and IoU with 95% bootstrap intervals, and Kalman; readouts_paired.csv: every readout minus all the
  layers on the same sequences;
- curves.csv: the probe's error on y and the decoder's correlation of every readout against the frames since the
  onset, per k;
- belief.csv: per surprise kind and pair, the probe's distance to the center and the decoder's correlation of every
  readout over the frames after the reappearance, the impossible member minus the possible one;
- errors.csv and error_curves.csv: the error units of every PredNet layer relative to their visible level;
- the figures layers.png, visible.png, belief.png, and errors.png.

Every choice is fixed on dev_v1; test_v1 is analyzed once, with the same choices.
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from peekaboo.eval.correction import HORIZON
from peekaboo.eval.layers import (CELLS, K_SHOWN, inside, layer_runs, order, plot_belief, plot_errors, plot_layers,
                                  plot_visible, read_layers)
from peekaboo.eval.surprise import KINDS
from peekaboo.eval.tables import analysis_folder, available_runs, baselines_folder
from peekaboo.paths import RUNS_DIR
from peekaboo.train.sweep import MODEL_SETS


def wide(rows: pd.DataFrame, value: str = "mean", digits: int = 2) -> str:
    """A table with one line per model and one column per readout, in the order of the readouts."""
    table = rows.pivot_table(index="who", columns="readout", values=value, aggfunc="first", sort=False)
    table = table[sorted(table.columns, key=order)]
    return table.to_string(float_format=lambda v: f"{v:.{digits}f}", na_rep="")


def report(tables: dict[str, pd.DataFrame], table: pd.DataFrame) -> None:
    """Print the readouts on the hidden and visible frames, the belief after a surprise, the error units, and the
    check that the probe on all the layers is the probe of the extraction pass."""
    summary = tables["readouts"]
    for condition in CELLS:
        for frames in ("hidden", "visible"):
            for measure, label, digits in (("probe_y", "probe |error| on y (px)", 2),
                                           ("decoder_corr", "decoder correlation", 2)):
                rows = summary[(summary["condition"] == condition) & (summary["frames"] == frames)
                               & (summary["measure"] == measure)]
                rows = rows.assign(readout=rows["readout"].replace({"": "Kalman"}))
                where = "hidden frames" if frames == "hidden" else "visible frames before the onset"
                print(f"\n{condition.replace('_', ' ')}, {where}: {label}, per readout")
                print(wide(rows, digits=digits))

    belief = tables["belief"]
    for measure, label in (("probe", "probe distance to the center (px)"), ("decoder_corr", "decoder correlation")):
        own = belief[belief["measure"] == measure]
        print(f"\nafter the reappearance (frames 0 to {HORIZON - 1}), {label}: impossible minus possible member, per "
              f"readout (AB - B, BA - A for vanish)")
        for event in KINDS:
            pair = "BA - A" if event == "vanish" else "AB - B"
            rows = own[(own["event"] == event) & (own["pair"] == pair)]
            rows = rows.assign(readout=rows["readout"].replace({"": "Kalman"}))
            if len(rows):
                print(f"{event}, {pair}")
                print(wide(rows))

    errors = tables["errors"]
    if len(errors):
        print("\nerror units relative to their level on the visible frames before the onset (1 = the same)")
        for condition in CELLS:
            rows = errors[errors["condition"] == condition]
            shown = rows.pivot_table(index=["who", "layer"], columns=["phase", "region"], values="mean", sort=False)
            print(f"{condition.replace('_', ' ')}: mean over the hidden frames, and over the reappearance frame and "
                  f"the next two")
            print(shown.to_string(float_format=lambda v: f"{v:.2f}"))

    check = table[table["t"] >= 2]
    difference = (check["probe_all_y"] - check["probe_y"]).abs().groupby(check["model"], sort=False).max()
    print("\ncheck, largest |probe on all the layers - probe of the extraction pass| on y (px): "
          + ", ".join(f"{model} {value:.1e}" for model, value in difference.items()))


def main() -> int:
    """Read every available model on the set, write the tables and figures, and print the main numbers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="dev_v1", help="a set with complete extractions (dev_v1 or test_v1)")
    parser.add_argument("--runs", type=Path, default=RUNS_DIR, help="folder of the runs")
    parser.add_argument("--models", choices=MODEL_SETS, default="sweep",
                        help="the model set: the sweep (Phase 3), or blackout (step 4.1: PredNet 5 layers and the "
                             "same model trained with blackouts, in runs/analysis/<set>_blackout/)")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default runs/analysis/<set>/layers)")
    args = parser.parse_args()

    started = time.perf_counter()
    runs = layer_runs(available_runs(args.runs, args.data, args.models), args.data)
    baselines = baselines_folder(args.runs, args.data)
    if not runs or not (baselines / "info.json").exists():
        print(f"no complete layer pass on {args.data}: python -m scripts.extract --status --layers --data {args.data} "
              f"--models {args.models}")
        return 1
    out = args.out or analysis_folder(args.runs, args.data, args.models) / "layers"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{args.data}: {', '.join(f'{run.label} (seed {run.seed})' for run in runs)}")
    table = read_layers(runs, args.data, baselines)
    print(f"{len(table)} rows read")

    tables = inside(table)
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    plot_layers(tables["readouts"], out / "layers.png",
                f"readouts of single layers on the hidden frames of the occlusion sequences (95% intervals; "
                f"{args.data})")
    plot_visible(tables["curves"], out / "visible.png",
                 f"readouts of all the layers fitted on every frame or on visible frames only, occlusion k = "
                 f"{K_SHOWN} (bands: standard errors; {args.data})")
    plot_belief(tables["belief"], out / "belief.png",
                f"probe distance to the center over the {HORIZON} frames after the reappearance, impossible minus "
                f"possible member, per layer read (95% intervals; {args.data})")
    plot_errors(tables["error_curves"], out / "errors.png",
                f"error units of every layer relative to their visible level, k = {K_SHOWN} ({args.data})")
    report(tables, table)
    print(f"\nsaved in {out}: {', '.join(f'{n}.csv' for n in tables)}, layers.png, visible.png, belief.png, "
          f"errors.png ({(time.perf_counter() - started) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
