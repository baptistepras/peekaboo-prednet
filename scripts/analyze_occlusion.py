"""Tracking through occlusion (RQ1, step 3.2): reads the extraction tables of every sweep model with a complete pass on
a set (scripts/extract.py) and the baselines of the set, and writes the tables and figures of
peekaboo/eval/occlusion.py. Nothing runs a model.

Saved in runs/analysis/<set>/occlusion/:
- curves.csv and curves_speed.csv: every readout against j, the frames since the onset, per k (and speed);
- hidden.csv and hidden_paired.csv: the readouts on the hidden frames with 95% bootstrap intervals, and the paired
  differences against the random initialization control, the trackers and templates, and the reference model;
- decay.csv: the slope of every readout over the hidden frames and its value at the last hidden frame, per k;
- emergence.csv, emergence_summary.csv, emergence_paired.csv, and timing.csv: the predicted frames when the digit
  comes out, against the copy of the last frame and the blank frame;
- the figures occlusion.png, occlusion_speed.png, hidden_bounce.png, decay.png, and emergence.png.

Every choice is fixed on dev_v1; test_v1 is analyzed once, with the same choices.
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from peekaboo.eval.occlusion import (COLUMNS, CONDITIONS, curves, decay, emergence, hidden_summary, plot_curves,
                                     plot_decay, plot_emergence, plot_speeds, timing)
from peekaboo.eval.tables import available_runs, baselines_folder, read_runs
from peekaboo.paths import RUNS_DIR
from peekaboo.train.sweep import REFERENCE, find_run


def interval(row: pd.Series, digits: int = 2) -> str:
    """A mean and its 95% interval."""
    return f"{row['mean']:.{digits}f} [{row['low']:.{digits}f}, {row['high']:.{digits}f}]"


def report(summary: pd.DataFrame, paired: pd.DataFrame, slopes: pd.DataFrame, emerging: pd.DataFrame,
           times: pd.DataFrame) -> None:
    """Print the main numbers of every model, per condition, on all k together."""
    for condition in CONDITIONS:
        print(f"\n{condition.replace('_', ' ')}, hidden frames (all k; 95% bootstrap intervals over sequences)")
        own = summary[(summary["condition"] == condition) & (summary["k"] == "all")]
        diff = paired[(paired["condition"] == condition) & (paired["k"] == "all")]
        lines = []
        for who in own.loc[own["kind"] == "model", "who"].unique():
            row = {"model": who}
            for name, label in (("probe_y", "probe y (px)"), ("probe_x", "probe x (px)"),
                                ("decoder_corr", "decoder corr")):
                row[label] = interval(own[(own["readout"] == name) & (own["who"] == who)].iloc[0])
            kalman = diff[(diff["readout"] == "probe_y") & (diff["model"] == who) & (diff["against"] == "Kalman")]
            row["probe y - Kalman"] = interval(kalman.iloc[0])
            lines.append(row)
        for who in own.loc[own["kind"] == "reference", "who"].unique():
            row = {"model": who}
            for name, label in (("probe_y", "probe y (px)"), ("probe_x", "probe x (px)"),
                                ("decoder_corr", "decoder corr")):
                match = own[(own["readout"] == name) & (own["who"] == who)]
                row[label] = interval(match.iloc[0]) if len(match) else ""
            lines.append(row)
        print(pd.DataFrame(lines).fillna("").to_string(index=False))

        decay_rows = slopes[(slopes["condition"] == condition) & (slopes["readout"] == "probe_y")
                            & (slopes["kind"] == "model")]
        print("\nslope of the probe error on y over the hidden frames (px/frame), per k")
        print(decay_rows.pivot(index="who", columns="k", values="slope_mean").round(3).to_string())

        out = emerging[(emerging["condition"] == condition) & (emerging["k"] == "all")]
        when = times[(times["condition"] == condition) & (times["k"] == "all")].set_index("who")
        print("\nemergence (frames 0 to 2 since the reappearance) and timing of the drawn digit")
        lines = []
        for who in when.index:
            row = {"model": who}
            for name, label in (("exit_ink", "exit ink"), ("readout_y", "readout y (px)"), ("bar_mse", "bar MSE")):
                match = out[(out["measure"] == name) & (out["who"] == who)]
                row[label] = interval(match.iloc[0], 5 if name == "bar_mse" else 2) if len(match) else ""
            for name in ("early", "on_time", "late", "never"):
                row[name.replace("_", " ")] = f"{100 * when.loc[who, name]:.0f}%"
            lines.append(row)
        print(pd.DataFrame(lines).to_string(index=False))


def main() -> int:
    """Read every available model on the set, write the tables and figures, and print the main numbers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="dev_v1", help="a set with complete extractions (dev_v1 or test_v1)")
    parser.add_argument("--runs", type=Path, default=RUNS_DIR, help="folder of the runs")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default runs/analysis/<set>/occlusion)")
    args = parser.parse_args()

    started = time.perf_counter()
    runs = available_runs(args.runs, args.data)
    baselines = baselines_folder(args.runs, args.data)
    if not runs or not (baselines / "info.json").exists():
        print(f"no complete extraction on {args.data}: python -m scripts.extract --status --data {args.data}")
        return 1
    out = args.out or args.runs / "analysis" / args.data / "occlusion"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{args.data}: {', '.join(f'{run.label} (seed {run.seed})' for run in runs)}")
    table = read_runs(runs, baselines, list(CONDITIONS), COLUMNS, roles=["none"])
    reference = find_run(REFERENCE).label
    sequences = table.groupby("condition")["sequence"].nunique()
    print(f"{len(table)} rows: {', '.join(f'{n} {c} sequences' for c, n in sequences.items())}")

    by_k, by_speed = curves(table), curves(table, by_speed=True)
    summary, paired = hidden_summary(table, reference)
    slopes = decay(table)
    emergence_curve, emerging, emerging_paired = emergence(table)
    times = timing(table)
    tables = {"curves": by_k, "curves_speed": by_speed, "hidden": summary, "hidden_paired": paired, "decay": slopes,
              "emergence": emergence_curve, "emergence_summary": emerging, "emergence_paired": emerging_paired,
              "timing": times}
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)

    plot_curves(by_k, "occlusion", ("probe_y", "probe_x", "decoder_corr"), reference, out / "occlusion.png",
                f"occlusion: readouts of the state against the frames since the onset, hidden frames shaded "
                f"(mean and standard error over sequences; {args.data})")
    plot_speeds(by_speed, "occlusion", "probe_y", reference, out / "occlusion_speed.png",
                f"occlusion: probe error on y per k and speed ({args.data})")
    plot_curves(by_k, "hidden_bounce", ("probe_x", "probe_y", "decoder_corr"), reference, out / "hidden_bounce.png",
                f"hidden bounce: readouts of the state against the frames since the onset ({args.data})")
    plot_decay(slopes, "occlusion", reference, out / "decay.png",
               f"occlusion: the state at the last hidden frame and the decay over the hidden frames, against k "
               f"(95% intervals; {args.data})")
    plot_emergence(emergence_curve, out / "emergence.png",
                   f"predicted frames when the digit comes out (mean and standard error; {args.data})")
    report(summary, paired, slopes, emerging, times)
    print(f"\nsaved in {out}: {', '.join(f'{n}.csv' for n in tables)}, occlusion.png, occlusion_speed.png, "
          f"hidden_bounce.png, decay.png, emergence.png ({(time.perf_counter() - started) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
