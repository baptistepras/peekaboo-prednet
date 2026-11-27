"""Check of step 4.1: refit the position probe with blackouts in its stream, and read it on the blackout frames.

For every seed of PredNet 5 layers and of the model trained with blackouts (the `blackout` model set), in turn:
1. fits the ridge and bin probes on the states of the model over a probe stream whose mix gives half of the
   sequences to blackouts (peekaboo/eval/blackout_probe.py), with the settings of scripts/fit_probes.py, and saves it
   in <run>/probes/position_blackout_mix.npz;
2. runs the model in open loop over the blackout and occlusion cells of the set, and reads the original probe
   (probes/position_model.npz) and the new one on the same states; saves the rows in
   <run>/eval/blackout_probe_<set>.parquet. A run whose rows exist is skipped, so a stopped check continues.
Then prints and saves the summary over the seeds (runs/analysis/<set>_blackout/blackout_probe/summary.csv): the probe
error on y on the blackout frames, the hidden frames of the occlusion cells, and the visible frames before the onset.
The original probe should give again the numbers of step 4.1 (6.93 and 1.27 px on the blackout frames).
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from peekaboo.data.conditions import GeneratorSettings
from peekaboo.data.dataset import OnTheFlyDataset
from peekaboo.data.mnist_pool import build_digit_pool
from peekaboo.data.render import RenderSettings
from peekaboo.data.store import StoredDataset
from peekaboo.device import describe_device, get_device
from peekaboo.eval.blackout_probe import BLACKOUT_SHARE, REGIONS, probe_mix, region_rows, summarize
from peekaboo.eval.extract import cell_range
from peekaboo.eval.tables import analysis_folder
from peekaboo.paths import DATASETS_DIR, RUNS_DIR
from peekaboo.probes.features import LAYER_LIMIT, collect_frames, iterate_frames
from peekaboo.probes.position import PositionProbe, fit_position_probe
from peekaboo.train.checkpoint import find_checkpoint, load_model
from peekaboo.train.sweep import MODEL_SETS, model_runs

MODES = ("blackout", "occlusion")
COLUMNS = ["sequence", "t", "condition", "k", "since_onset", "state", "y", "probe_y", "new_probe_y"]


def check_run(run_dir: Path, data_dir: Path, args: argparse.Namespace) -> pd.DataFrame:
    """Fit the probe with blackouts on one run and read both probes on the cells of the set."""
    device = get_device(args.device)
    path = find_checkpoint(run_dir, None)
    model, checkpoint = load_model(path, device)
    data_config = checkpoint["config"]["data"]
    digits = data_config["digits"]
    render_settings = RenderSettings.from_config(data_config)
    pool = build_digit_pool("train", digits["box_size"], digits["ink_threshold"], digits["val_size"],
                            digits["holdout_seed"], download=False)
    stream = OnTheFlyDataset(pool, GeneratorSettings.from_config(probe_mix(data_config)), render_settings, "probe",
                             args.seed)
    evaluation = StoredDataset(data_dir, render_settings=render_settings)  # drawn with the training colors
    frame_size = (evaluation.specs[0].frame_height, evaluation.specs[0].frame_width)
    options = {"batch_size": args.batch_size, "limit": LAYER_LIMIT, "workers": args.workers}
    print(f"{run_dir} ({path.name}, step {checkpoint['step']}) on {describe_device(device)}")

    started = time.perf_counter()
    frames = collect_frames(model, stream, args.train_sequences, device, window_only=True, **options)
    blackout = int((frames.rows["condition"] == "blackout").sum())
    new, _ = fit_position_probe(frames.features, frames.rows[["y", "x"]].to_numpy(),
                                frames.rows["sequence"].to_numpy(), frame_size, LAYER_LIMIT, seed=args.seed)
    new.save(run_dir / "probes" / "position_blackout_mix.npz")
    print(f"  probe with blackouts: {len(frames.rows)} frames ({blackout} of blackout sequences), ridge alpha "
          f"{new.alpha:g} ({time.perf_counter() - started:.0f} s)")
    del frames
    original = PositionProbe.load(run_dir / "probes" / "position_model.npz")

    parts = []
    for mode in MODES:
        first, size = cell_range(evaluation, mode)
        for features, rows, _ in iterate_frames(model, evaluation, size, device, start=first, window_only=True,
                                                **options):
            parts.append(rows.assign(probe_y=original.predict(features)[:, 0], new_probe_y=new.predict(features)[:, 0]))
        print(f"  {size} {mode} sequences read ({time.perf_counter() - started:.0f} s)")
    table = region_rows(pd.concat(parts, ignore_index=True))[COLUMNS + ["region"]]
    table.to_parquet(run_dir / "eval" / f"blackout_probe_{data_dir.name}.parquet", index=False)
    return table


def main() -> int:
    """Check every run of the blackout model set, then summarize over the seeds."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="test_v1", help="a set name in data/datasets/ or a folder")
    parser.add_argument("--runs", type=Path, default=RUNS_DIR, help="folder of the runs")
    parser.add_argument("--train-sequences", type=int, default=800, help="sequences of the probe stream")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0, help="probe stream and validation split, as fit_probes")
    parser.add_argument("--overwrite", action="store_true", help="check again the runs already checked")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    data_dir = Path(args.data) if Path(args.data).is_dir() else DATASETS_DIR / args.data
    labels = {run.key: run.label for run in MODEL_SETS["blackout"]}
    print(f"probe stream: {args.train_sequences} sequences, {100 * BLACKOUT_SHARE:.0f}% of them with a blackout")
    tables = []
    for run in model_runs("blackout"):
        run_dir = run.run_dir(args.runs)
        saved = run_dir / "eval" / f"blackout_probe_{data_dir.name}.parquet"
        if saved.exists() and not args.overwrite:
            print(f"{run_dir}: already checked ({saved.name})")
            table = pd.read_parquet(saved)
        else:
            table = check_run(run_dir, data_dir, args)
        tables.append(table.assign(model=labels[run.key], seed=run.seed))
    summary = summarize(pd.concat(tables, ignore_index=True))
    out = analysis_folder(args.runs, data_dir.name, "blackout") / "blackout_probe"
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "summary.csv", index=False)

    shown = summary.assign(value=summary.apply(lambda r: f"{r['mean']:.2f} [{r['low']:.2f}, {r['high']:.2f}]",
                                               axis=1))
    with pd.option_context("display.width", 250, "display.max_columns", None):
        for region in REGIONS:
            rows = shown[(shown["region"] == region) & (shown["k"] == "all")]
            print(f"\n{region}: probe |error| on y (px), {data_dir.name}, three seeds (95% interval)")
            print(rows.pivot(index="model", columns="probe", values="value").to_string())
        rows = shown[shown["region"] == "blackout"]
        print("\nblackout frames, per k: mean over the seeds (px)")
        print(rows.pivot_table(index=["model", "probe"], columns="k", values="mean").round(2).to_string())
        seeds = [c for c in summary if c.startswith("seed ")]
        print("\nblackout frames, every seed (px)")
        print(summary[(summary["region"] == "blackout") & (summary["k"] == "all")]
              .set_index(["model", "probe"])[seeds].round(2).to_string())
    print(f"\nsaved in {out}: summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
