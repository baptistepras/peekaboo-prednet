"""Record of the step 2.4 decision: two short pilots to find how PredNet can learn to draw the digits.

The first 10k step pilot, with PredNet's own loss, converged to the blank solution: it drew the bar and the background
but never the digit, whose red ink is only about 0.4% of the loss values. Two fixes were tried for 2000 steps each, at
a constant learning rate, with the same initialization and training sequences:
- white: the digit drawn on all three channels (three times its weight in the loss), PredNet and its loss unchanged;
- weighted: red digit as before, with the visible digit pixels of the target frames weighted by --weight in the loss.
White digits stayed at the blank solution (0.5% above its MSE at step 2000); the weighted loss reached 57% below it
and passed the gate D16 (see the README, "Pilot results"). The weighted loss became loss.digit_weight in the training
configs, which this script now uses. Each run goes to runs/trial_<variant>/seed0/ and ends with a table of the
validation MSE against the blank baseline (the frame with the digit erased) and two prediction sheets.
"""

import argparse
import shutil
import sys

import numpy as np
import pandas as pd
import torch

from peekaboo.config import load_config
from peekaboo.data.conditions import GeneratorSettings
from peekaboo.data.dataset import OnTheFlyDataset, prepare_batch
from peekaboo.data.mnist_pool import build_digit_pool
from peekaboo.data.occluder import VisibilityThresholds
from peekaboo.data.render import RenderSettings, compose_observed, render_amodal, render_sequence
from peekaboo.data.store import StoredDataset
from peekaboo.device import describe_device, get_device
from peekaboo.models import build_model
from peekaboo.paths import CONFIGS_DIR, DATASETS_DIR, PROJECT_ROOT, RUNS_DIR
from peekaboo.seeding import seed_everything
from peekaboo.train.checkpoint import load_model
from peekaboo.train.loop import TrainSettings, train
from peekaboo.train.metrics import read_csv
from peekaboo.viz.frames import describe
from peekaboo.viz.predictions import frames_around_event, prediction_sheet

REFERENCE = "the 10k pilot had a validation MSE of 0.00161 at step 2000, 9% above its blank baseline 0.00148"


def blank_mse(dataset: StoredDataset, render_settings: RenderSettings, count: int) -> float:
    """MSE per pixel of the blank baseline (the frames without the digit) on frames 1 and later of the first
    `count` sequences, with the given colors."""
    total, pixels = 0.0, 0
    for spec in dataset.specs[:count]:
        amodal = render_amodal(spec, dataset.sprites[spec.digit_index])
        observed = compose_observed(amodal, spec, render_settings)[1:] / 255.0
        empty = compose_observed(np.zeros_like(amodal), spec, render_settings)[1:] / 255.0
        total += float(((observed - empty) ** 2).sum())
        pixels += observed.size
    return total / pixels


def main() -> int:
    """Train one trial variant for a few thousand steps, then compare it with the blank baseline and draw it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("white", "weighted"), required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--weight", type=float, default=10.0, help="weight of the digit pixels (weighted variant)")
    parser.add_argument("--val-sequences", type=int, default=None, help="default: all of val_v1")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = load_config(CONFIGS_DIR / "train" / "pilot_prednet5l.yaml")
    config["name"] = f"trial_{args.variant}"
    config["train"]["steps"] = args.steps
    config["optimizer"]["lr_drop_at"] = None  # constant rate, as in the first 5000 steps of the 10k pilot
    config["validation"]["every"] = min(500, args.steps)
    config["validation"]["sequences"] = args.val_sequences
    data_config = load_config(PROJECT_ROOT / config["data"])
    if args.variant == "white":
        data_config["render"]["digit_rgb"] = [1.0, 1.0, 1.0]
    config["loss"] = {"digit_weight": args.weight if args.variant == "weighted" else 1.0}
    run_config = {"name": config["name"], "train_config": config,
                  "model": load_config(PROJECT_ROOT / config["model"]), "data": data_config}
    run_dir = RUNS_DIR / config["name"] / "seed0"
    if run_dir.exists():
        if not args.overwrite:
            print(f"{run_dir} already exists: pass --overwrite to replace it")
            return 1
        shutil.rmtree(run_dir)

    seed_everything(0)
    device = get_device(args.device)
    render_settings = RenderSettings.from_config(data_config)
    digits = data_config["digits"]
    pool = build_digit_pool("train", digits["box_size"], digits["ink_threshold"], digits["val_size"],
                            digits["holdout_seed"], download=False)
    train_data = OnTheFlyDataset(pool, GeneratorSettings.from_config(data_config), render_settings, "train", 0)
    val_data = StoredDataset(DATASETS_DIR / "val_v1", render_settings=render_settings)
    val_count = min(args.val_sequences or len(val_data), len(val_data))
    reference = blank_mse(val_data, render_settings, val_count)
    model = build_model(run_config["model"]).to(device)
    settings = TrainSettings.from_config(config)
    detail = "white digits" if args.variant == "white" else f"digit pixels weighted x{args.weight:g}"
    print(f"{config['name']}: {detail}, {args.steps} steps on {describe_device(device)}, blank baseline MSE "
          f"{reference:.5f} on {val_count} validation sequences")
    train(run_dir, model, train_data, val_data, settings, run_config, device)

    rows = read_csv(run_dir / "val_metrics.csv")
    table = pd.DataFrame({"step": [int(r["step"]) for r in rows], "val_mse": [float(r["val_mse"]) for r in rows]})
    table["blank_mse"] = reference
    table["reduction_vs_blank"] = 1.0 - table["val_mse"] / reference
    print(f"\n{config['name']}: validation MSE against the blank baseline (draws digits when clearly above 0)")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"for comparison, {REFERENCE}")

    best, _ = load_model(run_dir / "best.pt", device)
    thresholds = VisibilityThresholds(**val_data.info["thresholds"])
    picks = [i for i, spec in enumerate(val_data.specs) if spec.condition == "occlusion"][:2]
    batch = prepare_batch(torch.utils.data.default_collate([val_data[i] for i in picks]), device)
    with torch.no_grad():
        predictions = best(batch["frames"])["prediction"].permute(0, 1, 3, 4, 2).cpu().numpy()
    out = run_dir / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    for i, index in enumerate(picks):
        spec = val_data.specs[index]
        rendered = render_sequence(spec, val_data.sprites[spec.digit_index], render_settings, thresholds)
        title = f"{config['name']} ({detail}), best step, val_v1 #{index}: {describe(spec)}"
        prediction_sheet(rendered, predictions[i], frames_around_event(spec), title).save(
            out / f"occlusion_{index:05d}.png")
    print(f"saved {len(picks)} prediction sheets in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
