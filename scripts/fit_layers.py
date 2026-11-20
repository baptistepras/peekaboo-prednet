"""Fit the readouts of single layers of a trained model (step 3.5, peekaboo/probes/layers.py): the ridge position
probe and the amodal decoder on every layer alone, on every layer but one, on all the layers, and on all the layers
from the fully visible frames only.

1. Probes: the pooled states of the analysis windows of the same training stream sequences as scripts/fit_probes.py
   (split "probe", same seed and count), so the probe on all the layers is the ridge probe of that script.
2. Decoders: trained together on the same batches of the stream as scripts/fit_decoder.py (split "decoder").

Saves everything in <run>/probes/layers/ (position.npz, amodal_<readout>.npz, info.json) and prints the validation
error of every probe and the last loss of every decoder. Nothing is evaluated here: the readouts are read on a stored
set by python -m scripts.extract --run <run> --data <set> --layers, then compared by scripts/analyze_layers.py.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from peekaboo.data.conditions import GeneratorSettings
from peekaboo.data.dataset import OnTheFlyDataset
from peekaboo.data.mnist_pool import build_digit_pool
from peekaboo.data.render import RenderSettings
from peekaboo.device import describe_device, get_device
from peekaboo.probes.features import LAYER_LIMIT, collect_frames, layer_columns
from peekaboo.probes.layers import (VISIBLE_STATE, fit_layer_decoders, fit_layer_probes, layer_subsets, save_readouts,
                                    state_shapes)
from peekaboo.train.checkpoint import find_checkpoint, load_model


def main() -> int:
    """Fit the probes and decoders of every layer subset, print their validation scores, and save them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="run folder, for example runs/sweep_prednet5l/seed0")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint file name inside the run folder (default: best.pt, last.pt, or model.pt)")
    parser.add_argument("--train-sequences", type=int, default=800, help="training stream sequences for the probes")
    parser.add_argument("--steps", type=int, default=400, help="training batches of the decoders")
    parser.add_argument("--batch-size", type=int, default=8, help="sequences per training batch of the decoders")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0, help="probe and decoder streams, validation split")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    started = time.perf_counter()
    device = get_device(args.device)
    path = find_checkpoint(args.run, args.checkpoint)
    model, checkpoint = load_model(path, device)
    data_config = checkpoint["config"]["data"]
    digits = data_config["digits"]
    render_settings = RenderSettings.from_config(data_config)
    pool = build_digit_pool("train", digits["box_size"], digits["ink_threshold"], digits["val_size"],
                            digits["holdout_seed"], download=False)
    generator = GeneratorSettings.from_config(data_config)
    probe_stream = OnTheFlyDataset(pool, generator, render_settings, "probe", args.seed)
    decoder_stream = OnTheFlyDataset(pool, generator, render_settings, "decoder", args.seed, include_amodal=True)
    frame_size = tuple(probe_stream[0]["frames"].shape[-2:])
    shapes = state_shapes(model, frame_size, device)
    subsets = layer_subsets(len(shapes))
    print(f"{args.run} ({path.name}, step {checkpoint['step']}) on {describe_device(device)}: {len(shapes)} layers "
          f"{shapes}, {len(subsets)} readouts")

    frames = collect_frames(model, probe_stream, args.train_sequences, device, window_only=True,
                            workers=args.workers, batch_size=16, limit=LAYER_LIMIT)
    visible = int((frames.rows["state"] == VISIBLE_STATE).sum())
    print(f"probes: {len(frames.rows)} frames ({visible} fully visible) x {frames.features.shape[1]} features "
          f"({time.perf_counter() - started:.0f} s)")
    probes, errors = fit_layer_probes(frames.features, frames.rows, layer_columns(shapes, LAYER_LIMIT), LAYER_LIMIT,
                                      seed=args.seed)
    del frames
    print(f"  fitted ({time.perf_counter() - started:.0f} s)")

    print(f"decoders: {args.steps} batches of {args.batch_size} sequences")
    decoders, losses = fit_layer_decoders(model, decoder_stream, args.steps, device, args.batch_size, args.lr,
                                          workers=args.workers)
    table = pd.DataFrame({
        "layers": {name: " ".join(map(str, layers)) for name, layers in subsets.items()},
        "probe features": {name: len(columns) for name, columns in probes.columns.items()},
        "probe alpha": probes.alpha,
        "probe validation RMSE per axis (px)": {name: float(np.sqrt(error)) for name, error in errors.items()},
        "decoder loss, last 50": {name: float(np.mean(loss[-50:])) for name, loss in losses.items()}})
    print("\nvalidation (held out training sequences; the visible readouts on visible frames only):")
    print(table.to_string(float_format=lambda v: f"{v:.4g}"))

    info = {"run": str(args.run), "checkpoint": path.name, "step": int(checkpoint["step"]), "seed": args.seed,
            "train_sequences": args.train_sequences, "decoder_steps": args.steps, "shapes": shapes,
            "subsets": {name: list(layers) for name, layers in subsets.items()},
            "probes": {name: {"alpha": probes.alpha[name], "validation_mse": errors[name]} for name in subsets},
            "decoders": {name: {"loss_first": loss[0], "loss_last_50": float(np.mean(loss[-50:]))}
                         for name, loss in losses.items()}}
    folder = save_readouts(args.run, probes, decoders, info)
    print(f"\nsaved in {folder} ({(time.perf_counter() - started) / 60:.1f} min)")
    print(f"read them on a set with: PYTORCH_ENABLE_MPS_FALLBACK=0 python -m scripts.extract --run {args.run} "
          f"--data dev_v1 --layers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
