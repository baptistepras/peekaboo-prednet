"""Run a trained model once over a stored set and save per frame everything the Phase 3 analyses read (step 3.1).

With --run: the next frame scores of the model and of both baselines, what its predicted frames show, the position
probe and the amodal decoder of the run (and of its random initialization control), and for PredNet the activity of
its error units, in <run>/eval/frames_<set>/. Needs the run's probes and decoders (scripts/fit_probes.py and
scripts/fit_decoder.py). With --baselines: the programmed trackers and templates, once per set, in
runs/baselines/<set>/. With --status: what is extracted for the sweep models on a set, and the next command.

The rows are written by chunks of sequences; a stopped pass started again with the same command continues at the
first missing chunk.

With --layers (step 3.5): the readouts of single layers fitted by scripts/fit_layers.py, in a second, lighter pass
of the model, in <run>/eval/layers_<set>/; with --status, the state of these passes too.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from peekaboo.data.render import RenderSettings
from peekaboo.data.store import StoredDataset
from peekaboo.data.truth import STATE_NAMES
from peekaboo.device import describe_device, get_device
from peekaboo.eval.extract import (BASELINES_FORMAT, FRAMES_FORMAT, LAYERS_FORMAT, RANDOM_SEED, baseline_frames,
                                   chunks_written, extract_frames, is_complete, layer_frames, pass_format, read_frames,
                                   run_chunks)
from peekaboo.eval.tables import baselines_folder, frames_folder, layers_folder
from peekaboo.models import build_model
from peekaboo.paths import DATASETS_DIR, RUNS_DIR
from peekaboo.probes.amodal import AmodalDecoder
from peekaboo.probes.layers import has_readouts, load_readouts
from peekaboo.probes.position import PositionProbe
from peekaboo.train.checkpoint import find_checkpoint, load_model
from peekaboo.train.sweep import MPS, SWEEP


def status(runs_dir: Path, data: str, count: int, chunk: int, layers: bool) -> int:
    """Print what is extracted on a set for the baselines and every sweep model (or, with `layers`, the layer passes
    of every model), and the next command. A pass written with an older version of the columns is to be written
    again with --overwrite."""
    total = -(-count // chunk)
    jobs = [] if layers else [("baselines", baselines_folder(runs_dir, data), BASELINES_FORMAT,
                               f"python -m scripts.extract --baselines --data {data}")]
    for run in SWEEP:
        run_dir, name = run.run_dir(runs_dir), f"{run.key}/seed{run.seed}"
        command = f"{MPS} python -m scripts.extract --run runs/{name} --data {data}"
        if not layers:
            jobs.append((name, frames_folder(run_dir, data), FRAMES_FORMAT, command))
        elif has_readouts(run_dir):
            jobs.append((name, layers_folder(run_dir, data), LAYERS_FORMAT, f"{command} --layers"))
        else:  # the readouts are fitted first, once per run
            jobs.append((f"{name} (no readouts yet)", layers_folder(run_dir, data), LAYERS_FORMAT,
                         f"{MPS} python -m scripts.fit_layers --run runs/{name}"))
    print(f"{'layer passes' if layers else 'extraction'} on {data} ({count} sequences, {total} chunks of {chunk})\n")
    following = None
    for name, folder, version, command in jobs:
        done = is_complete(folder)
        written = chunks_written(folder)
        old = written > 0 and pass_format(folder) != version
        if old:
            state, done, command = f"older columns (format {pass_format(folder)})", False, f"{command} --overwrite"
        else:
            state = "complete" if done else (f"{written} of {total} chunks" if written else "not started")
        print(f"  {name:32s} {state}")
        if not done and following is None:
            following = command
    print(f"\nnext: {following}" if following else "\nevery extraction on this set is complete")
    return 0


def main() -> int:
    """Extract one run or the baselines on a set, or print the status of a set."""
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--run", type=Path, help="run folder, for example runs/sweep_prednet5l/seed0")
    target.add_argument("--baselines", action="store_true", help="extract the programmed baselines of the set")
    target.add_argument("--status", action="store_true", help="what is extracted on the set, and the next command")
    parser.add_argument("--data", default="dev_v1", help="a set name in data/datasets/ or a folder")
    parser.add_argument("--n", type=int, default=None, help="first sequences of the set (default: all)")
    parser.add_argument("--chunk", type=int, default=400, help="sequences per saved chunk")
    parser.add_argument("--checkpoint", default=None, help="checkpoint file name (default: best.pt, last.pt, model.pt)")
    parser.add_argument("--no-control", action="store_true",
                        help="skip the random initialization control (about halves the time)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--runs", type=Path, default=RUNS_DIR, help="folder of the runs (for --baselines, --status)")
    parser.add_argument("--overwrite", action="store_true", help="start again a pass written with other settings")
    parser.add_argument("--layers", action="store_true",
                        help="with --run: the pass of the readouts of single layers (scripts/fit_layers.py); with "
                             "--status: the state of these passes")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    data_dir = Path(args.data) if Path(args.data).is_dir() else DATASETS_DIR / args.data
    probe_set = StoredDataset(data_dir)
    count = min(args.n or len(probe_set), len(probe_set))
    if args.status:
        return status(args.runs, data_dir.name, count, args.chunk, args.layers)

    started = time.perf_counter()
    settings = {"data": data_dir.name, "count": count, "chunk": args.chunk}
    if args.baselines:
        settings["format"] = BASELINES_FORMAT
        dataset = StoredDataset(data_dir, include_amodal=True)
        folder = baselines_folder(args.runs, data_dir.name)
        print(f"baselines of {data_dir.name}: {count} sequences, in {folder}")

        def make_part(start: int, size: int) -> pd.DataFrame:
            """The baselines of one chunk."""
            return baseline_frames(dataset, start, size, args.batch_size, args.workers)
    elif args.layers:
        if not has_readouts(args.run):
            print(f"no readouts of single layers in {args.run}: run scripts.fit_layers first")
            return 1
        device = get_device(args.device)
        path = find_checkpoint(args.run, args.checkpoint)
        model, checkpoint = load_model(path, device)
        probes, decoders = load_readouts(args.run)
        dataset = StoredDataset(data_dir, render_settings=RenderSettings.from_config(checkpoint["config"]["data"]),
                                include_amodal=True)  # drawn with the training colors
        folder = layers_folder(args.run, data_dir.name)
        settings.update(format=LAYERS_FORMAT, checkpoint=path.name, step=int(checkpoint["step"]))
        print(f"{args.run} ({path.name}, step {checkpoint['step']}) on {describe_device(device)}: readouts of "
              f"{len(decoders)} layer subsets on {count} sequences of {data_dir.name}, in {folder}")

        def make_part(start: int, size: int) -> pd.DataFrame:
            """The rows of one chunk."""
            return layer_frames(model, dataset, start, size, device, probes, decoders, args.batch_size,
                                args.workers)
    else:
        device = get_device(args.device)
        path = find_checkpoint(args.run, args.checkpoint)
        model, checkpoint = load_model(path, device)
        config = checkpoint["config"]
        missing = [name for name in ("position_model", "position_random", "amodal_model", "amodal_random")
                   if not (args.run / "probes" / f"{name}.npz").exists()]
        if missing:
            print(f"missing in {args.run}/probes: {', '.join(missing)}; run scripts.fit_probes and scripts.fit_decoder "
                  f"first")
            return 1
        probe = PositionProbe.load(args.run / "probes" / "position_model.npz")
        decoder = AmodalDecoder.load(args.run / "probes" / "amodal_model.npz")
        control = None
        if not args.no_control:
            torch.manual_seed(RANDOM_SEED)  # the random model the control probe and decoder were fitted on
            control = (build_model(config["model"]).to(device).eval(),
                       PositionProbe.load(args.run / "probes" / "position_random.npz"),
                       AmodalDecoder.load(args.run / "probes" / "amodal_random.npz"))
        dataset = StoredDataset(data_dir, render_settings=RenderSettings.from_config(config["data"]),
                                include_amodal=True)  # drawn with the training colors
        folder = frames_folder(args.run, data_dir.name)
        settings.update(format=FRAMES_FORMAT, checkpoint=path.name, step=int(checkpoint["step"]),
                        control=control is not None)
        print(f"{args.run} ({path.name}, step {checkpoint['step']}) on {describe_device(device)}: {count} sequences "
              f"of {data_dir.name}{'' if control else ', without the random initialization control'}, in {folder}")

        def make_part(start: int, size: int) -> pd.DataFrame:
            """The rows of one chunk."""
            return extract_frames(model, dataset, start, size, device, probe, decoder, control, args.batch_size,
                                  args.workers)

    if args.overwrite and folder.exists():
        for old in list(folder.glob("part_*")) + [folder / "info.json"]:
            old.unlink(missing_ok=True)
    run_chunks(folder, count, args.chunk, make_part, settings)
    table = read_frames(folder)
    minutes = (time.perf_counter() - started) / 60
    print(f"\n{len(table)} rows x {table.shape[1]} columns in {folder} ({minutes:.1f} min)")
    if args.layers and not args.baselines:  # a quick look: the hidden frames of the occlusion sequences
        hidden = table[(table["state"] == STATE_NAMES.index("occluded")) & table["in_window"]
                       & (table["condition"] == "occlusion")]
        names = [c.removeprefix("decoder_").removesuffix("_corr") for c in table if c.endswith("_corr")]
        print(f"fully hidden frames of occlusion sequences ({len(hidden)}), probe y error in px and decoder "
              f"correlation:")
        print(pd.DataFrame({"probe y": [np.abs(hidden[f"probe_{n}_y"] - hidden["y"]).mean() for n in names],
                            "decoder": [hidden[f"decoder_{n}_corr"].mean() for n in names]}, index=names)
              .to_string(float_format=lambda v: f"{v:.2f}"))
    elif not args.baselines:  # a quick look, to compare with the step 2.10 numbers
        scored = table[table["t"] >= 1]
        print(f"MSE on every frame: model {scored['model_mse'].mean():.5f}, copy {scored['copy_mse'].mean():.5f}, "
              f"blank {scored['blank_mse'].mean():.5f}")
        hidden = table[(table["state"] == STATE_NAMES.index("occluded")) & table["in_window"]
                       & (table["condition"] == "occlusion")]
        print(f"fully hidden frames of occlusion sequences ({len(hidden)}): probe y error "
              f"{np.abs(hidden['probe_y'] - hidden['y']).mean():.2f} px, decoder correlation "
              f"{hidden['decoder_corr'].mean():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
