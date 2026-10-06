"""Draw a trained model's predictions on stored sequences, next to the truth: actual, predicted, and error rows.

Loads the checkpoint of a run, predicts every frame of a few sequences of a stored set, and saves in
<run>/predictions/ one sheet per sequence (<condition>_<index>.png) and a GIF with the sequences side by side.
With --probe, a yellow cross marks the position read in the model's state by the probe fitted with
scripts/fit_probes.py. Nothing is trained: the figures can be redrawn at any time from the saved model.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from peekaboo.data.dataset import CONDITION_NAMES, prepare_batch
from peekaboo.data.occluder import VisibilityThresholds
from peekaboo.data.render import render_sequence
from peekaboo.data.store import StoredDataset
from peekaboo.device import describe_device, get_device
from peekaboo.eval.next_frame import MIN_CONTEXT
from peekaboo.paths import DATASETS_DIR
from peekaboo.probes.features import frame_features
from peekaboo.probes.position import PositionProbe
from peekaboo.train.checkpoint import find_checkpoint, load_model
from peekaboo.viz.frames import describe, save_gif
from peekaboo.viz.predictions import frames_around_event, prediction_animation, prediction_sheet


def main() -> int:
    """Load the model, predict the chosen sequences, and save one sheet per sequence and a GIF."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="run folder, for example runs/bench/prednet_5l_...")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint file name inside the run folder (default: best.pt, last.pt, or model.pt, "
                             "the first that exists)")
    parser.add_argument("--data", default="val_v1", help="a set name in data/datasets/ or a folder")
    parser.add_argument("--condition", default="occlusion", choices=CONDITION_NAMES + ("any",))
    parser.add_argument("--k", type=int, default=None, help="keep only sequences with this k")
    parser.add_argument("--speed", type=int, default=None, help="keep only sequences with this speed")
    parser.add_argument("--n", type=int, default=4, help="number of sequences")
    parser.add_argument("--skip", type=int, default=0, help="skip the first matching sequences")
    parser.add_argument("--frames", choices=("event", "all"), default="event",
                        help="sheets show the frames around the crossing, or every frame")
    parser.add_argument("--step", type=int, default=1, help="sheets show one frame out of this many")
    parser.add_argument("--columns", type=int, default=10)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--gain", type=float, default=2.0, help="brightness gain of the error row")
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--probe", action="store_true",
                        help="draw the position read by the probe <run>/probes/position_model.npz (yellow cross)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None,
                        help="default: <run>/predictions, or <run>/predictions_probe with --probe")
    args = parser.parse_args()

    device = get_device(args.device)
    path = find_checkpoint(args.run, args.checkpoint)
    model, checkpoint = load_model(path, device)
    run_name = f"{args.run.parent.name}/{args.run.name} ({path.name})"

    data_dir = Path(args.data) if Path(args.data).is_dir() else DATASETS_DIR / args.data
    dataset = StoredDataset(data_dir)
    picks = [i for i, spec in enumerate(dataset.specs)
             if args.condition in ("any", spec.condition)
             and args.k in (None, spec.k_target) and args.speed in (None, spec.speed)][args.skip:args.skip + args.n]
    if not picks:
        print(f"no sequence of {data_dir.name} matches condition={args.condition}, k={args.k}, speed={args.speed}")
        return 1
    print(f"step {checkpoint['step']} of {run_name} on {describe_device(device)}, "
          f"{len(picks)} sequences of {data_dir.name}")

    batch = prepare_batch(torch.utils.data.default_collate([dataset[i] for i in picks]), device)
    with torch.no_grad():
        result = model(batch["frames"], return_states=args.probe)
    frames = batch["frames"].permute(0, 1, 3, 4, 2).cpu().numpy()
    predictions = result["prediction"].permute(0, 1, 3, 4, 2).cpu().numpy()
    beliefs = [None] * len(picks)
    if args.probe:
        probe = PositionProbe.load(args.run / "probes" / "position_model.npz")
        features = frame_features(result["R"], probe.layer_limit).cpu().numpy()
        beliefs = probe.predict(features.reshape(-1, features.shape[2])).reshape(len(picks), -1, 2)
        beliefs[:, :MIN_CONTEXT] = np.nan  # the probe reads frames from MIN_CONTEXT on

    thresholds = VisibilityThresholds(**dataset.info["thresholds"])
    out = args.out or args.run / ("predictions_probe" if args.probe else "predictions")
    out.mkdir(parents=True, exist_ok=True)
    renders, labels, saved = [], [], []
    for i, index in enumerate(picks):
        spec = dataset.specs[index]
        rendered = render_sequence(spec, dataset.sprites[spec.digit_index], dataset.render_settings, thresholds)
        renders.append(rendered)
        # frames 1 and later: frame 0 is predicted before any input
        model_mse = float(((predictions[i, 1:] - frames[i, 1:]) ** 2).mean())
        copy_mse = float(((frames[i, :-1] - frames[i, 1:]) ** 2).mean())
        title = (f"{run_name}, step {checkpoint['step']}, {data_dir.name} #{index}: {describe(spec)}; "
                 f"MSE {model_mse:.5f} (copy last frame {copy_mse:.5f})")
        print(f"#{index}: {describe(spec)}\n    MSE {model_mse:.5f}, copy last frame {copy_mse:.5f}")
        times = frames_around_event(spec) if args.frames == "event" else list(range(1, spec.seq_len))
        path = out / f"{spec.condition}_{index:05d}.png"
        prediction_sheet(rendered, predictions[i], times[::args.step], title, args.scale, args.columns,
                         args.gain, beliefs[i]).save(path)
        saved.append(path)
        labels.append(f"#{index} {spec.condition} k={spec.k_target} v={spec.speed}")

    gif = out / f"{args.condition}.gif"
    images = prediction_animation(renders, list(predictions), labels, list(range(1, renders[0].spec.seq_len)),
                                  args.scale, args.gain, list(beliefs))
    saved.append(save_gif(images, gif, args.fps))
    print(f"saved {len(saved)} files in {out}:")
    for path in saved:
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
