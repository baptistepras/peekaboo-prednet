"""Draw the error maps of every layer of a trained PredNet on stored sequences (step 3.5, peekaboo/viz/errors.py):
under the actual and the predicted frames, the activity of the error units of each layer, frame by frame.

Picks a few sequences of a condition (as scripts/show_predictions.py does), or with --surprise the four members of
one surprise tuple (A, B, AB, BA), and saves in <run>/errors/ one sheet per sequence and a GIF with the sequences side
by side. Nothing is trained; the ConvLSTM has no error units.
"""

import argparse
import sys
from pathlib import Path

import torch

from peekaboo.data.dataset import CONDITION_NAMES, prepare_batch
from peekaboo.data.occluder import VisibilityThresholds
from peekaboo.data.render import render_sequence
from peekaboo.data.splicing import SURPRISES
from peekaboo.data.store import StoredDataset
from peekaboo.device import describe_device, get_device
from peekaboo.paths import DATASETS_DIR
from peekaboo.train.checkpoint import find_checkpoint, load_model
from peekaboo.viz.errors import error_animation, error_maps, error_sheet, normalize
from peekaboo.viz.frames import describe, save_gif
from peekaboo.viz.predictions import frames_around_event

ROLES = ("A", "B", "AB", "BA")


def pick(dataset: StoredDataset, args: argparse.Namespace) -> list[int]:
    """The indices of the sequences to draw: the members of the --skip-th tuple of the --surprise kind, in the order
    of ROLES, or the first --n sequences of the condition (after --skip) with the given k and speed."""
    specs = dataset.specs
    if args.surprise:
        tuples = [s.tuple_id for s in specs if s.tuple_role == "AB" and s.surprise_type == args.surprise
                  and args.k in (None, s.k_target) and args.speed in (None, s.speed)]
        if len(tuples) <= args.skip:
            return []
        members = {s.tuple_role: i for i, s in enumerate(specs) if s.tuple_id == tuples[args.skip]}
        return [members[role] for role in ROLES if role in members]
    return [i for i, spec in enumerate(specs)
            if args.condition in ("any", spec.condition) and spec.tuple_role == "none"
            and args.k in (None, spec.k_target) and args.speed in (None, spec.speed)][args.skip:args.skip + args.n]


def main() -> int:
    """Load the model, predict the chosen sequences, and save one sheet per sequence and a GIF."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="run folder, for example runs/sweep_prednet5l/seed0")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint file name inside the run folder (default: best.pt, last.pt, or model.pt)")
    parser.add_argument("--data", default="dev_v1", help="a set name in data/datasets/ or a folder")
    parser.add_argument("--condition", default="occlusion", choices=CONDITION_NAMES + ("any",))
    parser.add_argument("--surprise", default=None, choices=SURPRISES,
                        help="draw the four members of a surprise tuple of this kind instead")
    parser.add_argument("--k", type=int, default=None, help="keep only sequences (or tuples) with this k")
    parser.add_argument("--speed", type=int, default=None, help="keep only sequences (or tuples) with this speed")
    parser.add_argument("--n", type=int, default=3, help="number of sequences (not with --surprise)")
    parser.add_argument("--skip", type=int, default=0, help="skip the first matching sequences (or tuples)")
    parser.add_argument("--frames", choices=("event", "all"), default="event",
                        help="sheets show the frames around the crossing, or every frame")
    parser.add_argument("--columns", type=int, default=12)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None, help="default: <run>/errors")
    args = parser.parse_args()

    device = get_device(args.device)
    path = find_checkpoint(args.run, args.checkpoint)
    model, checkpoint = load_model(path, device)
    run_name = f"{args.run.parent.name}/{args.run.name} ({path.name})"
    data_dir = Path(args.data) if Path(args.data).is_dir() else DATASETS_DIR / args.data
    dataset = StoredDataset(data_dir)
    picks = pick(dataset, args)
    if not picks:
        print(f"no sequence of {data_dir.name} matches condition={args.condition}, surprise={args.surprise}, "
              f"k={args.k}, speed={args.speed}, skip={args.skip}")
        return 1
    batch = prepare_batch(torch.utils.data.default_collate([dataset[i] for i in picks]), device)
    with torch.no_grad():
        result = model(batch["frames"], return_states=True)
    if "E" not in result:
        print(f"{run_name} has no error units (only PredNet has them)")
        return 1
    print(f"step {checkpoint['step']} of {run_name} on {describe_device(device)}, {len(picks)} sequences of "
          f"{data_dir.name}")
    frame_size = tuple(batch["frames"].shape[-2:])
    maps = [normalize(m) for m in error_maps(result["E"], frame_size)]
    predictions = list(result["prediction"].permute(0, 1, 3, 4, 2).cpu().numpy())

    thresholds = VisibilityThresholds(**dataset.info["thresholds"])
    out = args.out or args.run / "errors"
    out.mkdir(parents=True, exist_ok=True)
    renders, labels, saved = [], [], []
    for i, index in enumerate(picks):
        spec = dataset.specs[index]
        rendered = render_sequence(spec, dataset.sprites[spec.digit_index], dataset.render_settings, thresholds)
        renders.append(rendered)
        role = f" {spec.tuple_role} ({spec.surprise_type})" if spec.tuple_role != "none" else ""
        print(f"#{index}{role}: {describe(spec)}")
        times = frames_around_event(spec) if args.frames == "event" else list(range(1, spec.seq_len))
        name = f"{args.surprise}_{spec.tuple_role}" if args.surprise else spec.condition
        sheet = out / f"{name}_{index:05d}.png"
        error_sheet(rendered, predictions[i], maps[i], times,
                    f"{run_name}, step {checkpoint['step']}, {data_dir.name} #{index}{role}: {describe(spec)}",
                    args.scale, args.columns).save(sheet)
        saved.append(sheet)
        labels.append(f"#{index} {spec.tuple_role if args.surprise else spec.condition} k={spec.k_target} "
                      f"v={spec.speed}")
    images = error_animation(renders, predictions, maps, labels, list(range(1, renders[0].spec.seq_len)),
                             args.scale)
    saved.append(save_gif(images, out / f"{args.surprise or args.condition}.gif", args.fps))
    print(f"saved {len(saved)} files in {out}:")
    for saved_path in saved:
        print(f"  {saved_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
