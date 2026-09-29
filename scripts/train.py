"""Train a model from a training config, or resume a run.

The run goes to runs/<name>/seed<seed>/ (see peekaboo/train/loop.py for its files). A run folder that already exists
is never replaced unless --overwrite is given, and --resume continues it from last.pt with the same settings.
"""

import argparse
import shutil
import sys
from pathlib import Path

from peekaboo.config import apply_overrides, load_config
from peekaboo.data.conditions import GeneratorSettings
from peekaboo.data.dataset import OnTheFlyDataset
from peekaboo.data.mnist_pool import build_digit_pool
from peekaboo.data.render import RenderSettings
from peekaboo.data.store import StoredDataset
from peekaboo.device import describe_device, get_device
from peekaboo.models import build_model
from peekaboo.models.prednet import count_parameters
from peekaboo.paths import DATASETS_DIR, PROJECT_ROOT, RUNS_DIR
from peekaboo.seeding import seed_everything
from peekaboo.train.loop import TrainSettings, train


def resolve(path: str) -> Path:
    """A path from a config, relative to the project root unless absolute."""
    return Path(path) if Path(path).is_absolute() else PROJECT_ROOT / path


def main() -> int:
    """Build the run settings, the model, and the data, then train."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="a training config, for example "
                                                                    "configs/train/smoke.yaml")
    parser.add_argument("--seed", type=int, default=None, help="overrides train.seed")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, for example --set train.steps=100 (repeatable)")
    parser.add_argument("--resume", action="store_true", help="continue the run from its last.pt")
    parser.add_argument("--stop-at", type=int, default=None, help="stop after this step and save last.pt")
    parser.add_argument("--overwrite", action="store_true", help="delete an existing run folder first")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=RUNS_DIR)
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args.set)
    if args.seed is not None:
        config["train"]["seed"] = args.seed
    seed = int(config["train"]["seed"])
    run_dir = args.out / config["name"] / f"seed{seed}"
    run_config = {"name": config["name"], "train_config": config,
                  "model": load_config(resolve(config["model"])), "data": load_config(resolve(config["data"]))}

    if args.resume:
        if not (run_dir / "last.pt").exists():
            print(f"cannot resume: {run_dir / 'last.pt'} does not exist")
            return 1
        if load_config(run_dir / "config.yaml") != run_config:
            print(f"cannot resume: the settings differ from {run_dir / 'config.yaml'}; resume with the same config "
                  f"and overrides, or start a new run")
            return 1
    elif run_dir.exists() and any(run_dir.iterdir()):
        if not args.overwrite:
            print(f"{run_dir} already exists: pass --resume to continue it or --overwrite to replace it")
            return 1
        shutil.rmtree(run_dir)

    seed_everything(seed)
    device = get_device(args.device)
    settings = TrainSettings.from_config(config)
    data_config = run_config["data"]
    digits = data_config["digits"]
    pool = build_digit_pool("train", digits["box_size"], digits["ink_threshold"], digits["val_size"],
                            digits["holdout_seed"], download=False)
    train_data = OnTheFlyDataset(pool, GeneratorSettings.from_config(data_config),
                                 RenderSettings.from_config(data_config), "train", seed)
    val_data = StoredDataset(DATASETS_DIR / config["validation"]["data"])
    model = build_model(run_config["model"]).to(device)
    print(f"{config['name']} seed {seed}: {count_parameters(model):,} parameters on {describe_device(device)}, "
          f"{settings.steps} steps of {settings.batch_size} x {settings.seq_len} frames, validation on "
          f"{min(settings.val_sequences or len(val_data), len(val_data))} sequences of "
          f"{config['validation']['data']} every {settings.val_every} steps")
    print(f"run folder: {run_dir}" + (" (resumed)" if args.resume else ""))
    train(run_dir, model, train_data, val_data, settings, run_config, device, resume=args.resume,
          stop_at=args.stop_at)
    return 0


if __name__ == "__main__":
    sys.exit(main())
