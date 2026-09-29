"""Save and load training checkpoints: model and optimizer weights, step, settings, and random number states.

The file holds only tensors and plain Python values, so it loads with torch.load(weights_only=True) and never runs
code from the file. Tensors are loaded on the CPU by default; the model's own device is kept by load_state_dict.
"""

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from peekaboo.models import build_model

CHECKPOINT_VERSION = 1
CHECKPOINT_NAMES = ("best.pt", "last.pt", "model.pt")  # a training run's best and last, a benchmark's model


def rng_state() -> dict[str, Any]:
    """Capture the Python, numpy, and torch random number states (CPU, and CUDA or MPS when present)."""
    kind, keys, pos, has_gauss, gauss = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {"kind": kind, "keys": torch.from_numpy(keys.astype(np.int64)), "pos": int(pos),
                  "has_gauss": int(has_gauss), "gauss": float(gauss)},
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    """Restore random number states saved by rng_state. States of devices missing on this machine are skipped."""
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy["kind"], numpy["keys"].numpy().astype(np.uint32), numpy["pos"], numpy["has_gauss"],
                         numpy["gauss"]))
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


def save_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None,
                    step: int = 0, config: dict[str, Any] | None = None,
                    extra: dict[str, Any] | None = None) -> Path:
    """Write a checkpoint. It goes to a temporary file first, so an interrupted save never leaves a broken file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "version": CHECKPOINT_VERSION,
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "config": config or {},
        "rng": rng_state(),
        "extra": extra or {},
    }
    partial = path.with_name(path.name + ".partial")
    torch.save(checkpoint, partial)
    partial.replace(path)
    return path


def load_checkpoint(path: str | Path, model: torch.nn.Module | None = None,
                    optimizer: torch.optim.Optimizer | None = None, restore_rng: bool = False,
                    map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Read a checkpoint, load its weights into the model and optimizer when given, and return its contents."""
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=True)
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"{path} has checkpoint version {checkpoint.get('version')}, expected {CHECKPOINT_VERSION}.")
    if model is not None:
        model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        if checkpoint["optimizer"] is None:
            raise ValueError(f"{path} has no optimizer state.")
        optimizer.load_state_dict(checkpoint["optimizer"])
    if restore_rng:
        set_rng_state(checkpoint["rng"])
    return checkpoint


def find_checkpoint(run_dir: str | Path, name: str | None = None) -> Path:
    """The checkpoint `name` of a run folder, or by default the first of best.pt, last.pt, and model.pt that exists."""
    run_dir = Path(run_dir)
    names = [name] if name else list(CHECKPOINT_NAMES)
    for candidate in names:
        if (run_dir / candidate).exists():
            return run_dir / candidate
    raise FileNotFoundError(f"No checkpoint {' or '.join(names)} in {run_dir}.")


def load_model(path: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Rebuild a model from the settings stored in its checkpoint, load its weights, and put it in eval mode on the
    device. Returns the model and the checkpoint."""
    checkpoint = load_checkpoint(path)
    model = build_model(checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint
