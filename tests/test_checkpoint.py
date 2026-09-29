"""Tests for checkpoints: weights, optimizer, step, settings, and random states survive a save and a load."""

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from peekaboo.models import build_model
from peekaboo.train.checkpoint import load_checkpoint, save_checkpoint

CONFIG = {"model": {"model": "prednet", "stack_sizes": [3, 4, 8], "layer_loss_weights": "L0"},
          "args": {"lr": "0.01", "steps": "3"}}


def train_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer, frames: torch.Tensor) -> None:
    """One Adam step on a fixed batch."""
    optimizer.zero_grad(set_to_none=True)
    model.loss(model(frames)["layer_errors"]).backward()
    optimizer.step()


def test_round_trip_resumes_identically(tmp_path: Path) -> None:
    """A loaded model predicts exactly as the saved one, and one more training step keeps both identical."""
    torch.manual_seed(0)
    frames = torch.rand(2, 4, 3, 16, 24)
    model = build_model(CONFIG["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    train_step(model, optimizer, frames)
    path = save_checkpoint(tmp_path / "run" / "model.pt", model, optimizer, step=7, config=CONFIG)
    assert path.exists() and not path.with_name("model.pt.partial").exists()

    fresh = build_model(CONFIG["model"])
    fresh_optimizer = torch.optim.Adam(fresh.parameters(), lr=1e-2)
    checkpoint = load_checkpoint(path, fresh, fresh_optimizer)
    assert checkpoint["step"] == 7 and checkpoint["config"] == CONFIG
    assert torch.equal(model(frames)["prediction"], fresh(frames)["prediction"])

    train_step(model, optimizer, frames)
    train_step(fresh, fresh_optimizer, frames)
    for name, value in model.state_dict().items():
        assert torch.equal(value, fresh.state_dict()[name]), name


def test_random_states_are_restored(tmp_path: Path) -> None:
    """After a load with restore_rng, Python, numpy, and torch draw the same numbers as right after the save."""
    path = save_checkpoint(tmp_path / "model.pt", build_model(CONFIG["model"]))
    expected = (random.random(), np.random.random(), torch.rand(3))
    load_checkpoint(path, restore_rng=True)
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    assert torch.equal(torch.rand(3), expected[2])


def test_missing_optimizer_state_is_an_error(tmp_path: Path) -> None:
    """Asking for an optimizer state that was not saved fails loudly."""
    model = build_model(CONFIG["model"])
    path = save_checkpoint(tmp_path / "model.pt", model)
    with pytest.raises(ValueError):
        load_checkpoint(path, model, torch.optim.Adam(model.parameters()))


def test_unknown_model_is_an_error() -> None:
    """build_model rejects a config whose model it does not know."""
    with pytest.raises(ValueError):
        build_model({"model": "unknown"})
