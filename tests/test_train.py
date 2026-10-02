"""Tests for the training loop: a tiny CPU run, exact resume, the learning rate schedule, losses, and validation."""

from pathlib import Path

import pytest
import torch

from peekaboo.config import load_config
from peekaboo.data.conditions import GeneratorSettings
from peekaboo.data.dataset import OnTheFlyDataset
from peekaboo.data.mnist_pool import DigitPool
from peekaboo.data.render import RenderSettings
from peekaboo.models import build_model
from peekaboo.paths import CONFIGS_DIR, PROJECT_ROOT
from peekaboo.train.checkpoint import load_checkpoint
from peekaboo.train.loop import (TrainSettings, format_duration, learning_rate, next_frame_l1, train, training_loss,
                                 validate)
from peekaboo.train.metrics import read_csv

RENDER = RenderSettings.from_config(load_config(CONFIGS_DIR / "data" / "base.yaml"))
MODEL = {"model": "prednet", "stack_sizes": [3, 4, 8], "layer_loss_weights": "L0"}
SETTINGS = TrainSettings(steps=4, batch_size=2, seq_len=4, log_every=1, lr=1e-2, lr_drop_at=0.5, val_every=2,
                         val_sequences=4, val_batch_size=2)
CPU = torch.device("cpu")


class CopyLastFrame(torch.nn.Module):
    """A stub model whose prediction of frame t is frame t - 1 (and zeros for frame 0)."""

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        """Shift the frames by one step."""
        return {"prediction": torch.cat([torch.zeros_like(frames[:, :1]), frames[:, :-1]], dim=1)}


def streams(pool: DigitPool, settings: GeneratorSettings) -> tuple[OnTheFlyDataset, OnTheFlyDataset]:
    """A training stream and a validation stream from the synthetic pool."""
    return (OnTheFlyDataset(pool, settings, RENDER, "train", 0),
            OnTheFlyDataset(pool, settings, RENDER, "val", 0, length=8))


def run(run_dir: Path, pool: DigitPool, settings: GeneratorSettings, resume: bool = False,
        stop_at: int | None = None) -> torch.nn.Module:
    """Build the tiny model with a fixed initialization and train it."""
    torch.manual_seed(0)
    model = build_model(MODEL)
    train_data, val_data = streams(pool, settings)
    train(run_dir, model, train_data, val_data, SETTINGS, {"name": "test", "model": MODEL}, CPU, resume=resume,
          stop_at=stop_at, log=lambda line: None)
    return model


@pytest.mark.filterwarnings("error")
def test_tiny_run_writes_every_file(tmp_path: Path, pool: DigitPool, settings: GeneratorSettings) -> None:
    """A 4 step run logs every step, validates at steps 0, 2, and 4, and saves its checkpoints and curves, without any
    warning."""
    run(tmp_path, pool, settings)
    for name in ("config.yaml", "train_metrics.csv", "val_metrics.csv", "curves.png", "last.pt", "best.pt"):
        assert (tmp_path / name).exists(), name
    assert [int(r["step"]) for r in read_csv(tmp_path / "train_metrics.csv")] == [1, 2, 3, 4]
    assert [int(r["step"]) for r in read_csv(tmp_path / "val_metrics.csv")] == [0, 2, 4]
    assert load_checkpoint(tmp_path / "last.pt")["step"] == 4
    rates = [float(r["lr"]) for r in read_csv(tmp_path / "train_metrics.csv")]
    assert rates == pytest.approx([1e-2, 1e-2, 1e-3, 1e-3])


def test_resume_equals_an_uninterrupted_run(tmp_path: Path, pool: DigitPool, settings: GeneratorSettings) -> None:
    """Stopping after step 3 and resuming gives the same weights and the same logged losses as one run."""
    whole = run(tmp_path / "whole", pool, settings)
    run(tmp_path / "split", pool, settings, stop_at=3)
    assert load_checkpoint(tmp_path / "split" / "last.pt")["step"] == 3
    resumed = run(tmp_path / "split", pool, settings, resume=True)
    for name, value in whole.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[name]), name
    for log in ("train_metrics.csv", "val_metrics.csv"):
        a, b = read_csv(tmp_path / "whole" / log), read_csv(tmp_path / "split" / log)
        assert [r["step"] for r in a] == [r["step"] for r in b]
        key = "loss" if log.startswith("train") else "val_l1"
        assert [r[key] for r in a] == [r[key] for r in b]


def test_learning_rate_drops_once() -> None:
    """The rate is divided by 10 from the drop step on, and stays constant without a drop."""
    assert [learning_rate(s, SETTINGS) for s in range(4)] == pytest.approx([1e-2, 1e-2, 1e-3, 1e-3])
    constant = TrainSettings(steps=4, batch_size=2, seq_len=4, lr=1e-2)
    assert {learning_rate(s, constant) for s in range(4)} == {1e-2}


def test_prednet_loss_is_half_the_next_frame_l1() -> None:
    """With the L0 weights, PredNet's own loss equals the common loss of other models."""
    torch.manual_seed(0)
    model = build_model(MODEL)
    frames = torch.rand(2, 5, 3, 16, 24)
    out = model(frames)
    assert torch.allclose(training_loss(model, out, frames), 0.5 * next_frame_l1(out["prediction"], frames),
                          atol=1e-7)
    assert torch.allclose(training_loss(CopyLastFrame(), CopyLastFrame()(frames), frames),
                          0.5 * (frames[:, 1:] - frames[:, :-1]).abs().mean())


def test_validation_of_a_copy_model_equals_the_copy_baseline(pool: DigitPool, settings: GeneratorSettings) -> None:
    """A model that copies the last frame gets exactly the copy baseline, which is positive on moving digits."""
    _, val_data = streams(pool, settings)
    metrics = validate(CopyLastFrame(), val_data, 4, 2, 6, CPU)
    assert metrics["val_l1"] == pytest.approx(metrics["copy_l1"])
    assert metrics["val_mse"] == pytest.approx(metrics["copy_mse"])
    assert metrics["copy_l1"] > 0


def test_durations_read_well() -> None:
    """Short times are given in seconds or minutes, long ones in hours."""
    assert [format_duration(s) for s in (12.4, 150, 5400)] == ["12 s", "2 min", "1.5 h"]


@pytest.mark.parametrize("path", sorted((CONFIGS_DIR / "train").glob("*.yaml")), ids=lambda p: p.stem)
def test_training_configs_are_complete(path: Path) -> None:
    """Every training config gives valid settings, points to existing model and data configs, and builds its model."""
    config = load_config(path)
    settings = TrainSettings.from_config(config)
    assert settings.steps > 0 and settings.val_every > 0 and 1 <= settings.seq_len <= 40
    for key in ("model", "data"):
        assert (PROJECT_ROOT / config[key]).exists(), config[key]
    build_model(load_config(PROJECT_ROOT / config["model"]))
