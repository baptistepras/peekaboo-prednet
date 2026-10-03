"""The training loop: step based, over the on the fly stream, with validation, checkpoints, logs, and exact resume.

Step s trains on the sequences s * batch_size to (s + 1) * batch_size - 1 of the stream, so every sample is new and a
resumed run sees exactly the samples it would have seen without the interruption. The learning rate is a function of
the step only. Validation measures the next frame error on a stored set, and the best checkpoint is chosen on the
validation value of the training objective alone: occlusion metrics never take part in model selection.

Every model returns a dictionary with "prediction" (B, T, C, H, W), where prediction[:, t] predicts frame t from the
frames before it. The training objective is in peekaboo/train/losses.py (decision D12).
"""

import math
import shutil
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from peekaboo.config import save_config
from peekaboo.data.dataset import make_loader, prepare_batch
from peekaboo.train.checkpoint import load_checkpoint, save_checkpoint
from peekaboo.train.losses import training_loss, weighted_pixel_loss
from peekaboo.train.metrics import TRAIN_FIELDS, VAL_FIELDS, CsvLog
from peekaboo.viz.curves import plot_training


@dataclass(frozen=True)
class TrainSettings:
    """Length, batches, optimizer, and validation settings of a run."""

    steps: int
    batch_size: int
    seq_len: int
    workers: int = 0
    log_every: int = 50
    lr: float = 1e-3
    lr_drop_at: float | None = None  # fraction of the steps after which lr is multiplied by lr_drop_factor
    lr_drop_factor: float = 0.1
    grad_clip: float | None = None   # maximum gradient norm, None for no clipping
    val_every: int = 500
    val_sequences: int | None = None  # None: the whole validation set
    val_batch_size: int = 16
    digit_weight: float = 1.0         # weight of the visible digit pixels in the loss, 1 for PredNet's own loss

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TrainSettings":
        """Build the settings from a training config (sections train, optimizer, validation, and optionally loss)."""
        train, optimizer, validation = config["train"], config["optimizer"], config["validation"]
        return cls(steps=int(train["steps"]), batch_size=int(train["batch_size"]), seq_len=int(train["seq_len"]),
                   workers=int(train.get("workers", 0)), log_every=int(train.get("log_every", 50)),
                   lr=float(optimizer["lr"]), lr_drop_at=optimizer.get("lr_drop_at"),
                   lr_drop_factor=float(optimizer.get("lr_drop_factor", 0.1)), grad_clip=optimizer.get("grad_clip"),
                   val_every=int(validation["every"]), val_sequences=validation.get("sequences"),
                   val_batch_size=int(validation.get("batch_size", 16)),
                   digit_weight=float(config.get("loss", {}).get("digit_weight", 1.0)))


def learning_rate(step: int, settings: TrainSettings) -> float:
    """Learning rate of step `step` (counted from 0): lr, then lr * lr_drop_factor from the drop step on."""
    if settings.lr_drop_at is not None and step >= round(settings.lr_drop_at * settings.steps):
        return settings.lr * settings.lr_drop_factor
    return settings.lr


def format_duration(seconds: float) -> str:
    """A duration in seconds, minutes, or hours, whichever reads best."""
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


@torch.no_grad()
def validate(model: torch.nn.Module, dataset: Dataset, count: int, batch_size: int, seq_len: int,
             device: torch.device, digit_weight: float = 1.0) -> dict[str, float]:
    """Validation scores on the first `count` sequences, over frames 1 to seq_len - 1.

    val_loss is the weighted pixel loss of the training objective, which chooses the best checkpoint. The plain L1
    and squared errors per pixel of the model and of copying the last frame are given for reference.
    """
    was_training = model.training
    model.eval()
    sums = {"val_loss": 0.0, "val_l1": 0.0, "val_mse": 0.0, "copy_l1": 0.0, "copy_mse": 0.0}
    pixels = 0
    for batch in make_loader(dataset, batch_size, start=0, count=count):
        frames = prepare_batch(batch, device)["frames"][:, :seq_len]
        prediction = model(frames)["prediction"]
        target = frames[:, 1:]
        sums["val_loss"] += float(weighted_pixel_loss(prediction, frames, digit_weight)) * target.numel()
        for name, guess in (("val", prediction[:, 1:]), ("copy", frames[:, :-1])):
            difference = guess - target
            sums[f"{name}_l1"] += float(difference.abs().sum())
            sums[f"{name}_mse"] += float(difference.square().sum())
        pixels += target.numel()
    model.train(was_training)
    return {name: value / pixels for name, value in sums.items()}


def train(run_dir: str | Path, model: torch.nn.Module, train_data: Dataset, val_data: Dataset,
          settings: TrainSettings, run_config: dict[str, Any], device: torch.device, resume: bool = False,
          stop_at: int | None = None, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Train `model` (already on `device`) and return a summary.

    Writes in run_dir: config.yaml, train_metrics.csv, val_metrics.csv, curves.png, last.pt (at every validation and
    when stopping), and best.pt (lowest validation loss). With resume, training continues from last.pt. With stop_at,
    training stops after that step and saves last.pt, as a job with a time limit would. When the last step is done,
    last.pt is deleted: it is only needed to resume, and best.pt is the model of the run.
    """
    run_dir = Path(run_dir)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.lr)
    start, best = 0, {"val_loss": math.inf, "step": -1}
    if resume:
        checkpoint = load_checkpoint(run_dir / "last.pt", model, optimizer, restore_rng=True)
        start, best = checkpoint["step"], checkpoint["extra"]["best"]
    else:
        save_config(run_config, run_dir / "config.yaml")
    train_log = CsvLog(run_dir / "train_metrics.csv", TRAIN_FIELDS, keep_until=start if resume else None)
    val_log = CsvLog(run_dir / "val_metrics.csv", VAL_FIELDS, keep_until=start if resume else None)
    val_count = min(settings.val_sequences or len(val_data), len(val_data))
    end = min(settings.steps, stop_at) if stop_at is not None else settings.steps

    def run_validation(step: int) -> dict[str, float]:
        """Validate, log the row, and return the metrics."""
        metrics = validate(model, val_data, val_count, settings.val_batch_size, settings.seq_len, device,
                           settings.digit_weight)
        if step > 0 and metrics["val_loss"] < best["val_loss"]:
            best.update(val_loss=metrics["val_loss"], step=step)
        val_log.write(step=step, **metrics, best_step=best["step"])
        log(f"validation at step {step}: loss {metrics['val_loss']:.5f}, L1 {metrics['val_l1']:.5f} (copy last frame "
            f"{metrics['copy_l1']:.5f}), MSE {metrics['val_mse']:.5f} (copy {metrics['copy_mse']:.5f}), best step "
            f"{best['step']}")
        return metrics

    if start == 0:
        run_validation(0)
    if start >= end:
        log(f"nothing to train: the run is at step {start}, the end is step {end}")
        return {"step": start, "best": best}

    model.train()
    loader = make_loader(train_data, settings.batch_size, start=start * settings.batch_size,
                         count=(end - start) * settings.batch_size, num_workers=settings.workers)
    iterator = iter(loader)
    recent_losses, recent_seconds = deque(maxlen=settings.log_every), deque(maxlen=50)
    last_saved = start
    for step in range(start, end):
        began = time.perf_counter()
        frames = prepare_batch(next(iterator), device)["frames"][:, :settings.seq_len]
        lr = learning_rate(step, settings)
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss = training_loss(model, model(frames), frames, settings.digit_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip or math.inf)
        optimizer.step()
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise RuntimeError(f"The loss is {loss_value} at step {step + 1}; the last good checkpoint is last.pt.")
        done = step + 1
        seconds = time.perf_counter() - began
        train_log.write(step=done, loss=loss_value, grad_norm=float(grad_norm), lr=lr, seconds=seconds)
        recent_losses.append(loss_value)
        recent_seconds.append(seconds)

        if done % settings.log_every == 0 or done == end:
            per_step = sum(recent_seconds) / len(recent_seconds)
            log(f"step {done}/{settings.steps}: loss {sum(recent_losses) / len(recent_losses):.5f}, lr {lr:g}, "
                f"{per_step:.2f} s/step, {format_duration((settings.steps - done) * per_step)} left")
        validating = done % settings.val_every == 0 or done == settings.steps
        if validating:
            run_validation(done)
        if validating or done == end:
            save_checkpoint(run_dir / "last.pt", model, optimizer, done, run_config, extra={"best": dict(best)})
            last_saved = done
            if best["step"] == done:  # the validation of this step is the best so far
                shutil.copyfile(run_dir / "last.pt", run_dir / "best.pt")
            plot_training(run_dir, f"{run_config.get('name', '')} {run_dir.name}".strip())

    if last_saved == settings.steps and (run_dir / "best.pt").exists():
        (run_dir / "last.pt").unlink()
    log(f"stopped at step {last_saved} of {settings.steps}; best validation loss {best['val_loss']:.5f} at step "
        f"{best['step']}")
    return {"step": last_saved, "best": best}
