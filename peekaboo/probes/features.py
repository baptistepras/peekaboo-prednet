"""Features for the probes: the internal states R of a frozen model, one vector per frame.

R[t] is the state that makes the prediction of frame t: it has seen frames 0 to t - 1. Each layer is average pooled
over space, by the smallest power of 2 that leaves at most `limit` values, and the layers are concatenated. For
PredNet 5 layers on 64 x 96 frames, with the default limit of 2048, this gives 5,760 features per frame; for the
ConvLSTM baseline, 6,144. The pooled maps keep a coarse spatial layout, from which a linear probe can read a position.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from peekaboo.data.dataset import CONDITION_NAMES, make_loader, prepare_batch
from peekaboo.eval.next_frame import MIN_CONTEXT

LAYER_LIMIT = 2048  # pooled values per layer


def pool_layer(r: torch.Tensor, limit: int = LAYER_LIMIT) -> torch.Tensor:
    """Average pool one layer's state (B, C, H, W) by the smallest power of 2 that leaves at most `limit` values, and
    flatten it to (B, C * H' * W')."""
    _, channels, height, width = r.shape
    factor = 1
    while channels * (height // factor) * (width // factor) > limit and factor < min(height, width):
        factor *= 2
    if factor > 1:
        r = F.avg_pool2d(r, factor)
    return r.flatten(1)


def frame_features(states: list[list[torch.Tensor]], limit: int = LAYER_LIMIT) -> torch.Tensor:
    """(B, T, F) features from the states of a forward pass with return_states: one list of per layer tensors per
    time step."""
    return torch.stack([torch.cat([pool_layer(r, limit) for r in layers], dim=1) for layers in states], dim=1)


@dataclass(frozen=True)
class FrameSet:
    """Features of the selected frames and what is known about each frame."""

    features: np.ndarray  # (N, F) float32
    rows: pd.DataFrame    # one row per frame, in the same order


def frame_rows(batch: dict[str, torch.Tensor], first: int) -> pd.DataFrame:
    """One row per frame of a CPU batch: sequence number, frame, true centroid, state, and event information."""
    size, seq_len = batch["state"].shape
    t = np.tile(np.arange(seq_len), size)
    onset = np.repeat(batch["onset_frame"].numpy(), seq_len)
    center = batch["center"].numpy().reshape(-1, 2)
    return pd.DataFrame({
        "sequence": np.repeat(np.arange(first, first + size), seq_len),
        "t": t,
        "y": center[:, 0],
        "x": center[:, 1],
        "state": batch["state"].numpy().ravel(),
        "episode": batch["episode"].numpy().ravel(),
        "in_window": batch["in_window"].numpy().ravel(),
        "condition": np.repeat([CONDITION_NAMES[c] for c in batch["condition"].numpy()], seq_len),
        "k": np.repeat(batch["k_target"].numpy(), seq_len),
        "speed": np.repeat(batch["speed"].numpy(), seq_len),
        "since_onset": np.where(onset >= 0, t - onset, np.iinfo(np.int64).min),
    })


def iterate_frames(model: torch.nn.Module, dataset: Dataset, count: int, device: torch.device, start: int = 0,
                   batch_size: int = 16, seq_len: int = 40, limit: int = LAYER_LIMIT, window_only: bool = False,
                   workers: int = 0) -> Iterator[tuple[np.ndarray, pd.DataFrame, dict[str, torch.Tensor]]]:
    """Run the frozen model over `count` sequences from index `start` and yield, per batch, the features and rows of
    the selected frames, and the CPU batch. Selected frames: from frame MIN_CONTEXT on, with the digit present, and
    inside the analysis window if `window_only`."""
    model.eval()
    first = start
    for batch in make_loader(dataset, batch_size, start, count, workers):
        batch = {name: value[:, :seq_len] if value.dim() > 1 else value for name, value in batch.items()}
        with torch.no_grad():
            states = model(prepare_batch(batch, device)["frames"], return_states=True)["R"]
            features = frame_features(states, limit).flatten(0, 1).float().cpu().numpy()
        rows = frame_rows(batch, first)
        keep = (rows["t"] >= MIN_CONTEXT) & rows["y"].notna()
        if window_only:
            keep &= rows["in_window"]
        first += len(batch["state"])
        yield features[keep.to_numpy()], rows[keep].reset_index(drop=True), batch


def collect_frames(model: torch.nn.Module, dataset: Dataset, count: int, device: torch.device, **options) -> FrameSet:
    """All selected frames of `count` sequences, in one FrameSet (see iterate_frames for the options)."""
    parts = list(iterate_frames(model, dataset, count, device, **options))
    return FrameSet(features=np.concatenate([p[0] for p in parts]),
                    rows=pd.concat([p[1] for p in parts], ignore_index=True))
