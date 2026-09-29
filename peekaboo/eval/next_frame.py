"""Next frame quality: MSE, MAE, and SSIM of a model's predictions against copying the last frame.

Every frame of every sequence gets one row in a table, so the scores can be grouped by condition, speed, k, or state.
The go/no-go gate (decision D16) uses the visible moving frames: frames t >= 2 where the digit is fully visible in
frames t - 1 and t. Every visible digit moves (|vx| >= 2), the model has seen at least two frames to estimate the
motion, and nothing is hidden, so copying the last frame is a fair baseline there. The gate passes when the model's
MSE on these frames is at least 30% below the copy's. A model that barely beats the copy has not learned the motion,
and its behavior under occlusion would say little (Rane et al. found PredNet close to copying on fast motion, hence
the scores per speed).
"""

from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from peekaboo.data.dataset import CONDITION_NAMES, make_loader, prepare_batch
from peekaboo.data.occluder import STATE_VISIBLE

MIN_CONTEXT = 2       # frames seen before a prediction counts: two are needed to estimate the motion
GATE_REDUCTION = 0.3  # D16: the model's MSE must be at least 30% below the copy's
SCORES = ("mse", "mae", "ssim")


def gaussian_window(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """A normalized 1D Gaussian of the given size."""
    x = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    window = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return window / window.sum()


def ssim(x: torch.Tensor, y: torch.Tensor, size: int = 11, sigma: float = 1.5,
         data_range: float = 1.0) -> torch.Tensor:
    """Structural similarity of images (N, C, H, W) in float32, one value per image (Wang et al., 2004).

    Local statistics use a Gaussian window of the given size and sigma, applied without padding, with the usual
    constants (0.01 L)^2 and (0.03 L)^2. The SSIM map is averaged over pixels and channels.
    """
    x, y = x.float(), y.float()
    channels = x.shape[1]
    window = gaussian_window(size, sigma).to(x.device)
    rows = window.view(1, 1, size, 1).repeat(channels, 1, 1, 1)
    columns = window.view(1, 1, 1, size).repeat(channels, 1, 1, 1)

    def blur(image: torch.Tensor) -> torch.Tensor:
        """Separable Gaussian filter, one channel at a time."""
        return F.conv2d(F.conv2d(image, rows, groups=channels), columns, groups=channels)

    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu_x, mu_y = blur(x), blur(y)
    var_x = blur(x * x) - mu_x ** 2
    var_y = blur(y * y) - mu_y ** 2
    cov = blur(x * y) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))
    return ssim_map.mean(dim=(1, 2, 3))


def frame_scores(prediction: torch.Tensor, frames: torch.Tensor) -> dict[str, torch.Tensor]:
    """MSE, MAE, and SSIM of every predicted frame (B, T); frame 0, predicted before any input, is NaN."""
    difference = prediction[:, 1:] - frames[:, 1:]
    batch, steps = difference.shape[:2]
    scores = {"mse": difference.square().mean(dim=(2, 3, 4)), "mae": difference.abs().mean(dim=(2, 3, 4)),
              "ssim": ssim(prediction[:, 1:].flatten(0, 1), frames[:, 1:].flatten(0, 1)).view(batch, steps)}
    first = torch.full((batch, 1), float("nan"), device=frames.device)
    return {name: torch.cat([first, value], dim=1) for name, value in scores.items()}


def visible_moving(state: torch.Tensor, min_context: int = MIN_CONTEXT) -> torch.Tensor:
    """Frames t >= min_context where the digit is fully visible in frames t - 1 and t (B, T) bool."""
    visible = state == STATE_VISIBLE
    mask = torch.zeros_like(visible)
    mask[:, 1:] = visible[:, 1:] & visible[:, :-1]
    mask[:, :min_context] = False
    return mask


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataset: Dataset, device: torch.device, batch_size: int = 16,
             seq_len: int | None = None, count: int | None = None) -> pd.DataFrame:
    """Score every frame of the first `count` sequences (default: all), for the model and for copying the last frame.

    Returns one row per sequence and frame t >= 1, with the sequence's index, condition, speed, and k, the frame's
    state, whether it is a visible moving frame, and the model_* and copy_* scores.
    """
    model.eval()
    count = len(dataset) if count is None else min(count, len(dataset))
    tables = []
    for batch in make_loader(dataset, batch_size, start=0, count=count):
        batch = prepare_batch(batch, device)
        frames = batch["frames"][:, :seq_len]
        state = batch["state"][:, :frames.shape[1]]
        model_scores = frame_scores(model(frames)["prediction"], frames)
        copy = torch.cat([torch.zeros_like(frames[:, :1]), frames[:, :-1]], dim=1)
        copy_scores = frame_scores(copy, frames)
        b, t = state.shape
        columns: dict[str, Any] = {
            "index": batch["index"][:, None].expand(b, t),
            "condition": batch["condition"][:, None].expand(b, t),
            "speed": batch["speed"][:, None].expand(b, t),
            "k": batch["k_target"][:, None].expand(b, t),
            "t": torch.arange(t, device=device)[None].expand(b, t),
            "state": state,
            "visible_moving": visible_moving(state),
        }
        columns.update({f"model_{name}": value for name, value in model_scores.items()})
        columns.update({f"copy_{name}": value for name, value in copy_scores.items()})
        table = pd.DataFrame({name: value.reshape(-1).cpu().numpy() for name, value in columns.items()})
        tables.append(table[table["t"] >= 1])
    table = pd.concat(tables, ignore_index=True)
    table["condition"] = [CONDITION_NAMES[i] for i in table["condition"]]
    return table


def summarize(table: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Mean scores of the model and the copy, and the MSE reduction 1 - model / copy, overall or per group."""
    columns = [f"{who}_{name}" for who in ("model", "copy") for name in SCORES]
    groups = table.groupby(by) if by else table.assign(all="all").groupby("all")
    summary = groups[columns].mean()
    summary.insert(0, "frames", groups.size())
    summary["mse_reduction"] = 1.0 - summary["model_mse"] / summary["copy_mse"]
    return summary


def gate(table: pd.DataFrame, reduction: float = GATE_REDUCTION) -> dict[str, Any]:
    """Decision D16 on the visible moving frames: does the model's MSE fall at least `reduction` below the copy's?"""
    visible = table[table["visible_moving"]]
    model_mse, copy_mse = float(visible["model_mse"].mean()), float(visible["copy_mse"].mean())
    achieved = 1.0 - model_mse / copy_mse
    return {"frames": int(len(visible)), "model_mse": model_mse, "copy_mse": copy_mse, "mse_reduction": achieved,
            "required_reduction": reduction, "passed": bool(achieved >= reduction)}


def tables_to_dict(summary: pd.DataFrame) -> dict[str, dict[str, float]]:
    """A summary table as plain nested dictionaries, for JSON."""
    return {str(group): {name: float(value) for name, value in row.items()} for group, row in summary.iterrows()}

