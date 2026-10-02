"""Training losses shared by every learned model (decision D12).

The common objective is half the mean absolute next frame error, over frames 1 and later, with the pixels of the
visible digit in the target frame weighted by `digit_weight`. With digit_weight = 1 it is exactly PredNet's pixel layer
loss (the mean activity of its error units E_0), and PredNet keeps its own loss.

Why a weight: the red digit covers about 1.2% of the pixels on one channel out of three, so about 0.4% of the loss
values. With the plain L1 loss, the 10k step pilot settled on the blank solution: it drew the bar and the background
perfectly and never the digit, whose position it would have to know to the pixel to gain anything. Weighting the digit
pixels by 10 made the model draw it within 500 steps (see the README, "Pilot results").

The weight changes only the training objective. PredNet's dynamics and error units are unchanged, and the mask holds
only ink visible in the target frame, read from its pixels: the model gets no information about a hidden digit.
"""

from typing import Any

import torch

from peekaboo.models.prednet import PredNet


def visible_digit_mask(frames: torch.Tensor) -> torch.Tensor:
    """Pixels of frames (B, T, 3, H, W) that show the digit (B, T, H, W).

    The digit is drawn in pure red, on a black background, under a gray bar: its visible pixels are those with some
    red and no green. The bar, the background, and blackout frames are never selected, nor is ink hidden by the bar.
    """
    return (frames[:, :, 0] > 0) & (frames[:, :, 1] == 0)


def next_frame_l1(prediction: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    """Mean absolute error of the predictions of frames 1 and later (frame 0 is predicted before any input)."""
    return (prediction[:, 1:] - frames[:, 1:]).abs().mean()


def weighted_pixel_loss(prediction: torch.Tensor, frames: torch.Tensor, digit_weight: float = 1.0) -> torch.Tensor:
    """Half the mean absolute error of frames 1 and later, with the visible digit pixels of the target weighted."""
    difference = (prediction[:, 1:] - frames[:, 1:]).abs()
    if digit_weight != 1.0:
        weights = 1.0 + (digit_weight - 1.0) * visible_digit_mask(frames[:, 1:]).unsqueeze(2).to(difference.dtype)
        difference = weights * difference
    return 0.5 * difference.mean()


def training_loss(model: torch.nn.Module, out: dict[str, Any], frames: torch.Tensor,
                  digit_weight: float = 1.0) -> torch.Tensor:
    """The training objective of any model.

    PredNet keeps its own loss when digit_weight = 1. Otherwise its pixel layer term becomes the weighted pixel loss,
    and the terms of its upper layers (nonzero only with "Lall" weights) are unchanged. Other models train on the
    weighted pixel loss.
    """
    if not isinstance(model, PredNet):
        return weighted_pixel_loss(out["prediction"], frames, digit_weight)
    if digit_weight == 1.0:
        return model.loss(out["layer_errors"])
    upper = out["layer_errors"].clone()
    upper[:, :, 0] = 0.0
    return model.layer_weights[0] * weighted_pixel_loss(out["prediction"], frames, digit_weight) + model.loss(upper)
