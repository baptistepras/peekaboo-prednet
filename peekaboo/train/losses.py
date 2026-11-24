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

Blackout frames in training (step 4.1) are scored in one of three ways, `loss.blackout` of the training config:
- plain (the default; the v1 runs had no blackout in training): the observed frames are the targets, so the model
  is asked to predict black;
- masked: the predictions of the blackout frames are left out of the loss, so the model is never asked to predict a
  black frame, and the first frame after the blackout still rewards keeping the digit; the loss stays self supervised;
- unblacked: the targets are the frames drawn without the blackout (what Loci-Looped does: it blacks out the input
  only), so the model is told where the digit is during the blackout.
"""

from typing import Any

import torch

from peekaboo.models.prednet import PredNet

BLACKOUT_LOSSES = ("plain", "masked", "unblacked")


def visible_digit_mask(frames: torch.Tensor) -> torch.Tensor:
    """Pixels of frames (B, T, 3, H, W) that show the digit (B, T, H, W).

    The digit is drawn in pure red, on a black background, under a gray bar: its visible pixels are those with some
    red and no green. The bar, the background, and blackout frames are never selected, nor is ink hidden by the bar.
    """
    return (frames[:, :, 0] > 0) & (frames[:, :, 1] == 0)


def blackout_frames(frames: torch.Tensor) -> torch.Tensor:
    """Frames of frames (B, T, 3, H, W) that are blackout frames (B, T): every pixel is zero. The bar is drawn on
    every other frame, so no other frame is all black."""
    return frames.flatten(2).amax(dim=2) == 0


def loss_targets(batch: dict[str, torch.Tensor], seq_len: int,
                 blackout_loss: str = "plain") -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """The target frames (None: the input frames) and the frames counted in the loss (None: every frame) of a batch,
    for one of BLACKOUT_LOSSES. The unblacked mode needs the "unblacked" frames in the batch."""
    if blackout_loss == "plain":
        return None, None
    if blackout_loss == "masked":
        return None, ~blackout_frames(batch["frames"][:, :seq_len])
    if blackout_loss == "unblacked":
        return batch["unblacked"][:, :seq_len], None
    raise ValueError(f"unknown blackout loss '{blackout_loss}', expected one of {BLACKOUT_LOSSES}")


def next_frame_l1(prediction: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    """Mean absolute error of the predictions of frames 1 and later (frame 0 is predicted before any input)."""
    return (prediction[:, 1:] - frames[:, 1:]).abs().mean()


def weighted_pixel_loss(prediction: torch.Tensor, frames: torch.Tensor, digit_weight: float = 1.0,
                        keep: torch.Tensor | None = None) -> torch.Tensor:
    """Half the mean absolute error of frames 1 and later against the target frames, with the visible digit pixels of
    the target weighted. With keep (B, T), the mean is over the kept frames only."""
    difference = (prediction[:, 1:] - frames[:, 1:]).abs()
    if digit_weight != 1.0:
        weights = 1.0 + (digit_weight - 1.0) * visible_digit_mask(frames[:, 1:]).unsqueeze(2).to(difference.dtype)
        difference = weights * difference
    if keep is None:
        return 0.5 * difference.mean()
    kept = keep[:, 1:].to(difference.dtype)[:, :, None, None, None]
    count = kept.sum() * difference[0, 0].numel()
    return 0.5 * (kept * difference).sum() / count.clamp_min(1.0)


def training_loss(model: torch.nn.Module, out: dict[str, Any], frames: torch.Tensor, digit_weight: float = 1.0,
                  target: torch.Tensor | None = None, keep: torch.Tensor | None = None) -> torch.Tensor:
    """The training objective of any model, on the input frames, or on `target` frames and the `keep` frames when
    given (loss_targets).

    PredNet keeps its own loss when digit_weight = 1 on the input frames, every frame counted. Otherwise its pixel
    layer term becomes the weighted pixel loss, and the terms of its upper layers (nonzero only with "Lall" weights)
    are unchanged (and not masked: they are zero with the "L0" weights of every run). Other models train on the
    weighted pixel loss.
    """
    if isinstance(model, PredNet) and digit_weight == 1.0 and target is None and keep is None:
        return model.loss(out["layer_errors"])
    pixel = weighted_pixel_loss(out["prediction"], frames if target is None else target, digit_weight, keep)
    if not isinstance(model, PredNet):
        return pixel
    upper = out["layer_errors"].clone()
    upper[:, :, 0] = 0.0
    return model.layer_weights[0] * pixel + model.loss(upper)
