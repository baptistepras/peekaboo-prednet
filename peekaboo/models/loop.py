"""Closed loop inputs, shared by the models (step 3.6).

In closed loop, the model reads its own prediction of a frame in place of the frame. The frames read in closed loop
are given per sequence and per frame by a boolean mask (B, T): True where the input of frame t is the model's
prediction of frame t, made from the inputs before t. Frame 0 is always read, since nothing predicted it.
"""

import torch


def closed_mask(frames: torch.Tensor, extrapolate_from: int | None = None,
                closed: torch.Tensor | None = None) -> torch.Tensor | None:
    """The mask (B, T) of the frames read in closed loop for frames (B, T, C, H, W): `closed` itself, or every frame
    from `extrapolate_from` on for every sequence, or None for an open loop pass. Frame 0 is never closed."""
    if closed is not None and extrapolate_from is not None:
        raise ValueError("give either extrapolate_from or closed, not both")
    batch, steps = frames.shape[:2]
    if extrapolate_from is not None:
        closed = (torch.arange(steps, device=frames.device) >= extrapolate_from).expand(batch, steps)
    if closed is None:
        return None
    if tuple(closed.shape) != (batch, steps):
        raise ValueError(f"the closed loop mask has shape {tuple(closed.shape)}, expected {(batch, steps)}")
    closed = closed.to(device=frames.device, dtype=torch.bool).clone()
    closed[:, 0] = False
    return closed
