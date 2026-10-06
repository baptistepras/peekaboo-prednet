"""Amodal decoder (option C): read from a frozen model's state the whole digit, including the part under the bar.

The decoder is a deliberately minimal readout: a 1 x 1 convolution of each layer's state R_l, upsampled to the frame
and summed, then a sigmoid. With a 1 x 1 kernel, this is exactly a 1 x 1 convolution of all the layers upsampled and
stacked: each pixel of the decoded digit is a weighted sum of the states at that place. It has one weight per state
channel (244 for PredNet 5 layers), so it cannot draw a digit by itself: a shape appears only where the state holds
one. It is trained on the training stream to output the amodal frame (the digit as if there were no bar), with the
ink pixels weighted as in the training loss (decision D12), since the digit covers about 1% of the pixels.

Controls, as for the position probe: the same decoder on the same architecture at random initialization, and a
template baseline that copies the last fully visible digit and moves it to the position given by a tracker.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from peekaboo.data.dataset import make_loader, prepare_batch
from peekaboo.eval.next_frame import MIN_CONTEXT
from peekaboo.probes.features import frame_rows
from peekaboo.trackers.baselines import run_trackers
from peekaboo.trackers.detector import Detections, detect, digit_mask

INK = 0.5  # intensity above which a pixel counts as ink, for the IoU


class AmodalDecoder(nn.Module):
    """Sum over layers of a convolution of R_l upsampled to the frame, then a sigmoid: the decoded intensity."""

    def __init__(self, channels: tuple[int, ...], frame_size: tuple[int, int], kernel_size: int = 1) -> None:
        """One convolution per layer, no bias, and one shared bias that starts low, since most pixels are empty."""
        super().__init__()
        self.channels, self.frame_size, self.kernel_size = tuple(channels), tuple(frame_size), kernel_size
        self.convs = nn.ModuleList(nn.Conv2d(c, 1, kernel_size, padding=kernel_size // 2, bias=False)
                                   for c in channels)
        self.bias = nn.Parameter(torch.full((1,), -3.0))

    def forward(self, states: list[torch.Tensor]) -> torch.Tensor:
        """Logits (N, 1, H, W) from the states of each layer (N, C_l, H_l, W_l)."""
        out = self.bias.view(1, 1, 1, 1)
        for conv, r in zip(self.convs, states):
            out = out + F.interpolate(conv(r), size=self.frame_size, mode="bilinear", align_corners=False)
        return out

    def imagine(self, states: list[torch.Tensor]) -> torch.Tensor:
        """Decoded intensity (N, H, W) in [0, 1]."""
        return torch.sigmoid(self.forward(states))[:, 0]

    def save(self, path: str | Path) -> Path:
        """Save as .npz, plain arrays only."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        weights = {f"weight_{l}": conv.weight.detach().cpu().numpy() for l, conv in enumerate(self.convs)}
        np.savez(path, channels=np.array(self.channels), frame_size=np.array(self.frame_size),
                 kernel_size=self.kernel_size, bias=self.bias.detach().cpu().numpy(), **weights)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "AmodalDecoder":
        """Load a decoder saved by save, on the CPU."""
        with np.load(path, allow_pickle=False) as data:
            decoder = cls(tuple(int(c) for c in data["channels"]), tuple(int(s) for s in data["frame_size"]),
                          int(data["kernel_size"]))
            with torch.no_grad():
                decoder.bias.copy_(torch.from_numpy(data["bias"]))
                for l, conv in enumerate(decoder.convs):
                    conv.weight.copy_(torch.from_numpy(data[f"weight_{l}"]))
        return decoder


def state_channels(states: list[list[torch.Tensor]]) -> tuple[int, ...]:
    """Channels of each layer, from the states of one forward pass."""
    return tuple(r.shape[1] for r in states[0])


def select_states(states: list[list[torch.Tensor]], seq: torch.Tensor, t: torch.Tensor) -> list[torch.Tensor]:
    """The states of the chosen frames (seq[i], t[i]), per layer (N, C_l, H_l, W_l), from a forward pass with
    return_states (one list of per layer tensors per time step)."""
    return [torch.stack([step[l] for step in states], dim=1)[seq, t] for l in range(len(states[0]))]


def decoder_loss(decoder: AmodalDecoder, states: list[torch.Tensor], target: torch.Tensor,
                 digit_weight: float = 10.0) -> torch.Tensor:
    """Binary cross entropy against the amodal intensity (N, H, W) in [0, 1], ink pixels weighted by digit_weight."""
    weights = 1.0 + (digit_weight - 1.0) * (target > 0).float()
    return F.binary_cross_entropy_with_logits(decoder(states)[:, 0], target, weight=weights)


def selected_frames(batch: dict[str, torch.Tensor], window_only: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """(sequence, frame) indices of the frames to decode: from MIN_CONTEXT on, digit present, inside the analysis
    window if asked."""
    keep = ~torch.isnan(batch["center"][..., 0])
    keep[:, :MIN_CONTEXT] = False
    if window_only:
        keep &= batch["in_window"]
    return torch.nonzero(keep, as_tuple=True)


def fit_decoder(model: nn.Module, dataset: Dataset, steps: int, device: torch.device, batch_size: int = 8,
                lr: float = 1e-2, digit_weight: float = 10.0, kernel_size: int = 1, workers: int = 0,
                log: Callable[[str], None] = print) -> tuple[AmodalDecoder, list[float]]:
    """Train a decoder on the frozen model's states over `steps` batches of the stream (analysis windows only), with
    Adam. The dataset must include the amodal frames. Returns the decoder and the loss of every step."""
    model.eval()
    decoder, optimizer, losses = None, None, []
    for step, batch in enumerate(make_loader(dataset, batch_size, 0, steps * batch_size, workers)):
        batch = prepare_batch(batch, device)
        with torch.no_grad():
            states = model(batch["frames"], return_states=True)["R"]
        if decoder is None:
            decoder = AmodalDecoder(state_channels(states), batch["frames"].shape[-2:], kernel_size).to(device)
            optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
        seq, t = selected_frames(batch, window_only=True)
        loss = decoder_loss(decoder, select_states(states, seq, t), batch["amodal"][seq, t, 0], digit_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (step + 1) % 50 == 0 or step == 0:
            log(f"  step {step + 1}/{steps}: loss {np.mean(losses[-50:]):.4f}")
    return decoder.cpu(), losses


def ink_scores(image: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per frame (N, H, W): the Pearson correlation of the intensities, and the intersection over union of the ink
    (above INK). NaN when undefined (a constant image, or no ink in either)."""
    x = image.flatten(1) - image.flatten(1).mean(dim=1, keepdim=True)
    y = target.flatten(1) - target.flatten(1).mean(dim=1, keepdim=True)
    denominator = x.norm(dim=1) * y.norm(dim=1)
    correlation = torch.where(denominator > 0, (x * y).sum(dim=1) / denominator.clamp_min(1e-12),
                              torch.full_like(denominator, float("nan")))
    a, b = image.flatten(1) > INK, target.flatten(1) > INK
    union = (a | b).sum(dim=1).float()
    iou = torch.where(union > 0, (a & b).sum(dim=1).float() / union.clamp_min(1), torch.full_like(union, float("nan")))
    return correlation, iou


def shift(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Move an image (H, W) by whole pixels, filling with zeros. A move of the whole height or width, or more (a
    tracker far outside the frame), leaves an empty image."""
    out = np.zeros_like(image)
    height, width = image.shape
    if abs(dy) >= height or abs(dx) >= width:
        return out
    out[max(dy, 0):height - max(-dy, 0), max(dx, 0):width - max(-dx, 0)] = \
        image[max(-dy, 0):height - max(dy, 0), max(-dx, 0):width - max(dx, 0)]
    return out


def template_images(frames: np.ndarray, detections: Detections, positions: np.ndarray) -> np.ndarray:
    """The template baseline (T, H, W) in [0, 1]: at frame t, the digit of the last frame before t where it was
    detected (fully visible), moved by the change of position from that frame to the tracker's prediction for t.
    NaN before the first detection."""
    out = np.full((len(frames),) + frames.shape[2:], np.nan, dtype=np.float32)
    last = None
    for t in range(len(frames)):
        if last is not None and not np.isnan(positions[t]).any():
            dy, dx = np.rint(positions[t] - detections.center[last]).astype(int)
            template = frames[last, 0] * digit_mask(frames[last]) / 255.0
            out[t] = shift(template, int(dy), int(dx))
        if detections.detected[t]:
            last = t
    return out


TEMPLATES = {"template_kalman": "kalman", "template_walls": "kalman_walls"}


def evaluate_decoders(decoders: dict[str, tuple[nn.Module, AmodalDecoder]], dataset: Dataset, count: int,
                      device: torch.device, batch_size: int = 16) -> pd.DataFrame:
    """One row per frame from MIN_CONTEXT on with the digit present, with the correlation (<method>_corr) and the ink
    IoU (<method>_iou) against the amodal frame, for every (model, decoder) pair and for the template baselines
    moved by the constant velocity Kalman filter and by the filter with walls. The dataset must include the amodal
    frames."""
    parts, first = [], 0
    for batch in make_loader(dataset, batch_size, 0, count):
        rows = frame_rows(batch, first)
        seq, t = selected_frames(batch, window_only=False)
        rows = rows.iloc[(seq * batch["state"].shape[1] + t).numpy()].reset_index(drop=True)
        target = batch["amodal"][seq, t, 0].float() / 255.0
        scores = {}
        moved = prepare_batch(batch, device)
        for name, (model, decoder) in decoders.items():
            with torch.no_grad():
                states = model(moved["frames"], return_states=True)["R"]
                image = decoder.to(device).imagine(select_states(states, seq.to(device), t.to(device))).cpu()
            scores[f"{name}_corr"], scores[f"{name}_iou"] = ink_scores(image, target)
        frames = batch["frames"].numpy()
        templates = {name: [] for name in TEMPLATES}
        for i, (sequence, center, state) in enumerate(zip(frames, batch["center"].numpy(), batch["state"].numpy())):
            detections = detect(sequence)
            tracks = run_trackers(sequence, center, state)
            for name, tracker in TEMPLATES.items():
                templates[name].append(template_images(sequence, detections, tracks[tracker].prediction))
        for name in TEMPLATES:
            image = torch.from_numpy(np.stack(templates[name])[seq.numpy(), t.numpy()])
            corr, iou = ink_scores(torch.nan_to_num(image), target)
            missing = torch.isnan(image.flatten(1)).any(dim=1)
            corr[missing], iou[missing] = float("nan"), float("nan")
            scores[f"{name}_corr"], scores[f"{name}_iou"] = corr, iou
        parts.append(rows.assign(**{key: value.numpy() for key, value in scores.items()}))
        first += len(batch["state"])
    return pd.concat(parts, ignore_index=True)
