"""Pixel rendering of a SequenceSpec: amodal digit frames, observed RGB frames, and modal masks.

Only the amodal frames (one uint8 channel) need to be stored: the observed frames and the masks are rebuilt exactly
from them and the spec.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from peekaboo.data.occluder import VisibilityThresholds
from peekaboo.data.spec import SequenceSpec
from peekaboo.data.truth import FrameTruth, compute_truth


@dataclass(frozen=True)
class RenderSettings:
    """Colors: the digit's RGB weights (applied to its intensity) and the bar's gray level on every channel."""

    digit_rgb: tuple[float, float, float] = (1.0, 0.0, 0.0)
    bar_value: int = 128

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RenderSettings":
        """Build the settings from the render section of a data config."""
        render = config["render"]
        return cls(digit_rgb=tuple(float(c) for c in render["digit_rgb"]),
                   bar_value=int(round(float(render["bar_gray"]) * 255)))


@dataclass(frozen=True)
class RenderedSequence:
    """All pixel arrays of one sequence, with its exact ground truth."""

    spec: SequenceSpec
    observed: np.ndarray    # (T, H, W, 3) uint8, what the models see
    amodal: np.ndarray      # (T, H, W) uint8, digit intensity without occluder or blackout
    modal_mask: np.ndarray  # (T, H, W) bool, ink pixels visible on screen
    truth: FrameTruth

    @property
    def amodal_mask(self) -> np.ndarray:
        """All ink pixels, hidden or not."""
        return self.amodal > 0


def render_amodal(spec: SequenceSpec, sprite: np.ndarray) -> np.ndarray:
    """Return the digit intensity of every frame, ignoring the bar and blackouts, and empty where the digit is absent."""
    frames = np.zeros((spec.seq_len, spec.frame_height, spec.frame_width), dtype=np.uint8)
    height, width = sprite.shape
    for t in np.flatnonzero(spec.digit_present):
        y, x = spec.positions[t]
        frames[t, y:y + height, x:x + width] = sprite
    return frames


def bar_columns(spec: SequenceSpec) -> np.ndarray:
    """Return a boolean mask over image columns: True under the bar."""
    columns = np.zeros(spec.frame_width, dtype=bool)
    columns[spec.bar_left:spec.bar_left + spec.bar_width] = True
    return columns


def compose_observed(amodal: np.ndarray, spec: SequenceSpec, settings: RenderSettings) -> np.ndarray:
    """Build the observed RGB frames: colored digit, bar drawn on top, blackout frames all zero."""
    observed = np.zeros(amodal.shape + (3,), dtype=np.uint8)
    for channel, weight in enumerate(settings.digit_rgb):
        if weight == 1.0:
            observed[..., channel] = amodal
        elif weight > 0.0:
            observed[..., channel] = np.rint(amodal * weight).astype(np.uint8)
    observed[:, :, bar_columns(spec), :] = settings.bar_value
    observed[list(spec.blackout_frames)] = 0
    return observed


def modal_mask(amodal: np.ndarray, spec: SequenceSpec) -> np.ndarray:
    """Return the ink pixels that are visible on screen (not under the bar, not blacked out)."""
    mask = amodal > 0
    mask[:, :, bar_columns(spec)] = False
    mask[list(spec.blackout_frames)] = False
    return mask


def render_sequence(spec: SequenceSpec, sprite: np.ndarray, settings: RenderSettings,
                    thresholds: VisibilityThresholds) -> RenderedSequence:
    """Render every pixel array of a spec and attach its ground truth."""
    amodal = render_amodal(spec, sprite)
    return RenderedSequence(spec=spec, observed=compose_observed(amodal, spec, settings), amodal=amodal,
                            modal_mask=modal_mask(amodal, spec), truth=compute_truth(spec, sprite, thresholds))


def _centroids(weights: np.ndarray) -> np.ndarray:
    """Return the (y, x) centroid of each frame of a (T, H, W) weight array, NaN for empty frames."""
    mass = weights.sum(axis=(1, 2))
    ys = np.arange(weights.shape[1])[None, :, None]
    xs = np.arange(weights.shape[2])[None, None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        center = np.stack([(weights * ys).sum(axis=(1, 2)), (weights * xs).sum(axis=(1, 2))], axis=1) / mass[:, None]
    center[mass <= 0] = np.nan
    return center


def measure_from_pixels(rendered: RenderedSequence) -> dict[str, np.ndarray]:
    """Recompute the visible fraction and both centers from the pixels, to check them against the exact truth."""
    amodal = rendered.amodal.astype(np.float64)
    visible = amodal * rendered.modal_mask
    total = amodal.sum(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.where(total > 0, visible.sum(axis=(1, 2)) / total, 0.0)
    return {"visible_fraction": fraction, "center": _centroids(amodal), "modal_center": _centroids(visible)}
