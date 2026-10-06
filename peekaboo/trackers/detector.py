"""A programmed digit detector for the observed frames, the front end of the trackers.

The digit is the only pure red element of a frame (some red, no green, no blue) and the bar the only gray one (the same
nonzero value on the three channels), so both are found exactly by their color. A scene holds one digit and no noise,
so all red pixels belong to the digit and no connected component analysis is needed.

A digit partly hidden by the bar is reported as missing, not detected (critique C11): the centroid of its visible ink
is biased toward the visible side, and a tracker fed with it would learn a wrong velocity just before the occlusion.
Two rules catch a partial view:
- at the bar: red ink within `margin` columns of the bar. The detector cannot tell a digit partly under the bar from
  one just next to it, so both are missing;
- narrower: the ink is narrower than the widest digit seen earlier in the sequence. Some digits have a fragment
  separated from the rest by empty columns; when the bar hides only that fragment, the visible ink can be far from
  the bar, but the digit is narrower than when it was seen whole. The digit's width never changes, so a narrower
  digit is always partly hidden.
A detected digit is therefore fully visible, and its centroid is exact, once the whole digit has been seen in the
sequence.
"""

from dataclasses import dataclass

import numpy as np

DETECTED, NO_DIGIT, AT_BAR, NARROWER = 0, 1, 2, 3
STATUS_NAMES = ("detected", "no digit", "at the bar", "narrower")
MARGIN = 2  # columns: an empty column at the edge of a digit must not hide that the bar covers it


@dataclass(frozen=True)
class Detections:
    """Detections of one sequence, in pixels and (y, x) order, NaN when the digit is not detected."""

    center: np.ndarray  # (T, 2) intensity weighted centroid of the red ink
    extent: np.ndarray  # (T, 4) distances from the centroid to the ink edges: up, down, left, right
    status: np.ndarray  # (T,) DETECTED, NO_DIGIT, AT_BAR, or NARROWER

    @property
    def detected(self) -> np.ndarray:
        """(T,) frames where the digit was detected."""
        return self.status == DETECTED


def digit_mask(frame: np.ndarray) -> np.ndarray:
    """(H, W) pixels of the digit in a frame (3, H, W): some red, no green, no blue."""
    return (frame[0] > 0) & (frame[1] == 0) & (frame[2] == 0)


def bar_columns(frame: np.ndarray) -> np.ndarray:
    """(W,) columns that hold the gray bar in a frame (3, H, W): the same nonzero value on the three channels."""
    gray = (frame[0] > 0) & (frame[0] == frame[1]) & (frame[1] == frame[2])
    return gray.any(axis=0)


def detect_frame(frame: np.ndarray, margin: int = MARGIN) -> tuple[np.ndarray, np.ndarray, int]:
    """Detect the digit in one frame (3, H, W) uint8. Returns its centroid (y, x), its extent, and its status."""
    missing = np.full(2, np.nan), np.full(4, np.nan)
    mask = digit_mask(frame)
    if not mask.any():
        return *missing, NO_DIGIT
    columns = np.flatnonzero(mask.any(axis=0))
    bar = np.flatnonzero(bar_columns(frame))
    if bar.size and np.abs(columns[:, None] - bar[None, :]).min() <= margin:
        return *missing, AT_BAR
    weights = frame[0].astype(np.float64) * mask
    ys, xs = np.nonzero(mask)
    total = weights.sum()
    center = np.array([(weights.sum(axis=1) * np.arange(frame.shape[1])).sum() / total,
                       (weights.sum(axis=0) * np.arange(frame.shape[2])).sum() / total])
    extent = np.array([center[0] - ys.min(), ys.max() - center[0], center[1] - xs.min(), xs.max() - center[1]])
    return center, extent, DETECTED


def detect(frames: np.ndarray, margin: int = MARGIN) -> Detections:
    """Detect the digit in every frame of a sequence (T, 3, H, W) uint8. A digit narrower than the widest one detected
    in an earlier frame is partly hidden, and is reported as NARROWER instead of detected."""
    centers, extents, statuses = [], [], []
    widest = 0
    for frame in frames:
        center, extent, status = detect_frame(frame, margin)
        if status == DETECTED:
            width = round(extent[2] + extent[3])  # columns from the first to the last ink column, exact once rounded
            if width < widest:
                center, extent, status = np.full(2, np.nan), np.full(4, np.nan), NARROWER
            widest = max(widest, width)
        centers.append(center)
        extents.append(extent)
        statuses.append(status)
    return Detections(center=np.stack(centers), extent=np.stack(extents), status=np.array(statuses, dtype=np.int8))
