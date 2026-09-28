"""Per frame and per sequence ground truth, computed exactly from a spec and its digit sprite (no pixels needed).

Frame states extend the occluder states with blackout and absent frames. Each frame also says whether the digit is
in contact with the bar as part of the main event (the one the sequence was built around), as part of another
contact outside the analysis window, or not at all.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from peekaboo.data.occluder import FULLY_VISIBLE, STATE_OCCLUDED, VisibilityThresholds, visibility_states
from peekaboo.data.spec import SequenceSpec

STATE_BLACKOUT, STATE_ABSENT = 3, 4
STATE_NAMES = ("visible", "partial", "occluded", "blackout", "absent")
EPISODE_NONE, EPISODE_MAIN, EPISODE_OTHER = 0, 1, 2


@dataclass(frozen=True)
class FrameTruth:
    """Ground truth per frame, in pixels and (y, x) order, with pixel centers at integer coordinates."""

    center: np.ndarray            # (T, 2) intensity weighted centroid of all the ink (amodal), NaN when absent
    modal_center: np.ndarray      # (T, 2) centroid of the visible ink only, NaN when no ink is visible
    velocity: np.ndarray          # (T, 2) velocity leaving each frame, px/frame
    visible_fraction: np.ndarray  # (T,) share of the ink visible on screen (0 when absent or blacked out)
    state: np.ndarray             # (T,) index into STATE_NAMES
    episode: np.ndarray           # (T,) EPISODE_NONE, EPISODE_MAIN, or EPISODE_OTHER
    in_window: np.ndarray         # (T,) inside the analysis window


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return the (first, last) indices of each stretch of consecutive True values."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return [(int(a), int(b) - 1) for a, b in zip(edges[::2], edges[1::2])]


def main_reference_frame(spec: SequenceSpec) -> int:
    """Return a frame inside the main event: the entry frame, or the reappearance if the digit appears from nowhere."""
    if spec.entry_frame >= 0:
        return spec.entry_frame
    return spec.actual_reappear_frame


def compute_truth(spec: SequenceSpec, sprite: np.ndarray, thresholds: VisibilityThresholds) -> FrameTruth:
    """Compute the ground truth of every frame from the spec and the digit sprite."""
    seq_len = spec.seq_len
    height, width = sprite.shape
    ink = sprite.astype(np.float64) / 255.0
    col_ink = ink.sum(axis=0)                               # ink of each sprite column
    col_y = (ink * np.arange(height)[:, None]).sum(axis=0)  # y moment of each column
    col_x = col_ink * np.arange(width)                      # x moment of each column
    total = col_ink.sum()

    # the bar covers whole columns, so visibility is decided column by column
    cols = spec.positions[:, 1][:, None] + np.arange(width)[None, :]
    visible_cols = ~((cols >= spec.bar_left) & (cols < spec.bar_left + spec.bar_width))
    visible_ink = (visible_cols * col_ink).sum(axis=1)
    bar_fraction = visible_ink / total

    present = spec.digit_present.astype(bool)
    blackout = np.zeros(seq_len, dtype=bool)
    blackout[list(spec.blackout_frames)] = True
    on_screen = present & ~blackout

    offset = spec.positions.astype(np.float64)
    center = offset + np.array([col_y.sum(), col_x.sum()]) / total
    center[~present] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        modal = offset + np.stack([(visible_cols * col_y).sum(axis=1), (visible_cols * col_x).sum(axis=1)],
                                  axis=1) / visible_ink[:, None]
    modal[~on_screen | (visible_ink <= 0)] = np.nan

    state = visibility_states(bar_fraction, thresholds)
    state[blackout] = STATE_BLACKOUT
    state[~present] = STATE_ABSENT

    # contact episodes: the one containing the main event, and any other
    episode = np.full(seq_len, EPISODE_NONE, dtype=np.int8)
    reference = main_reference_frame(spec)
    for first, last in runs(present & (bar_fraction < FULLY_VISIBLE)):
        episode[first:last + 1] = EPISODE_MAIN if first <= reference <= last else EPISODE_OTHER

    in_window = np.zeros(seq_len, dtype=bool)
    if spec.window_start >= 0:
        in_window[spec.window_start:spec.window_end + 1] = True
    return FrameTruth(center=center, modal_center=modal, velocity=spec.velocities.copy(),
                      visible_fraction=np.where(on_screen, bar_fraction, 0.0), state=state, episode=episode,
                      in_window=in_window)


def sequence_summary(truth: FrameTruth) -> dict[str, Any]:
    """Return per sequence counts: measured k of the main event, and the other contacts and occlusions."""
    occluded = truth.state == STATE_OCCLUDED
    main = truth.episode == EPISODE_MAIN
    other = truth.episode == EPISODE_OTHER
    other_runs = runs(other)
    return {
        "measured_k": int((occluded & main).sum()),
        "occluded_frames": int(occluded.sum()),
        "other_contacts": len(other_runs),
        "other_occlusions": sum(1 for first, last in other_runs if occluded[first:last + 1].any()),
        "other_frames_in_window": int((other & truth.in_window).sum()),
    }
