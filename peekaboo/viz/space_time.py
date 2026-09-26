"""Space time diagrams: x horizontally, frames downward, the bar as a gray band, the ink colored by visibility."""

import matplotlib.pyplot as plt
import numpy as np

from peekaboo.data.occluder import (STATE_OCCLUDED, STATE_PARTIAL, STATE_VISIBLE, VisibilityThresholds, column_ink,
                                    visibility_states, visible_fraction)
from peekaboo.data.spec import SequenceSpec

STATE_COLORS = {STATE_VISIBLE: "tab:green", STATE_PARTIAL: "tab:orange", STATE_OCCLUDED: "tab:red"}
MARK_FIELDS = (("entry_frame", "entry"), ("onset_frame", "onset"), ("expected_reappear_frame", "expected"),
               ("actual_reappear_frame", "reappear"), ("exit_frame", "exit"), ("surprise_frame", "splice"))


def draw_space_time(ax: plt.Axes, x_positions: np.ndarray, ink_width: int, fractions: np.ndarray, bar_left: int,
                    bar_width: int, frame_width: int, thresholds: VisibilityThresholds, marks: dict[str, int],
                    title: str, present: np.ndarray | None = None, blackout: tuple[int, ...] = ()) -> None:
    """Draw one sequence: absent frames are left empty, blackout frames are drawn in black."""
    seq_len = len(x_positions)
    present = np.ones(seq_len, dtype=bool) if present is None else present
    if bar_width > 0:
        ax.axvspan(bar_left, bar_left + bar_width, color="0.8", zorder=0)
    states = visibility_states(fractions, thresholds)
    for t in range(seq_len):
        if not present[t]:
            continue
        x = x_positions[t]
        color = "black" if t in blackout else STATE_COLORS[int(states[t])]
        ax.plot([x, x + ink_width], [t, t], color=color, lw=2.5, solid_capstyle="butt")
    # marks that share a frame are written on one line
    by_frame: dict[int, list[str]] = {}
    for name, frame in marks.items():
        if frame >= 0:
            by_frame.setdefault(frame, []).append(name)
    for frame, names in by_frame.items():
        ax.axhline(frame, color="black", lw=0.5, ls=":")
        ax.text(frame_width + 1, frame, ", ".join(names), fontsize=6, va="center")
    ax.set_xlim(0, frame_width)
    ax.set_ylim(seq_len - 0.5, -0.5)
    ax.set_xlabel("x (px)", fontsize=7)
    ax.set_ylabel("frame", fontsize=7)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)


def draw_spec(ax: plt.Axes, spec: SequenceSpec, sprite: np.ndarray, thresholds: VisibilityThresholds,
              title: str) -> None:
    """Draw a SequenceSpec, with its event frames marked."""
    fractions = visible_fraction(column_ink(sprite), spec.positions[:, 1], spec.bar_left, spec.bar_width)
    marks = {label: getattr(spec, name) for name, label in MARK_FIELDS}
    if marks["expected"] == marks["reappear"]:
        marks.pop("expected")
    draw_space_time(ax, spec.positions[:, 1], spec.ink_width, fractions, spec.bar_left, spec.bar_width,
                    spec.frame_width, thresholds, marks, title, present=spec.digit_present,
                    blackout=spec.blackout_frames)
