"""Full height occluder bar: exact visible ink fractions, occlusion episodes, and a solver that places the bar for a target k.

The solver works event first: it fixes the frame where full occlusion starts, puts the digit at the bar edge there,
builds the trajectory outward from there, and keeps the placement only if the measured k and all window rules hold.
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from peekaboo.data.trajectory import Bounds, Trajectory, build_trajectory, has_bounce

STATE_VISIBLE, STATE_PARTIAL, STATE_OCCLUDED = 0, 1, 2
_FULLY_VISIBLE = 1.0 - 1e-9  # tolerance on float sums
FULLY_VISIBLE = _FULLY_VISIBLE  # public name for the other modules


@dataclass(frozen=True)
class VisibilityThresholds:
    """Occluded if the visible fraction is at most occluded_max, visible if at least visible_min, partial otherwise."""

    occluded_max: float = 0.02
    visible_min: float = 0.95


def column_ink(sprite: np.ndarray) -> np.ndarray:
    """Return the ink of each sprite column, in units of full intensity pixels."""
    return sprite.astype(np.float64).sum(axis=0) / 255.0


def visible_fraction(col_ink: np.ndarray, x_positions: np.ndarray, bar_left: int, bar_width: int) -> np.ndarray:
    """Return, for each frame, the fraction of the digit's ink that the bar does not cover."""
    x = np.asarray(x_positions)
    if bar_width <= 0:
        return np.ones(len(x))
    cols = x[:, None] + np.arange(len(col_ink))[None, :]
    covered = (cols >= bar_left) & (cols < bar_left + bar_width)
    return 1.0 - (covered * col_ink[None, :]).sum(axis=1) / col_ink.sum()


def visibility_states(fractions: np.ndarray, thresholds: VisibilityThresholds) -> np.ndarray:
    """Label each frame STATE_VISIBLE, STATE_PARTIAL, or STATE_OCCLUDED."""
    states = np.full(len(fractions), STATE_PARTIAL, dtype=np.int8)
    states[fractions >= thresholds.visible_min] = STATE_VISIBLE
    states[fractions <= thresholds.occluded_max] = STATE_OCCLUDED
    return states


@dataclass(frozen=True)
class CrossingEvent:
    """Key frames of one pass behind the bar. k is the number of fully occluded frames."""

    entry_frame: int     # first frame where the bar covers some ink
    onset_frame: int     # first fully occluded frame
    reappear_frame: int  # first frame after the onset with visible ink again
    exit_frame: int      # last frame where the bar covers some ink

    @property
    def k(self) -> int:
        """Occlusion duration in frames."""
        return self.reappear_frame - self.onset_frame


def find_crossing(fractions: np.ndarray, thresholds: VisibilityThresholds,
                  frame: int | None = None) -> CrossingEvent | None:
    """Locate the occlusion episode that contains `frame` (the first episode if frame is None).

    Returns None if there is no such episode or if it is still going on at the last frame.
    """
    occluded = fractions <= thresholds.occluded_max
    if frame is None:
        if not occluded.any():
            return None
        frame = int(np.argmax(occluded))
    if not occluded[frame]:
        return None
    onset = frame
    while onset > 0 and occluded[onset - 1]:
        onset -= 1
    reappear = frame
    while reappear < len(fractions) and occluded[reappear]:
        reappear += 1
    if reappear == len(fractions):
        return None

    covered = fractions < _FULLY_VISIBLE
    entry = onset
    while entry > 0 and covered[entry - 1]:
        entry -= 1
    exit_frame = reappear - 1  # the last occluded frame is always covered
    while exit_frame + 1 < len(fractions) and covered[exit_frame + 1]:
        exit_frame += 1
    if exit_frame == len(fractions) - 1:
        return None
    return CrossingEvent(entry, onset, reappear, exit_frame)


@dataclass(frozen=True)
class CrossingSettings:
    """Scene size and window rules used by the placement solver (see configs/data/base.yaml)."""

    frame_height: int
    frame_width: int
    seq_len: int
    y_velocities: tuple[int, ...]
    k_speeds: dict[int, tuple[int, ...]]
    n_context: int
    n_post: int
    n_clean: int
    thresholds: VisibilityThresholds
    allow_contact_outside_window: bool = True
    max_attempts: int = 500
    control_min_width: int = 1  # narrowest bar of a control sequence

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CrossingSettings":
        """Build the settings from a data config dictionary."""
        motion, occluder, visibility = config["motion"], config["occluder"], config["visibility"]
        return cls(frame_height=config["frame_height"], frame_width=config["frame_width"],
                   seq_len=config["seq_len"], y_velocities=tuple(motion["y_velocities"]),
                   k_speeds={int(k): tuple(v) for k, v in occluder["k_speeds"].items()},
                   n_context=motion["n_context"], n_post=motion["n_post"], n_clean=motion["n_clean"],
                   thresholds=VisibilityThresholds(**visibility),
                   allow_contact_outside_window=occluder["allow_contact_outside_window"],
                   max_attempts=occluder["max_attempts"],
                   control_min_width=occluder.get("control_min_width", 1))


@dataclass(frozen=True)
class OcclusionPlan:
    """A trajectory and a bar that together produce one crossing with exactly k fully occluded frames."""

    trajectory: Trajectory
    bar_left: int
    bar_width: int
    event: CrossingEvent
    fractions: np.ndarray  # visible fraction per frame
    speed: int             # |vx|
    attempts: int          # placement attempts used, for efficiency reports


def windows_ok(traj: Trajectory, fractions: np.ndarray, event: CrossingEvent, settings: CrossingSettings,
               x_bounces: int = 0) -> bool:
    """Check the context, post, other contact, and bounce rules around one event.

    The analysis window runs from n_context frames before entry to n_post frames after exit. Around the event,
    exactly `x_bounces` wall bounces on x are allowed, and each of them must happen while the digit is fully hidden.
    """
    fully_visible = fractions >= _FULLY_VISIBLE
    window_start = event.entry_frame - settings.n_context
    window_end = event.exit_frame + settings.n_post
    if window_start < 0 or window_end > len(fractions) - 1:
        return False
    # the digit is fully visible just before and just after the event
    if not fully_visible[window_start:event.entry_frame].all():
        return False
    if not fully_visible[event.exit_frame + 1:window_end + 1].all():
        return False
    # other contacts with the bar, if allowed at all, stay outside the window
    if not settings.allow_contact_outside_window:
        outside = np.r_[fully_visible[:window_start], fully_visible[window_end + 1:]]
        if not outside.all():
            return False
    # clean motion on x from approach to departure, except the allowed hidden bounces
    c = settings.n_clean
    first = max(event.entry_frame - c, 0)
    last = min(event.exit_frame + c, len(fractions) - 1)
    x_hits = first + 1 + np.flatnonzero(traj.bounced[first + 1:last + 1, 1])
    if len(x_hits) != x_bounces:
        return False
    occluded = fractions <= settings.thresholds.occluded_max
    if not all(occluded[t - 1] and occluded[t] for t in x_hits):
        return False
    # no y bounce right around entry and exit
    if has_bounce(traj, event.entry_frame - c, event.entry_frame, axes=(0,)):
        return False
    return not has_bounce(traj, event.exit_frame, event.exit_frame + c, axes=(0,))


def find_contact(fractions: np.ndarray, frame: int) -> CrossingEvent | None:
    """Locate the frames around `frame` where the bar covers some ink, for a digit that never gets fully hidden.

    Onset and reappear both point to the most covered frame, so k is 0. Returns None if the contact touches either end.
    """
    covered = fractions < _FULLY_VISIBLE
    if not covered[frame]:
        return None
    entry = frame
    while entry > 0 and covered[entry - 1]:
        entry -= 1
    exit_frame = frame
    while exit_frame + 1 < len(fractions) and covered[exit_frame + 1]:
        exit_frame += 1
    if entry == 0 or exit_frame == len(fractions) - 1:
        return None
    peak = entry + int(np.argmin(fractions[entry:exit_frame + 1]))
    return CrossingEvent(entry, peak, peak, exit_frame)


def _random_motion(rng: np.random.Generator, speeds: list[int], settings: CrossingSettings) -> tuple[int, int, int]:
    """Draw |vx|, the x direction, and vy for one attempt."""
    speed = int(rng.choice(speeds))
    direction = int(rng.choice([-1, 1]))
    vy = int(rng.choice(settings.y_velocities))
    return speed, direction, vy


def _try_crossing(rng: np.random.Generator, col_ink: np.ndarray, bounds: Bounds, width: int, k: int,
                  speeds: list[int], settings: CrossingSettings) -> tuple | None:
    """Make one random crossing attempt. Returns (trajectory, bar_left, bar_width, event, fractions, speed) or None."""
    speed, direction, vy = _random_motion(rng, speeds, settings)

    # loose time bounds: entry is at or before the onset, exit at or after reappear - 1; the rules check the rest
    onset_max = settings.seq_len - settings.n_post - k
    if settings.n_context > onset_max:
        return None
    onset = int(rng.integers(settings.n_context, onset_max + 1))

    # the digit enters through a fixed bar edge and sits just inside it at the onset frame
    phase = int(rng.integers(0, speed))
    entry_edge = int(rng.integers(0, settings.frame_width + 1))
    if direction > 0:
        x_onset = entry_edge + phase           # ink left edge just inside the bar's left edge
    else:
        x_onset = entry_edge - phase - width   # ink right edge just inside the bar's right edge
    y_onset = int(rng.integers(bounds.y_min + 1, bounds.y_max))  # strictly inside, so any vy is valid
    try:
        traj = build_trajectory(onset, (y_onset, x_onset), (vy, direction * speed), settings.seq_len, bounds)
    except ValueError:  # anchor outside the frame, or on a wall with outward velocity
        return None

    # width + phase + (k - 1) * speed hides the digit for exactly k frames geometrically. Narrower bars compensate
    # digits whose edge columns hold so little ink that a nearly hidden frame already counts as occluded
    # (one step per faint side), and a wider one covers the sub step range
    base_width = width + phase + (k - 1) * speed
    for slack in rng.permutation(np.arange(-2 * speed, speed)):
        bar_width = base_width + int(slack)
        bar_left = entry_edge if direction > 0 else entry_edge - bar_width
        if bar_width < 1 or bar_left < 0 or bar_left + bar_width > settings.frame_width:
            continue
        fractions = visible_fraction(col_ink, traj.positions[:, 1], bar_left, bar_width)
        event = find_crossing(fractions, settings.thresholds, frame=onset)
        if event is not None and event.k == k and windows_ok(traj, fractions, event, settings):
            return traj, bar_left, bar_width, event, fractions, speed
    return None


def _try_hidden_bounce(rng: np.random.Generator, col_ink: np.ndarray, bounds: Bounds, width: int, k: int,
                       speeds: list[int], settings: CrossingSettings) -> tuple | None:
    """Make one attempt with the bar against a wall: the digit bounces off the wall while hidden and comes back out."""
    speed, side, vy = _random_motion(rng, speeds, settings)  # side +1: bar on the right wall

    # anchor one frame before the wall bounce, 1 to speed pixels from the wall, moving toward it
    t_wall = int(rng.integers(settings.n_context, settings.seq_len - settings.n_post))
    gap = int(rng.integers(1, speed + 1))
    x_anchor = bounds.x_max - gap if side > 0 else bounds.x_min + gap
    y_anchor = int(rng.integers(bounds.y_min + 1, bounds.y_max))
    try:
        traj = build_trajectory(t_wall, (y_anchor, x_anchor), (vy, side * speed), settings.seq_len, bounds)
    except ValueError:
        return None

    # the digit stays hidden for about 1 + 2 (bar width - ink width) / speed frames: try widths around that
    base_width = width + math.ceil((k - 1) * speed / 2)
    for slack in rng.permutation(np.arange(-2 * speed, 2 * speed)):
        bar_width = base_width + int(slack)
        if bar_width < 1 or bar_width > settings.frame_width:
            continue
        bar_left = settings.frame_width - bar_width if side > 0 else 0
        fractions = visible_fraction(col_ink, traj.positions[:, 1], bar_left, bar_width)
        event = find_crossing(fractions, settings.thresholds, frame=t_wall)
        if event is not None and event.k == k and windows_ok(traj, fractions, event, settings, x_bounces=1):
            return traj, bar_left, bar_width, event, fractions, speed
    return None


def _try_contact(rng: np.random.Generator, col_ink: np.ndarray, bounds: Bounds, width: int,
                 speeds: list[int], settings: CrossingSettings) -> tuple | None:
    """Make one attempt with a bar narrower than the digit, which passes behind it without ever being fully hidden."""
    speed, direction, vy = _random_motion(rng, speeds, settings)
    t_anchor = int(rng.integers(settings.n_context, settings.seq_len - settings.n_post))
    bar_width = int(rng.integers(settings.control_min_width, width))  # narrower than the digit
    bar_left = int(rng.integers(0, settings.frame_width - bar_width + 1))
    # at the anchor frame, the bar's middle column lies somewhere under the ink
    x_anchor = bar_left + bar_width // 2 - int(rng.integers(0, width))
    y_anchor = int(rng.integers(bounds.y_min + 1, bounds.y_max))
    try:
        traj = build_trajectory(t_anchor, (y_anchor, x_anchor), (vy, direction * speed), settings.seq_len, bounds)
    except ValueError:
        return None
    fractions = visible_fraction(col_ink, traj.positions[:, 1], bar_left, bar_width)
    if (fractions <= settings.thresholds.occluded_max).any():  # never fully hidden, anywhere in the sequence
        return None
    event = find_contact(fractions, t_anchor)
    if event is None or not windows_ok(traj, fractions, event, settings):
        return None
    return traj, bar_left, bar_width, event, fractions, speed


def _search(rng: np.random.Generator, sprite: np.ndarray, settings: CrossingSettings, attempt: Any,
            description: str) -> OcclusionPlan:
    """Repeat placement attempts until one succeeds, and wrap it in an OcclusionPlan."""
    height, width = sprite.shape
    col_ink = column_ink(sprite)
    bounds = Bounds.for_sprite(settings.frame_height, settings.frame_width, height, width)
    for i in range(1, settings.max_attempts + 1):
        found = attempt(rng, col_ink, bounds, width)
        if found is not None:
            traj, bar_left, bar_width, event, fractions, speed = found
            return OcclusionPlan(traj, bar_left, bar_width, event, fractions, speed, attempts=i)
    raise RuntimeError(f"No valid placement for {description}, sprite {height}x{width} "
                       f"after {settings.max_attempts} attempts.")


def plan_crossing(rng: np.random.Generator, sprite: np.ndarray, k: int, settings: CrossingSettings,
                  speeds: list[int] | None = None) -> OcclusionPlan:
    """Place a trajectory and a bar so the digit crosses behind it, fully hidden for exactly k frames.

    Placements are drawn uniformly among the valid ones by rejection. Raises RuntimeError if none is found.
    """
    if k < 1:
        raise ValueError("k must be at least 1. Use plan_contact for sequences without full occlusion.")
    speeds = list(speeds if speeds is not None else settings.k_speeds[k])
    return _search(rng, sprite, settings,
                   lambda r, ink, b, w: _try_crossing(r, ink, b, w, k, speeds, settings),
                   f"a crossing with k={k}, speeds={speeds}")


def plan_hidden_bounce(rng: np.random.Generator, sprite: np.ndarray, k: int, settings: CrossingSettings,
                       speeds: list[int] | None = None) -> OcclusionPlan:
    """Place a bar against a wall so the digit bounces off the wall while hidden for exactly k frames."""
    if k < 1:
        raise ValueError("k must be at least 1.")
    speeds = list(speeds if speeds is not None else settings.k_speeds[k])
    return _search(rng, sprite, settings,
                   lambda r, ink, b, w: _try_hidden_bounce(r, ink, b, w, k, speeds, settings),
                   f"a hidden bounce with k={k}, speeds={speeds}")


def plan_contact(rng: np.random.Generator, sprite: np.ndarray, settings: CrossingSettings,
                 speeds: list[int]) -> OcclusionPlan:
    """Place a bar narrower than the digit, which passes behind it without ever being fully hidden (k = 0).

    Raises RuntimeError at once if the digit is not wider than the minimum control bar.
    """
    if sprite.shape[1] <= settings.control_min_width:
        raise RuntimeError(f"A {sprite.shape[1]} px wide digit cannot pass behind a bar of at least "
                           f"{settings.control_min_width} px without being fully hidden.")
    return _search(rng, sprite, settings,
                   lambda r, ink, b, w: _try_contact(r, ink, b, w, list(speeds), settings),
                   f"a narrow bar contact, speeds={list(speeds)}")
