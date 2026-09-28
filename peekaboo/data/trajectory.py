"""Integer trajectories that bounce off the frame borders, built forward and backward from an anchor frame.

Each axis is treated as uniform motion on an unfolded line, folded back into the allowed range by a triangle wave.
This makes bounces exactly reversible in time, so a trajectory can be built outward from the occlusion event.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Bounds:
    """Allowed range of the sprite's top left corner, inclusive on both ends."""

    y_min: int
    y_max: int
    x_min: int
    x_max: int

    @classmethod
    def for_sprite(cls, frame_height: int, frame_width: int, height: int, width: int) -> "Bounds":
        """Return the bounds that keep a sprite of the given ink size fully inside the frame."""
        if not (0 < height < frame_height and 0 < width < frame_width):
            raise ValueError(f"Sprite {height}x{width} does not fit with room to move "
                             f"in a {frame_height}x{frame_width} frame.")
        return cls(0, frame_height - height, 0, frame_width - width)

    def axis(self, axis: int) -> tuple[int, int]:
        """Return (min, max) for axis 0 (y) or axis 1 (x)."""
        return (self.y_min, self.y_max) if axis == 0 else (self.x_min, self.x_max)


@dataclass(frozen=True)
class Trajectory:
    """Per frame integer state of the sprite's top left corner, as (seq_len, 2) arrays in (y, x) order.

    velocities[t] moves the sprite from frame t to frame t + 1, and bounced[t] marks a wall bounce between frames t - 1 and t.
    """

    positions: np.ndarray
    velocities: np.ndarray
    bounced: np.ndarray

    def __len__(self) -> int:
        """Number of frames."""
        return len(self.positions)


def _fold(unfolded: np.ndarray, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
    """Map unfolded coordinates to (position, direction). At a wall the direction always points inward."""
    period = 2 * (hi - lo)
    r = np.mod(unfolded, period)
    forward = r < hi - lo
    position = np.where(forward, lo + r, lo + period - r)
    return position, np.where(forward, 1, -1)


def axis_path(anchor_frame: int, anchor_pos: int, anchor_vel: int, seq_len: int,
              lo: int, hi: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return positions, velocities, and bounce flags on one axis, passing through anchor_pos at anchor_frame."""
    if not lo <= anchor_pos <= hi:
        raise ValueError(f"Anchor position {anchor_pos} is outside [{lo}, {hi}].")
    speed = abs(anchor_vel)
    if speed >= hi - lo:
        raise ValueError(f"Speed {speed} must be smaller than the range {hi - lo}.")
    frames = np.arange(seq_len)
    if speed == 0:
        return np.full(seq_len, anchor_pos), np.zeros(seq_len, dtype=np.int64), np.zeros(seq_len, dtype=bool)

    # a sprite resting on a wall must move inward, otherwise the state is ambiguous
    if (anchor_pos == hi and anchor_vel > 0) or (anchor_pos == lo and anchor_vel < 0):
        raise ValueError("Anchor sits on a wall with its velocity pointing outward.")
    offset = anchor_pos - lo
    unfolded_anchor = offset if anchor_vel > 0 else 2 * (hi - lo) - offset
    unfolded = unfolded_anchor + speed * (frames - anchor_frame)
    positions, directions = _fold(unfolded, lo, hi)
    bounced = np.zeros(seq_len, dtype=bool)
    bounced[1:] = directions[1:] != directions[:-1]
    return positions.astype(np.int64), (directions * speed).astype(np.int64), bounced


def build_trajectory(anchor_frame: int, anchor_pos: tuple[int, int], anchor_vel: tuple[int, int],
                     seq_len: int, bounds: Bounds) -> Trajectory:
    """Build a full trajectory whose state at anchor_frame is (anchor_pos, anchor_vel), both in (y, x) order."""
    columns = [axis_path(anchor_frame, anchor_pos[a], anchor_vel[a], seq_len, *bounds.axis(a)) for a in (0, 1)]
    positions, velocities, bounced = (np.stack([columns[0][i], columns[1][i]], axis=1) for i in range(3))
    return Trajectory(positions=positions, velocities=velocities, bounced=bounced)


def continue_from(trajectory: Trajectory, frame: int, pos: tuple[int, int], vel: tuple[int, int],
                  bounds: Bounds) -> Trajectory:
    """Keep the trajectory before `frame`, then restart it from a new state at `frame` (used for scripted events).

    The change at `frame` itself is not flagged as a bounce, since it is a scripted event and not a wall contact.
    """
    new = build_trajectory(frame, pos, vel, len(trajectory), bounds)
    positions = trajectory.positions.copy()
    velocities = trajectory.velocities.copy()
    bounced = trajectory.bounced.copy()
    positions[frame:] = new.positions[frame:]
    velocities[frame:] = new.velocities[frame:]
    bounced[frame:] = new.bounced[frame:]
    bounced[frame] = False
    return Trajectory(positions=positions, velocities=velocities, bounced=bounced)


def has_bounce(trajectory: Trajectory, first: int, last: int, axes: tuple[int, ...] = (0, 1)) -> bool:
    """Return True if a wall bounce happens between frames first and last (any transition inside the window)."""
    first = max(first, 0)
    last = min(last, len(trajectory) - 1)
    if last <= first:
        return False
    return bool(trajectory.bounced[first + 1:last + 1][:, list(axes)].any())


def sample_velocity(rng: np.random.Generator, x_speeds: list[int], y_velocities: list[int]) -> tuple[int, int]:
    """Draw an integer velocity (vy, vx): |vx| from x_speeds with a random sign, vy from y_velocities."""
    vx = int(rng.choice(x_speeds)) * int(rng.choice([-1, 1]))
    vy = int(rng.choice(y_velocities))
    return vy, vx
