"""Tests for reversible bouncing trajectories and the SequenceSpec container."""

import json

import numpy as np
import pytest

from peekaboo.data.spec import SequenceSpec
from peekaboo.data.trajectory import (Bounds, axis_path, build_trajectory, continue_from, has_bounce,
                                      sample_velocity)
from peekaboo.seeding import make_rng

BOUNDS = Bounds.for_sprite(64, 15, 12)  # y in [0, 49], x in [0, 52]


def naive_forward(pos: int, vel: int, n: int, lo: int, hi: int) -> list[int]:
    """Reference simulation: step, and mirror the overshoot when the position leaves [lo, hi]."""
    out = [pos]
    for _ in range(n - 1):
        pos += vel
        if pos > hi:
            pos, vel = 2 * hi - pos, -vel
        elif pos < lo:
            pos, vel = 2 * lo - pos, -vel
        out.append(pos)
    return out


def random_trajectory(seed: int, seq_len: int = 40):
    """Return a random trajectory with its anchor, for property tests."""
    rng = make_rng("test_trajectory", seed)
    anchor_frame = int(rng.integers(0, seq_len))
    anchor_pos = (int(rng.integers(1, BOUNDS.y_max)), int(rng.integers(1, BOUNDS.x_max)))
    anchor_vel = sample_velocity(rng, [2, 3, 4], [-2, -1, 0, 1, 2])
    return build_trajectory(anchor_frame, anchor_pos, anchor_vel, seq_len, BOUNDS), anchor_frame, anchor_pos, anchor_vel


@pytest.mark.parametrize("seed", range(50))
def test_trajectory_properties(seed: int) -> None:
    """Anchor state is exact, positions stay in bounds, motion matches a naive mirror simulation."""
    traj, t_a, pos_a, vel_a = random_trajectory(seed)
    assert tuple(traj.positions[t_a]) == pos_a
    assert tuple(traj.velocities[t_a]) == vel_a
    for axis in (0, 1):
        lo, hi = BOUNDS.axis(axis)
        p = traj.positions[:, axis]
        assert p.min() >= lo and p.max() <= hi
        expected = naive_forward(int(p[0]), int(traj.velocities[0, axis]), len(traj), lo, hi)
        assert p.tolist() == expected
        # without a bounce, the step equals the velocity
        steps = np.diff(p)
        free = ~traj.bounced[1:, axis]
        assert np.array_equal(steps[free], traj.velocities[:-1, axis][free])


@pytest.mark.parametrize("seed", range(20))
def test_reanchoring_gives_same_trajectory(seed: int) -> None:
    """Rebuilding from the state at any other frame reproduces the same trajectory (bounces are reversible)."""
    traj, _, _, _ = random_trajectory(seed)
    for t in (0, 7, 23, len(traj) - 1):
        again = build_trajectory(t, tuple(traj.positions[t]), tuple(traj.velocities[t]), len(traj), BOUNDS)
        assert np.array_equal(again.positions, traj.positions)
        assert np.array_equal(again.velocities, traj.velocities)
        assert np.array_equal(again.bounced, traj.bounced)


def test_bounce_on_right_wall() -> None:
    """A sprite two steps from the right wall bounces, and the flag and velocity flip land on the right frame."""
    positions, velocities, bounced = axis_path(anchor_frame=0, anchor_pos=46, anchor_vel=3, seq_len=5, lo=0, hi=50)
    assert positions.tolist() == [46, 49, 48, 45, 42]
    assert velocities.tolist() == [3, 3, -3, -3, -3]
    assert bounced.tolist() == [False, False, True, False, False]


def test_landing_exactly_on_wall() -> None:
    """Landing exactly on the wall counts as the bounce frame, with the velocity already pointing inward."""
    positions, velocities, bounced = axis_path(0, 46, 2, 5, 0, 50)
    assert positions.tolist() == [46, 48, 50, 48, 46]
    assert velocities.tolist() == [2, 2, -2, -2, -2]
    assert bounced.tolist() == [False, False, True, False, False]


def test_backward_through_left_wall() -> None:
    """Building backward from an anchor also reflects off the walls."""
    positions, _, bounced = axis_path(anchor_frame=4, anchor_pos=5, anchor_vel=4, seq_len=5, lo=0, hi=50)
    assert positions.tolist() == [11, 7, 3, 1, 5]
    assert bounced.tolist() == [False, False, False, True, False]


def test_zero_velocity_is_static() -> None:
    """With zero velocity the sprite stays in place and never bounces."""
    positions, velocities, bounced = axis_path(3, 10, 0, 6, 0, 50)
    assert set(positions.tolist()) == {10}
    assert not velocities.any() and not bounced.any()


def test_invalid_anchors_raise() -> None:
    """Outward velocity on a wall, positions out of range, and speeds too large are rejected."""
    with pytest.raises(ValueError):
        axis_path(0, 50, 2, 5, 0, 50)
    with pytest.raises(ValueError):
        axis_path(0, 0, -2, 5, 0, 50)
    with pytest.raises(ValueError):
        axis_path(0, 51, 2, 5, 0, 50)
    with pytest.raises(ValueError):
        axis_path(0, 10, 50, 5, 0, 50)
    with pytest.raises(ValueError):
        Bounds.for_sprite(64, 64, 10)


def test_continue_from_keeps_the_past() -> None:
    """A scripted restart keeps earlier frames, starts from the new state, and is not flagged as a bounce."""
    traj = build_trajectory(0, (20, 10), (0, 3), 20, BOUNDS)
    changed = continue_from(traj, 8, (20, 34), (0, -3), BOUNDS)
    assert np.array_equal(changed.positions[:8], traj.positions[:8])
    assert tuple(changed.positions[8]) == (20, 34)
    assert tuple(changed.velocities[8]) == (0, -3)
    assert not changed.bounced[8].any()
    assert changed.positions[9, 1] == 31


def test_has_bounce_windows() -> None:
    """has_bounce only looks at transitions inside the window."""
    traj = build_trajectory(0, (20, 45), (0, 3), 10, BOUNDS)  # x: 45, 48, 51, then 54 mirrors to 50 at frame 3
    assert traj.bounced[3, 1]
    assert has_bounce(traj, 0, 3)
    assert has_bounce(traj, 2, 5, axes=(1,))
    assert not has_bounce(traj, 3, 9)
    assert not has_bounce(traj, 0, 9, axes=(0,))


def test_sample_velocity_values() -> None:
    """Sampled velocities come from the allowed sets and use both x directions."""
    rng = np.random.default_rng(0)
    draws = [sample_velocity(rng, [2, 3, 4], [-1, 0, 1]) for _ in range(200)]
    assert {abs(vx) for _, vx in draws} == {2, 3, 4}
    assert {vy for vy, _ in draws} == {-1, 0, 1}
    assert min(vx for _, vx in draws) < 0 < max(vx for _, vx in draws)


def make_spec() -> SequenceSpec:
    """Build a small spec with every field set, for serialization tests."""
    traj = build_trajectory(5, (20, 10), (1, 3), 12, BOUNDS)
    return SequenceSpec(split="test", index=3, seed=42, frame_size=64, seq_len=12, digit_index=7,
                        mnist_index=1234, label=5, ink_height=15, ink_width=12, positions=traj.positions,
                        velocities=traj.velocities, digit_present=np.ones(12, dtype=bool), bar_left=30,
                        bar_width=24, blackout_frames=(6, 7), condition="occlusion", k_target=4, onset_frame=6,
                        expected_reappear_frame=10, actual_reappear_frame=10, extra={"note": "demo"})


def test_spec_round_trip_through_json() -> None:
    """A spec survives to_dict, JSON, and from_dict unchanged."""
    spec = make_spec()
    again = SequenceSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert again == spec
    assert again.positions.dtype == np.int64 and again.digit_present.dtype == bool


def test_spec_metadata_is_flat() -> None:
    """Metadata rows hold only scalars, with extra parameters prefixed."""
    row = make_spec().metadata()
    assert "positions" not in row and row["extra_note"] == "demo"
    assert row["blackout_frames"] == "6,7"
    assert all(not isinstance(value, (list, dict, np.ndarray)) for value in row.values())
