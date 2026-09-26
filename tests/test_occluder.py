"""Tests for visible fractions, occlusion episodes, and the occluder placement solver."""

import numpy as np
import pytest

from peekaboo.config import load_config
from peekaboo.data.occluder import (STATE_OCCLUDED, STATE_PARTIAL, STATE_VISIBLE, CrossingSettings,
                                    VisibilityThresholds, column_ink, find_crossing, plan_crossing,
                                    visibility_states, visible_fraction)
from peekaboo.data.trajectory import has_bounce
from peekaboo.paths import CONFIGS_DIR
from peekaboo.seeding import make_rng

THRESHOLDS = VisibilityThresholds(occluded_max=0.02, visible_min=0.95)
SETTINGS = CrossingSettings.from_config(load_config(CONFIGS_DIR / "data" / "base.yaml"))


def synthetic_sprite(seed: int, height: int = 14, width: int = 12) -> np.ndarray:
    """Return a random digit like sprite whose first and last columns contain ink."""
    rng = np.random.default_rng(seed)
    sprite = (rng.random((height, width)) < 0.4) * rng.integers(26, 256, size=(height, width))
    sprite[height // 2, 0] = 255
    sprite[height // 2, -1] = 255
    return sprite.astype(np.uint8)


def test_visible_fraction_by_hand() -> None:
    """Covering columns removes exactly their share of the ink."""
    ink = np.array([1.0, 2.0, 1.0])  # column ink of a 3 column sprite, 4 in total
    x = np.array([0, 10, 8, 9])      # sprite left edge per frame
    fractions = visible_fraction(ink, x, bar_left=9, bar_width=3)  # bar covers columns 9, 10, 11
    assert fractions.tolist() == pytest.approx([1.0, 0.25, 0.25, 0.0])


def test_column_ink_units() -> None:
    """Column ink is measured in full intensity pixels."""
    sprite = np.array([[255, 0], [255, 51]], dtype=np.uint8)
    assert column_ink(sprite).tolist() == pytest.approx([2.0, 0.2])


def test_no_bar_means_fully_visible() -> None:
    """A bar of width zero leaves every frame fully visible."""
    assert visible_fraction(np.ones(4), np.arange(5), 0, 0).tolist() == [1.0] * 5


def test_visibility_states() -> None:
    """Thresholds split frames into visible, partial, and occluded."""
    states = visibility_states(np.array([1.0, 0.95, 0.5, 0.03, 0.02, 0.0]), THRESHOLDS)
    assert states.tolist() == [STATE_VISIBLE, STATE_VISIBLE, STATE_PARTIAL, STATE_PARTIAL, STATE_OCCLUDED,
                               STATE_OCCLUDED]


def test_find_crossing_by_hand() -> None:
    """Entry, onset, reappearance, and exit are the right frames, and k counts the hidden frames."""
    fractions = np.array([1, 1, 0.6, 0.1, 0, 0, 0, 0.3, 0.8, 1, 1])
    event = find_crossing(fractions, THRESHOLDS)
    assert (event.entry_frame, event.onset_frame, event.reappear_frame, event.exit_frame) == (2, 4, 7, 8)
    assert event.k == 3


def test_find_crossing_jump_to_fully_visible() -> None:
    """A thin fast digit can go straight from hidden to fully visible: exit is then the last hidden frame."""
    event = find_crossing(np.array([1, 0.5, 0, 0, 1, 1]), THRESHOLDS)
    assert (event.entry_frame, event.onset_frame, event.reappear_frame, event.exit_frame) == (1, 2, 4, 3)


def test_find_crossing_with_anchor_frame() -> None:
    """With two episodes, the anchor frame selects which one is returned."""
    fractions = np.array([1, 0, 1, 1, 0.5, 0, 0, 0.5, 1])
    later = find_crossing(fractions, THRESHOLDS, frame=5)
    assert (later.entry_frame, later.onset_frame, later.reappear_frame, later.exit_frame) == (4, 5, 7, 7)
    first = find_crossing(fractions, THRESHOLDS)
    assert (first.entry_frame, first.onset_frame, first.reappear_frame, first.exit_frame) == (1, 1, 2, 1)
    assert find_crossing(fractions, THRESHOLDS, frame=3) is None


def test_find_crossing_none_cases() -> None:
    """No full occlusion, occlusion until the end, or cover until the end give None."""
    assert find_crossing(np.array([1, 0.5, 0.1, 1]), THRESHOLDS) is None
    assert find_crossing(np.array([1, 0.5, 0, 0]), THRESHOLDS) is None
    assert find_crossing(np.array([1, 0, 0.5, 0.7]), THRESHOLDS) is None


CASES = [(k, v, seed) for k, speeds in SETTINGS.k_speeds.items() for v in speeds for seed in range(3)]


@pytest.mark.parametrize("k,speed,seed", CASES)
def test_plan_crossing_meets_every_rule(k: int, speed: int, seed: int) -> None:
    """Placed crossings have exactly k hidden frames and respect the context, post, and bounce rules."""
    sprite = synthetic_sprite(seed, width=[4, 10, 12][seed])
    plan = plan_crossing(make_rng("test_occluder", k, speed, seed), sprite, k, SETTINGS, speeds=[speed])
    e, f = plan.event, plan.fractions
    assert e.k == k and plan.speed == speed
    assert (f[e.onset_frame:e.reappear_frame] <= THRESHOLDS.occluded_max).all()
    assert f[e.reappear_frame] > THRESHOLDS.occluded_max
    assert 0 <= plan.bar_left and plan.bar_left + plan.bar_width <= SETTINGS.frame_width
    assert plan.bar_width >= sprite.shape[1]
    # fully visible context before entry and post frames after exit
    assert e.entry_frame >= SETTINGS.n_context
    assert (f[e.entry_frame - SETTINGS.n_context:e.entry_frame] == 1).all()
    assert e.exit_frame + SETTINGS.n_post <= SETTINGS.seq_len - 1
    assert (f[e.exit_frame + 1:e.exit_frame + SETTINGS.n_post + 1] == 1).all()
    assert not has_bounce(plan.trajectory, e.entry_frame - SETTINGS.n_clean, e.exit_frame + SETTINGS.n_clean,
                          axes=(1,))
    # the digit leaves on the far side of the bar
    x = plan.trajectory.positions[:, 1]
    assert np.sign(x[e.exit_frame + 1] - x[e.entry_frame - 1]) == np.sign(plan.trajectory.velocities[e.onset_frame, 1])


def test_plan_crossing_is_deterministic() -> None:
    """The same random stream gives the same placement."""
    sprite = synthetic_sprite(0)
    a = plan_crossing(make_rng("det", 1), sprite, 4, SETTINGS)
    b = plan_crossing(make_rng("det", 1), sprite, 4, SETTINGS)
    assert (a.bar_left, a.bar_width, a.event) == (b.bar_left, b.bar_width, b.event)
    assert np.array_equal(a.trajectory.positions, b.trajectory.positions)


def test_plan_crossing_rejects_impossible_requests() -> None:
    """k = 0 is refused, and a k longer than the sequence raises after the attempts run out."""
    sprite = synthetic_sprite(0)
    with pytest.raises(ValueError):
        plan_crossing(make_rng("x"), sprite, 0, SETTINGS)
    with pytest.raises(RuntimeError):
        plan_crossing(make_rng("x"), sprite, 40, SETTINGS, speeds=[2])
    # a 16 px wide digit at 2 px/frame needs 8 + 8 + 12 + 8 + 4 = 40 frames for k = 12: one too many
    with pytest.raises(RuntimeError):
        plan_crossing(make_rng("x"), synthetic_sprite(0, width=16), 12, SETTINGS, speeds=[2])
