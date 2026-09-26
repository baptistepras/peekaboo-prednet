"""Tests for the control, occlusion, hidden bounce, and blackout conditions."""

import numpy as np
import pytest

from peekaboo.data.conditions import CONDITIONS, GeneratorSettings, sample_spec, spec_visible_fraction
from peekaboo.data.mnist_pool import DigitPool
from peekaboo.data.occluder import column_ink, find_crossing, visible_fraction
from peekaboo.data.spec import SequenceSpec


def bar_fractions(spec: SequenceSpec, pool: DigitPool) -> np.ndarray:
    """Visible fraction per frame due to the bar alone."""
    return visible_fraction(column_ink(pool.sprite(spec.digit_index)), spec.positions[:, 1], spec.bar_left,
                            spec.bar_width)


@pytest.mark.parametrize("condition", CONDITIONS)
@pytest.mark.parametrize("index", range(6))
def test_condition_rules(condition: str, index: int, pool: DigitPool, settings: GeneratorSettings) -> None:
    """Every condition respects the analysis window, and its own definition."""
    spec = sample_spec(pool, settings, "test", index, 0, condition=condition)
    cs = settings.crossing
    occ_max = cs.thresholds.occluded_max
    f = bar_fractions(spec, pool)
    assert spec.condition == condition
    assert 0 <= spec.window_start <= spec.entry_frame <= spec.exit_frame <= spec.window_end <= spec.seq_len - 1
    assert (f[spec.window_start:spec.entry_frame] == 1).all()
    assert (f[spec.exit_frame + 1:spec.exit_frame + cs.n_post + 1] == 1).all()

    if condition in ("control", "blackout"):
        assert spec.bar_width < spec.ink_width
        assert (f > occ_max).all()  # never fully hidden by the bar
    if condition in ("occlusion", "hidden_bounce"):
        event = find_crossing(f, cs.thresholds, frame=spec.onset_frame)
        assert event.k == spec.k_target >= 2
        assert spec.expected_reappear_frame == spec.actual_reappear_frame == event.reappear_frame
    if condition == "occlusion":
        # the digit leaves on the far side
        assert np.sign(spec.velocities[spec.entry_frame, 1]) == np.sign(spec.velocities[spec.exit_frame, 1])
    if condition == "hidden_bounce":
        assert spec.bar_left == 0 or spec.bar_left + spec.bar_width == spec.frame_width
        # it comes back out on the side it entered
        assert np.sign(spec.velocities[spec.entry_frame, 1]) == -np.sign(spec.velocities[spec.exit_frame, 1])
    if condition == "control":
        assert spec.k_target == 0 and spec.onset_frame == spec.actual_reappear_frame
    if condition == "blackout":
        frames = spec.blackout_frames
        assert len(frames) == spec.k_target and list(frames) == list(range(frames[0], frames[0] + len(frames)))
        assert spec.onset_frame == frames[0] and spec.actual_reappear_frame == frames[-1] + 1
        assert (spec_visible_fraction(spec, pool.sprite(spec.digit_index))[list(frames)] == 0).all()


def test_sample_spec_is_deterministic(pool: DigitPool, settings: GeneratorSettings) -> None:
    """The same (seed, split, index) always gives the same spec, and another index a different one."""
    a = sample_spec(pool, settings, "train", 7, 3)
    b = sample_spec(pool, settings, "train", 7, 3)
    c = sample_spec(pool, settings, "train", 8, 3)
    assert a == b
    assert a != c


def test_training_mix_draws(pool: DigitPool, settings: GeneratorSettings) -> None:
    """Without a fixed condition, only training conditions and configured k values come out."""
    specs = [sample_spec(pool, settings, "train", i, 0) for i in range(40)]
    assert {s.condition for s in specs} <= set(settings.train_mix)
    assert {s.k_target for s in specs if s.condition != "control"} <= set(settings.k_weights)
    assert len({s.condition for s in specs}) >= 2


def test_fixed_cell_is_respected(pool: DigitPool, settings: GeneratorSettings) -> None:
    """Fixing condition, k, and speed gives exactly that cell."""
    spec = sample_spec(pool, settings, "test", 0, 0, condition="occlusion", k=8, speed=3)
    assert (spec.k_target, spec.speed) == (8, 3)


def test_unknown_condition_raises(pool: DigitPool, settings: GeneratorSettings) -> None:
    """A misspelled condition is rejected."""
    with pytest.raises(ValueError):
        sample_spec(pool, settings, "test", 0, 0, condition="occlusions", k=4, speed=2)
