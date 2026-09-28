"""Tests for PLATO style surprise tuples."""

import numpy as np
import pytest

from peekaboo.data.conditions import GeneratorSettings
from peekaboo.data.mnist_pool import DigitPool
from peekaboo.data.occluder import column_ink, visible_fraction
from peekaboo.data.spec import SequenceSpec
from peekaboo.data.splicing import SURPRISES, make_surprise_tuple, sample_surprise_tuple, surprise_speeds

# B keeps a training speed: speed_fast runs from 2 px/frame (to 4), speed_slow from 4 (to 2), the others at 2 and 3
SPEED_CELLS = {"speed_fast": [(6, 2), (4, 2)], "speed_slow": [(4, 4), (6, 4)]}
CELLS = [(kind, k, speed) for kind in SURPRISES for k, speed in SPEED_CELLS.get(kind, [(6, 2), (4, 3)])]


def bar_fractions(spec: SequenceSpec, pool: DigitPool) -> np.ndarray:
    """Visible fraction per frame due to the bar alone."""
    return visible_fraction(column_ink(pool.sprite(spec.digit_index)), spec.positions[:, 1], spec.bar_left,
                            spec.bar_width)


@pytest.mark.parametrize("kind,k,speed", CELLS)
def test_tuple_structure(kind: str, k: int, speed: int, pool: DigitPool, settings: GeneratorSettings) -> None:
    """AB and BA are exact splices of A and B, cut at a frame where the digit is hidden in both."""
    t = sample_surprise_tuple(pool, settings, "test", 0, 0, kind, k, speed)
    a, b, ab, ba = t.specs()
    ts = t.splice_frame
    occ_max = settings.crossing.thresholds.occluded_max

    for spec, role in zip(t.specs(), ("A", "B", "AB", "BA")):
        assert spec.tuple_role == role and spec.tuple_id == a.tuple_id and spec.surprise_frame == ts
        assert (spec.digit_index, spec.bar_left, spec.bar_width) == (a.digit_index, a.bar_left, a.bar_width)
        assert 0 <= spec.window_start <= spec.window_end <= spec.seq_len - 1

    # the change happens strictly while hidden
    assert a.onset_frame < ts < a.expected_reappear_frame
    fa, fab = bar_fractions(a, pool), bar_fractions(ab, pool)
    assert fa[ts - 1] <= occ_max and fa[ts] <= occ_max
    assert fab[ts - 1] <= occ_max and fab[ts] <= occ_max
    assert np.array_equal(ab.positions[:ts], a.positions[:ts])
    assert np.array_equal(ba.positions[ts:], a.positions[ts:])

    if kind == "vanish":
        assert not b.digit_present.any()
        assert ab.digit_present[:ts].all() and not ab.digit_present[ts:].any()
        assert not ba.digit_present[:ts].any() and ba.digit_present[ts:].all()
        assert ab.actual_reappear_frame == -1
        return

    fb = bar_fractions(b, pool)
    assert fb[ts - 1] <= occ_max and fb[ts] <= occ_max
    assert np.array_equal(ab.positions[ts:], b.positions[ts:])
    assert np.array_equal(ba.positions[:ts], b.positions[:ts])
    assert ab.surprise_type == kind

    center = a.bar_left + a.bar_width / 2
    side_in = np.sign(a.positions[a.entry_frame, 1] + a.ink_width / 2 - center)
    side_out = np.sign(ab.positions[ab.actual_reappear_frame, 1] + ab.ink_width / 2 - center)
    if kind == "direction":
        assert side_out == side_in  # comes back out where it went in
    else:
        assert side_out == -side_in
    if kind in ("speed_fast", "early"):
        assert ab.actual_reappear_frame < ab.expected_reappear_frame
    if kind == "speed_slow":
        assert ab.actual_reappear_frame > ab.expected_reappear_frame
    if kind == "offset":
        assert abs(ab.positions[ts, 0] - a.positions[ts, 0]) == settings.offset_px


@pytest.mark.parametrize("kind,speed", [("speed_fast", 2), ("early", 2), ("early", 3), ("speed_slow", 4)])
@pytest.mark.parametrize("index", range(8))
def test_timing_surprises_move_reappearance(kind: str, speed: int, index: int, pool: DigitPool,
                                            settings: GeneratorSettings) -> None:
    """Timing surprises always change the reappearance frame, even when the splice falls late in the occlusion."""
    ab = sample_surprise_tuple(pool, settings, "test", index, 0, kind, 6, speed).impossible_ab
    if kind == "speed_slow":
        assert ab.actual_reappear_frame > ab.expected_reappear_frame
    else:
        assert ab.actual_reappear_frame < ab.expected_reappear_frame


def test_tuples_are_deterministic(pool: DigitPool, settings: GeneratorSettings) -> None:
    """The same arguments give the same tuple."""
    t1 = sample_surprise_tuple(pool, settings, "test", 3, 0, "direction", 6, 2)
    t2 = sample_surprise_tuple(pool, settings, "test", 3, 0, "direction", 6, 2)
    assert all(x == y for x, y in zip(t1.specs(), t2.specs()))


def test_surprise_speeds_keep_b_in_training_range(settings: GeneratorSettings) -> None:
    """With training speeds {2, 3, 4}, speed_fast starts from 2 and speed_slow from 4; the others use every speed."""
    assert surprise_speeds("speed_fast", settings) == [2]
    assert surprise_speeds("speed_slow", settings) == [4]
    assert surprise_speeds("direction", settings) == [2, 3, 4]


def test_invalid_surprise_requests(pool: DigitPool, settings: GeneratorSettings) -> None:
    """Speeds that would take B out of the training range, a too short "early", and unknown kinds are rejected."""
    rng = np.random.default_rng(0)
    identity = {"split": "test", "index": 0, "seed": 0, "frame_height": 64, "frame_width": 96}
    for kind, speed in (("speed_slow", 3), ("speed_slow", 2), ("speed_fast", 3), ("speed_fast", 4)):
        with pytest.raises(ValueError):
            make_surprise_tuple(rng, pool, 0, kind, 4, speed, settings, identity, 0)
    with pytest.raises(ValueError):
        make_surprise_tuple(rng, pool, 0, "early", 2, 2, settings, identity, 0)
    with pytest.raises(ValueError):
        make_surprise_tuple(rng, pool, 0, "teleport", 4, 2, settings, identity, 0)
