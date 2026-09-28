"""Tests for the pixel renderer and the exact ground truth."""

import numpy as np
import pytest

from peekaboo.config import load_config
from peekaboo.data.conditions import CONDITIONS, GeneratorSettings, sample_spec
from peekaboo.data.mnist_pool import DigitPool
from peekaboo.data.occluder import STATE_OCCLUDED
from peekaboo.data.render import (RenderedSequence, RenderSettings, bar_columns, compose_observed,
                                  measure_from_pixels, render_sequence)
from peekaboo.data.spec import SequenceSpec
from peekaboo.data.splicing import sample_surprise_tuple
from peekaboo.data.truth import (EPISODE_MAIN, EPISODE_OTHER, STATE_ABSENT, STATE_BLACKOUT, runs,
                                 sequence_summary)
from peekaboo.paths import CONFIGS_DIR

RENDER = RenderSettings.from_config(load_config(CONFIGS_DIR / "data" / "base.yaml"))
CASES = [("condition", c, i) for c in CONDITIONS for i in range(3)] + \
        [("surprise", kind, role) for kind in ("direction", "vanish") for role in range(4)]


def make_case(case: tuple, pool: DigitPool, settings: GeneratorSettings) -> SequenceSpec:
    """Build the spec of one test case: a condition sequence, or one role of a surprise tuple."""
    family, name, number = case
    if family == "condition":
        return sample_spec(pool, settings, "test", number, 0, condition=name)
    return sample_surprise_tuple(pool, settings, "test", 0, 0, name, 6, 2).specs()[number]


def render(spec: SequenceSpec, pool: DigitPool, settings: GeneratorSettings) -> RenderedSequence:
    """Render a spec with the base colors."""
    return render_sequence(spec, pool.sprite(spec.digit_index), RENDER, settings.crossing.thresholds)


def test_render_settings_from_config() -> None:
    """The base config draws a red digit and a mid gray bar."""
    assert RENDER.digit_rgb == (1.0, 0.0, 0.0) and RENDER.bar_value == 128


@pytest.mark.parametrize("case", CASES, ids=str)
def test_pixels_match_exact_truth(case: tuple, pool: DigitPool, settings: GeneratorSettings) -> None:
    """Visible fraction, amodal center, and modal center measured on pixels equal the exact ground truth."""
    r = render(make_case(case, pool, settings), pool, settings)
    measured = measure_from_pixels(r)
    np.testing.assert_allclose(measured["visible_fraction"], r.truth.visible_fraction, atol=1e-12)
    np.testing.assert_allclose(measured["center"], r.truth.center, atol=1e-9)  # NaN where absent, in both
    np.testing.assert_allclose(measured["modal_center"], r.truth.modal_center, atol=1e-9)


@pytest.mark.parametrize("case", CASES, ids=str)
def test_observed_frames(case: tuple, pool: DigitPool, settings: GeneratorSettings) -> None:
    """Observed frames: red digit, gray bar on top, black blackouts; ink is conserved in the amodal frames."""
    spec = make_case(case, pool, settings)
    r = render(spec, pool, settings)
    sprite = pool.sprite(spec.digit_index)
    under_bar = bar_columns(spec)
    shown = np.ones(spec.seq_len, dtype=bool)
    shown[list(spec.blackout_frames)] = False

    assert (r.observed[~shown] == 0).all()
    obs = r.observed[shown]
    assert (obs[:, :, under_bar, :] == RENDER.bar_value).all()
    assert np.array_equal(obs[:, :, ~under_bar, 0], r.amodal[shown][:, :, ~under_bar])
    assert (obs[:, :, ~under_bar, 1:] == 0).all()
    assert np.array_equal(compose_observed(r.amodal, spec, RENDER), r.observed)

    sums = r.amodal.reshape(spec.seq_len, -1).sum(axis=1)
    assert (sums[spec.digit_present] == int(sprite.sum())).all()
    assert (sums[~spec.digit_present] == 0).all()
    assert not r.modal_mask[~shown].any() and not r.modal_mask[:, :, under_bar].any()


@pytest.mark.parametrize("case", CASES, ids=str)
def test_states_and_episodes(case: tuple, pool: DigitPool, settings: GeneratorSettings) -> None:
    """States mark blackouts and absent frames, the main episode covers the event, no other contact is in the window."""
    spec = make_case(case, pool, settings)
    truth = render(spec, pool, settings).truth
    summary = sequence_summary(truth)
    assert (truth.state[list(spec.blackout_frames)] == STATE_BLACKOUT).all()
    assert ((truth.state == STATE_ABSENT) == ~spec.digit_present).all()
    assert summary["other_frames_in_window"] == 0
    assert truth.in_window.sum() == (spec.window_end - spec.window_start + 1)
    if spec.condition in ("occlusion", "hidden_bounce"):
        assert summary["measured_k"] == spec.k_target
    if spec.entry_frame >= 0 and spec.exit_frame >= 0:
        assert (truth.episode[spec.entry_frame:spec.exit_frame + 1] == EPISODE_MAIN).all()
    if spec.condition == "empty":
        assert (truth.episode == 0).all()
    # every other episode lies outside the window
    for first, last in runs(truth.episode == EPISODE_OTHER):
        assert last < spec.window_start or first > spec.window_end


def test_occluded_frames_are_exactly_k(pool: DigitPool, settings: GeneratorSettings) -> None:
    """In an occlusion sequence, the main event has exactly k frames in the occluded state, all consecutive."""
    spec = sample_spec(pool, settings, "test", 11, 0, condition="occlusion", k=6, speed=3)
    truth = render(spec, pool, settings).truth
    hidden = (truth.state == STATE_OCCLUDED) & (truth.episode == EPISODE_MAIN)
    assert runs(hidden) == [(spec.onset_frame, spec.onset_frame + 5)]
