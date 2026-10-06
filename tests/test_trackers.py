"""Tests for the programmed baselines: the detector, the wall bounds, the Kalman filters, and the last seen position."""

import numpy as np
import pytest

from peekaboo.config import load_config
from peekaboo.data.conditions import GeneratorSettings, sample_spec
from peekaboo.data.mnist_pool import DigitPool
from peekaboo.data.render import RenderSettings, RenderedSequence, render_sequence
from peekaboo.data.trajectory import Bounds
from peekaboo.data.truth import STATE_NAMES
from peekaboo.paths import CONFIGS_DIR
from peekaboo.trackers.baselines import last_seen, oracle_measurements, run_trackers
from peekaboo.trackers.detector import AT_BAR, DETECTED, NO_DIGIT, detect
from peekaboo.trackers.kalman import KalmanSettings, kalman_track, reflect, wall_bounds

RENDER = RenderSettings.from_config(load_config(CONFIGS_DIR / "data" / "base.yaml"))
TOLERANCE = 0.05  # px


def rendered(pool: DigitPool, settings: GeneratorSettings, condition: str, index: int,
             k: int = 4, speed: int = 3) -> tuple[RenderedSequence, np.ndarray]:
    """One rendered sequence and its observed frames (T, 3, H, W), as in the dataset samples."""
    spec = sample_spec(pool, settings, "test", index, 0, condition=condition, k=k, speed=speed)
    r = render_sequence(spec, pool.sprite(spec.digit_index), RENDER, settings.crossing.thresholds)
    return r, r.observed.transpose(0, 3, 1, 2)


def error(positions: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """(T,) distance in pixels between estimated and true centroids."""
    return np.linalg.norm(positions - truth, axis=1)


@pytest.mark.parametrize("condition", ["control", "occlusion", "hidden_bounce", "blackout"])
def test_detector_is_exact_and_never_reports_a_partial_view(condition: str, pool: DigitPool,
                                                            settings: GeneratorSettings) -> None:
    """A detected digit is fully visible and its centroid is exact; a partly hidden digit is never detected; a digit
    with no visible ink is reported as absent. Most fully visible frames are detected."""
    full, detected = 0, 0
    for index in range(10):
        r, frames = rendered(pool, settings, condition, index)
        found = detect(frames)
        fraction = r.truth.visible_fraction
        hit = found.status == DETECTED
        assert np.all(fraction[hit] == 1.0)
        assert np.allclose(found.center[hit], r.truth.center[hit], atol=1e-9)
        assert np.all(found.status[(fraction > 0) & (fraction < 1)] == AT_BAR)
        assert np.all(found.status[fraction == 0] == NO_DIGIT)
        full, detected = full + int((fraction == 1).sum()), detected + int(hit.sum())
    assert detected > 0.8 * full


def test_wall_bounds_are_the_generator_range(pool: DigitPool, settings: GeneratorSettings) -> None:
    """The centroid range built from a detected ink extent is the generator's range of the sprite corner, shifted by
    the centroid's offset in the sprite."""
    r, frames = rendered(pool, settings, "occlusion", 0)
    found = detect(frames)
    t = int(np.flatnonzero(found.status == DETECTED)[0])
    spec = r.spec
    height, width = pool.sprite(spec.digit_index).shape
    corner = Bounds.for_sprite(spec.frame_height, spec.frame_width, height, width)
    offset = r.truth.center[t] - spec.positions[t]
    expected = np.array([[corner.y_min, corner.y_max], [corner.x_min, corner.x_max]]) + offset[:, None]
    assert np.allclose(wall_bounds(found.extent[t], (spec.frame_height, spec.frame_width)), expected)


def test_kalman_crosses_an_occlusion_at_constant_velocity(pool: DigitPool, settings: GeneratorSettings) -> None:
    """When the digit keeps its velocity from the first frame to its reappearance, the constant velocity filter
    predicts it exactly while it is hidden and when it comes back."""
    checked = 0
    for index in range(20):
        r, frames = rendered(pool, settings, "occlusion", index, k=8)
        spec = r.spec
        end = spec.actual_reappear_frame
        if not np.all(spec.velocities[:end] == spec.velocities[0]):
            continue  # a wall bounce, seen or hidden, breaks constant velocity
        track = kalman_track(detect(frames).center)
        hidden = np.arange(spec.entry_frame, end + 1)
        assert error(track.prediction[hidden], r.truth.center[hidden]).max() < TOLERANCE
        checked += 1
    assert checked >= 3


def test_hidden_bounce_needs_the_walls(pool: DigitPool, settings: GeneratorSettings) -> None:
    """After a bounce while hidden, the constant velocity filter expects the digit on the wrong side, while the filter
    with walls and the oracle find it exactly."""
    for index in range(5):
        r, frames = rendered(pool, settings, "hidden_bounce", index, k=8)
        spec = r.spec
        tracks = run_trackers(frames, r.truth.center, r.truth.state)
        back = spec.actual_reappear_frame
        hidden = np.arange(spec.onset_frame, back + 1)
        assert error(tracks["kalman"].prediction[[back]], r.truth.center[[back]])[0] > 2.0
        for name in ("kalman_walls", "oracle"):
            assert error(tracks[name].prediction[hidden], r.truth.center[hidden]).max() < TOLERANCE, name


def test_walls_follow_every_occlusion_and_blackout(pool: DigitPool, settings: GeneratorSettings) -> None:
    """With walls, the filter is exact from the entry to the end of the analysis window, through occlusions and
    blackouts, with or without bounces."""
    for condition in ("occlusion", "blackout"):
        for index in range(8):
            r, frames = rendered(pool, settings, condition, index, k=4)
            spec = r.spec
            tracks = run_trackers(frames, r.truth.center, r.truth.state)
            window = np.arange(spec.entry_frame, spec.window_end + 1)
            assert error(tracks["kalman_walls"].prediction[window], r.truth.center[window]).max() < TOLERANCE


def test_reflection_mirrors_position_velocity_and_covariance() -> None:
    """A position 5 px past the top wall comes back 5 px inside it, with the velocity reversed on that axis; a position
    inside the walls is unchanged."""
    state = np.array([5.0, 32.0, 3.0, 1.0])
    covariance = np.arange(16, dtype=float).reshape(4, 4)
    covariance = covariance @ covariance.T
    bounds = np.array([[10.0, 50.0], [8.0, 80.0]])
    mirrored, mirrored_covariance = reflect(state, covariance, bounds)
    assert np.allclose(mirrored, [15.0, 32.0, -3.0, 1.0])
    assert np.allclose(np.diag(mirrored_covariance), np.diag(covariance))
    assert np.allclose(reflect(np.array([20.0, 30.0, 1.0, 1.0]), covariance, bounds)[0], [20.0, 30.0, 1.0, 1.0])


def test_last_seen_and_oracle_inputs() -> None:
    """The last seen position holds through missing frames and comes one frame later as a prediction; the oracle sees
    the truth on fully visible frames only."""
    z = np.array([[1.0, 2.0], [3.0, 4.0], [np.nan, np.nan], [np.nan, np.nan], [9.0, 9.0]])
    track = last_seen(z)
    assert np.allclose(track.estimate, [[1, 2], [3, 4], [3, 4], [3, 4], [9, 9]])
    assert np.isnan(track.prediction[0]).all() and np.allclose(track.prediction[1:], track.estimate[:-1])
    state = np.array([STATE_NAMES.index(s) for s in ("visible", "partial", "occluded", "visible", "blackout")])
    center = np.arange(10, dtype=float).reshape(5, 2)
    oracle = oracle_measurements(center, state)
    assert np.allclose(oracle[[0, 3]], center[[0, 3]]) and np.isnan(oracle[[1, 2, 4]]).all()


def test_walls_need_bounds() -> None:
    """The walls option without bounds is an error."""
    with pytest.raises(ValueError):
        kalman_track(np.zeros((3, 2)), KalmanSettings(walls=True))
