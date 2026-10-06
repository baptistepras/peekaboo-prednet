"""Constant velocity Kalman filter on the digit centroid, in numpy, with an optional reflection at the frame walls.

The state is (y, x, vy, vx) in pixels and px/frame. Each frame, the filter first predicts the position from the past
frames, then corrects it with the detection when there is one. While the digit is not detected (hidden, at the bar,
or blacked out), it only predicts, so it carries the digit through the occlusion at its last estimated velocity.

The generator moves the digit at a constant integer velocity and mirrors it off the frame borders. With the walls
option, the filter mirrors its prediction off the borders too, using the digit's ink extent seen in a full detection,
so it also follows a bounce that happens while the digit is hidden (the hidden_bounce condition).
"""

from dataclasses import dataclass

import numpy as np

TRANSITION = np.array([[1.0, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
OBSERVATION = np.array([[1.0, 0, 0, 0], [0, 1, 0, 0]])


@dataclass(frozen=True)
class KalmanSettings:
    """Noise levels of the filter and the walls option."""

    measurement_std: float = 0.1       # px: the detected centroid is exact
    acceleration_std: float = 0.5      # px/frame^2: lets the velocity follow a bounce seen on screen within a few frames
    initial_velocity_std: float = 5.0  # px/frame: prior on the velocity at the first detection, about the top speed
    walls: bool = False                # mirror the prediction off the frame borders


@dataclass(frozen=True)
class Track:
    """Positions of the digit given by a tracker, (T, 2) in pixels and (y, x) order, NaN before the first detection."""

    prediction: np.ndarray  # position at frame t from the frames before t, the counterpart of a model's prediction
    estimate: np.ndarray    # position at frame t once frame t is seen (the prediction when the digit is not detected)


def process_noise(acceleration_std: float) -> np.ndarray:
    """(4, 4) noise of a random acceleration constant over one frame, on each axis."""
    return acceleration_std ** 2 * np.kron(np.array([[0.25, 0.5], [0.5, 1.0]]), np.eye(2))


def wall_bounds(extent: np.ndarray, frame_size: tuple[int, int]) -> np.ndarray:
    """(2, 2) range [low, high] of the centroid on each axis, for a digit whose ink reaches `extent` (up, down, left,
    right) from its centroid: the ink touches a border at either end, as in the generator."""
    height, width = frame_size
    up, down, left, right = extent
    return np.array([[up, height - 1 - down], [left, width - 1 - right]])


def reflect(state: np.ndarray, covariance: np.ndarray, bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mirror the position off the walls and reverse the velocity on that axis, with the covariance."""
    for axis in (0, 1):
        low, high = bounds[axis]
        edge = high if state[axis] > high else low if state[axis] < low else None
        if edge is not None:
            state = state.copy()
            state[axis] = 2 * edge - state[axis]
            state[axis + 2] = -state[axis + 2]
            flip = np.eye(4)
            flip[axis, axis] = flip[axis + 2, axis + 2] = -1.0
            covariance = flip @ covariance @ flip.T
    return state, covariance


def kalman_track(measurements: np.ndarray, settings: KalmanSettings = KalmanSettings(),
                 bounds: np.ndarray | None = None) -> Track:
    """Track the digit from its detected centroids (T, 2), NaN where it is not detected. `bounds` (from wall_bounds)
    is needed by the walls option."""
    if settings.walls and bounds is None:
        raise ValueError("The walls option needs the bounds of the centroid.")
    seq_len = len(measurements)
    prediction, estimate = np.full((seq_len, 2), np.nan), np.full((seq_len, 2), np.nan)
    noise = process_noise(settings.acceleration_std)
    measurement_noise = settings.measurement_std ** 2 * np.eye(2)
    state, covariance = None, None
    for t in range(seq_len):
        if state is not None:
            state = TRANSITION @ state
            covariance = TRANSITION @ covariance @ TRANSITION.T + noise
            if settings.walls:
                state, covariance = reflect(state, covariance, bounds)
            prediction[t] = state[:2]
        z = measurements[t]
        if not np.isnan(z).any():
            if state is None:  # first detection: the position is known, the velocity is not
                state = np.array([z[0], z[1], 0.0, 0.0])
                covariance = np.diag([settings.measurement_std ** 2] * 2 + [settings.initial_velocity_std ** 2] * 2)
            else:
                innovation_covariance = OBSERVATION @ covariance @ OBSERVATION.T + measurement_noise
                gain = covariance @ OBSERVATION.T @ np.linalg.inv(innovation_covariance)
                state = state + gain @ (z - OBSERVATION @ state)
                covariance = (np.eye(4) - gain @ OBSERVATION) @ covariance
        if state is not None:
            estimate[t] = state[:2]
    return Track(prediction=prediction, estimate=estimate)

