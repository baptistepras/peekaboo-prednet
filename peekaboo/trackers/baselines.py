"""The programmed trackers of the project, run on one sequence: last seen position, Kalman filters, and an oracle.

Every tracker reads only the detections of the observed frames, except the oracle, which receives the true centroid
on every frame where the digit is fully visible (state "visible"). The oracle removes the detector's misses at the
bar, so it shows how well a tracker can do with perfect detection and the true motion model.
"""

import dataclasses

import numpy as np

from peekaboo.data.truth import STATE_NAMES
from peekaboo.trackers.detector import detect
from peekaboo.trackers.kalman import KalmanSettings, Track, kalman_track, wall_bounds

TRACKER_NAMES = ("last_seen", "kalman", "kalman_walls", "oracle")


def last_seen(measurements: np.ndarray) -> Track:
    """The last detected position (T, 2), held while the digit is not detected."""
    estimate = np.full((len(measurements), 2), np.nan)
    last = np.full(2, np.nan)
    for t, z in enumerate(measurements):
        if not np.isnan(z).any():
            last = z
        estimate[t] = last
    prediction = np.vstack([np.full((1, 2), np.nan), estimate[:-1]])
    return Track(prediction=prediction, estimate=estimate)


def oracle_measurements(center: np.ndarray, state: np.ndarray) -> np.ndarray:
    """The true centroid (T, 2) on the frames where the digit is fully visible, NaN elsewhere."""
    measurements = np.full_like(center, np.nan, dtype=np.float64)
    visible = state == STATE_NAMES.index("visible")
    measurements[visible] = center[visible]
    return measurements


def run_trackers(frames: np.ndarray, center: np.ndarray, state: np.ndarray,
                 settings: KalmanSettings = KalmanSettings()) -> dict[str, Track]:
    """Run every tracker on one sequence: observed frames (T, 3, H, W) uint8, and the true centroid (T, 2) and state
    (T,) for the oracle. The walls of the Kalman filters come from the ink extent of the first full detection."""
    detections = detect(frames)
    detected = np.flatnonzero(detections.detected)
    bounds = wall_bounds(detections.extent[detected[0]], frames.shape[2:]) if detected.size else None
    walls = dataclasses.replace(settings, walls=bounds is not None)
    plain = dataclasses.replace(settings, walls=False)
    return {
        "last_seen": last_seen(detections.center),
        "kalman": kalman_track(detections.center, plain),
        "kalman_walls": kalman_track(detections.center, walls, bounds),
        "oracle": kalman_track(oracle_measurements(center, state), walls, bounds),
    }
