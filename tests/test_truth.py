"""Tests for the exact ground truth on hand made cases."""

import numpy as np

from peekaboo.data.occluder import STATE_OCCLUDED, VisibilityThresholds
from peekaboo.data.spec import SequenceSpec
from peekaboo.data.truth import EPISODE_MAIN, STATE_ABSENT, compute_truth, sequence_summary


def appearing_thin_digit() -> tuple[SequenceSpec, np.ndarray]:
    """A 2 px wide digit that appears from nowhere behind a bar, then jumps from hidden to fully visible.

    The bar covers columns 40 to 49. The digit is absent before frame 3, hidden at frames 3 and 4, and fully
    visible from frame 5 on, so its reappearance frame never touches the bar.
    """
    seq_len = 10
    x = 30 + 4 * np.arange(seq_len)  # 30, 34, 38, 42, 46, 50, ...
    positions = np.stack([np.full(seq_len, 10), x], axis=1)
    velocities = np.tile([0, 4], (seq_len, 1))
    spec = SequenceSpec(split="test", index=0, seed=0, frame_height=64, frame_width=96, seq_len=seq_len,
                        digit_index=0, mnist_index=0, label=1, ink_height=4, ink_width=2, positions=positions,
                        velocities=velocities, digit_present=np.arange(seq_len) >= 3, bar_left=40, bar_width=10,
                        condition="surprise", surprise_type="appear", k_target=2, speed=4,
                        actual_reappear_frame=5, exit_frame=4, window_start=0, window_end=8, surprise_frame=3,
                        tuple_role="BA")
    return spec, np.full((4, 2), 255, dtype=np.uint8)


def test_appearing_digit_has_one_main_episode() -> None:
    """Without an entry frame, the main event is found from the last hidden frame, not from the reappearance."""
    spec, sprite = appearing_thin_digit()
    truth = compute_truth(spec, sprite, VisibilityThresholds())
    assert truth.state[:3].tolist() == [STATE_ABSENT] * 3
    assert truth.state[3:5].tolist() == [STATE_OCCLUDED] * 2
    assert truth.visible_fraction[5] == 1.0  # fully visible on its reappearance frame
    assert truth.episode[3:5].tolist() == [EPISODE_MAIN] * 2
    assert sequence_summary(truth)["other_frames_in_window"] == 0
