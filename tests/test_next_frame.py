"""Tests for the next frame evaluation: SSIM, baselines, visible moving frames, the per frame table, and gate D16."""

import numpy as np
import pytest
import torch

from peekaboo.config import load_config
from peekaboo.data.conditions import GeneratorSettings, sample_spec
from peekaboo.data.dataset import OnTheFlyDataset
from peekaboo.data.mnist_pool import DigitPool
from peekaboo.data.occluder import STATE_OCCLUDED, STATE_PARTIAL, STATE_VISIBLE
from peekaboo.data.render import RenderSettings, compose_observed, render_sequence
from peekaboo.eval.next_frame import (blank_prediction, copy_prediction, evaluate, gate, ssim, summarize,
                                      visible_moving)
from peekaboo.paths import CONFIGS_DIR

RENDER = RenderSettings.from_config(load_config(CONFIGS_DIR / "data" / "base.yaml"))
CPU = torch.device("cpu")


class Stub(torch.nn.Module):
    """A stub model: "copy" copies the last frame, "blank" erases the digit, and "perfect" cheats with the frame."""

    def __init__(self, mode: str) -> None:
        """Remember the mode."""
        super().__init__()
        self.mode = mode

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict the frames according to the mode."""
        if self.mode == "copy":
            return {"prediction": copy_prediction(frames)}
        if self.mode == "blank":
            return {"prediction": blank_prediction(frames)}
        return {"prediction": frames.clone()}


def stream(pool: DigitPool, settings: GeneratorSettings) -> OnTheFlyDataset:
    """Six sequences of the training mix from the synthetic pool."""
    return OnTheFlyDataset(pool, settings, RENDER, "val", 0, length=6)


def test_ssim_of_identical_images_is_one() -> None:
    """SSIM is 1 for identical images, symmetric, and lower for noisier copies."""
    torch.manual_seed(0)
    x = torch.rand(4, 3, 32, 48)
    assert torch.allclose(ssim(x, x), torch.ones(4), atol=1e-5)
    y = (x + 0.1 * torch.randn_like(x)).clamp(0, 1)
    z = (x + 0.3 * torch.randn_like(x)).clamp(0, 1)
    assert torch.allclose(ssim(x, y), ssim(y, x), atol=1e-6)
    assert (ssim(x, z) < ssim(x, y)).all() and (ssim(x, y) < 1).all()


def test_ssim_of_constant_images_is_the_luminance_term() -> None:
    """For two flat images a and b, SSIM reduces to (2ab + c1) / (a^2 + b^2 + c1)."""
    a, b, c1 = 0.2, 0.6, 0.01 ** 2
    x, y = torch.full((1, 3, 16, 16), a), torch.full((1, 3, 16, 16), b)
    assert ssim(x, y).item() == pytest.approx((2 * a * b + c1) / (a * a + b * b + c1), rel=1e-5)


def test_visible_moving_frames() -> None:
    """A frame counts from frame 2 on, when the digit is fully visible in it and in the frame before."""
    v, p, o = STATE_VISIBLE, STATE_PARTIAL, STATE_OCCLUDED
    state = torch.tensor([[v, v, v, v, p, o, p, v, v]])
    expected = [False, False, True, True, False, False, False, False, True]
    assert visible_moving(state)[0].tolist() == expected


def test_blank_frame_is_the_scene_without_the_digit(pool: DigitPool, settings: GeneratorSettings) -> None:
    """Erasing the red digit gives exactly the frames rendered without any digit, blackouts included."""
    for condition in ("occlusion", "blackout"):
        spec = sample_spec(pool, settings, "test", 0, 0, condition=condition, k=6, speed=2)
        r = render_sequence(spec, pool.sprite(spec.digit_index), RENDER, settings.crossing.thresholds)
        frames = torch.from_numpy(r.observed).permute(0, 3, 1, 2)[None].float() / 255.0
        empty = compose_observed(np.zeros_like(r.amodal), spec, RENDER)
        expected = torch.from_numpy(empty).permute(0, 3, 1, 2)[None].float() / 255.0
        assert torch.equal(blank_prediction(frames), expected), condition


@pytest.mark.parametrize("mode", ["copy", "blank"])
def test_baseline_models_score_exactly_their_baseline(mode: str, pool: DigitPool,
                                                      settings: GeneratorSettings) -> None:
    """A model that copies the last frame, or never draws the digit, gets that baseline's scores and fails the gate."""
    table = evaluate(Stub(mode), stream(pool, settings), CPU, batch_size=4, seq_len=12)
    assert len(table) == 6 * 11 and table["t"].min() == 1
    for score in ("mse", "mae", "ssim"):
        assert (table[f"model_{score}"] == table[f"{mode}_{score}"]).all(), score
    visible = table["visible_moving"]
    assert visible.any() and (table["copy_mse"][visible] > 0).all() and (table["blank_mse"][visible] > 0).all()
    result = gate(table)
    assert result[f"reduction_vs_{mode}"] == pytest.approx(0.0) and result["mse_reduction"] <= 1e-9
    assert not result["passed"]


def test_perfect_model_passes_the_gate(pool: DigitPool, settings: GeneratorSettings) -> None:
    """A model that returns the true frame has zero error, SSIM 1, and a reduction of 100%."""
    table = evaluate(Stub("perfect"), stream(pool, settings), CPU, batch_size=4, seq_len=12)
    assert (table["model_mse"] == 0).all() and table["model_ssim"].to_numpy() == pytest.approx(1.0, abs=1e-5)
    result = gate(table)
    assert result["mse_reduction"] == pytest.approx(1.0) and result["passed"]
    assert result["reduction_vs_copy"] == pytest.approx(1.0) and result["reduction_vs_blank"] == pytest.approx(1.0)
    by_speed = summarize(table[table["visible_moving"]], "speed")
    assert by_speed["frames"].sum() == result["frames"]
    assert (by_speed["reduction_vs_blank"] == 1.0).all()
    assert set(table["condition"]) <= {"control", "occlusion", "hidden_bounce"}
