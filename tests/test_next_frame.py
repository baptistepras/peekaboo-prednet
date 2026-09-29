"""Tests for the next frame evaluation: SSIM, visible moving frames, the per frame table, and the gate D16."""

import pytest
import torch

from peekaboo.config import load_config
from peekaboo.data.conditions import GeneratorSettings
from peekaboo.data.dataset import OnTheFlyDataset
from peekaboo.data.mnist_pool import DigitPool
from peekaboo.data.occluder import STATE_OCCLUDED, STATE_PARTIAL, STATE_VISIBLE
from peekaboo.data.render import RenderSettings
from peekaboo.eval.next_frame import evaluate, gate, ssim, summarize, visible_moving
from peekaboo.paths import CONFIGS_DIR

RENDER = RenderSettings.from_config(load_config(CONFIGS_DIR / "data" / "base.yaml"))
CPU = torch.device("cpu")


class Shift(torch.nn.Module):
    """A stub model: with lag 1 it copies the last frame, with lag 0 it cheats and returns the frame itself."""

    def __init__(self, lag: int) -> None:
        """Remember the lag."""
        super().__init__()
        self.lag = lag

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        """Shift the frames by `lag` steps, with zeros at the start."""
        if self.lag == 0:
            return {"prediction": frames.clone()}
        return {"prediction": torch.cat([torch.zeros_like(frames[:, :1]), frames[:, :-1]], dim=1)}


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


def test_copy_model_scores_exactly_the_baseline(pool: DigitPool, settings: GeneratorSettings) -> None:
    """A model that copies the last frame gets the copy's scores on every frame and fails the gate."""
    table = evaluate(Shift(1), stream(pool, settings), CPU, batch_size=4, seq_len=12)
    assert len(table) == 6 * 11 and table["t"].min() == 1
    for score in ("mse", "mae", "ssim"):
        assert (table[f"model_{score}"] == table[f"copy_{score}"]).all(), score
    assert table["visible_moving"].any() and (table["copy_mse"][table["visible_moving"]] > 0).all()
    result = gate(table)
    assert result["mse_reduction"] == pytest.approx(0.0) and not result["passed"]


def test_perfect_model_passes_the_gate(pool: DigitPool, settings: GeneratorSettings) -> None:
    """A model that returns the true frame has zero error, SSIM 1, and a reduction of 100%."""
    table = evaluate(Shift(0), stream(pool, settings), CPU, batch_size=4, seq_len=12)
    assert (table["model_mse"] == 0).all() and table["model_ssim"].to_numpy() == pytest.approx(1.0, abs=1e-5)
    result = gate(table)
    assert result["mse_reduction"] == pytest.approx(1.0) and result["passed"]
    by_speed = summarize(table[table["visible_moving"]], "speed")
    assert by_speed["frames"].sum() == result["frames"]
    assert set(table["condition"]) <= {"control", "occlusion", "hidden_bounce"}
