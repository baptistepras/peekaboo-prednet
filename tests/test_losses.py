"""Tests for the training losses: the visible digit mask and the digit weighted loss (decision D12)."""

import pytest
import torch

from peekaboo.config import load_config
from peekaboo.data.conditions import GeneratorSettings, sample_spec
from peekaboo.data.mnist_pool import DigitPool
from peekaboo.data.render import RenderSettings, render_sequence
from peekaboo.eval.next_frame import blank_prediction
from peekaboo.models import build_model
from peekaboo.paths import CONFIGS_DIR
from peekaboo.train.losses import training_loss, visible_digit_mask, weighted_pixel_loss

RENDER = RenderSettings.from_config(load_config(CONFIGS_DIR / "data" / "base.yaml"))


def rendered_frames(pool: DigitPool, settings: GeneratorSettings, condition: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Frames (1, T, 3, H, W) in [0, 1] of one sequence, and its visible ink mask (1, T, H, W) from the renderer."""
    spec = sample_spec(pool, settings, "test", 0, 0, condition=condition, k=6, speed=2)
    r = render_sequence(spec, pool.sprite(spec.digit_index), RENDER, settings.crossing.thresholds)
    frames = torch.from_numpy(r.observed).permute(0, 3, 1, 2)[None].float() / 255.0
    return frames, torch.from_numpy(r.modal_mask)[None]


@pytest.mark.parametrize("condition", ["occlusion", "blackout"])
def test_mask_is_exactly_the_visible_ink(condition: str, pool: DigitPool, settings: GeneratorSettings) -> None:
    """The color rule finds the visible ink of the renderer, never the bar, the background, or hidden ink."""
    frames, visible = rendered_frames(pool, settings, condition)
    assert visible.any()
    assert torch.equal(visible_digit_mask(frames), visible)


def test_weight_one_is_prednet_own_loss() -> None:
    """With weight 1, the common loss equals PredNet's loss on its error units, with L0 and with Lall weights."""
    torch.manual_seed(0)
    frames = torch.rand(2, 5, 3, 16, 24)
    for weights in ("L0", "Lall"):
        model = build_model({"model": "prednet", "stack_sizes": [3, 4, 8], "layer_loss_weights": weights})
        out = model(frames)
        assert torch.equal(training_loss(model, out, frames, 1.0), model.loss(out["layer_errors"]))
    l0 = build_model({"model": "prednet", "stack_sizes": [3, 4, 8], "layer_loss_weights": "L0"})
    out = l0(frames)
    assert torch.allclose(weighted_pixel_loss(out["prediction"], frames), l0.loss(out["layer_errors"]), atol=1e-7)


def test_weight_multiplies_the_error_on_the_digit(pool: DigitPool, settings: GeneratorSettings) -> None:
    """A prediction without the digit errs only on the visible digit, so a weight of 10 multiplies its loss by 10."""
    frames, _ = rendered_frames(pool, settings, "occlusion")
    blank = blank_prediction(frames)
    plain = weighted_pixel_loss(blank, frames, 1.0)
    assert plain > 0
    assert weighted_pixel_loss(blank, frames, 10.0) == pytest.approx(10.0 * float(plain), rel=1e-5)
    # an error away from the digit keeps its weight of 1
    shifted = frames.clone()
    shifted[:, :, 1] += 0.1 * (~visible_digit_mask(frames)).float()
    assert weighted_pixel_loss(shifted, frames, 10.0) == pytest.approx(float(weighted_pixel_loss(shifted, frames)))


def test_prednet_weighted_loss_keeps_its_upper_layers(pool: DigitPool, settings: GeneratorSettings) -> None:
    """With Lall weights, only the pixel term changes with the weight, and gradients reach every parameter."""
    torch.manual_seed(0)
    frames, _ = rendered_frames(pool, settings, "occlusion")
    frames = frames[:, :5]
    model = build_model({"model": "prednet", "stack_sizes": [3, 4, 8], "layer_loss_weights": "Lall"})
    out = model(frames)
    change = training_loss(model, out, frames, 10.0) - training_loss(model, out, frames, 1.0)
    pixel_change = weighted_pixel_loss(out["prediction"], frames, 10.0) - weighted_pixel_loss(out["prediction"], frames)
    assert torch.allclose(change, pixel_change, atol=1e-7)
    training_loss(model, out, frames, 10.0).backward()
    assert all(p.grad is not None for p in model.parameters())
