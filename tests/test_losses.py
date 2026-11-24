"""Tests for the training losses: the visible digit mask, the digit weighted loss (decision D12), and the ways of
scoring blackout frames (step 4.1)."""

import pytest
import torch

from peekaboo.config import load_config
from peekaboo.data.conditions import GeneratorSettings, sample_spec
from peekaboo.data.dataset import OnTheFlyDataset, prepare_batch
from peekaboo.data.mnist_pool import DigitPool
from peekaboo.data.render import RenderSettings, render_sequence
from peekaboo.eval.next_frame import blank_prediction
from peekaboo.models import build_model
from peekaboo.paths import CONFIGS_DIR
from peekaboo.train.losses import (blackout_frames, loss_targets, training_loss, visible_digit_mask,
                                   weighted_pixel_loss)

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


def blackout_batch(pool: DigitPool, settings: GeneratorSettings) -> dict[str, torch.Tensor]:
    """A batch of two blackout sequences, k = 4, with their frames drawn without the blackout, in [0, 1]."""
    data = OnTheFlyDataset(pool, settings, RENDER, "test", 0, condition="blackout", k=4, include_unblacked=True)
    batch = {name: torch.stack([data[i][name] for i in range(2)]) for name in ("frames", "unblacked")}
    return prepare_batch(batch, torch.device("cpu"))


def test_blackout_frames_are_the_black_frames(pool: DigitPool, settings: GeneratorSettings) -> None:
    """The blackout frames found from the pixels are exactly those of the spec, and no frame of an occlusion."""
    data = OnTheFlyDataset(pool, settings, RENDER, "test", 0, condition="blackout", k=4)
    frames = data[0]["frames"][None].float()
    expected = torch.zeros(frames.shape[1], dtype=torch.bool)
    expected[list(data.spec(0).blackout_frames)] = True
    assert expected.sum() == 4 and torch.equal(blackout_frames(frames)[0], expected)
    occlusion, _ = rendered_frames(pool, settings, "occlusion")
    assert not blackout_frames(occlusion).any()


def test_masked_loss_leaves_the_blackout_frames_out(pool: DigitPool, settings: GeneratorSettings) -> None:
    """The masked loss does not depend on the predictions of the blackout frames, is the mean over the other frames,
    and equals the plain loss when no frame is left out."""
    batch = blackout_batch(pool, settings)
    frames = batch["frames"]
    target, keep = loss_targets(batch, frames.shape[1], "masked")
    assert target is None and torch.equal(keep, ~blackout_frames(frames))
    torch.manual_seed(0)
    prediction = torch.rand_like(frames)
    changed = prediction.clone()
    changed[~keep] = 1.0
    masked = weighted_pixel_loss(prediction, frames, 10.0, keep)
    assert torch.equal(masked, weighted_pixel_loss(changed, frames, 10.0, keep))
    assert not torch.isclose(weighted_pixel_loss(prediction, frames, 10.0),
                             weighted_pixel_loss(changed, frames, 10.0))
    one = weighted_pixel_loss(prediction[:1], frames[:1], 10.0, keep[:1])
    kept = keep[0, 1:]
    assert torch.allclose(one, weighted_pixel_loss(torch.cat([prediction[:1, :1], prediction[:1, 1:][:, kept]], 1),
                                                   torch.cat([frames[:1, :1], frames[:1, 1:][:, kept]], 1), 10.0))
    every = torch.ones_like(keep)
    assert torch.allclose(weighted_pixel_loss(prediction, frames, 10.0, every),
                          weighted_pixel_loss(prediction, frames, 10.0))


def test_blackout_losses_of_prednet(pool: DigitPool, settings: GeneratorSettings) -> None:
    """PredNet trains on the unblacked frames or on the kept frames when asked, even with weight 1, and on its own
    loss otherwise; gradients do not reach the predictions of masked frames."""
    batch = blackout_batch(pool, settings)
    frames = batch["frames"]
    model = build_model({"model": "prednet", "stack_sizes": [3, 4, 8], "layer_loss_weights": "L0"})
    out = model(frames)
    target, keep = loss_targets(batch, frames.shape[1], "unblacked")
    assert keep is None and torch.equal(target, batch["unblacked"])
    assert torch.allclose(training_loss(model, out, frames, 1.0, target=target),
                          weighted_pixel_loss(out["prediction"], batch["unblacked"]), atol=1e-7)
    assert not torch.isclose(training_loss(model, out, frames, 1.0, target=target), training_loss(model, out, frames))
    _, keep = loss_targets(batch, frames.shape[1], "masked")
    prediction = out["prediction"].detach().requires_grad_(True)
    training_loss(model, {**out, "prediction": prediction}, frames, 10.0, keep=keep).backward()
    assert prediction.grad[~keep].abs().sum() == 0 and prediction.grad[:, 1:][keep[:, 1:]].abs().sum() > 0
    assert loss_targets(batch, frames.shape[1]) == (None, None)
    with pytest.raises(ValueError):
        loss_targets(batch, frames.shape[1], "black")
