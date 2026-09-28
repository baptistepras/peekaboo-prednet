"""Tests for the PredNet implementation: shapes, parameter count, reference semantics, loss, and devices."""

import pytest
import torch

from peekaboo.config import load_config
from peekaboo.device import get_device
from peekaboo.models.prednet import PredNet, PredNetConfig, count_parameters, hard_sigmoid
from peekaboo.paths import CONFIGS_DIR

SMALL = PredNetConfig(stack_sizes=(3, 4, 8), layer_loss_weights=(1.0, 0.1, 0.1))


def expected_parameters(stack: tuple[int, ...], k: int = 3) -> int:
    """Hand count of PredNet parameters (R channels equal A channels), from the equations of the paper."""
    n, total = len(stack), 0
    for l in range(n):
        gate_in = 2 * stack[l] + stack[l] + (stack[l + 1] if l < n - 1 else 0)
        total += 4 * (gate_in * stack[l] * k * k + stack[l])       # LSTM gates
        total += stack[l] * stack[l] * k * k + stack[l]             # Ahat
        if l < n - 1:
            total += 2 * stack[l] * stack[l + 1] * k * k + stack[l + 1]  # A of the next layer
    return total


def test_hard_sigmoid_is_keras_2() -> None:
    """Slope 0.2 around 0.5, clipped to [0, 1]."""
    x = torch.tensor([-10.0, -2.5, 0.0, 1.0, 2.5, 10.0])
    assert torch.allclose(hard_sigmoid(x), torch.tensor([0.0, 0.0, 0.5, 0.7, 1.0, 1.0]))


def test_parameter_counts_of_the_project_models() -> None:
    """The 5 and 4 layer configs match the hand count, about 3.1 million parameters each."""
    for name, expected in (("prednet_5l", 3_131_628), ("prednet_4l", 3_076_524)):
        config = PredNetConfig.from_config(load_config(CONFIGS_DIR / "models" / f"{name}.yaml"))
        model = PredNet(config)
        assert count_parameters(model) == expected_parameters(config.stack_sizes) == expected


def test_shapes_and_first_prediction() -> None:
    """Outputs have the right shapes, and the first prediction is uniform since R and E start at zero."""
    model = PredNet(SMALL)
    frames = torch.rand(2, 5, 3, 16, 24)
    out = model(frames, return_states=True)
    assert out["prediction"].shape == frames.shape
    assert out["layer_errors"].shape == (2, 5, 3)
    assert len(out["R"]) == 5 and out["R"][0][2].shape == (2, 8, 4, 6)
    first = out["prediction"][:, 0]
    assert torch.allclose(first, first[..., :1, :1].expand_as(first))
    assert 0.0 <= out["prediction"].min() and out["prediction"].max() <= 1.0


def test_pixel_error_is_half_the_absolute_error() -> None:
    """E_0 splits |frame - prediction| into two ReLU populations, so its mean is half the mean absolute error."""
    model = PredNet(SMALL)
    frames = torch.rand(2, 4, 3, 16, 24)
    out = model(frames)
    mae = (frames - out["prediction"]).abs().mean(dim=(2, 3, 4))
    assert torch.allclose(out["layer_errors"][:, :, 0], mae / 2, atol=1e-6)


def test_loss_ignores_the_first_step_and_trains() -> None:
    """The loss weights the first step by zero, and one Adam step on a fixed batch lowers it."""
    torch.manual_seed(0)
    model = PredNet(SMALL)
    errors = torch.rand(2, 4, 3)
    changed = errors.clone()
    changed[:, 0] += 5.0
    assert torch.allclose(model.loss(errors), model.loss(changed))
    frames = torch.rand(2, 6, 3, 16, 24)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    before = model.loss(model(frames)["layer_errors"])
    before.backward()
    optimizer.step()
    after = model.loss(model(frames)["layer_errors"])
    assert after < before


def test_closed_loop_ignores_later_inputs() -> None:
    """When extrapolating from step s, frames from s on do not change any prediction."""
    torch.manual_seed(0)
    model = PredNet(SMALL)
    frames = torch.rand(1, 6, 3, 16, 24)
    other = frames.clone()
    other[:, 3:] = torch.rand(1, 3, 3, 16, 24)
    a = model(frames, extrapolate_from=3)["prediction"]
    b = model(other, extrapolate_from=3)["prediction"]
    assert torch.equal(a, b)
    assert not torch.equal(model(frames)["prediction"], model(other)["prediction"])


def test_frame_size_must_divide() -> None:
    """Frames whose size does not divide by 2^(L-1) are rejected."""
    with pytest.raises(ValueError):
        PredNet(SMALL)(torch.rand(1, 2, 3, 15, 24))


def test_runs_on_the_selected_device() -> None:
    """Forward and backward run on the auto selected device (MPS on the Mac, CUDA on the cluster)."""
    device = get_device("auto")
    model = PredNet(SMALL).to(device)
    frames = torch.rand(2, 4, 3, 16, 24, device=device)
    loss = model.loss(model(frames)["layer_errors"])
    loss.backward()
    assert loss.device.type == device.type and torch.isfinite(loss)
