"""Tests for the PredNet implementation: shapes, parameter count, reference semantics, loss, and devices."""

import dataclasses

import pytest
import torch

from peekaboo.config import load_config
from peekaboo.device import get_device
from peekaboo.models.prednet import PredNet, PredNetConfig, count_parameters, hard_sigmoid
from peekaboo.paths import CONFIGS_DIR

SMALL = PredNetConfig(stack_sizes=(3, 4, 8), layer_loss_weights=(1.0, 0.1, 0.1))
SMALL_CONCAT = dataclasses.replace(SMALL, error_mode="concat")


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


PROJECT_MODELS = {"prednet_5l": 3_131_628, "prednet_4l": 3_076_524, "prednet_3l": 3_078_924,
                  "prednet_5l_concat": 3_131_628}


@pytest.mark.parametrize("name", PROJECT_MODELS)
def test_parameter_counts_of_the_project_models(name: str) -> None:
    """Every model config matches the hand count and stays within 10% of the main model (decision D9)."""
    config = PredNetConfig.from_config(load_config(CONFIGS_DIR / "models" / f"{name}.yaml"))
    model = PredNet(config)
    assert count_parameters(model) == expected_parameters(config.stack_sizes) == PROJECT_MODELS[name]
    assert abs(PROJECT_MODELS[name] / PROJECT_MODELS["prednet_5l"] - 1) < 0.1


def test_project_models_differ_only_where_intended() -> None:
    """The concat config is the 5 layer model with only error_mode changed."""
    main = PredNetConfig.from_config(load_config(CONFIGS_DIR / "models" / "prednet_5l.yaml"))
    concat = PredNetConfig.from_config(load_config(CONFIGS_DIR / "models" / "prednet_5l_concat.yaml"))
    assert main.error_mode == "split" and concat.error_mode == "concat"
    assert dataclasses.replace(concat, error_mode="split") == main


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


def test_concat_mode_has_the_same_parameters_and_outputs() -> None:
    """Concat mode keeps every parameter shape, so split weights load into it, and returns the same outputs."""
    split, concat = PredNet(SMALL), PredNet(SMALL_CONCAT)
    concat.load_state_dict(split.state_dict())
    frames = torch.rand(2, 4, 3, 16, 24)
    a, b = split(frames, return_states=True), concat(frames, return_states=True)
    assert set(a) == set(b)
    for key in ("prediction", "layer_errors"):
        assert a[key].shape == b[key].shape
    assert [e.shape for e in a["E"][-1]] == [e.shape for e in b["E"][-1]]
    # the first prediction comes from zero states in both modes, the later ones differ
    assert torch.equal(a["prediction"][:, 0], b["prediction"][:, 0])
    assert not torch.allclose(a["prediction"][:, 1:], b["prediction"][:, 1:])


@pytest.mark.parametrize("config", [SMALL, SMALL_CONCAT], ids=["split", "concat"])
def test_each_layer_passes_on_its_signal(config: PredNetConfig) -> None:
    """Split mode passes E_l on; concat mode passes [A_l, Ahat_l], where Ahat_0 is the prediction and A_0 the frame.
    In both modes the returned errors are E_l = [ReLU(A_l - Ahat_l), ReLU(Ahat_l - A_l)]."""
    torch.manual_seed(0)
    model = PredNet(config)
    frame = torch.rand(2, 3, 16, 24)
    r, c, x = model.initial_state(2, 16, 24, frame)
    x = [torch.rand_like(v) for v in x]  # a nonzero previous signal
    prediction, _, _, new_x, new_e = model.step(frame, r, c, x)
    assert torch.equal(new_e[0], torch.cat([torch.relu(frame - prediction), torch.relu(prediction - frame)], dim=1))
    for signal, error in zip(new_x, new_e):
        assert signal.shape == error.shape
    if config.error_mode == "split":
        assert all(s is e for s, e in zip(new_x, new_e))
    else:
        assert torch.equal(new_x[0], torch.cat([frame, prediction], dim=1))
        # A_l and Ahat_l are recovered from the errors: A - Ahat = E+ - E-
        for signal, error in zip(new_x[1:], new_e[1:]):
            target, ahat = signal.chunk(2, dim=1)
            plus, minus = error.chunk(2, dim=1)
            assert torch.allclose(target - ahat, plus - minus, atol=1e-6)


def test_concat_mode_trains_and_runs_on_the_selected_device() -> None:
    """One Adam step lowers the loss of a concat model, on the auto selected device."""
    torch.manual_seed(0)
    device = get_device("auto")
    model = PredNet(SMALL_CONCAT).to(device)
    frames = torch.rand(2, 6, 3, 16, 24, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    before = model.loss(model(frames)["layer_errors"])
    before.backward()
    assert all(p.grad is not None for p in model.parameters())
    optimizer.step()
    assert model.loss(model(frames)["layer_errors"]) < before


def test_unknown_error_mode_is_rejected() -> None:
    """Only "split" and "concat" exist."""
    with pytest.raises(ValueError):
        PredNet(dataclasses.replace(SMALL, error_mode="diff"))
