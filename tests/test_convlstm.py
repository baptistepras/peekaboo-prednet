"""Tests for the ConvLSTM baseline: peephole equations, parameter count, causality, interface, and devices."""

import math
from pathlib import Path

import pytest
import torch

from peekaboo.config import load_config
from peekaboo.device import get_device
from peekaboo.models import build_model
from peekaboo.models.convlstm import ConvLSTM, ConvLSTMCell, ConvLSTMConfig
from peekaboo.models.prednet import count_parameters
from peekaboo.paths import CONFIGS_DIR
from peekaboo.train.checkpoint import load_model, save_checkpoint
from peekaboo.train.losses import training_loss

SMALL = ConvLSTMConfig(hidden_sizes=(4, 6), kernel_size=3, patch_size=2)
PROJECT = ConvLSTMConfig.from_config(load_config(CONFIGS_DIR / "models" / "convlstm.yaml"))
PREDNET_5L = 3_131_628


def expected_parameters(config: ConvLSTMConfig) -> int:
    """Hand count from Eq. 3 of Shi et al., with one peephole weight per channel and gate."""
    k, patch = config.kernel_size, config.channels * config.patch_size ** 2
    total, inputs = 0, patch
    for hidden in config.hidden_sizes:
        total += (inputs + hidden) * 4 * hidden * k * k + 4 * hidden  # the four gates, from X and H
        total += 3 * hidden                                           # peepholes W_ci, W_cf, W_co
        inputs = hidden
    return total + sum(config.hidden_sizes) * patch + patch           # 1 x 1 output


def sigmoid(x: float) -> float:
    """Logistic function."""
    return 1.0 / (1.0 + math.exp(-x))


def test_cell_follows_the_peephole_equations() -> None:
    """A one channel cell with a 1 x 1 kernel matches Eq. 3 computed by hand."""
    cell = ConvLSTMCell(1, 1, 1)
    w = {"i": (0.5, -0.3, 0.1), "f": (0.2, 0.4, -0.2), "g": (-0.7, 0.6, 0.3), "o": (0.9, -0.1, 0.05)}  # W_x, W_h, b
    with torch.no_grad():
        cell.conv.weight.copy_(torch.tensor([[[[w[g][0]]], [[w[g][1]]]] for g in "ifgo"]))
        cell.conv.bias.copy_(torch.tensor([w[g][2] for g in "ifgo"]))
        cell.peephole.copy_(torch.tensor([0.3, -0.4, 0.8]).view(3, 1, 1, 1))
    x, h, c = 0.8, -0.2, 0.6
    i = sigmoid(w["i"][0] * x + w["i"][1] * h + 0.3 * c + w["i"][2])
    f = sigmoid(w["f"][0] * x + w["f"][1] * h - 0.4 * c + w["f"][2])
    c_new = f * c + i * math.tanh(w["g"][0] * x + w["g"][1] * h + w["g"][2])
    o = sigmoid(w["o"][0] * x + w["o"][1] * h + 0.8 * c_new + w["o"][2])
    h_new = o * math.tanh(c_new)
    out_h, out_c = cell(*(torch.full((1, 1, 1, 1), v) for v in (x, h, c)))
    assert out_c.item() == pytest.approx(c_new, abs=1e-6)
    assert out_h.item() == pytest.approx(h_new, abs=1e-6)


@pytest.mark.parametrize("config", [SMALL, PROJECT], ids=["small", "project"])
def test_parameter_count(config: ConvLSTMConfig) -> None:
    """The model has exactly the parameters of the hand count."""
    assert count_parameters(ConvLSTM(config)) == expected_parameters(config)


def test_project_config_matches_prednet() -> None:
    """The project config has 3,188,528 parameters, within 10% of PredNet 5 layers (decision D9)."""
    assert expected_parameters(PROJECT) == 3_188_528
    assert abs(expected_parameters(PROJECT) / PREDNET_5L - 1) < 0.1


def test_shapes_states_and_first_prediction() -> None:
    """Outputs have the right shapes, R has one hidden state per layer and step, and the first prediction, made from
    zero states, is black."""
    model = ConvLSTM(SMALL)
    frames = torch.rand(2, 5, 3, 16, 24)
    out = model(frames, return_states=True)
    assert out["prediction"].shape == frames.shape
    assert len(out["R"]) == 5 and [r.shape for r in out["R"][0]] == [(2, 4, 8, 12), (2, 6, 8, 12)]
    assert torch.equal(out["prediction"][:, 0], torch.zeros_like(frames[:, 0]))
    assert 0.0 <= out["prediction"].min() and out["prediction"].max() <= 1.0


def test_prediction_of_frame_t_uses_only_earlier_frames() -> None:
    """Changing frames t and later leaves the predictions of frames 0 to t unchanged, and changes the later ones."""
    torch.manual_seed(0)
    model = ConvLSTM(SMALL)
    with torch.no_grad():
        model.output.bias.fill_(0.3)  # keep the output above the ReLU so that every input shows
    frames = torch.rand(1, 6, 3, 16, 24)
    other = frames.clone()
    other[:, 3:] = torch.rand(1, 3, 3, 16, 24)
    a, b = model(frames)["prediction"], model(other)["prediction"]
    assert torch.equal(a[:, :4], b[:, :4])
    assert not torch.equal(a[:, 4:], b[:, 4:])


def test_closed_loop_ignores_later_inputs() -> None:
    """When extrapolating from step s, frames from s on do not change any prediction."""
    torch.manual_seed(0)
    model = ConvLSTM(SMALL)
    frames = torch.rand(1, 6, 3, 16, 24)
    other = frames.clone()
    other[:, 3:] = torch.rand(1, 3, 3, 16, 24)
    assert torch.equal(model(frames, extrapolate_from=3)["prediction"], model(other, extrapolate_from=3)["prediction"])


def test_frame_size_must_divide() -> None:
    """Frames whose size does not divide by the patch size are rejected."""
    with pytest.raises(ValueError):
        ConvLSTM(SMALL)(torch.rand(1, 2, 3, 15, 24))


def test_trains_with_the_common_loss_on_the_selected_device() -> None:
    """Built from a config, the model trains with the digit weighted loss: every parameter gets a gradient and one
    Adam step lowers the loss, on the auto selected device."""
    torch.manual_seed(0)
    device = get_device("auto")
    model = build_model({"model": "convlstm", "hidden_sizes": [4, 6], "kernel_size": 3, "patch_size": 2}).to(device)
    with torch.no_grad():
        model.output.bias.fill_(0.3)
    frames = torch.rand(2, 6, 3, 16, 24, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    before = training_loss(model, model(frames), frames, 10.0)
    before.backward()
    assert all(p.grad is not None for p in model.parameters())
    optimizer.step()
    assert training_loss(model, model(frames), frames, 10.0) < before


def test_checkpoint_rebuilds_the_model(tmp_path: Path) -> None:
    """load_model rebuilds a ConvLSTM from the settings stored in its checkpoint."""
    torch.manual_seed(0)
    config = {"model": {"model": "convlstm", "hidden_sizes": [4, 6], "kernel_size": 3, "patch_size": 2}}
    model = build_model(config["model"])
    save_checkpoint(tmp_path / "best.pt", model, config=config)
    loaded, _ = load_model(tmp_path / "best.pt", torch.device("cpu"))
    frames = torch.rand(1, 4, 3, 16, 24)
    assert isinstance(loaded, ConvLSTM)
    assert torch.equal(loaded(frames)["prediction"], model(frames)["prediction"])
