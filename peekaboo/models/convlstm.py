"""ConvLSTM baseline (Shi et al., NeurIPS 2015), written from the paper, with the interface of PredNet.

The cell follows Eq. 3 of the paper, with peepholes from the cell state to the three gates:
- i_t = sigmoid(W_xi * X_t + W_hi * H_{t-1} + W_ci o C_{t-1} + b_i)
- f_t = sigmoid(W_xf * X_t + W_hf * H_{t-1} + W_cf o C_{t-1} + b_f)
- C_t = f_t o C_{t-1} + i_t o tanh(W_xc * X_t + W_hc * H_{t-1} + b_c)
- o_t = sigmoid(W_xo * X_t + W_ho * H_{t-1} + W_co o C_t + b_o)
- H_t = o_t o tanh(C_t)
where * is a convolution and o the elementwise product. One deviation: the peephole weights W_c are one weight per
channel, not one per channel and pixel as in the paper. Weights per pixel would give the model a code of absolute
position that PredNet, fully convolutional, does not have, which would bias the comparison of the position probes.

The model cuts each frame into 4 x 4 patches (space to depth, as in OpenSTL for Moving MNIST), runs a stack of ConvLSTM
layers on the 16 x 24 grid, and predicts the next frame from the hidden states of all layers with a 1 x 1 convolution,
as in the paper's forecasting network. The output goes through ReLU and is clipped at pixel_max, as PredNet's pixel
layer (SatLU), so both models can predict an exactly black background. No code was copied (see THIRD_PARTY_NOTICES.md).
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ConvLSTMConfig:
    """Architecture settings."""

    hidden_sizes: tuple[int, ...] = (64, 64, 64, 64)
    kernel_size: int = 5
    patch_size: int = 4
    channels: int = 3
    pixel_max: float = 1.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ConvLSTMConfig":
        """Build the settings from a model config dictionary."""
        return cls(hidden_sizes=tuple(int(h) for h in config["hidden_sizes"]),
                   kernel_size=int(config.get("kernel_size", 5)), patch_size=int(config.get("patch_size", 4)),
                   channels=int(config.get("channels", 3)), pixel_max=float(config.get("pixel_max", 1.0)))


class ConvLSTMCell(nn.Module):
    """One ConvLSTM layer with peepholes (Shi et al., Eq. 3)."""

    def __init__(self, in_channels: int, hidden: int, kernel_size: int) -> None:
        """One convolution of [X, H] gives the four gates; the peepholes are one weight per channel and gate."""
        super().__init__()
        self.conv = nn.Conv2d(in_channels + hidden, 4 * hidden, kernel_size, padding=kernel_size // 2)
        self.peephole = nn.Parameter(torch.zeros(3, hidden, 1, 1))  # W_ci, W_cf, W_co

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One step: the new hidden state H_t and cell state C_t."""
        i, f, g, o = torch.chunk(self.conv(torch.cat([x, h], dim=1)), 4, dim=1)
        i = torch.sigmoid(i + self.peephole[0] * c)
        f = torch.sigmoid(f + self.peephole[1] * c)
        c = f * c + i * torch.tanh(g)
        o = torch.sigmoid(o + self.peephole[2] * c)  # the output gate sees the new cell state
        return o * torch.tanh(c), c


class ConvLSTM(nn.Module):
    """Stacked ConvLSTM over sequences of frames (B, T, C, H, W) in [0, 1]. H and W must divide by patch_size."""

    def __init__(self, config: ConvLSTMConfig) -> None:
        """Create the layers and the 1 x 1 output convolution."""
        super().__init__()
        self.config = config
        patch_channels = config.channels * config.patch_size ** 2
        inputs = (patch_channels,) + config.hidden_sizes[:-1]
        self.cells = nn.ModuleList(ConvLSTMCell(i, h, config.kernel_size) for i, h in zip(inputs, config.hidden_sizes))
        self.output = nn.Conv2d(sum(config.hidden_sizes), patch_channels, 1)
        for conv in self.modules():
            if isinstance(conv, nn.Conv2d):
                nn.init.xavier_uniform_(conv.weight)
                nn.init.zeros_(conv.bias)

    def initial_state(self, batch: int, height: int, width: int,
                      reference: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Zero H and C for every layer, on the device of `reference`."""
        p = self.config.patch_size
        if height % p or width % p:
            raise ValueError(f"Frames of {height}x{width} do not divide by the patch size {p}.")
        h = [reference.new_zeros((batch, n, height // p, width // p)) for n in self.config.hidden_sizes]
        return h, [z.clone() for z in h]

    def readout(self, h: list[torch.Tensor]) -> torch.Tensor:
        """The predicted frame (B, C, H, W) from the hidden states of all layers."""
        patches = torch.clamp(F.relu(self.output(torch.cat(h, dim=1))), max=self.config.pixel_max)
        return F.pixel_shuffle(patches, self.config.patch_size)

    def step(self, frame: torch.Tensor, h: list[torch.Tensor],
             c: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Read one frame: the new H and C of every layer, from the bottom layer up."""
        x = F.pixel_unshuffle(frame, self.config.patch_size)
        new_h, new_c = [], []
        for cell, h_l, c_l in zip(self.cells, h, c):
            h_l, c_l = cell(x, h_l, c_l)
            new_h.append(h_l)
            new_c.append(c_l)
            x = h_l
        return new_h, new_c

    def forward(self, frames: torch.Tensor, extrapolate_from: int | None = None,
                return_states: bool = False) -> dict[str, Any]:
        """Run over a sequence. prediction[:, t] predicts frame t from frames before t.

        From step extrapolate_from on, the prediction of frame t replaces frame t as input (closed loop). The output
        holds "prediction" (B, T, C, H, W) and, if return_states, "R": one list per time step of the hidden states H
        of each layer that make the prediction of frame t, the counterpart of PredNet's R.
        """
        batch, seq_len, _, height, width = frames.shape
        h, c = self.initial_state(batch, height, width, frames)
        predictions, states = [], []
        for t in range(seq_len):
            prediction = self.readout(h)
            predictions.append(prediction)
            if return_states:
                states.append(h)
            if t == seq_len - 1:
                break  # the last frame would only update states that predict nothing
            closed = extrapolate_from is not None and t >= extrapolate_from
            h, c = self.step(prediction if closed else frames[:, t], h, c)
        out = {"prediction": torch.stack(predictions, dim=1)}
        if return_states:
            out["R"] = states
        return out
