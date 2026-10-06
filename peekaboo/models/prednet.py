"""PredNet (Lotter, Kreiman, and Cox, ICLR 2017), reimplemented in PyTorch from the paper and the original code.

The details follow coxlab/prednet (prednet.py, kitti_train.py), as checked in the project plan:
- each layer l has a target A_l, a prediction Ahat_l, an error E_l, and a convolutional LSTM representation R_l;
- E_l = [ReLU(A_l - Ahat_l), ReLU(Ahat_l - A_l)]; Ahat_l = ReLU(conv(R_l)), clipped at pixel_max for l = 0 (SatLU);
- A_{l+1} = maxpool(ReLU(conv(E_l)));
- R_l is updated from E_l and R_l at the previous step and from the upsampled R_{l+1} at the current step;
- each step first updates R from the top layer down, then computes Ahat, E, and A from the bottom layer up;
- the LSTM gates use the Keras 2 hard sigmoid, clip(0.2 x + 0.5, 0, 1), with no peepholes;
- the loss is the weighted mean activity of the error units: layer weights (L0: 1 on layer 0, 0 elsewhere) and time
  weights (0 for the first step, 1 / (T - 1) after);
- kernels use Glorot uniform initialization and biases start at zero, as in Keras.
The ablation "no explicit error" (error_mode "concat", decision D11) passes [A_l, Ahat_l] instead of E_l to R_l and to
the next layer. The channels and the parameters are the same, so it removes only the inductive bias of computing the
error: a convolution can still learn the difference. The error units are still computed for the loss and the readouts.
No code was copied: the reference repository is read only (see THIRD_PARTY_NOTICES.md).
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


ERROR_MODES = ("split", "concat")


def hard_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Keras 2 hard sigmoid, clip(0.2 x + 0.5, 0, 1). PyTorch's F.hardsigmoid has slope 1/6 and is not the same."""
    return torch.clamp(0.2 * x + 0.5, 0.0, 1.0)


@dataclass(frozen=True)
class PredNetConfig:
    """Architecture and loss settings. R channels equal A channels, as in the original models."""

    stack_sizes: tuple[int, ...] = (3, 16, 32, 64, 128)
    kernel_size: int = 3
    pixel_max: float = 1.0
    layer_loss_weights: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 0.0)  # "L0"
    error_mode: str = "split"  # "split": E_l is passed on; "concat": [A_l, Ahat_l] is passed on instead

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PredNetConfig":
        """Build the settings from a model config dictionary."""
        stack = tuple(int(s) for s in config["stack_sizes"])
        weights = config.get("layer_loss_weights", "L0")
        if weights == "L0":
            weights = (1.0,) + (0.0,) * (len(stack) - 1)
        elif weights == "Lall":
            weights = (1.0,) + (0.1,) * (len(stack) - 1)
        return cls(stack_sizes=stack, kernel_size=int(config.get("kernel_size", 3)),
                   pixel_max=float(config.get("pixel_max", 1.0)),
                   layer_loss_weights=tuple(float(w) for w in weights),
                   error_mode=str(config.get("error_mode", "split")))

    @property
    def n_layers(self) -> int:
        """Number of layers."""
        return len(self.stack_sizes)


class PredNet(nn.Module):
    """PredNet over sequences of frames (B, T, C, H, W) with values in [0, 1]. H and W must divide by 2^(L-1)."""

    def __init__(self, config: PredNetConfig) -> None:
        """Create the convolutions of every layer."""
        super().__init__()
        self.config = config
        s, k, n = config.stack_sizes, config.kernel_size, config.n_layers
        if len(config.layer_loss_weights) != n:
            raise ValueError("layer_loss_weights needs one weight per layer.")
        if config.error_mode not in ERROR_MODES:
            raise ValueError(f"error_mode must be one of {ERROR_MODES}, not {config.error_mode!r}.")
        pad = k // 2
        # the four LSTM gates (i, f, candidate, o) share one convolution with 4 R_l output channels
        self.gates = nn.ModuleList(
            nn.Conv2d(2 * s[l] + s[l] + (s[l + 1] if l < n - 1 else 0), 4 * s[l], k, padding=pad) for l in range(n))
        self.ahat = nn.ModuleList(nn.Conv2d(s[l], s[l], k, padding=pad) for l in range(n))
        self.a = nn.ModuleList(nn.Conv2d(2 * s[l], s[l + 1], k, padding=pad) for l in range(n - 1))
        for conv in self.modules():
            if isinstance(conv, nn.Conv2d):
                nn.init.xavier_uniform_(conv.weight)
                nn.init.zeros_(conv.bias)
        self.register_buffer("layer_weights", torch.tensor(config.layer_loss_weights, dtype=torch.float32))

    def initial_state(self, batch: int, height: int, width: int,
                      reference: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Zero R, C, and the passed on signal (E or [A, Ahat]) for every layer, on the device of `reference`."""
        s, n = self.config.stack_sizes, self.config.n_layers
        if height % 2 ** (n - 1) or width % 2 ** (n - 1):
            raise ValueError(f"Frames of {height}x{width} do not divide by 2^{n - 1}.")
        shapes = [(height // 2 ** l, width // 2 ** l) for l in range(n)]
        zeros = [reference.new_zeros((batch, s[l]) + shapes[l]) for l in range(n)]
        signals = [reference.new_zeros((batch, 2 * s[l]) + shapes[l]) for l in range(n)]
        return list(zeros), [z.clone() for z in zeros], signals

    def step(self, frame: torch.Tensor, r: list[torch.Tensor], c: list[torch.Tensor], x: list[torch.Tensor]
             ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """One time step: returns the frame prediction Ahat_0, the new R and C, the signal X that each layer passes
        to its representation and to the layer above (E_l, or [A_l, Ahat_l] in "concat" mode), and the errors E."""
        n = self.config.n_layers
        new_r, new_c = [None] * n, [None] * n
        # top down: representations, from the previous signals and the upper layer's new representation
        for l in reversed(range(n)):
            inputs = [x[l], r[l]]
            if l < n - 1:
                inputs.append(F.interpolate(new_r[l + 1], scale_factor=2, mode="nearest"))
            i, f, g, o = torch.chunk(self.gates[l](torch.cat(inputs, dim=1)), 4, dim=1)
            new_c[l] = hard_sigmoid(f) * c[l] + hard_sigmoid(i) * torch.tanh(g)
            new_r[l] = hard_sigmoid(o) * torch.tanh(new_c[l])
        # bottom up: predictions, errors, and targets of the next layer
        new_x, new_e = [], []
        target = frame
        prediction = None
        for l in range(n):
            ahat = F.relu(self.ahat[l](new_r[l]))
            if l == 0:
                ahat = torch.clamp(ahat, max=self.config.pixel_max)  # SatLU
                prediction = ahat
            new_e.append(torch.cat([F.relu(target - ahat), F.relu(ahat - target)], dim=1))
            new_x.append(new_e[l] if self.config.error_mode == "split" else torch.cat([target, ahat], dim=1))
            if l < n - 1:
                target = F.max_pool2d(F.relu(self.a[l](new_x[l])), 2)
        return prediction, new_r, new_c, new_x, new_e

    def forward(self, frames: torch.Tensor, extrapolate_from: int | None = None,
                return_states: bool = False) -> dict[str, Any]:
        """Run over a sequence. prediction[:, t] predicts frame t from frames before t.

        From step extrapolate_from on, the model's previous prediction replaces the input (closed loop). The output
        holds "prediction" (B, T, C, H, W), "layer_errors" (B, T, L), the mean activity of each layer's error units,
        and, if return_states, the lists "R" and "E" with one list of per layer tensors per time step.
        """
        batch, seq_len, _, height, width = frames.shape
        r, c, x = self.initial_state(batch, height, width, frames)
        predictions, layer_errors, states_r, states_e = [], [], [], []
        previous = None
        for t in range(seq_len):
            frame = frames[:, t]
            if extrapolate_from is not None and t >= extrapolate_from and previous is not None:
                frame = previous
            previous, r, c, x, e = self.step(frame, r, c, x)
            predictions.append(previous)
            layer_errors.append(torch.stack([err.mean(dim=(1, 2, 3)) for err in e], dim=1))
            if return_states:
                states_r.append(r)
                states_e.append(e)
        out = {"prediction": torch.stack(predictions, dim=1), "layer_errors": torch.stack(layer_errors, dim=1)}
        if return_states:
            out["R"], out["E"] = states_r, states_e
        return out

    def loss(self, layer_errors: torch.Tensor) -> torch.Tensor:
        """Weighted error activity: layer weights, then time weights 0 for the first step and 1 / (T - 1) after."""
        seq_len = layer_errors.shape[1]
        time_weights = torch.full((seq_len,), 1.0 / (seq_len - 1), device=layer_errors.device)
        time_weights[0] = 0.0
        per_step = (layer_errors * self.layer_weights).sum(dim=2)  # (B, T)
        return (per_step * time_weights).sum(dim=1).mean()


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
