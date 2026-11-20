"""Readouts of single layers (step 3.5): which layers of a frozen model hold the hidden digit's position and shape.

The position probe of step 2.8 and the amodal decoder of step 2.9 read every layer of the state R at once. Here the
same readouts are fitted on subsets of the layers, on the same frames of the training stream:
- `all`: every layer, the readout of steps 2.8 and 2.9 (the ridge probe is the very same, fitted on the same frames
  with the same split);
- `only_<l>`: layer l alone, what that layer holds;
- `without_<l>`: every layer but l, what is lost without it;
- `visible`: every layer, fitted on the fully visible frames only. If it still finds the hidden digit, the state
  codes a hidden digit as it codes a visible one (the main limit of the probe of step 2.8, which saw hidden frames in
  training).

The probes are the ridge regressions of (y, x) only (the logistic bins of step 2.8 are left out), on the same pooled
features (peekaboo/probes/features.py), standardized once on the training frames; a subset reads its columns,
centered on the frames it is fitted on. Each decoder is the minimal decoder of step 2.9 on the layers of its subset.
Everything is saved in <run>/probes/layers/ as plain arrays; layer_frames (peekaboo/eval/extract.py) reads every
readout on every frame of a stored set.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from peekaboo.data.dataset import make_loader, prepare_batch
from peekaboo.data.truth import STATE_NAMES
from peekaboo.probes.amodal import AmodalDecoder, decoder_loss, select_states, selected_frames, state_channels
from peekaboo.probes.position import Standardizer, fit_ridge, validation_split

VISIBLE = "visible"                         # the readout fitted on the fully visible frames only
VISIBLE_STATE = STATE_NAMES.index("visible")
FOLDER = Path("probes") / "layers"          # inside a run folder


def layer_subsets(n_layers: int) -> dict[str, tuple[int, ...]]:
    """The layers read by every readout: all, visible (all, on visible frames), only_<l>, and without_<l>."""
    every = tuple(range(n_layers))
    out = {"all": every, VISIBLE: every}
    out.update({f"only_{l}": (l,) for l in every})
    if n_layers > 1:
        out.update({f"without_{l}": tuple(m for m in every if m != l) for l in every})
    return out


def state_shapes(model: nn.Module, frame_size: tuple[int, int], device: torch.device) -> list[tuple[int, int, int]]:
    """The shape (C, H, W) of the state of every layer, from a forward pass on two blank frames."""
    with torch.no_grad():
        states = model(torch.zeros(1, 2, 3, *frame_size, device=device), return_states=True)["R"]
    return [tuple(int(s) for s in r.shape[1:]) for r in states[0]]


@dataclass(frozen=True)
class LayerProbes:
    """A ridge probe of (y, x) per readout, each on its columns of the standardized features."""

    standardizer: Standardizer
    columns: dict[str, np.ndarray]    # feature columns read by every probe
    coef: dict[str, np.ndarray]       # (columns, 2)
    intercept: dict[str, np.ndarray]  # (2,)
    alpha: dict[str, float]
    layer_limit: int                  # pooling limit of the features (peekaboo/probes/features.py)

    def predict(self, features: np.ndarray) -> dict[str, np.ndarray]:
        """(N, 2) predicted centroid (y, x) in pixels of every probe."""
        z = self.standardizer(features)
        return {name: z[:, columns] @ self.coef[name].astype(np.float32) + self.intercept[name]
                for name, columns in self.columns.items()}

    def save(self, path: str | Path) -> Path:
        """Save as .npz, plain arrays only."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for name in self.columns:
            arrays.update({f"{name}.columns": self.columns[name], f"{name}.coef": self.coef[name],
                           f"{name}.intercept": self.intercept[name], f"{name}.alpha": self.alpha[name]})
        np.savez(path, names=np.array(list(self.columns)), mean=self.standardizer.mean,
                 scale=self.standardizer.scale, layer_limit=self.layer_limit, **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "LayerProbes":
        """Load the probes saved by save."""
        with np.load(path, allow_pickle=False) as data:
            names = [str(name) for name in data["names"]]
            return cls(standardizer=Standardizer(data["mean"], data["scale"]),
                       columns={n: data[f"{n}.columns"] for n in names}, coef={n: data[f"{n}.coef"] for n in names},
                       intercept={n: data[f"{n}.intercept"] for n in names},
                       alpha={n: float(data[f"{n}.alpha"]) for n in names}, layer_limit=int(data["layer_limit"]))


def fit_layer_probes(features: np.ndarray, rows: pd.DataFrame, columns: list[np.ndarray], layer_limit: int,
                     validation_share: float = 0.2, seed: int = 0) -> tuple[LayerProbes, dict[str, float]]:
    """Fit the probe of every readout on frames (N, F) with their rows (true centroid, sequence, state) and the
    columns of every layer (peekaboo/probes/features.py layer_columns). The held out sequences, the standardization,
    and the penalties are those of fit_position_probe, so `all` is its ridge probe. Returns the probes and the
    validation MSE of every readout at its chosen penalty."""
    positions = rows[["y", "x"]].to_numpy(dtype=np.float64)
    train = validation_split(rows["sequence"].to_numpy(), validation_share, seed)
    standardizer = Standardizer.fit(features[train])
    z = standardizer(features)
    visible = (rows["state"] == VISIBLE_STATE).to_numpy()
    out: dict[str, dict[str, Any]] = {"columns": {}, "coef": {}, "intercept": {}, "alpha": {}}
    errors = {}
    for name, layers in layer_subsets(len(columns)).items():
        own = np.concatenate([columns[l] for l in layers])
        keep = visible if name == VISIBLE else np.ones(len(z), dtype=bool)
        # fit_ridge has no constant feature, so the features are centered on the frames it is fitted on (the visible
        # frames do not have the mean of all the frames), and the shift goes into the intercept; for the readouts
        # fitted on every frame the shift is 0 up to rounding
        own_z = z[np.ix_(keep, own)]
        center = own_z[train[keep]].mean(axis=0, dtype=np.float64)
        own_z -= center.astype(np.float32)
        coef, intercept, alpha, validation = fit_ridge(own_z, positions[keep], train[keep])
        intercept = intercept - center @ coef
        out["columns"][name], out["coef"][name], out["intercept"][name] = own, coef, intercept
        out["alpha"][name], errors[name] = alpha, validation[alpha]
    return LayerProbes(standardizer=standardizer, layer_limit=layer_limit, **out), errors


def fit_layer_decoders(model: nn.Module, dataset: Dataset, steps: int, device: torch.device, batch_size: int = 8,
                       lr: float = 1e-2, digit_weight: float = 10.0, workers: int = 0,
                       log: Callable[[str], None] = print) -> tuple[dict[str, AmodalDecoder], dict[str, list[float]]]:
    """Train the decoder of every readout on the same batches of the stream, as fit_decoder trains one (analysis
    windows, Adam, ink weighted loss); the `visible` decoder sees only the fully visible frames of each batch.
    Returns the decoders (on the CPU) and the loss of every step of each."""
    model.eval()
    subsets: dict[str, tuple[int, ...]] = {}
    decoders: dict[str, AmodalDecoder] = {}
    optimizers: dict[str, torch.optim.Optimizer] = {}
    losses: dict[str, list[float]] = {}
    for step, batch in enumerate(make_loader(dataset, batch_size, 0, steps * batch_size, workers)):
        batch = prepare_batch(batch, device)
        with torch.no_grad():
            states = model(batch["frames"], return_states=True)["R"]
        if not decoders:
            channels = state_channels(states)
            subsets = layer_subsets(len(channels))
            for name, layers in subsets.items():
                decoders[name] = AmodalDecoder(tuple(channels[l] for l in layers),
                                               batch["frames"].shape[-2:]).to(device)
                optimizers[name] = torch.optim.Adam(decoders[name].parameters(), lr=lr)
                losses[name] = []
        seq, t = selected_frames(batch, window_only=True)
        chosen = select_states(states, seq, t)
        target = batch["amodal"][seq, t, 0]
        visible = batch["state"][seq, t] == VISIBLE_STATE
        for name, layers in subsets.items():
            keep = visible if name == VISIBLE else torch.ones_like(visible)
            if not keep.any():
                continue
            loss = decoder_loss(decoders[name], [chosen[l][keep] for l in layers], target[keep], digit_weight)
            optimizers[name].zero_grad(set_to_none=True)
            loss.backward()
            optimizers[name].step()
            losses[name].append(loss.item())
        if (step + 1) % 50 == 0 or step == 0:
            log(f"  step {step + 1}/{steps}: loss of all {np.mean(losses['all'][-50:]):.4f}, "
                f"visible {np.mean(losses[VISIBLE][-50:]) if losses[VISIBLE] else float('nan'):.4f}")
    return {name: decoder.cpu() for name, decoder in decoders.items()}, losses


def save_readouts(run_dir: Path, probes: LayerProbes, decoders: dict[str, AmodalDecoder],
                  info: dict[str, Any]) -> Path:
    """Save the probes, the decoders, and a description (the layers of every readout and the fit) in
    <run>/probes/layers/."""
    folder = run_dir / FOLDER
    probes.save(folder / "position.npz")
    for name, decoder in decoders.items():
        decoder.save(folder / f"amodal_{name}.npz")
    (folder / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return folder


def load_readouts(run_dir: Path) -> tuple[LayerProbes, dict[str, tuple[tuple[int, ...], AmodalDecoder]]]:
    """The probes, and the layers and decoder of every readout, saved by save_readouts."""
    folder = run_dir / FOLDER
    info = json.loads((folder / "info.json").read_text(encoding="utf-8"))
    decoders = {name: (tuple(layers), AmodalDecoder.load(folder / f"amodal_{name}.npz"))
                for name, layers in info["subsets"].items()}
    return LayerProbes.load(folder / "position.npz"), decoders


def has_readouts(run_dir: Path) -> bool:
    """Whether the readouts of a run were fitted."""
    return (run_dir / FOLDER / "info.json").exists()
