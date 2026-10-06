"""Evaluation of the position probes on held out sequences, next to the controls of decision D15.

Every frame from MIN_CONTEXT on with the digit present gets a row with its truth and, for each method, the position
believed at frame t from frames before t:
- the probe of each model (for example the trained model and the same architecture at random initialization);
- the programmed trackers (last seen position, constant velocity Kalman filter with and without walls);
- the bar center, for x only: while the digit is hidden, x is bounded by the bar (critique C4).
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from peekaboo.data.dataset import make_loader
from peekaboo.data.truth import STATE_NAMES
from peekaboo.eval.next_frame import MIN_CONTEXT
from peekaboo.probes.features import frame_rows, iterate_frames
from peekaboo.probes.position import AXES, PositionProbe, position_bins
from peekaboo.trackers.baselines import run_trackers
from peekaboo.trackers.detector import bar_columns

BASELINES = ("last_seen", "kalman", "kalman_walls")


def probe_positions(model: torch.nn.Module, probe: PositionProbe, dataset: Dataset, count: int,
                    device: torch.device, **options) -> pd.DataFrame:
    """Rows of the selected frames with the probe's position (y, x) and the probability it gives to the true bin of
    each axis."""
    parts = []
    for features, rows, _ in iterate_frames(model, dataset, count, device, **options):
        position = probe.predict(features)
        right = position_bins(rows[["y", "x"]].to_numpy(), probe.bin_edges)
        p = probe.bin_probabilities(features)
        parts.append(rows.assign(pred_y=position[:, 0], pred_x=position[:, 1],
                                 p_bin_y=p[np.arange(len(p)), 0, right[:, 0]],
                                 p_bin_x=p[np.arange(len(p)), 1, right[:, 1]]))
    return pd.concat(parts, ignore_index=True)


def baseline_positions(dataset: Dataset, count: int, start: int = 0, seq_len: int = 40) -> pd.DataFrame:
    """Rows of every frame with the positions predicted by the programmed trackers and the bar center x."""
    parts, first = [], start
    for batch in make_loader(dataset, 16, start, count):
        batch = {name: value[:, :seq_len] if value.dim() > 1 else value for name, value in batch.items()}
        rows = frame_rows(batch, first)
        columns = {f"{name}_{axis}": [] for name in BASELINES for axis in AXES}
        bar_x = []
        for frames, center, state in zip(batch["frames"].numpy(), batch["center"].numpy(), batch["state"].numpy()):
            tracks = run_trackers(frames, center, state)
            for name in BASELINES:
                for a, axis in enumerate(AXES):
                    columns[f"{name}_{axis}"].append(tracks[name].prediction[:, a])
            bar = np.flatnonzero(bar_columns(frames[0]))
            bar_x.append(np.full(len(frames), bar.mean() if bar.size else np.nan))
        parts.append(rows.assign(**{name: np.concatenate(v) for name, v in columns.items()},
                                 bar_center_x=np.concatenate(bar_x)))
        first += len(batch["state"])
    return pd.concat(parts, ignore_index=True)


def position_table(probes: dict[str, pd.DataFrame], baselines: pd.DataFrame) -> pd.DataFrame:
    """One row per selected frame with the positions of every method, columns <method>_y and <method>_x (and
    <probe>_p_bin_y, <probe>_p_bin_x for the probes)."""
    keys = ["sequence", "t"]
    table = baselines[(baselines["t"] >= MIN_CONTEXT) & baselines["y"].notna()].reset_index(drop=True)
    for name, rows in probes.items():
        renamed = rows[keys + ["pred_y", "pred_x", "p_bin_y", "p_bin_x"]].rename(
            columns={"pred_y": f"{name}_y", "pred_x": f"{name}_x", "p_bin_y": f"{name}_p_bin_y",
                     "p_bin_x": f"{name}_p_bin_x"})
        table = table.merge(renamed, on=keys, how="left", validate="one_to_one")
    table["state_name"] = [STATE_NAMES[s] for s in table["state"]]
    return table


def add_errors(table: pd.DataFrame, methods: tuple[str, ...]) -> pd.DataFrame:
    """Add the absolute error per axis (<method>_err_y, <method>_err_x) and the distance (<method>_err) of each
    method, and the x error of the bar center."""
    out = table.copy()
    for m in methods:
        out[f"{m}_err_y"] = (out[f"{m}_y"] - out["y"]).abs()
        out[f"{m}_err_x"] = (out[f"{m}_x"] - out["x"]).abs()
        out[f"{m}_err"] = np.hypot(out[f"{m}_err_y"], out[f"{m}_err_x"])
    out["bar_center_err_x"] = (out["bar_center_x"] - out["x"]).abs()
    return out


def summarize_errors(table: pd.DataFrame, methods: tuple[str, ...], by: list[str], axis: str) -> pd.DataFrame:
    """Mean absolute error on one axis ("y", "x", or "" for the distance) of every method, per group."""
    suffix = f"_err_{axis}" if axis else "_err"
    columns = [f"{m}{suffix}" for m in methods]
    if axis == "x":
        columns.append("bar_center_err_x")
    summary = table.groupby(by)[columns].mean()
    summary.columns = [c.removesuffix(suffix) if c != "bar_center_err_x" else "bar_center" for c in columns]
    summary.insert(0, "frames", table.groupby(by).size())
    return summary
