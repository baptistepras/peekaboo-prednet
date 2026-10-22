"""One pass of a trained model over a stored set, saving per frame everything the Phase 3 analyses need (step 3.1).

The model runs once over the set, so no later analysis runs it again. Every frame of every sequence gets one row:
- what the frame is: sequence, frame, condition, k, speed, state, episode, analysis window, frames since the onset and
  since the reappearance, the tuple and its role for surprise sequences, and the true and visible ink centers;
- the next frame scores of the model, the copy of the last frame, and the blank frame (peekaboo/eval/next_frame.py):
  MSE, MAE, and SSIM on the whole frame, and the MSE in two regions, the bar with a margin on each side, where the
  digit emerges, and the box around the true digit;
- what the predicted frame shows (option A): the predicted ink, its centroid off the bar, and the ink on each side of
  the bar, next to the same quantities on the true frame;
- what the state holds: the position probe and the amodal decoder saved in the run (steps 2.8 and 2.9), for the
  trained model and for the same architecture at random initialization, the control of both readouts;
- for PredNet, the mean activity of the error units of every layer, on the whole frame and over the bar.

Row t always describes frame t: the prediction of frame t, made from frames 0 to t - 1, and the state R[t] that makes
it. The programmed trackers do not depend on the model, so they are extracted once per set (baseline_frames).

The rows are written by chunks of sequences in a folder (part_<start>.parquet), each chunk renamed into place only
once complete, so a stopped pass resumes at the first missing chunk.
"""

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from peekaboo.data.dataset import ROLE_NAMES, SURPRISE_NAMES, make_loader, prepare_batch
from peekaboo.eval.next_frame import blank_prediction, copy_prediction, frame_scores
from peekaboo.probes.amodal import TEMPLATES, AmodalDecoder, ink_scores, template_images
from peekaboo.probes.features import frame_features, frame_rows
from peekaboo.probes.position import PositionProbe, position_bins
from peekaboo.trackers.baselines import TRACKER_NAMES, run_trackers
from peekaboo.trackers.detector import bar_columns, detect

BAR_MARGIN = 16      # px on each side of the bar: the widest digit, so the emerging digit is inside the region
DIGIT_MARGIN = 4     # px around the true digit's ink box
INK_FLOOR = 0.1      # predicted ink below this intensity is left out of the readout (faint blur everywhere)
MIN_INK = 1.0        # summed ink below which a frame shows no digit, and the centroid is undefined
MISSING = np.iinfo(np.int64).min  # frames since an event that did not happen
FRAMES_FORMAT = 2    # version of the columns of a model pass: 2 takes the ink centroid off the bar
BASELINES_FORMAT = 1  # version of the columns of the baselines
RANDOM_SEED = 0      # seed of the random initialization control, as in scripts/fit_probes.py and fit_decoder.py
WHO = ("model", "copy", "blank")


def bar_mask(first_frames: torch.Tensor) -> torch.Tensor:
    """(B, W) columns of the bar, from the first frame of every sequence (B, 3, H, W) in [0, 1]: the gray pixels,
    the same nonzero value on the three channels. Frame 0 is never blacked out."""
    r, g, b = first_frames[:, 0], first_frames[:, 1], first_frames[:, 2]
    return ((r > 0) & (r == g) & (g == b)).any(dim=1)


def dilate(mask: torch.Tensor, margin: int) -> torch.Tensor:
    """A boolean mask (..., N) grown by `margin` along its last axis."""
    shape = mask.shape
    grown = F.max_pool1d(mask.float().reshape(-1, 1, shape[-1]), 2 * margin + 1, stride=1, padding=margin)
    return grown.reshape(shape) > 0


def regions(bar: torch.Tensor, amodal: torch.Tensor) -> dict[str, torch.Tensor]:
    """Pixel regions (B, T, H, W) of every frame: the bar columns with BAR_MARGIN on each side, and the box of the true
    digit's ink with DIGIT_MARGIN (empty when the digit is absent). amodal is (B, T, 1, H, W)."""
    batch, steps, _, height, width = amodal.shape
    columns = dilate(bar, BAR_MARGIN)[:, None, None, :].expand(batch, steps, height, width)
    ink = amodal[:, :, 0] > 0
    rows = dilate(ink.any(dim=3), DIGIT_MARGIN)
    cols = dilate(ink.any(dim=2), DIGIT_MARGIN)
    return {"bar": columns, "digit": rows[..., :, None] & cols[..., None, :]}


def region_mse(prediction: torch.Tensor, frames: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
    """MSE (B, T) over the pixels of a region (B, T, H, W) and the three channels; NaN for frame 0 and empty regions."""
    squared = (prediction - frames).square().sum(dim=2)
    size = region.sum(dim=(2, 3)).float()
    mse = (squared * region).sum(dim=(2, 3)) / (3 * size).clamp_min(1)
    mse = torch.where(size > 0, mse, torch.full_like(mse, float("nan")))
    mse[:, 0] = float("nan")
    return mse


def ink_readout(images: torch.Tensor, bar: torch.Tensor) -> dict[str, torch.Tensor]:
    """What an image shows of the digit (B, T, 3, H, W): its red ink (red minus green, which is 0 on the gray bar and
    the black background) above INK_FLOOR, summed (ink), the centroid of the ink off the bar (ink_y, ink_x, NaN below
    MIN_INK of ink off the bar), and the ink left and right of the bar. The centroid leaves the bar out because the
    bar is opaque, so the true digit is never seen there, and some models tint the whole bar red, which would pull the
    centroid to the bar."""
    ink = (images[:, :, 0] - images[:, :, 1]).clamp_min(0)
    ink = ink * (ink > INK_FLOOR)
    height, width = ink.shape[-2:]
    total = ink.sum(dim=(2, 3))
    off = ink * (~bar)[:, None, None, :]
    off_total = off.sum(dim=(2, 3))
    ys = torch.arange(height, device=ink.device, dtype=ink.dtype)
    xs = torch.arange(width, device=ink.device, dtype=ink.dtype)
    nan = torch.full_like(total, float("nan"))
    defined = off_total >= MIN_INK
    centroid_y = torch.where(defined, (off.sum(dim=3) * ys).sum(dim=2) / off_total.clamp_min(1e-12), nan)
    centroid_x = torch.where(defined, (off.sum(dim=2) * xs).sum(dim=2) / off_total.clamp_min(1e-12), nan)
    index = torch.arange(width, device=ink.device)
    has_bar = bar.any(dim=1)
    first = torch.where(has_bar, bar.float().argmax(dim=1), torch.full_like(has_bar, width, dtype=torch.int64))
    last = torch.where(has_bar, width - 1 - bar.flip(1).float().argmax(dim=1), torch.full_like(first, -1))
    left = (index[None] < first[:, None]).float()[:, None, None, :]
    right = (index[None] > last[:, None]).float()[:, None, None, :]
    return {"ink": total, "ink_y": centroid_y, "ink_x": centroid_x,
            "ink_left": (ink * left).sum(dim=(2, 3)), "ink_right": (ink * right).sum(dim=(2, 3))}


def error_activity(errors: list[list[torch.Tensor]], bar: torch.Tensor) -> dict[str, torch.Tensor]:
    """Mean activity of the error units of every layer (B, T), on the whole frame (e<l>) and over the bar columns with
    their margin, taken at the layer's resolution (e<l>_bar). errors is one list of per layer tensors per time step."""
    out = {}
    columns = dilate(bar, BAR_MARGIN).float()
    for l in range(len(errors[0])):
        stacked = torch.stack([step[l] for step in errors], dim=1)  # (B, T, C, H_l, W_l)
        out[f"e{l}"] = stacked.mean(dim=(2, 3, 4))
        width = stacked.shape[-1]
        factor = columns.shape[1] // width
        at_layer = F.max_pool1d(columns[:, None], factor)[:, 0] if factor > 1 else columns  # (B, W_l): any bar column
        size = (at_layer.sum(dim=1) * stacked.shape[2] * stacked.shape[3]).clamp_min(1)
        out[f"e{l}_bar"] = (stacked * at_layer[:, None, None, None, :]).sum(dim=(2, 3, 4)) / size[:, None]
    return out


def readout_columns(prefix: str, model: torch.nn.Module, frames: torch.Tensor, center: np.ndarray,
                    amodal: torch.Tensor, probe: PositionProbe | None,
                    decoder: AmodalDecoder | None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The probe position (<prefix>probe_y, _x) and the probability of the true bin of each axis, and the decoder's
    correlation and IoU against the amodal frame, for every frame (B * T values each). Also returns the forward pass."""
    out = model(frames, return_states=True)
    columns: dict[str, np.ndarray] = {}
    states = out["R"]
    if probe is not None:
        features = frame_features(states, probe.layer_limit).flatten(0, 1).float().cpu().numpy()
        position = probe.predict(features)
        p = probe.bin_probabilities(features)
        known = ~np.isnan(center).any(axis=1)
        right = position_bins(np.where(known[:, None], center, 0.0), probe.bin_edges)
        for a, axis in enumerate(("y", "x")):
            columns[f"{prefix}probe_{axis}"] = position[:, a]
            columns[f"{prefix}probe_p_bin_{axis}"] = np.where(known, p[np.arange(len(p)), a, right[:, a]], np.nan)
    if decoder is not None:
        layers = [torch.stack([step[l] for step in states], dim=1).flatten(0, 1) for l in range(len(states[0]))]
        image = decoder.to(frames.device).imagine(layers)
        corr, iou = ink_scores(image, amodal.flatten(0, 1)[:, 0])
        columns[f"{prefix}decoder_corr"], columns[f"{prefix}decoder_iou"] = corr.cpu().numpy(), iou.cpu().numpy()
    return columns, out


def describe(batch: dict[str, torch.Tensor], first: int) -> pd.DataFrame:
    """What every frame of a CPU batch is: frame_rows plus the visible ink center, the visible fraction, the frames
    since the reappearance, and the tuple of surprise sequences."""
    rows = frame_rows(batch, first)
    size, steps = batch["state"].shape
    t = np.tile(np.arange(steps), size)
    reappear = np.repeat(batch["actual_reappear_frame"].numpy(), steps)
    modal = batch["modal_center"].numpy().reshape(-1, 2)
    return rows.assign(
        modal_y=modal[:, 0], modal_x=modal[:, 1],
        visible_fraction=batch["visible_fraction"].numpy().ravel(),
        since_reappear=np.where(reappear >= 0, t - reappear, MISSING),
        tuple_id=np.repeat(batch["tuple_id"].numpy(), steps),
        tuple_role=np.repeat([ROLE_NAMES[i] for i in batch["tuple_role"].numpy()], steps),
        surprise_type=np.repeat([SURPRISE_NAMES[i] for i in batch["surprise_type"].numpy()], steps),
        surprise_frame=np.repeat(batch["surprise_frame"].numpy(), steps))


@torch.no_grad()
def extract_frames(model: torch.nn.Module, dataset: Dataset, start: int, count: int, device: torch.device,
                   probe: PositionProbe | None = None, decoder: AmodalDecoder | None = None,
                   control: tuple[torch.nn.Module, PositionProbe | None, AmodalDecoder | None] | None = None,
                   batch_size: int = 16, workers: int = 0) -> pd.DataFrame:
    """One row per frame of sequences [start, start + count) of a dataset that includes the amodal frames, with every
    column described in the module docstring. `control` is the random initialization model with its probe and
    decoder, read the same way (random_probe_*, random_decoder_*)."""
    model.eval()
    parts, first = [], start
    for cpu_batch in make_loader(dataset, batch_size, start, count, workers):
        rows = describe(cpu_batch, first)
        batch = prepare_batch(cpu_batch, device)
        frames, amodal = batch["frames"], batch["amodal"]
        center = cpu_batch["center"].numpy().reshape(-1, 2)
        bar = bar_mask(frames[:, 0])
        columns, out = readout_columns("", model, frames, center, amodal, probe, decoder)
        if control is not None:
            random_model, random_probe, random_decoder = control
            columns.update(readout_columns("random_", random_model.eval(), frames, center, amodal, random_probe,
                                           random_decoder)[0])
        values: dict[str, torch.Tensor] = {}
        guesses = {"model": out["prediction"], "copy": copy_prediction(frames), "blank": blank_prediction(frames)}
        areas = regions(bar, amodal)
        for who, guess in guesses.items():
            values.update({f"{who}_{name}": v for name, v in frame_scores(guess, frames).items()})
            for area, mask in areas.items():
                values[f"{who}_mse_{area}"] = region_mse(guess, frames, mask)
        values.update({f"pred_{name}": v for name, v in ink_readout(out["prediction"], bar).items()})
        values.update({f"true_{name}": v for name, v in ink_readout(frames, bar).items()})
        if "E" in out:
            values.update(error_activity(out["E"], bar))
        columns.update({name: v.reshape(-1).float().cpu().numpy() for name, v in values.items()})
        parts.append(rows.assign(**{name: np.asarray(v, dtype=np.float32) for name, v in columns.items()}))
        first += len(cpu_batch["state"])
    return pd.concat(parts, ignore_index=True)


def baseline_frames(dataset: Dataset, start: int, count: int, batch_size: int = 16,
                    workers: int = 0) -> pd.DataFrame:
    """The programmed baselines on every frame of sequences [start, start + count) of a dataset that includes the
    amodal frames: the position predicted by every tracker (<tracker>_y, _x), the bar center x, and the correlation
    and IoU of the templates moved by the Kalman filters (peekaboo/probes/amodal.py) against the amodal frame."""
    parts, first = [], start
    for batch in make_loader(dataset, batch_size, start, count, workers):
        size, steps = batch["state"].shape
        columns: dict[str, list[np.ndarray]] = {name: [] for name in (
            [f"{tracker}_{axis}" for tracker in TRACKER_NAMES for axis in ("y", "x")] + ["bar_center_x"]
            + [f"{name}_{score}" for name in TEMPLATES for score in ("corr", "iou")])}
        for frames, center, state, amodal in zip(batch["frames"].numpy(), batch["center"].numpy(),
                                                 batch["state"].numpy(), batch["amodal"].numpy()):
            tracks = run_trackers(frames, center, state)
            for tracker in TRACKER_NAMES:
                for a, axis in enumerate(("y", "x")):
                    columns[f"{tracker}_{axis}"].append(tracks[tracker].prediction[:, a])
            bar = np.flatnonzero(bar_columns(frames[0]))
            columns["bar_center_x"].append(np.full(steps, bar.mean() if bar.size else np.nan))
            detections = detect(frames)
            target = torch.from_numpy(amodal[:, 0].astype(np.float32) / 255.0)
            for name, tracker in TEMPLATES.items():
                image = torch.from_numpy(template_images(frames, detections, tracks[tracker].prediction))
                corr, iou = ink_scores(torch.nan_to_num(image), target)
                missing = torch.isnan(image.flatten(1)).any(dim=1)
                corr[missing], iou[missing] = float("nan"), float("nan")
                columns[f"{name}_corr"].append(corr.numpy())
                columns[f"{name}_iou"].append(iou.numpy())
        parts.append(pd.DataFrame({"sequence": np.repeat(np.arange(first, first + size), steps),
                                   "t": np.tile(np.arange(steps), size),
                                   **{name: np.concatenate(v).astype(np.float32) for name, v in columns.items()}}))
        first += size
    return pd.concat(parts, ignore_index=True)


def part_path(folder: Path, start: int) -> Path:
    """The file of the chunk that starts at sequence `start`."""
    return folder / f"part_{start:06d}.parquet"


def run_chunks(folder: Path, count: int, chunk: int, make_part: Callable[[int, int], pd.DataFrame],
               settings: dict[str, Any], log: Callable[[str], None] = print) -> Path:
    """Write the rows of sequences [0, count) by chunks of `chunk` sequences into `folder`, skipping the chunks
    already written, so that a stopped pass resumes. `settings` describe the pass (checkpoint, set, count, chunk):
    a folder written with other settings is refused. Ends by marking the folder complete in its info.json."""
    folder.mkdir(parents=True, exist_ok=True)
    info_path = folder / "info.json"
    info = {"settings": settings, "complete": False}
    if info_path.exists():
        previous = json.loads(info_path.read_text(encoding="utf-8"))
        if previous["settings"] != settings:
            raise ValueError(f"{folder} was written with other settings ({previous['settings']}); use --overwrite")
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    starts = list(range(0, count, chunk))
    todo = [s for s in starts if not part_path(folder, s).exists()]
    log(f"{len(starts) - len(todo)} of {len(starts)} chunks already written")
    began = time.perf_counter()
    for done, start in enumerate(todo, 1):
        table = make_part(start, min(chunk, count - start))
        partial = part_path(folder, start).with_suffix(".partial")
        table.to_parquet(partial, index=False)
        partial.rename(part_path(folder, start))
        elapsed = time.perf_counter() - began
        log(f"chunk {starts.index(start) + 1}/{len(starts)} (sequences {start} to {min(start + chunk, count) - 1}): "
            f"{elapsed / 60:.1f} min, about {elapsed / done * (len(todo) - done) / 60:.0f} min left")
    info["complete"] = True
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return folder


def is_complete(folder: Path) -> bool:
    """Whether a pass wrote every chunk of its folder."""
    info_path = folder / "info.json"
    return info_path.exists() and json.loads(info_path.read_text(encoding="utf-8"))["complete"]


def chunks_written(folder: Path) -> int:
    """Number of chunks already in a folder."""
    return len(list(folder.glob("part_*.parquet"))) if folder.exists() else 0


def pass_format(folder: Path) -> int | None:
    """The column version a pass was written with, or None without an info.json."""
    info_path = folder / "info.json"
    return json.loads(info_path.read_text(encoding="utf-8"))["settings"].get("format") if info_path.exists() else None


def read_frames(folder: Path, columns: list[str] | None = None, filters: list[tuple] | None = None) -> pd.DataFrame:
    """The rows of a complete pass, in sequence order: every column, or only `columns`, and only the rows that pass
    the parquet `filters` (for example [("condition", "in", ["occlusion"])])."""
    if not is_complete(folder):
        raise FileNotFoundError(f"{folder} is not a complete extraction")
    return pd.concat([pd.read_parquet(p, columns=columns, filters=filters)
                      for p in sorted(folder.glob("part_*.parquet"))], ignore_index=True)
