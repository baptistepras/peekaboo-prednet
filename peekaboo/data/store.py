"""Stored datasets (validation and test sets) as small folders that render back to pixel identical frames.

A dataset folder holds:
- specs.jsonl: one SequenceSpec per line, in dataset order;
- sprites.npz: the sprites of the digits these specs use, so MNIST is not needed to read the set;
- truth.npz: the exact ground truth of every frame, stacked over sequences;
- metadata.parquet: one row per sequence (spec fields, summary counts, and frame checksums);
- info.yaml: format version, colors, thresholds, and how the set was built.
Frames are not stored: rendering is exact and takes about a millisecond per sequence, and the checksums prove that
the frames rendered today are the frames rendered when the set was written.
"""

import json
import shutil
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from peekaboo import __version__
from peekaboo.config import load_config, save_config
from peekaboo.data.dataset import TRUTH_KEYS, make_item, truth_arrays
from peekaboo.data.mnist_pool import DigitPool
from peekaboo.data.occluder import VisibilityThresholds
from peekaboo.data.render import RenderSettings, compose_observed, render_amodal
from peekaboo.data.spec import SequenceSpec
from peekaboo.data.truth import compute_truth, sequence_summary

FORMAT_VERSION = 1


def frames_checksum(frames: np.ndarray) -> int:
    """Return a CRC32 checksum of a frame array, to detect any change in rendering."""
    return zlib.crc32(np.ascontiguousarray(frames).tobytes())


def write_dataset(directory: str | Path, specs: list[SequenceSpec], pool: DigitPool, render_settings: RenderSettings,
                  thresholds: VisibilityThresholds, info: dict[str, Any], overwrite: bool = False) -> Path:
    """Write a stored dataset. The folder is built next to its destination, then moved in place when complete."""
    directory = Path(directory)
    if directory.exists() and not overwrite:
        raise FileExistsError(f"{directory} already exists. Pass overwrite=True to replace it.")
    if not specs:
        raise ValueError("A dataset needs at least one sequence.")
    staging = directory.with_name(directory.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    rows, truths = [], {key: [] for key in TRUTH_KEYS}
    with open(staging / "specs.jsonl", "w", encoding="utf-8") as handle:
        for spec in specs:
            handle.write(json.dumps(spec.to_dict()) + "\n")
            sprite = pool.sprite(spec.digit_index)
            amodal = render_amodal(spec, sprite)
            truth = compute_truth(spec, sprite, thresholds)
            for key, value in truth_arrays(truth).items():
                truths[key].append(value)
            row = spec.metadata()
            row.update(sequence_summary(truth))
            row["amodal_crc32"] = frames_checksum(amodal)
            row["observed_crc32"] = frames_checksum(compose_observed(amodal, spec, render_settings))
            rows.append(row)

    digits = np.array(sorted({spec.digit_index for spec in specs}), dtype=np.int64)
    np.savez(staging / "sprites.npz", digit_index=digits, sprites=pool.sprites[digits], heights=pool.heights[digits],
             widths=pool.widths[digits], centroids=pool.centroids[digits], mnist_indices=pool.mnist_indices[digits],
             labels=pool.labels[digits])
    np.savez(staging / "truth.npz", **{key: np.stack(values) for key, values in truths.items()})
    pd.DataFrame(rows).to_parquet(staging / "metadata.parquet", index=False)
    save_config({"format_version": FORMAT_VERSION, "n_sequences": len(specs),
                 "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "peekaboo_version": __version__,
                 "render": {"digit_rgb": list(render_settings.digit_rgb), "bar_value": render_settings.bar_value},
                 "thresholds": {"occluded_max": thresholds.occluded_max, "visible_min": thresholds.visible_min},
                 **info}, staging / "info.yaml")

    if directory.exists():
        shutil.rmtree(directory)
    staging.rename(directory)
    return directory


class StoredDataset(Dataset):
    """A stored dataset, returning the same samples as OnTheFlyDataset for the same specs."""

    def __init__(self, directory: str | Path, render_settings: RenderSettings | None = None,
                 include_amodal: bool = False) -> None:
        """Load specs, sprites, ground truth, and metadata; frames are rendered when a sample is requested."""
        self.directory = Path(directory)
        self.info = load_config(self.directory / "info.yaml")
        if self.info["format_version"] != FORMAT_VERSION:
            raise ValueError(f"{self.directory} has format {self.info['format_version']}, expected {FORMAT_VERSION}.")
        with open(self.directory / "specs.jsonl", encoding="utf-8") as handle:
            self.specs = [SequenceSpec.from_dict(json.loads(line)) for line in handle]
        with np.load(self.directory / "sprites.npz") as data:
            self.sprites = {int(d): data["sprites"][i, :data["heights"][i], :data["widths"][i]].copy()
                            for i, d in enumerate(data["digit_index"])}
        with np.load(self.directory / "truth.npz") as data:
            self.truth = {key: data[key] for key in TRUTH_KEYS}
        self.metadata = pd.read_parquet(self.directory / "metadata.parquet")
        render = self.info["render"]
        self.render_settings = render_settings or RenderSettings(digit_rgb=tuple(render["digit_rgb"]),
                                                                 bar_value=int(render["bar_value"]))
        self.include_amodal = include_amodal

    def __len__(self) -> int:
        """Number of sequences."""
        return len(self.specs)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        """Render sequence i and attach its stored ground truth."""
        spec = self.specs[i]
        amodal = render_amodal(spec, self.sprites[spec.digit_index])
        truth = {key: values[i] for key, values in self.truth.items()}
        return make_item(spec, amodal, self.render_settings, truth, self.include_amodal)

    def verify(self, i: int) -> bool:
        """Check that sequence i renders to exactly the frames it had when the dataset was written."""
        spec = self.specs[i]
        amodal = render_amodal(spec, self.sprites[spec.digit_index])
        row = self.metadata.iloc[i]
        return (frames_checksum(amodal) == row["amodal_crc32"]
                and frames_checksum(compose_observed(amodal, spec, self.render_settings)) == row["observed_crc32"])
