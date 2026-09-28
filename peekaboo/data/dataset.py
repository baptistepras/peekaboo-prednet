"""PyTorch datasets of occlusion sequences, generated on the fly from their index.

Every sample is a dictionary of tensors with the same keys, whether it comes from on the fly generation or from a
stored dataset (peekaboo.data.store). Frames stay uint8 until prepare_batch moves them to the device, which keeps
worker to main process transfers four times smaller.
"""

from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from peekaboo.data.conditions import CONDITIONS, GeneratorSettings, sample_spec
from peekaboo.data.mnist_pool import DigitPool
from peekaboo.data.render import RenderSettings, compose_observed, render_amodal
from peekaboo.data.spec import SequenceSpec
from peekaboo.data.splicing import SURPRISES
from peekaboo.data.truth import FrameTruth, compute_truth

CONDITION_NAMES = CONDITIONS + ("surprise", "empty")
ROLE_NAMES = ("none", "A", "B", "AB", "BA")
SURPRISE_NAMES = ("none",) + SURPRISES + ("appear",)
EVENT_FIELDS = ("k_target", "speed", "entry_frame", "onset_frame", "expected_reappear_frame",
                "actual_reappear_frame", "exit_frame", "window_start", "window_end", "surprise_frame", "tuple_id")
TRUTH_KEYS = ("center", "modal_center", "velocity", "visible_fraction", "state", "episode", "in_window")


def truth_arrays(truth: FrameTruth) -> dict[str, np.ndarray]:
    """Return the ground truth as compact arrays, the form stored on disk."""
    return {"center": truth.center.astype(np.float32), "modal_center": truth.modal_center.astype(np.float32),
            "velocity": truth.velocity.astype(np.int16), "visible_fraction": truth.visible_fraction.astype(np.float32),
            "state": truth.state.astype(np.int8), "episode": truth.episode.astype(np.int8),
            "in_window": truth.in_window.astype(bool)}


def make_item(spec: SequenceSpec, amodal: np.ndarray, render_settings: RenderSettings,
              truth: dict[str, np.ndarray], include_amodal: bool = False) -> dict[str, torch.Tensor]:
    """Build one sample: observed frames (T, 3, H, W) uint8, ground truth per frame, and the event scalars."""
    observed = compose_observed(amodal, spec, render_settings)
    item = {"frames": torch.from_numpy(np.ascontiguousarray(observed.transpose(0, 3, 1, 2)))}
    if include_amodal:
        item["amodal"] = torch.from_numpy(np.ascontiguousarray(amodal[:, None]))
    item["center"] = torch.from_numpy(truth["center"].astype(np.float32))
    item["modal_center"] = torch.from_numpy(truth["modal_center"].astype(np.float32))
    item["velocity"] = torch.from_numpy(truth["velocity"].astype(np.float32))
    item["visible_fraction"] = torch.from_numpy(truth["visible_fraction"].astype(np.float32))
    item["state"] = torch.from_numpy(truth["state"].astype(np.int64))
    item["episode"] = torch.from_numpy(truth["episode"].astype(np.int64))
    item["in_window"] = torch.from_numpy(truth["in_window"].astype(bool))
    item["index"] = torch.tensor(spec.index, dtype=torch.int64)
    item["condition"] = torch.tensor(CONDITION_NAMES.index(spec.condition), dtype=torch.int64)
    item["tuple_role"] = torch.tensor(ROLE_NAMES.index(spec.tuple_role), dtype=torch.int64)
    item["surprise_type"] = torch.tensor(SURPRISE_NAMES.index(spec.surprise_type), dtype=torch.int64)
    for name in EVENT_FIELDS:
        item[name] = torch.tensor(getattr(spec, name), dtype=torch.int64)
    return item


class OnTheFlyDataset(Dataset):
    """An endless stream of sequences: index i always gives the same sequence, whichever worker builds it.

    Nothing is random outside the index, so no worker seeding is needed and training can resume at any index.
    Fixing condition, k, or speed restricts the stream to that cell; otherwise they follow the training mix.
    """

    def __init__(self, pool: DigitPool, settings: GeneratorSettings, render_settings: RenderSettings, split: str,
                 base_seed: int, condition: str | None = None, k: int | None = None, speed: int | None = None,
                 include_amodal: bool = False, length: int = 2**31 - 1, max_retries: int = 5) -> None:
        """Store the generation settings; nothing is generated until an index is requested."""
        self.pool = pool
        self.settings = settings
        self.render_settings = render_settings
        self.split = split
        self.base_seed = base_seed
        self.condition, self.k, self.speed = condition, k, speed
        self.include_amodal = include_amodal
        self.length = length
        self.max_retries = max_retries

    def __len__(self) -> int:
        """Nominal length; use IndexRangeSampler to choose which indices a loader visits."""
        return self.length

    def spec(self, index: int) -> SequenceSpec:
        """Return the spec of sequence `index`, retrying with a derived stream in the rare case it cannot be placed."""
        for retry in range(self.max_retries):
            split = self.split if retry == 0 else f"{self.split}/retry{retry}"
            try:
                return sample_spec(self.pool, self.settings, split, index, self.base_seed,
                                   condition=self.condition, k=self.k, speed=self.speed)
            except RuntimeError:
                continue
        raise RuntimeError(f"Could not build sequence {index} of '{self.split}' after {self.max_retries} retries.")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Generate, render, and annotate sequence `index`."""
        spec = self.spec(int(index))
        sprite = self.pool.sprite(spec.digit_index)
        truth = truth_arrays(compute_truth(spec, sprite, self.settings.crossing.thresholds))
        return make_item(spec, render_amodal(spec, sprite), self.render_settings, truth, self.include_amodal)


class IndexRangeSampler(Sampler[int]):
    """Visit the indices start, start + 1, ..., start + count - 1 in order.

    With an on the fly dataset, step s of training with batch size b can use start = s * b, so every sample is new,
    the order is reproducible, and a resumed run continues exactly where it stopped.
    """

    def __init__(self, start: int, count: int) -> None:
        """Remember the first index and how many indices to visit."""
        self.start = start
        self.count = count

    def __iter__(self) -> Iterator[int]:
        """Yield the indices in order."""
        return iter(range(self.start, self.start + self.count))

    def __len__(self) -> int:
        """Number of indices visited."""
        return self.count


def make_loader(dataset: Dataset, batch_size: int, start: int, count: int, num_workers: int = 0) -> DataLoader:
    """Build a DataLoader over indices [start, start + count), with spawn workers on every platform.

    Spawn is the default on macOS; using it everywhere means code that works on the Mac also works on the cluster.
    """
    return DataLoader(dataset, batch_size=batch_size, sampler=IndexRangeSampler(start, count),
                      num_workers=num_workers, pin_memory=False, drop_last=False,
                      multiprocessing_context="spawn" if num_workers > 0 else None,
                      persistent_workers=num_workers > 0)


def prepare_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move a batch to the device and turn the uint8 frames into float32 values in [0, 1] there."""
    moved = {name: value.to(device) for name, value in batch.items()}
    moved["frames"] = moved["frames"].float().div_(255.0)
    if "amodal" in moved:
        moved["amodal"] = moved["amodal"].float().div_(255.0)
    return moved
