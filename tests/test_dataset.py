"""Tests for the on the fly dataset, the index sampler, and stored datasets."""

from pathlib import Path

import numpy as np
import pytest
import torch

from peekaboo.config import load_config
from peekaboo.data.conditions import GeneratorSettings
from peekaboo.data.dataset import (CONDITION_NAMES, IndexRangeSampler, OnTheFlyDataset, make_loader,
                                   prepare_batch)
from peekaboo.data.mnist_pool import DigitPool
from peekaboo.data.render import RenderSettings, render_sequence
from peekaboo.data.splicing import sample_surprise_tuple
from peekaboo.data.store import StoredDataset, write_dataset
from peekaboo.paths import CONFIGS_DIR

RENDER = RenderSettings.from_config(load_config(CONFIGS_DIR / "data" / "base.yaml"))


def assert_items_equal(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> None:
    """Check that two samples have the same keys and identical tensors (NaNs compared as equal)."""
    assert a.keys() == b.keys()
    for key in a:
        assert a[key].dtype == b[key].dtype, key
        assert torch.equal(a[key], b[key]) or torch.allclose(a[key], b[key], equal_nan=True), key


@pytest.fixture(scope="module")
def dataset(pool: DigitPool, settings: GeneratorSettings) -> OnTheFlyDataset:
    """An on the fly training stream over the synthetic pool."""
    return OnTheFlyDataset(pool, settings, RENDER, "train", base_seed=0, include_amodal=True)


def test_item_shapes_and_types(dataset: OnTheFlyDataset, settings: GeneratorSettings) -> None:
    """A sample has uint8 frames (T, 3, H, W), float truth, and integer event scalars."""
    item = dataset[5]
    cs = settings.crossing
    assert item["frames"].shape == (cs.seq_len, 3, cs.frame_height, cs.frame_width)
    assert item["frames"].dtype == torch.uint8 and item["amodal"].shape == (cs.seq_len, 1, cs.frame_height,
                                                                            cs.frame_width)
    assert item["center"].shape == (cs.seq_len, 2) and item["center"].dtype == torch.float32
    assert item["state"].dtype == torch.int64 and item["in_window"].dtype == torch.bool
    assert item["index"].item() == 5
    assert CONDITION_NAMES[item["condition"].item()] in settings.train_mix


def test_items_are_deterministic_and_match_the_renderer(dataset: OnTheFlyDataset,
                                                        settings: GeneratorSettings, pool: DigitPool) -> None:
    """The same index gives the same sample, other indices differ, and frames equal a direct rendering."""
    assert_items_equal(dataset[3], dataset[3])
    assert not torch.equal(dataset[3]["frames"], dataset[4]["frames"])
    spec = dataset.spec(3)
    rendered = render_sequence(spec, pool.sprite(spec.digit_index), RENDER, settings.crossing.thresholds)
    assert np.array_equal(dataset[3]["frames"].numpy(), rendered.observed.transpose(0, 3, 1, 2))


def test_fixed_cell_stream(pool: DigitPool, settings: GeneratorSettings) -> None:
    """Fixing condition, k, and speed gives only that cell."""
    stream = OnTheFlyDataset(pool, settings, RENDER, "test", 0, condition="occlusion", k=6, speed=3)
    for i in range(4):
        item = stream[i]
        assert CONDITION_NAMES[item["condition"].item()] == "occlusion"
        assert (item["k_target"].item(), item["speed"].item()) == (6, 3)


def test_index_range_sampler() -> None:
    """The sampler visits a contiguous range of indices in order."""
    sampler = IndexRangeSampler(start=40, count=5)
    assert list(sampler) == [40, 41, 42, 43, 44] and len(sampler) == 5


def test_spawn_workers_give_the_same_batches(dataset: OnTheFlyDataset) -> None:
    """Two spawn workers produce exactly the batches of the main process (spawn safe and deterministic)."""
    serial = list(make_loader(dataset, batch_size=3, start=100, count=6, num_workers=0))
    parallel = list(make_loader(dataset, batch_size=3, start=100, count=6, num_workers=2))
    assert len(serial) == len(parallel) == 2
    for a, b in zip(serial, parallel):
        assert_items_equal(a, b)


def test_prepare_batch_scales_frames(dataset: OnTheFlyDataset) -> None:
    """prepare_batch moves the batch and turns frames into floats in [0, 1]."""
    batch = next(iter(make_loader(dataset, batch_size=2, start=0, count=2)))
    moved = prepare_batch(batch, torch.device("cpu"))
    assert moved["frames"].dtype == torch.float32
    assert 0.0 <= moved["frames"].min() and moved["frames"].max() <= 1.0
    assert torch.equal(moved["frames"] * 255, batch["frames"].float())


def test_stored_dataset_round_trip(tmp_path: Path, dataset: OnTheFlyDataset, pool: DigitPool,
                                   settings: GeneratorSettings) -> None:
    """A written dataset reads back to exactly the samples of the on the fly stream, and verifies its checksums."""
    specs = [dataset.spec(i) for i in range(8)]
    specs += sample_surprise_tuple(pool, settings, "test", 0, 0, "vanish", 6, 2).specs()
    folder = write_dataset(tmp_path / "demo", specs, pool, RENDER, settings.crossing.thresholds,
                           info={"name": "demo", "base_seed": 0})
    stored = StoredDataset(folder, include_amodal=True)
    assert len(stored) == len(specs) == len(stored.metadata)
    assert stored.info["name"] == "demo" and stored.info["n_sequences"] == len(specs)
    for i in range(8):
        assert_items_equal(stored[i], dataset[i])
    assert all(stored.verify(i) for i in range(len(stored)))
    assert set(stored.metadata["tuple_role"]) == {"none", "A", "B", "AB", "BA"}
    assert not (tmp_path / "demo.partial").exists()


def test_write_refuses_to_overwrite(tmp_path: Path, dataset: OnTheFlyDataset, pool: DigitPool,
                                    settings: GeneratorSettings) -> None:
    """An existing dataset is only replaced when asked."""
    specs = [dataset.spec(0)]
    write_dataset(tmp_path / "demo", specs, pool, RENDER, settings.crossing.thresholds, info={})
    with pytest.raises(FileExistsError):
        write_dataset(tmp_path / "demo", specs, pool, RENDER, settings.crossing.thresholds, info={})
    write_dataset(tmp_path / "demo", specs, pool, RENDER, settings.crossing.thresholds, info={}, overwrite=True)
