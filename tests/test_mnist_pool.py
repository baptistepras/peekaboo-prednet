"""Tests for the MNIST digit pool. Most use synthetic images; one uses real MNIST if it is already downloaded."""

import numpy as np
import pytest

from peekaboo.data.mnist_pool import (build_digit_pool, build_pool_from_images, prepare_sprite, split_indices,
                                      sprite_centroid)
from peekaboo.paths import MNIST_DIR


def synthetic_digit(top: int, left: int, height: int, width: int) -> np.ndarray:
    """Return a 28x28 uint8 image with one bright rectangle and a faint halo around it."""
    image = np.zeros((28, 28), dtype=np.uint8)
    image[max(top - 1, 0):top + height + 1, max(left - 1, 0):left + width + 1] = 10  # below the ink threshold
    image[top:top + height, left:left + width] = 255
    return image


def test_split_indices_partition_and_determinism() -> None:
    """Train and val split the MNIST train file without overlap, the same way every time."""
    train = split_indices("train", 1000, val_size=100, holdout_seed=0)
    val = split_indices("val", 1000, val_size=100, holdout_seed=0)
    assert len(val) == 100 and len(train) == 900
    assert np.intersect1d(train, val).size == 0
    assert np.array_equal(np.sort(np.concatenate([train, val])), np.arange(1000))
    assert np.array_equal(val, split_indices("val", 1000, val_size=100, holdout_seed=0))
    assert not np.array_equal(val, split_indices("val", 1000, val_size=100, holdout_seed=1))
    assert np.array_equal(split_indices("test", 50), np.arange(50))


def test_split_indices_rejects_bad_input() -> None:
    """Unknown splits and impossible holdout sizes raise."""
    with pytest.raises(ValueError):
        split_indices("dev", 1000)
    with pytest.raises(ValueError):
        split_indices("val", 1000, val_size=1000)


def test_prepare_sprite_crops_and_removes_faint_pixels() -> None:
    """Without resizing, the sprite is exactly the bright rectangle and the faint halo is gone."""
    sprite = prepare_sprite(synthetic_digit(5, 8, 12, 6), box_size=28, ink_threshold=0.1)
    assert sprite.shape == (12, 6)
    assert sprite.min() == 255


def test_prepare_sprite_resizes() -> None:
    """Resizing 28 to 20 shrinks the ink by about 20/28 and keeps every pixel either zero or above threshold."""
    sprite = prepare_sprite(synthetic_digit(4, 4, 20, 14), box_size=20, ink_threshold=0.1)
    # 20 rows become about 14.3, plus up to one partially covered row on each side
    assert 14 <= sprite.shape[0] <= 17
    assert sprite.shape[1] <= 20
    assert ((sprite == 0) | (sprite >= 0.1 * 255)).all()


def test_prepare_sprite_empty_raises() -> None:
    """An image with no ink cannot become a sprite."""
    with pytest.raises(ValueError):
        prepare_sprite(np.zeros((28, 28), dtype=np.uint8))


def test_centroid_of_symmetric_block() -> None:
    """The centroid of a uniform block is its geometric center."""
    y, x = sprite_centroid(np.full((5, 3), 200, dtype=np.uint8))
    assert (y, x) == (2.0, 1.0)


def test_pool_from_synthetic_images() -> None:
    """The pool stores each sprite at the top left of its box, with matching sizes, labels, and ink sums."""
    images = np.stack([synthetic_digit(2, 3, 10, 4), synthetic_digit(6, 6, 8, 8), synthetic_digit(1, 1, 5, 5)])
    labels = np.array([7, 1, 3])
    pool = build_pool_from_images(images, labels, np.array([0, 2]), "val", box_size=28)
    assert len(pool) == 2
    assert pool.labels.tolist() == [7, 3]
    assert pool.mnist_indices.tolist() == [0, 2]
    assert pool.sprite(0).shape == (10, 4) and pool.sprite(1).shape == (5, 5)
    assert pool.ink_sums[0] == pytest.approx(40.0)
    assert pool.sprites[0, 10:, :].sum() == 0 and pool.sprites[0, :, 4:].sum() == 0
    rng = np.random.default_rng(0)
    assert all(0 <= pool.sample(rng) < 2 for _ in range(20))


@pytest.mark.skipif(not (MNIST_DIR / "MNIST" / "raw").exists(), reason="MNIST not downloaded yet")
def test_real_mnist_val_pool() -> None:
    """With real MNIST and the default settings, the val pool has 5000 digits that fit in a 20x20 box."""
    pool = build_digit_pool("val", download=False)
    assert len(pool) == 5000
    assert pool.heights.max() <= 20 and pool.widths.max() <= 20
    assert set(np.unique(pool.labels).tolist()) == set(range(10))
