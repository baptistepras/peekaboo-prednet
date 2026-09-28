"""MNIST digit pool: fixed train, val, and test splits, resized digits, and tight ink sprites for the generator."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from peekaboo.paths import MNIST_DIR
from peekaboo.seeding import make_rng

SPLITS = ("train", "val", "test")


def split_indices(split: str, n_images: int, val_size: int = 5000, holdout_seed: int = 0) -> np.ndarray:
    """Return the sorted MNIST indices of a split.

    Val is a fixed random holdout of the MNIST train file, train is the rest of that file, and test is the whole MNIST test file.
    """
    if split == "test":
        return np.arange(n_images)
    if split not in ("train", "val"):
        raise ValueError(f"Unknown split '{split}', expected one of {SPLITS}.")
    if not 0 < val_size < n_images:
        raise ValueError(f"val_size must be between 1 and {n_images - 1}, got {val_size}.")
    permutation = make_rng("mnist_holdout", holdout_seed).permutation(n_images)
    chosen = permutation[:val_size] if split == "val" else permutation[val_size:]
    return np.sort(chosen)


def prepare_sprite(image: np.ndarray, box_size: int = 20, ink_threshold: float = 0.1) -> np.ndarray:
    """Resize a 28x28 MNIST image to box_size, set pixels below the ink threshold to zero, and crop to the ink.

    After this step every nonzero pixel counts as ink, so ink masks and visible fractions are exact.
    """
    if image.dtype != np.uint8 or image.ndim != 2:
        raise ValueError("Expected a 2D uint8 image.")
    if image.shape != (box_size, box_size):
        image = cv2.resize(image, (box_size, box_size), interpolation=cv2.INTER_AREA)
    sprite = image.copy()
    sprite[sprite < ink_threshold * 255] = 0
    rows = np.flatnonzero(sprite.any(axis=1))
    cols = np.flatnonzero(sprite.any(axis=0))
    if rows.size == 0:
        raise ValueError("Image has no pixel above the ink threshold.")
    return sprite[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def sprite_centroid(sprite: np.ndarray) -> tuple[float, float]:
    """Return the intensity weighted (y, x) centroid of a sprite, in pixels from its top left corner."""
    weights = sprite.astype(np.float64)
    total = weights.sum()
    ys, xs = np.indices(sprite.shape)
    return float((ys * weights).sum() / total), float((xs * weights).sum() / total)


@dataclass(frozen=True)
class DigitPool:
    """All digit sprites of one split, padded into one array. Sprite i is sprites[i, :heights[i], :widths[i]]."""

    split: str
    box_size: int
    ink_threshold: float
    mnist_indices: np.ndarray  # row in the MNIST train file (train, val) or test file (test)
    labels: np.ndarray
    sprites: np.ndarray        # (n, box_size, box_size) uint8, sprite at the top left
    heights: np.ndarray        # ink height in pixels
    widths: np.ndarray         # ink width in pixels
    ink_sums: np.ndarray       # total ink, in units of full intensity pixels
    centroids: np.ndarray      # (n, 2) intensity weighted (y, x) inside the sprite

    def __len__(self) -> int:
        """Number of digits in the pool."""
        return len(self.labels)

    def sprite(self, i: int) -> np.ndarray:
        """Return the tight ink sprite of digit i as a uint8 array."""
        return self.sprites[i, :self.heights[i], :self.widths[i]]

    def sample(self, rng: np.random.Generator) -> int:
        """Draw a digit uniformly and return its position in the pool."""
        return int(rng.integers(len(self)))


def build_pool_from_images(images: np.ndarray, labels: np.ndarray, indices: np.ndarray, split: str,
                           box_size: int = 20, ink_threshold: float = 0.1) -> DigitPool:
    """Turn the selected MNIST images into a DigitPool. Works on any uint8 image stack, which keeps it testable."""
    n = len(indices)
    sprites = np.zeros((n, box_size, box_size), dtype=np.uint8)
    heights = np.zeros(n, dtype=np.int64)
    widths = np.zeros(n, dtype=np.int64)
    ink_sums = np.zeros(n, dtype=np.float64)
    centroids = np.zeros((n, 2), dtype=np.float64)
    for row, index in enumerate(indices):
        sprite = prepare_sprite(images[index], box_size, ink_threshold)
        h, w = sprite.shape
        sprites[row, :h, :w] = sprite
        heights[row], widths[row] = h, w
        ink_sums[row] = sprite.sum() / 255.0
        centroids[row] = sprite_centroid(sprite)
    return DigitPool(split=split, box_size=box_size, ink_threshold=ink_threshold,
                     mnist_indices=np.asarray(indices, dtype=np.int64), labels=np.asarray(labels)[indices],
                     sprites=sprites, heights=heights, widths=widths, ink_sums=ink_sums, centroids=centroids)


def load_mnist(train: bool, root: str | Path = MNIST_DIR, download: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Load the MNIST train or test file through torchvision, as uint8 images (n, 28, 28) and int64 labels."""
    import torchvision  # imported here so the rest of the module does not need it

    dataset = torchvision.datasets.MNIST(root=str(root), train=train, download=download)
    return dataset.data.numpy(), dataset.targets.numpy().astype(np.int64)


def build_digit_pool(split: str, box_size: int = 20, ink_threshold: float = 0.1, val_size: int = 5000,
                     holdout_seed: int = 0, root: str | Path = MNIST_DIR, download: bool = True) -> DigitPool:
    """Load MNIST and build the digit pool of one split (train, val, or test)."""
    images, labels = load_mnist(train=split != "test", root=root, download=download)
    indices = split_indices(split, len(images), val_size, holdout_seed)
    return build_pool_from_images(images, labels, indices, split, box_size, ink_threshold)


def pool_summary(pool: DigitPool) -> dict[str, Any]:
    """Return size, label counts, and ink size statistics of a pool, for reports and sanity checks."""
    percentiles = (1, 50, 99)
    return {
        "split": pool.split,
        "n_digits": len(pool),
        "label_counts": np.bincount(pool.labels, minlength=10).tolist(),
        "width_p1_p50_p99": np.percentile(pool.widths, percentiles).tolist(),
        "height_p1_p50_p99": np.percentile(pool.heights, percentiles).tolist(),
        "width_min_max": [int(pool.widths.min()), int(pool.widths.max())],
        "ink_sum_mean": float(pool.ink_sums.mean()),
    }
