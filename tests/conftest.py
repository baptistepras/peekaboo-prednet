"""Shared test fixtures: generator settings from the base config and a small synthetic digit pool."""

import numpy as np
import pytest

from peekaboo.config import load_config
from peekaboo.data.conditions import GeneratorSettings
from peekaboo.data.mnist_pool import DigitPool, build_pool_from_images
from peekaboo.paths import CONFIGS_DIR


@pytest.fixture(scope="session")
def settings() -> GeneratorSettings:
    """Generator settings from configs/data/base.yaml."""
    return GeneratorSettings.from_config(load_config(CONFIGS_DIR / "data" / "base.yaml"))


@pytest.fixture(scope="session")
def pool() -> DigitPool:
    """A pool of 40 digit like sprites, 6 to 14 px wide, so tests do not need MNIST."""
    rng = np.random.default_rng(0)
    n = 40
    images = np.zeros((n, 28, 28), dtype=np.uint8)
    for i in range(n):
        h, w = int(rng.integers(10, 17)), int(rng.integers(6, 15))
        top, left = int(rng.integers(0, 28 - h)), int(rng.integers(0, 28 - w))
        block = (rng.random((h, w)) < 0.5) * rng.integers(60, 256, size=(h, w))
        block[:, 0] = block[:, -1] = 200  # solid edge columns
        images[i, top:top + h, left:left + w] = block
    labels = rng.integers(0, 10, size=n)
    return build_pool_from_images(images, labels, np.arange(n), "test", box_size=28)
