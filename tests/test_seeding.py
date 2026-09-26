"""Tests for seed derivation and global seeding."""

import os
import subprocess
import sys

import torch

from peekaboo.paths import PROJECT_ROOT
from peekaboo.seeding import derive_seed, make_rng, seed_everything


def test_derive_seed_is_deterministic_and_key_sensitive() -> None:
    """The same keys give the same seed, and different keys give different seeds."""
    assert derive_seed(0, "train", 5) == derive_seed(0, "train", 5)
    assert derive_seed(0, "train", 5) != derive_seed(0, "train", 6)
    assert derive_seed(0, "train", 5) != derive_seed(0, "test", 5)
    assert 0 <= derive_seed(1, "x") < 2**63


def test_derive_seed_is_stable_across_processes() -> None:
    """The seed does not depend on Python's per process string hash salt."""
    code = "from peekaboo.seeding import derive_seed; print(derive_seed(7, 'val', 42))"
    values = set()
    for hash_seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        out = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, env=env,
                             capture_output=True, text=True, check=True)
        values.add(int(out.stdout.strip()))
    assert values == {derive_seed(7, "val", 42)}


def test_make_rng_reproducible() -> None:
    """Two generators built from the same keys produce identical draws."""
    a = make_rng(3, "train", 0).integers(0, 1000, size=10)
    b = make_rng(3, "train", 0).integers(0, 1000, size=10)
    assert (a == b).all()


def test_seed_everything_reproducible_torch() -> None:
    """Seeding twice with the same value repeats torch CPU random draws."""
    seed_everything(123)
    first = torch.randn(5)
    seed_everything(123)
    second = torch.randn(5)
    assert torch.equal(first, second)
