"""Seeding helpers. Data generation is fully reproducible; training on GPUs may not be bit exact."""

import hashlib
import random

import numpy as np
import torch


def derive_seed(*keys: int | str) -> int:
    """Map a tuple of keys (for example base seed, split, index) to a stable 63 bit seed.

    Unlike Python's hash(), the result is identical across processes, machines, and runs.
    """
    text = "/".join(str(key) for key in keys)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") >> 1


def make_rng(*keys: int | str) -> np.random.Generator:
    """Return a numpy Generator seeded from the given keys."""
    return np.random.default_rng(derive_seed(*keys))


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, numpy, and torch (all devices). Optionally request deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed % 2**32)  # legacy numpy seeds must fit in 32 bits
    torch.manual_seed(seed)
    if deterministic:
        # warn_only because some MPS and CUDA ops have no deterministic version
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker its own Python and numpy seed, derived from the torch worker seed."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
