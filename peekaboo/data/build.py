"""Build the specs of a validation or test set from a set config (a grid of cells and sizes).

A set config lists plain sequences by condition, k, and speed, surprise tuples by kind, k, and speed, and optionally
a number of sequences drawn from the training mix. Every sequence is determined by the set's base seed and its cell,
so rebuilding a set gives exactly the same specs.
"""

import dataclasses
from collections.abc import Callable, Iterator
from itertools import product
from typing import Any

from peekaboo.data.conditions import GeneratorSettings, sample_spec
from peekaboo.data.mnist_pool import DigitPool
from peekaboo.data.spec import SequenceSpec
from peekaboo.data.splicing import sample_surprise_tuple


def _as_list(value: Any) -> list:
    """Wrap a single value in a list, so a config can write `k: 4` or `k: [4, 8]`."""
    return list(value) if isinstance(value, (list, tuple)) else [value]


def cells(set_config: dict[str, Any]) -> Iterator[tuple[str, str, int | None, int | None, int]]:
    """Yield (family, name, k, speed, n) for every cell: family is "mix", "sequence", or "tuple"."""
    mix = set_config.get("mix")
    if mix:
        yield "mix", "mix", None, None, int(mix["n"])
    for group in set_config.get("sequences", []):
        ks = _as_list(group.get("k", [None]))
        for condition, k, speed in product(_as_list(group["condition"]), ks, _as_list(group["speed"])):
            yield "sequence", condition, k, speed, int(group["n"])
    for group in set_config.get("tuples", []):
        for kind, k, speed in product(_as_list(group["kind"]), _as_list(group["k"]), _as_list(group["speed"])):
            yield "tuple", kind, k, speed, int(group["n"])


def count_sequences(set_config: dict[str, Any]) -> int:
    """Number of sequences a set config produces (a tuple counts as four)."""
    return sum(n * (4 if family == "tuple" else 1) for family, _, _, _, n in cells(set_config))


def build_specs(set_config: dict[str, Any], pool: DigitPool, settings: GeneratorSettings,
                limit_per_cell: int | None = None,
                progress: Callable[[int], Any] | None = None) -> list[SequenceSpec]:
    """Build every spec of a set, cell by cell, with tuple ids numbered across the whole set.

    limit_per_cell caps the size of every cell, for quick trial builds with the same cells. progress, if given, is
    called with the number of specs added after each sequence or tuple.
    """
    name, base_seed = set_config["name"], int(set_config["base_seed"])
    specs: list[SequenceSpec] = []
    next_tuple = 0
    for family, cell_name, k, speed, n in cells(set_config):
        n = min(n, limit_per_cell) if limit_per_cell else n
        stream = f"{name}/{cell_name}/k{k}/v{speed}"  # each cell has its own random stream
        for i in range(n):
            before = len(specs)
            if family == "mix":
                specs.append(sample_spec(pool, settings, f"{name}/mix", i, base_seed))
            elif family == "sequence":
                specs.append(sample_spec(pool, settings, stream, i, base_seed, condition=cell_name, k=k, speed=speed))
            else:
                members = sample_surprise_tuple(pool, settings, stream, i, base_seed, cell_name, k, speed).specs()
                specs += [dataclasses.replace(s, tuple_id=next_tuple) for s in members]
                next_tuple += 1
            if progress is not None:
                progress(len(specs) - before)
    return specs
