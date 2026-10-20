"""Tests for building validation and test sets from set configs."""

from collections import Counter

from peekaboo.config import load_config
from peekaboo.data.build import build_specs, cells, count_sequences
from peekaboo.data.conditions import GeneratorSettings
from peekaboo.data.mnist_pool import DigitPool
from peekaboo.paths import CONFIGS_DIR

SMALL = {
    "name": "small", "split": "test", "base_seed": 5,
    "mix": {"n": 3},
    "sequences": [{"condition": "occlusion", "k": [2, 6], "speed": 3, "n": 2},
                  {"condition": "control", "speed": [2, 4], "n": 2}],
    "tuples": [{"kind": ["direction", "vanish"], "k": 6, "speed": 2, "n": 2}],
}


def test_cells_expand_the_grid() -> None:
    """Every combination of the listed values is one cell; a tuple counts as four sequences."""
    listed = list(cells(SMALL))
    assert ("sequence", "occlusion", 6, 3, 2) in listed and ("sequence", "control", None, 4, 2) in listed
    assert len(listed) == 1 + 2 + 2 + 2
    assert count_sequences(SMALL) == 3 + 4 + 4 + 2 * 2 * 4


def test_build_specs(pool: DigitPool, settings: GeneratorSettings) -> None:
    """Specs follow their cells, are rebuilt identically, and each tuple id marks exactly four sequences."""
    specs = build_specs(SMALL, pool, settings)
    assert len(specs) == count_sequences(SMALL)
    assert all(a == b for a, b in zip(specs, build_specs(SMALL, pool, settings)))
    occlusion = [s for s in specs if s.condition == "occlusion" and s.tuple_role == "none"]
    assert Counter((s.k_target, s.speed) for s in occlusion)[(6, 3)] >= 2
    ids = Counter(s.tuple_id for s in specs if s.tuple_role != "none")
    assert sorted(ids) == [0, 1, 2, 3] and set(ids.values()) == {4}


def test_limit_per_cell(pool: DigitPool, settings: GeneratorSettings) -> None:
    """A limit caps every cell without changing the cells."""
    specs = build_specs(SMALL, pool, settings, limit_per_cell=1)
    assert len(specs) == 1 + 2 + 2 + 2 * 4


def test_project_set_configs() -> None:
    """The validation set has 1000 sequences and the test set 22200 (11000 plain, 2800 tuples)."""
    assert count_sequences(load_config(CONFIGS_DIR / "data" / "val_v1.yaml")) == 1000
    assert count_sequences(load_config(CONFIGS_DIR / "data" / "test_v1.yaml")) == 22200


def test_dev_set_has_the_cells_of_the_test_set() -> None:
    """The development set has every cell of the test set, a tenth of the size, with held out training digits."""
    dev, test = (load_config(CONFIGS_DIR / "data" / f"{name}.yaml") for name in ("dev_v1", "test_v1"))
    assert [c[:4] for c in cells(dev)] == [c[:4] for c in cells(test)]
    assert all(d[4] * 10 == t[4] for d, t in zip(cells(dev), cells(test)))
    assert count_sequences(dev) == 2220 and dev["split"] == "val"
    assert len({c["base_seed"] for c in (dev, test, load_config(CONFIGS_DIR / "data" / "val_v1.yaml"))}) == 3
