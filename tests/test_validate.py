"""Tests for dataset validation: clean data passes, and each kind of corruption is caught."""

import dataclasses
from pathlib import Path

import pytest

from peekaboo.config import load_config
from peekaboo.data.conditions import GeneratorSettings, sample_spec
from peekaboo.data.mnist_pool import DigitPool
from peekaboo.data.render import RenderSettings
from peekaboo.data.spec import SequenceSpec
from peekaboo.data.splicing import sample_surprise_tuple
from peekaboo.data.store import StoredDataset, write_dataset
from peekaboo.data.validate import MNIST_TRAIN_SIZE, Validator, training_digits, validate_stored
from peekaboo.paths import CONFIGS_DIR

RENDER = RenderSettings.from_config(load_config(CONFIGS_DIR / "data" / "base.yaml"))


@pytest.fixture(scope="module")
def specs(pool: DigitPool, settings: GeneratorSettings) -> list[SequenceSpec]:
    """A small mix: training stream sequences, a blackout, and three complete surprise tuples."""
    out = [sample_spec(pool, settings, "test", i, 0) for i in range(10)]
    out.append(sample_spec(pool, settings, "test", 0, 0, condition="blackout"))
    for kind, speed in (("direction", 3), ("vanish", 2), ("speed_fast", 2)):
        out += sample_surprise_tuple(pool, settings, "test", 0, 0, kind, 6, speed).specs()
    return out


def run(specs: list[SequenceSpec], pool: DigitPool, settings: GeneratorSettings,
        forbidden: set[int] | None = None) -> dict:
    """Validate specs without stored data and return the report as a dictionary."""
    validator = Validator(settings, RENDER, forbidden)
    for i, spec in enumerate(specs):
        validator.add(i, spec, pool.sprite(spec.digit_index))
    return validator.finish().to_dict()


def test_clean_stored_dataset_passes(tmp_path: Path, specs: list[SequenceSpec], pool: DigitPool,
                                     settings: GeneratorSettings) -> None:
    """A freshly written dataset passes every check, and every check that applies is exercised."""
    folder = write_dataset(tmp_path / "set", specs, pool, RENDER, settings.crossing.thresholds, info={})
    report = validate_stored(StoredDataset(folder), settings, forbidden_digits=set())
    summary = report.to_dict()
    assert report.ok, summary["checks"]
    for name in ("render_matches_truth", "colors", "window_clean", "exact_k", "clean_motion", "splice_hidden",
                 "splice_identical", "holdout", "stored_truth", "checksums"):
        assert summary["checks"][name]["checked"] > 0, name
    assert summary["checks"]["splice_identical"]["checked"] == 3


def test_wrong_k_is_caught(specs: list[SequenceSpec], pool: DigitPool, settings: GeneratorSettings) -> None:
    """A spec that claims one hidden frame too many fails exact_k."""
    i = next(i for i, s in enumerate(specs) if s.condition == "occlusion")
    bad = list(specs)
    bad[i] = dataclasses.replace(specs[i], k_target=specs[i].k_target + 1)
    report = run(bad, pool, settings)
    assert report["checks"]["exact_k"]["first_failures"] == [i]


def test_broken_splice_is_caught(specs: list[SequenceSpec], pool: DigitPool, settings: GeneratorSettings) -> None:
    """If B moves on a frame where it is fully visible, AB no longer ends with B's frames and the splice check fails."""
    i = next(i for i, s in enumerate(specs) if s.tuple_role == "B" and s.digit_present.all())
    t = specs[i].exit_frame + 1  # fully visible by construction, and after the splice
    positions = specs[i].positions.copy()
    positions[t, 0] += 1 if positions[t, 0] == 0 else -1
    bad = list(specs)
    bad[i] = dataclasses.replace(specs[i], positions=positions)
    report = run(bad, pool, settings)
    assert report["checks"]["splice_identical"]["failed"] == 1


def test_incomplete_tuple_is_caught(specs: list[SequenceSpec], pool: DigitPool, settings: GeneratorSettings) -> None:
    """A tuple missing one of its four sequences counts as a failed splice."""
    i = next(i for i, s in enumerate(specs) if s.tuple_role == "BA")
    report = run(specs[:i] + specs[i + 1:], pool, settings)
    assert report["checks"]["splice_identical"]["failed"] == 1


def test_training_digit_is_caught(specs: list[SequenceSpec], pool: DigitPool, settings: GeneratorSettings) -> None:
    """A sequence using a forbidden digit fails the holdout check."""
    report = run(specs, pool, settings, forbidden={specs[0].mnist_index})
    assert 0 in report["checks"]["holdout"]["first_failures"]


def test_changed_checksum_is_caught(specs: list[SequenceSpec], pool: DigitPool, settings: GeneratorSettings) -> None:
    """Frames that do not match their stored checksums fail."""
    validator = Validator(settings, RENDER)
    validator.add(0, specs[0], pool.sprite(specs[0].digit_index), stored_checksums=(0, 0))
    assert validator.finish().to_dict()["checks"]["checksums"]["failed"] == 1


def test_training_digits_are_the_train_split() -> None:
    """The forbidden set is the train split: 55000 digits with the default holdout of 5000."""
    digits = training_digits(5000, 0)
    assert len(digits) == MNIST_TRAIN_SIZE - 5000
