"""Tests for config loading, saving, and overrides."""

from pathlib import Path

import pytest

from peekaboo.config import apply_overrides, load_config, save_config

EXAMPLE = {"data": {"seq_len": 40, "speeds": [2, 3, 4]}, "seed": 0, "name": "demo"}


@pytest.mark.parametrize("suffix", [".yaml", ".json"])
def test_round_trip(tmp_path: Path, suffix: str) -> None:
    """Saving then loading returns the same dictionary for both formats."""
    path = save_config(EXAMPLE, tmp_path / "sub" / f"config{suffix}")
    assert load_config(path) == EXAMPLE


def test_overrides_parse_types() -> None:
    """Override values are parsed as YAML scalars and lists, and the input is not modified."""
    result = apply_overrides(EXAMPLE, ["data.seq_len=30", "data.speeds=[2, 3]", "name=other"])
    assert result["data"]["seq_len"] == 30
    assert result["data"]["speeds"] == [2, 3]
    assert result["name"] == "other"
    assert EXAMPLE["data"]["seq_len"] == 40


def test_unknown_key_raises_when_strict() -> None:
    """A typo in an override key is caught in strict mode and allowed otherwise."""
    with pytest.raises(KeyError):
        apply_overrides(EXAMPLE, ["data.seq_lenn=30"])
    result = apply_overrides(EXAMPLE, ["extra.value=1"], strict=False)
    assert result["extra"]["value"] == 1


def test_bad_extension_raises(tmp_path: Path) -> None:
    """Only YAML and JSON files are accepted."""
    with pytest.raises(ValueError):
        save_config(EXAMPLE, tmp_path / "config.toml")
