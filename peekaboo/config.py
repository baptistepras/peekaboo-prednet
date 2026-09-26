"""Load, override, and save experiment configs as YAML or JSON dictionaries."""

import copy
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

_YAML_SUFFIXES = (".yaml", ".yml")


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML or JSON config file into a dictionary, chosen by file extension."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix in _YAML_SUFFIXES:
        config = yaml.safe_load(text)
    elif path.suffix == ".json":
        config = json.loads(text)
    else:
        raise ValueError(f"Unsupported config extension '{path.suffix}' for {path}.")
    if not isinstance(config, dict):
        raise ValueError(f"Config {path} must contain a mapping at the top level.")
    return config


def save_config(config: dict[str, Any], path: str | Path) -> Path:
    """Write the config as YAML or JSON (by extension), creating parent folders if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix in _YAML_SUFFIXES:
        text = yaml.safe_dump(config, sort_keys=False)
    elif path.suffix == ".json":
        text = json.dumps(config, indent=2) + "\n"
    else:
        raise ValueError(f"Unsupported config extension '{path.suffix}' for {path}.")
    path.write_text(text, encoding="utf-8")
    return path


def apply_overrides(config: dict[str, Any], overrides: Sequence[str],
                    strict: bool = True) -> dict[str, Any]:
    """Return a copy of the config with "a.b.c=value" overrides applied, values parsed as YAML ("3" gives an int).

    With strict=True, a key that is not already in the config raises a KeyError.
    """
    result = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must look like key.path=value.")
        dotted, raw_value = item.split("=", 1)
        keys = dotted.strip().split(".")
        node = result
        for key in keys[:-1]:
            if key not in node:
                if strict:
                    raise KeyError(f"Unknown config key '{dotted}'.")
                node[key] = {}
            node = node[key]
            if not isinstance(node, dict):
                raise KeyError(f"Config key '{dotted}' goes through a non mapping value.")
        if strict and keys[-1] not in node:
            raise KeyError(f"Unknown config key '{dotted}'.")
        node[keys[-1]] = yaml.safe_load(raw_value)
    return result
