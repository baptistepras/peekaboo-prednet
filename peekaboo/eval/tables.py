"""The extraction tables of step 3.1 as the Phase 3 analyses read them: which runs have a complete pass on a set, their
per frame rows joined with the baselines of the set, and the error of every position readout.

A model is a run of the training sweep (peekaboo/train/sweep.py), with every seed that has a complete pass; the
failed runs kept as records are left out. Rows hold `model` (the sweep label) and `seed`.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from peekaboo.eval.extract import FRAMES_FORMAT, is_complete, pass_format, read_frames
from peekaboo.train.sweep import FAILED, SWEEP

KEYS = ["sequence", "t"]
# position readouts: the y and x columns of each (None when it gives no y)
POSITIONS = {"probe": ("probe_y", "probe_x"), "random_probe": ("random_probe_y", "random_probe_x"),
             "last_seen": ("last_seen_y", "last_seen_x"), "kalman": ("kalman_y", "kalman_x"),
             "kalman_walls": ("kalman_walls_y", "kalman_walls_x"), "bar_center": (None, "bar_center_x")}


@dataclass(frozen=True)
class ModelRun:
    """A run with a complete pass on a set."""

    key: str
    label: str
    seed: int
    folder: Path  # the pass, <run>/eval/frames_<set>


def frames_folder(run_dir: Path, data: str) -> Path:
    """Where the pass of a run over a set is written."""
    return run_dir / "eval" / f"frames_{data}"


def baselines_folder(runs_dir: Path, data: str) -> Path:
    """Where the baselines of a set are written."""
    return runs_dir / "baselines" / data


def available_runs(runs_dir: Path, data: str) -> list[ModelRun]:
    """Every seed of every sweep model with a complete pass on the set, in the order of the sweep. A pass written
    with older columns is refused: it is to be written again (scripts/extract.py --overwrite)."""
    failed = {(run.key, run.seed) for run in FAILED}
    runs = []
    for run in SWEEP:
        seed_dirs = [p for p in (runs_dir / run.key).glob("seed*") if p.name[4:].isdigit()]
        for seed_dir in sorted(seed_dirs, key=lambda p: int(p.name[4:])):
            seed = int(seed_dir.name[4:])
            folder = frames_folder(seed_dir, data)
            if (run.key, seed) in failed or not is_complete(folder):
                continue
            if pass_format(folder) != FRAMES_FORMAT:
                raise ValueError(f"{folder} has the columns of format {pass_format(folder)}, not {FRAMES_FORMAT}: "
                                 f"write it again with python -m scripts.extract --run {seed_dir} --data {data} "
                                 f"--overwrite")
            runs.append(ModelRun(run.key, run.label, seed, folder))
    return runs


def read_runs(runs: list[ModelRun], baselines: Path, conditions: list[str], columns: list[str] | None = None,
              roles: list[str] | None = None) -> pd.DataFrame:
    """The rows of the given conditions (and tuple roles, all by default: "none" for the sequences of the cells) for
    every run, with `model` and `seed`, joined with the baselines and with the error of every position readout
    (add_errors)."""
    filters = [("condition", "in", conditions)] + ([("tuple_role", "in", roles)] if roles else [])
    parts = [read_frames(run.folder, columns, filters).assign(model=run.label, seed=run.seed) for run in runs]
    table = pd.concat(parts, ignore_index=True)
    trackers = read_frames(baselines)
    table = table.merge(trackers[trackers["sequence"].isin(table["sequence"].unique())], on=KEYS, how="left")
    return add_errors(table)


def add_errors(table: pd.DataFrame) -> pd.DataFrame:
    """For every position readout present, its absolute error on y and x against the true center (<name>_err_y,
    _err_x) and its distance to it (<name>_err)."""
    columns = {}
    for name, (y, x) in POSITIONS.items():
        if x not in table:
            continue
        columns[f"{name}_err_x"] = (table[x] - table["x"]).abs()
        if y is not None:
            columns[f"{name}_err_y"] = (table[y] - table["y"]).abs()
            columns[f"{name}_err"] = np.hypot(table[y] - table["y"], table[x] - table["x"])
    return table.assign(**columns)


def per_sequence(table: pd.DataFrame, column: str) -> pd.DataFrame:
    """One value per seed and sequence, the mean of a column over the rows of each: one row per seed, one column per
    sequence, as peekaboo/eval/stats.py takes them."""
    return table.groupby(["seed", "sequence"])[column].mean().unstack("sequence")
