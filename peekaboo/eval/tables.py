"""The extraction tables of step 3.1 as the Phase 3 analyses read them: which runs have a complete pass on a set, their
per frame rows joined with the baselines of the set, and the error of every position readout.

A model is a model of one of the model sets (peekaboo/train/sweep.py MODEL_SETS: the training sweep for Phase 3, by
default), with every seed that has a complete pass; the failed runs kept as records are left out. Rows hold `model`
(the label of the model's first run) and `seed`.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from peekaboo.eval.extract import FRAMES_FORMAT, frame_columns, is_complete, pass_format, read_frames
from peekaboo.train.sweep import FAILED, MODEL_SETS, SweepRun

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


def layers_folder(run_dir: Path, data: str) -> Path:
    """Where the layer pass of a run over a set is written (step 3.5)."""
    return run_dir / "eval" / f"layers_{data}"


def loop_folder(run_dir: Path, data: str, mode: str) -> Path:
    """Where the closed loop pass of a run over the cells of one condition of a set is written (step 3.6)."""
    return run_dir / "eval" / f"loop_{mode}_{data}"


def baselines_folder(runs_dir: Path, data: str) -> Path:
    """Where the baselines of a set are written."""
    return runs_dir / "baselines" / data


def analysis_folder(runs_dir: Path, data: str, models: str = "sweep") -> Path:
    """Where the analyses of a model set on a set are written: runs/analysis/<set>/ for the sweep, as in Phase 3, and
    runs/analysis/<set>_<models>/ for another model set."""
    return runs_dir / "analysis" / (data if models == "sweep" else f"{data}_{models}")


def seed_folders(runs_dir: Path, models: str = "sweep") -> Iterator[tuple[SweepRun, int, Path]]:
    """Every seed folder of every model of the set, in the order of the set, then of the seeds: the model, the seed,
    and the folder. The failed runs kept as records are left out."""
    failed = {(run.key, run.seed) for run in FAILED}
    for run in MODEL_SETS[models]:
        seed_dirs = [p for p in (runs_dir / run.key).glob("seed*") if p.name[4:].isdigit()]
        for seed_dir in sorted(seed_dirs, key=lambda p: int(p.name[4:])):
            if (run.key, int(seed_dir.name[4:])) not in failed:
                yield run, int(seed_dir.name[4:]), seed_dir


def available_runs(runs_dir: Path, data: str, models: str = "sweep") -> list[ModelRun]:
    """Every seed of every model of the set with a complete pass on the set, in the order of the set. A pass written
    with older columns is refused: it is to be written again (scripts/extract.py --overwrite)."""
    runs = []
    for run, seed, seed_dir in seed_folders(runs_dir, models):
        folder = frames_folder(seed_dir, data)
        if not is_complete(folder):
            continue
        if pass_format(folder) != FRAMES_FORMAT:
            raise ValueError(f"{folder} has the columns of format {pass_format(folder)}, not {FRAMES_FORMAT}: "
                             f"write it again with python -m scripts.extract --run {seed_dir} --data {data} "
                             f"--overwrite")
        runs.append(ModelRun(run.key, run.label, seed, folder))
    return runs


def read_runs(runs: list[ModelRun], baselines: Path, conditions: list[str], columns: list[str] | None = None,
              roles: list[str] | None = None, rows: list[list[tuple]] | None = None) -> pd.DataFrame:
    """The rows of the given conditions (and tuple roles, all by default: "none" for the sequences of the cells) for
    every run, with `model` and `seed`, joined with the baselines and with the error of every position readout
    (add_errors). `rows` keeps only the rows that match one of its alternatives, each a list of parquet conditions
    (for example [[("since_onset", "<", 0)], [("in_window", "==", True)]]). Columns a pass does not have (the error
    units of a model without them) are left out of its rows, and NaN in the joined table."""
    base = [("condition", "in", conditions)] + ([("tuple_role", "in", roles)] if roles else [])
    filters = [base + list(alternative) for alternative in rows] if rows else base
    parts = [read_frames(run.folder, None if columns is None else [c for c in columns if c in frame_columns(
        run.folder)], filters).assign(model=run.label, seed=run.seed) for run in runs]
    table = pd.concat(parts, ignore_index=True)
    trackers = read_frames(baselines)
    table = table.merge(trackers[trackers["sequence"].isin(table["sequence"].unique())], on=KEYS, how="left")
    return add_errors(table)


def add_errors(table: pd.DataFrame) -> pd.DataFrame:
    """For every position readout present, its absolute error on y and x against the true center (<name>_err_y,
    _err_x) and its distance to it (<name>_err). Nothing when the true center was not read."""
    columns = {}
    for name, (y, x) in POSITIONS.items():
        if x not in table or "x" not in table:
            continue
        columns[f"{name}_err_x"] = (table[x] - table["x"]).abs()
        if y is not None:
            columns[f"{name}_err_y"] = (table[y] - table["y"]).abs()
            columns[f"{name}_err"] = np.hypot(table[y] - table["y"], table[x] - table["x"])
    return table.assign(**columns)


def reference_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """The rows of the first model's first seed. The references (the trackers, the copy of the last frame) are the same
    for every run of a set, so they are read once: read on every seed of a model, they would count as seeds that agree
    exactly, and their standard error over seeds would be 0."""
    first = rows[rows["model"] == rows["model"].iloc[0]]
    return first[first["seed"] == first["seed"].min()]


def per_sequence(table: pd.DataFrame, column: str) -> pd.DataFrame:
    """One value per seed and sequence, the mean of a column over the rows of each: one row per seed, one column per
    sequence, as peekaboo/eval/stats.py takes them."""
    return table.groupby(["seed", "sequence"])[column].mean().unstack("sequence")
