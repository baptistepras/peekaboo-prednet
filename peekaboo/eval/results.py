"""The results of Phase 3 in one place (step 3.8), as scripts/make_results.py writes them.

- headline(folder, models): the main number of every research question for every model of a model set, read from
  the tables that the analyses wrote in runs/analysis/<set>/ (or runs/analysis/<set>_<models>/; nothing is
  recomputed), with the trackers where they apply.
- seed_table(runs_dir, data, key): every seed of a model (the reference model by default), read from its runs:
  training (best step, validation MSE), the gate D16, and on the set, the probe and the decoder on the hidden frames
  of the occlusion cells and the probe on the blackout frames in open and closed loop. These are plain means over
  sequences, one seed at a time, computed as the analyses compute them; the analyses give the same numbers over the
  seeds together.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from peekaboo.eval.extract import is_complete, read_frames
from peekaboo.eval.occlusion import event_rows, hidden
from peekaboo.eval.tables import frames_folder, loop_folder, per_sequence
from peekaboo.train.sweep import DATA, FAILED, MODEL_SETS, REFERENCE, find_run


@dataclass(frozen=True)
class Number:
    """One headline number: a column of a table of the analyses, on the rows that match `where` and the line `who`."""

    label: str
    table: str                       # path relative to runs/analysis/<set>/
    where: dict[str, str]
    value: str = "mean"
    interval: tuple[str, str] | None = ("low", "high")
    digits: int = 2
    percent: bool = False


HEADLINE = (
    Number("hidden digit, probe |error| on y (px)", "occlusion/hidden.csv",
           {"condition": "occlusion", "k": "all", "readout": "probe_y"}),
    Number("hidden bounce, probe |error| on y (px)", "occlusion/hidden.csv",
           {"condition": "hidden_bounce", "k": "all", "readout": "probe_y"}),
    Number("slope over 12 hidden frames (px/frame)", "occlusion/decay.csv",
           {"condition": "occlusion", "k": "12", "readout": "probe_y"}, "slope_mean", ("slope_low", "slope_high"), 3),
    Number("digit drawn again on time", "occlusion/timing.csv", {"condition": "occlusion", "k": "all"}, "on_time",
           None, percent=True),
    Number("after direction: drawn digit latency, AB - B (frames)", "correction/latency_paired.csv",
           {"event": "direction", "role": "AB", "scope": "all", "readout": "readout", "against": "B",
            "measure": "latency"}),
    Number("visible frames probe, hidden digit (px)", "layers/readouts.csv",
           {"condition": "occlusion", "frames": "hidden", "readout": "visible", "measure": "probe_y"}),
    Number("blackout, open loop probe (px)", "closed_loop/summary.csv",
           {"mode": "blackout", "frames": "hidden", "k": "all", "loop": "open", "measure": "probe_y"}),
    Number("blackout, closed loop probe (px)", "closed_loop/summary.csv",
           {"mode": "blackout", "frames": "hidden", "k": "all", "loop": "closed", "measure": "probe_y"}),
)
SURPRISE = ("surprise/index.csv", {"signal": "mse", "scope": "all"})  # the lowest AUROC over the surprise kinds
REFERENCES = ("Kalman",)  # the trackers shown next to the models


def pick(table: pd.DataFrame, where: dict[str, str], who: str) -> pd.Series | None:
    """The one row of a table with these values (compared as text, so that k = 12 matches "12") and this line."""
    keep = table["who"].astype(str) == who
    for column, value in where.items():
        if column not in table:
            return None
        keep &= table[column].fillna("").astype(str) == value
    rows = table[keep]
    if len(rows) > 1:
        raise ValueError(f"{len(rows)} rows of the table match {where} for {who}")
    return rows.iloc[0] if len(rows) else None


def show(row: pd.Series | None, number: Number) -> str:
    """A number as text: the value, with its interval when it has one; empty when there is no row."""
    if row is None or pd.isna(row[number.value]):
        return ""
    if number.percent:
        return f"{100 * row[number.value]:.0f}%"
    text = f"{row[number.value]:.{number.digits}f}"
    if number.interval:
        low, high = number.interval
        text += f" [{row[low]:.{number.digits}f}, {row[high]:.{number.digits}f}]"
    return text


def lowest_auroc(table: pd.DataFrame, who: str) -> str:
    """The lowest AUROC of the next frame MSE over the surprise kinds and pairs, with the kind and its interval."""
    _, where = SURPRISE
    keep = table["who"] == who
    for column, value in where.items():
        keep &= table[column].astype(str) == value
    rows = table[keep & table["auroc"].notna()]
    if not len(rows):
        return ""
    row = rows.loc[rows["auroc"].idxmin()]
    return f"{row['auroc']:.2f} [{row['auroc_low']:.2f}, {row['auroc_high']:.2f}] ({row['event']} {row['pair']})"


def headline(folder: Path, models: str = "sweep") -> pd.DataFrame:
    """One row per model of the set, in its order, then the trackers; one column per headline number. A table the
    analyses have not written yet leaves its column empty."""
    tables = {}
    for name in {number.table for number in HEADLINE} | {SURPRISE[0]}:
        path = folder / name
        tables[name] = pd.read_csv(path) if path.exists() else None
    rows = []
    for who in [run.label for run in MODEL_SETS[models]] + list(REFERENCES):
        row = {"model": who}
        for number in HEADLINE:
            table = tables[number.table]
            row[number.label] = "" if table is None else show(pick(table, number.where, who), number)
        surprise = tables[SURPRISE[0]]
        row["surprise: lowest AUROC of the next frame MSE"] = "" if surprise is None else lowest_auroc(surprise, who)
        rows.append(row)
    return pd.DataFrame(rows)


def seed_numbers(frames: pd.DataFrame, blackout: pd.DataFrame | None) -> dict[str, float]:
    """The numbers of one seed on a set: the probe's error on y and the decoder's correlation on the hidden frames of
    the occlusion cells (from the rows of its pass), and the probe's error on y on the blackout frames in open and
    closed loop (from the rows of its closed loop pass on the blackout cells, if any); means over sequences."""
    rows = hidden(event_rows(frames, "occlusion")).assign(probe_err_y=lambda r: (r["probe_y"] - r["y"]).abs())
    out = {"hidden probe |error| on y (px)": float(per_sequence(rows, "probe_err_y").mean(axis=1).iloc[0]),
           "hidden decoder correlation": float(per_sequence(rows, "decoder_corr").mean(axis=1).iloc[0])}
    if blackout is not None:
        closed = blackout[blackout["closed"]]
        for loop in ("open", "closed"):
            errors = closed.assign(error=(closed[f"{loop}_probe_y"] - closed["y"]).abs())
            out[f"blackout, {loop} loop probe (px)"] = float(per_sequence(errors, "error").mean(axis=1).iloc[0])
    return out


FRAME_COLUMNS = ["sequence", "t", "condition", "tuple_role", "in_window", "k", "since_onset", "y", "probe_y",
                 "decoder_corr"]
LOOP_COLUMNS = ["sequence", "t", "closed", "y", "open_probe_y", "closed_probe_y"]


def training_numbers(run_dir: Path) -> dict[str, float]:
    """The best step (the checkpoint kept), the validation MSE there and at the last step, and the gate D16."""
    validation = pd.read_csv(run_dir / "val_metrics.csv")
    best = int(validation["best_step"].iloc[-1])
    at_best = validation.loc[validation["step"] == best, "val_mse"]
    gate = json.loads((run_dir / "eval" / f"next_frame_{DATA}.json").read_text(encoding="utf-8"))["gate"]
    return {"best step": best, "validation MSE, best step": float(at_best.iloc[0]) if len(at_best) else np.nan,
            "validation MSE, last step": float(validation["val_mse"].iloc[-1]),
            "gate D16 (below the better baseline)": gate["mse_reduction"]}


def seed_table(runs_dir: Path, data: str, key: str = REFERENCE) -> pd.DataFrame:
    """One row per seed of a model (a sweep model, or the model of step 4.1) with a complete pass on the set (failed
    runs left out)."""
    run = find_run(key)
    failed = {(r.key, r.seed) for r in FAILED}
    rows = []
    for seed_dir in sorted((runs_dir / key).glob("seed*"), key=lambda p: int(p.name[4:]) if p.name[4:].isdigit()
                           else -1):
        if not seed_dir.name[4:].isdigit() or (key, int(seed_dir.name[4:])) in failed:
            continue
        folder = frames_folder(seed_dir, data)
        if not is_complete(folder):
            continue
        frames = read_frames(folder, FRAME_COLUMNS, [("condition", "==", "occlusion")]).assign(seed=0)
        loops = loop_folder(seed_dir, data, "blackout")
        blackout = (read_frames(loops, LOOP_COLUMNS, [("in_window", "==", True)]).assign(seed=0)
                    if is_complete(loops) else None)
        rows.append({"model": run.label, "seed": int(seed_dir.name[4:]), **training_numbers(seed_dir),
                     **seed_numbers(frames, blackout)})
    return pd.DataFrame(rows)
