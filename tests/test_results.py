"""Tests for the results of step 3.8: the headline numbers read from the tables of the analyses, the numbers of one
seed, and the commands that make every result."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from peekaboo.eval.results import HEADLINE, headline, pick, seed_numbers
from peekaboo.train.sweep import BLACKOUT, REFERENCE, SWEEP
from scripts.make_results import ANALYSES, EXAMPLES, commands, seeded

MODEL = SWEEP[0].label


def write(folder: Path, name: str, rows: list[dict]) -> None:
    """A table of the analyses, as they write them."""
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_headline_reads_the_tables(tmp_path: Path) -> None:
    """Each number is the row of its table that matches, with its interval; k = 12 read back as a number matches;
    a share is shown in percent; the lowest AUROC names its kind; a missing table leaves its column empty."""
    interval = {"mean": 0.72, "low": 0.64, "high": 0.80}
    write(tmp_path, "occlusion/hidden.csv", [
        {"condition": "occlusion", "k": "all", "readout": "probe_y", "who": MODEL, **interval},
        {"condition": "occlusion", "k": "all", "readout": "probe_x", "who": MODEL, **interval, "mean": 9.0},
        {"condition": "occlusion", "k": "all", "readout": "probe_y", "who": "Kalman", "mean": 2.38, "low": 2.21,
         "high": 2.54}])
    write(tmp_path, "occlusion/decay.csv", [
        {"condition": "occlusion", "k": 12, "readout": "probe_y", "who": MODEL, "slope_mean": 0.0123,
         "slope_low": 0.001, "slope_high": 0.02},
        {"condition": "occlusion", "k": 2, "readout": "probe_y", "who": MODEL, "slope_mean": 9.0, "slope_low": 0.0,
         "slope_high": 0.0}])
    write(tmp_path, "occlusion/timing.csv", [{"condition": "occlusion", "k": "all", "who": MODEL, "on_time": 0.871}])
    write(tmp_path, "surprise/index.csv", [
        {"event": event, "pair": "AB - B", "scope": "all", "signal": "mse", "who": MODEL, "auroc": auroc,
         "auroc_low": auroc - 0.01, "auroc_high": auroc + 0.01} for event, auroc in (("direction", 1.0),
                                                                                    ("early", 0.95))])
    table = headline(tmp_path).set_index("model")
    assert list(table.index) == [run.label for run in SWEEP] + ["Kalman"]
    labels = [number.label for number in HEADLINE]
    assert table.loc[MODEL, labels[0]] == "0.72 [0.64, 0.80]"
    assert table.loc["Kalman", labels[0]] == "2.38 [2.21, 2.54]"
    assert table.loc[MODEL, labels[2]] == "0.012 [0.001, 0.020]"
    assert table.loc[MODEL, labels[3]] == "87%"
    assert table.loc[MODEL, "surprise: lowest AUROC of the next frame MSE"] == "0.95 [0.94, 0.96] (early AB - B)"
    assert table.loc[MODEL, labels[1]] == "" and table.loc[MODEL, labels[-1]] == ""  # no such row, no such table
    assert table.loc[SWEEP[1].label, labels[0]] == ""
    blackout = headline(tmp_path, "blackout").set_index("model")
    assert list(blackout.index) == [MODEL, BLACKOUT[0].label, "Kalman"]
    assert blackout.loc[MODEL, labels[0]] == "0.72 [0.64, 0.80]" and blackout.loc[BLACKOUT[0].label, labels[0]] == ""


def test_pick_refuses_an_ambiguous_row() -> None:
    """Two matching rows are an error, not a silent choice."""
    table = pd.DataFrame({"who": ["A", "A"], "k": ["all", "all"], "mean": [1.0, 2.0]})
    try:
        pick(table, {"k": "all"}, "A")
    except ValueError:
        return
    raise AssertionError("two rows matched without an error")


def test_seed_numbers() -> None:
    """Means over sequences of the hidden frames of the occlusion cells and of the blackout frames of each loop."""
    rows = []
    for sequence, error in ((0, 1.0), (1, 3.0)):
        t = np.arange(10)
        j = t - 4
        rows.append(pd.DataFrame({
            "sequence": sequence, "t": t, "condition": "occlusion", "tuple_role": "none", "in_window": True, "k": 2,
            "since_onset": j, "y": 30.0, "probe_y": 30.0 + np.where((j >= 0) & (j < 2), error, 9.0),
            "decoder_corr": np.where((j >= 0) & (j < 2), 0.5, 0.9), "seed": 0}))
    frames = pd.concat(rows, ignore_index=True)
    blackout = pd.DataFrame({"sequence": [5, 5, 6, 6], "t": [1, 2, 1, 2], "closed": [False, True, True, True],
                             "y": 30.0, "open_probe_y": [30.0, 38.0, 34.0, 34.0], "closed_probe_y": 31.0, "seed": 0})
    out = seed_numbers(frames, blackout)
    assert np.isclose(out["hidden probe |error| on y (px)"], 2.0)
    assert np.isclose(out["hidden decoder correlation"], 0.5)
    assert np.isclose(out["blackout, open loop probe (px)"], 6.0)
    assert np.isclose(out["blackout, closed loop probe (px)"], 1.0)
    assert "blackout, open loop probe (px)" not in seed_numbers(frames, None)


def test_commands_cover_every_result(tmp_path: Path) -> None:
    """The sweep comparison, every analysis on the set, the examples of every model, and the error maps; every
    command is a module of the project."""
    plan = commands("test_v1", tmp_path)
    assert [c[0] for c in plan["analyses"]] == [f"scripts.analyze_{name}" for name in ANALYSES]
    assert all(c[c.index("--data") + 1] == "test_v1" for c in plan["analyses"] + plan["examples"])
    assert len(plan["examples"]) == len(SWEEP) * len(EXAMPLES) + 2
    assert all(str(tmp_path / "analysis" / "test_v1" / "examples") in c[c.index("--out") + 1]
               for c in plan["examples"])
    for command in plan["sweep"] + plan["analyses"] + plan["examples"]:
        assert importlib.util.find_spec(command[0]) is not None


def test_commands_of_the_blackout_set(tmp_path: Path) -> None:
    """For step 4.1: no sweep comparison, every analysis on this set, the examples of both models in the folder of the
    set, and the seeds of both models in the summary."""
    plan = commands("test_v1", tmp_path, "blackout")
    assert plan["sweep"] == []
    assert all(c[c.index("--models") + 1] == "blackout" for c in plan["analyses"])
    assert len(plan["examples"]) == 2 * len(EXAMPLES)
    assert all(c[0] == "scripts.show_predictions" and str(tmp_path / "analysis" / "test_v1_blackout" / "examples")
               in c[c.index("--out") + 1] for c in plan["examples"])
    assert {c[c.index("--run") + 1] for c in plan["examples"]} == {
        str(tmp_path / key / "seed0") for key in (REFERENCE, BLACKOUT[0].key)}
    assert seeded("sweep") == [REFERENCE] and seeded("blackout") == [REFERENCE, BLACKOUT[0].key]
