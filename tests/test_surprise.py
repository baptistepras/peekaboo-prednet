"""Tests for the RQ3 analysis on fake extraction tables with known answers: the window start of every tuple, the
surprise index and AUROC of a model that is surprised, of one that is not, and of the copy, the error units of the
models that have them, the vanish pairs, the key contrast, the figures, and the rows read from passes with and
without error units."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from peekaboo.eval.extract import FRAMES_FORMAT, MISSING
from peekaboo.eval.surprise import (AFTER, BEFORE, COLUMNS, CONDITIONS, ROWS, WINDOW, plot_auroc, plot_curves,
                                    plot_layers, surprise, tuple_rows)
from peekaboo.eval.tables import ModelRun, read_runs

STEPS, REAPPEAR = 20, 10
BASE, SURPRISED, BOUNCE = 1e-3, 5e-3, 3e-3  # M1's MSE: everywhere, in the window of AB and BA, after a hidden bounce
LAYERS = 2                                    # the error units of M1 (M2 has none)
# the tuples: kind, reappearance of B (None: the empty scene), and their window start
TUPLES = {"direction": (10, 10), "early": (8, 8), "vanish": (None, 10)}


def sequences() -> list[dict]:
    """The cells, then the four members of three tuples of every kind, with their reappearance frame."""
    out = [{"sequence": s, "condition": condition, "role": "none", "tuple_id": -1, "type": "none", "k": 4,
            "speed": 2, "reappear": REAPPEAR, "start": REAPPEAR}
           for s, condition in [(s, "occlusion") for s in range(4)] + [(s, "hidden_bounce") for s in range(10, 14)]]
    for i, (kind, (b_back, start)) in enumerate(TUPLES.items()):
        for j in range(3):
            tid, base = 100 + 10 * i + j, 100 + 40 * i + 4 * j
            ab_type, ba_type = ("vanish", "appear") if kind == "vanish" else (kind, kind)
            members = [("A", "occlusion", "none", 4, 2, REAPPEAR),
                       ("B", "empty" if b_back is None else "occlusion", "none", 6, 3, b_back),
                       ("AB", "surprise", ab_type, 4, 2, b_back), ("BA", "surprise", ba_type, 6, 3, REAPPEAR)]
            out += [{"sequence": base + m, "condition": condition, "role": role, "tuple_id": tid, "type": kind_of,
                     "k": k, "speed": speed, "reappear": back, "start": start}
                    for m, (role, condition, kind_of, k, speed, back) in enumerate(members)]
    return out


def fake_table() -> pd.DataFrame:
    """Two models on every sequence: M1's MSE rises in the window of the impossible members and after a hidden
    bounce, and it has error units that follow its MSE; M2's MSE is flat. The copy and blank MSE depend on t only."""
    parts = []
    t = np.arange(STEPS)
    for model in ("M1", "M2"):
        for s in sequences():
            in_window = (t - s["start"] >= 0) & (t - s["start"] < WINDOW)
            mse = np.full(STEPS, BASE)
            if model == "M1" and s["role"] in ("AB", "BA"):
                mse = np.where(in_window, SURPRISED, mse)
            if model == "M1" and s["condition"] == "hidden_bounce":
                mse = np.where(in_window, BOUNCE, mse)
            frame = pd.DataFrame({
                "sequence": s["sequence"], "t": t, "condition": s["condition"], "k": s["k"], "speed": s["speed"],
                "since_reappear": MISSING if s["reappear"] is None else t - s["reappear"],
                "tuple_id": s["tuple_id"], "tuple_role": s["role"], "surprise_type": s["type"],
                "model_mse": mse, "model_mse_bar": 2 * mse, "copy_mse": 2e-3 + 1e-4 * t, "blank_mse": 4e-3,
                "copy_mse_bar": 3e-3 + 1e-4 * t, "blank_mse_bar": 5e-3, "model": model, "seed": 0})
            if model == "M1":
                frame = frame.assign(**{f"e{l}": (l + 1) * mse for l in range(LAYERS)},
                                     **{f"e{l}_bar": 2 * (l + 1) * mse for l in range(LAYERS)})
            parts.append(frame)
    return pd.concat(parts, ignore_index=True)


@pytest.fixture(scope="module")
def tables() -> dict[str, pd.DataFrame]:
    """The analysis of the fake table."""
    return surprise(fake_table())


def test_window_start() -> None:
    """The window starts at the earlier reappearance of A and B: that of B when it comes early, that of A when B is
    the empty scene."""
    rows = tuple_rows(fake_table())
    rows = rows[rows["model"] == "M1"]
    for kind, (_, start) in TUPLES.items():
        own = rows[rows["event"] == kind]
        assert set(own["t"] - own["f"]) == {start}
    assert set(rows.loc[rows["surprise_type"] == "appear", "event"]) == {"vanish"}
    b = rows[(rows["role"] == "B") & (rows["event"] == "direction")]
    assert set(b["group_k"]) == {4} and set(b["group_speed"]) == {2}


def test_index_and_auroc(tables: dict[str, pd.DataFrame]) -> None:
    """M1 is surprised on every kind and pair (index SURPRISED - BASE, AUROC 1), M2 is not (0 and 0.5), and the copy
    sees the same frames in AB and B (exactly 0)."""
    index = tables["index"]
    index = index[index["scope"] == "all"].set_index(["event", "pair", "signal", "who"])
    for kind in TUPLES:
        for pair in ("AB - B", "BA - A"):
            m1, m2 = index.loc[(kind, pair, "mse", "M1")], index.loc[(kind, pair, "mse", "M2")]
            assert np.isclose(m1["index_mean"], SURPRISED - BASE) and m1["auroc"] == 1.0 and m1["higher"] == 1.0
            assert m1["tuples"] == 3 and np.isclose(index.loc[(kind, pair, "mse_bar", "M1"), "index_mean"],
                                                    2 * (SURPRISED - BASE))
            assert m2["index_mean"] == 0.0 and m2["auroc"] == 0.5 and m2["higher"] == 0.0
            assert index.loc[(kind, pair, "mse", "copy"), "index_mean"] == 0.0


def test_error_units_and_scopes(tables: dict[str, pd.DataFrame]) -> None:
    """The error units are read only for the model that has them, on all tuples only; the MSE per k and speed of A."""
    index = tables["index"]
    assert set(index.loc[index["signal"].str.match(r"e\d"), "who"]) == {"M1"}
    assert set(index.loc[index["signal"].str.match(r"e\d"), "signal"]) == {f"e{l}{s}" for l in range(LAYERS)
                                                                         for s in ("", "_bar")}
    e1 = index[(index["signal"] == "e1") & (index["event"] == "direction") & (index["pair"] == "AB - B")]
    assert np.isclose(e1["index_mean"].iloc[0], 2 * (SURPRISED - BASE)) and set(e1["scope"]) == {"all"}
    scopes = index.loc[(index["signal"] == "mse") & (index["event"] == "direction"), "scope"]
    assert set(scopes) == {"all", "k=4", "speed=2"}


def test_contrast_and_curves(tables: dict[str, pd.DataFrame]) -> None:
    """The hidden bounce is above occlusion and below the direction surprise for M1, and not for M2; the curves are
    aligned on the window start."""
    contrast = tables["contrast"]
    contrast = contrast[contrast["signal"] == "mse"].set_index(["positive", "negative", "who"])
    assert contrast.loc[("hidden_bounce", "occlusion", "M1"), "auroc"] == 1.0
    assert contrast.loc[("direction AB", "hidden_bounce", "M1"), "auroc"] == 1.0
    assert contrast.loc[("hidden_bounce", "occlusion", "M2"), "auroc"] == 0.5
    assert contrast.loc[("hidden_bounce", "occlusion", "M1"), "n"] == 8
    curve = tables["curves"]
    line = curve[(curve["event"] == "early") & (curve["role"] == "AB") & (curve["signal"] == "mse")
                 & (curve["who"] == "M1")].set_index("f")["mean"]
    assert line.index.min() == -BEFORE and line.index.max() == AFTER
    assert line.loc[0] == SURPRISED and line.loc[-1] == BASE and line.loc[WINDOW] == BASE


def test_figures(tables: dict[str, pd.DataFrame], tmp_path: Path) -> None:
    """Every figure is written."""
    paths = [plot_auroc(tables["index"], tables["contrast"], tmp_path / "a.png", "t"),
             plot_layers(tables["index"], tmp_path / "b.png", "t"),
             plot_curves(tables["curves"], tmp_path / "c.png", "t")]
    assert all(path.stat().st_size > 0 for path in paths)


def test_rows_read_from_passes(tmp_path: Path) -> None:
    """Every frame of the tuple members and the cell frames around the reappearance are read, and a pass without
    error units gives NaN in their columns. The baselines have positions but the true center is not read, so no
    position error is added."""
    table = fake_table()
    baselines = tmp_path / "baselines"
    runs = []
    for model in ("M1", "M2"):
        own = table[table["model"] == model]
        folder = tmp_path / model
        runs.append(ModelRun("fake", model, 0, folder))
        for path, frame in ((folder, own[[c for c in COLUMNS if c in own and own[c].notna().any()]]),
                            (baselines, own[["sequence", "t"]].assign(kalman_y=30.0, kalman_x=20.0))):
            path.mkdir(exist_ok=True)
            frame.to_parquet(path / "part_000000.parquet", index=False)
            (path / "info.json").write_text(json.dumps({"settings": {"format": FRAMES_FORMAT}, "complete": True}))
    read = read_runs(runs, baselines, list(CONDITIONS), COLUMNS, rows=ROWS)
    cells = read[read["tuple_role"] == "none"]
    assert cells["since_reappear"].between(-BEFORE, AFTER).all() and len(cells) == 2 * 8 * (BEFORE + AFTER + 1)
    assert len(read[read["tuple_role"] != "none"]) == 2 * 9 * 4 * STEPS
    assert read.loc[read["model"] == "M1", "e0"].notna().all() and read.loc[read["model"] == "M2", "e0"].isna().all()
    assert "kalman_x" in read and "kalman_err" not in read
