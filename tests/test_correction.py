"""Tests for the RQ2 analysis on fake extraction tables with known answers: the latency on synthetic curves
(immediate, late, censored, short), the alignment on the reappearance frame with the tuples paired by their members,
the thresholds, the figures, and the rows read from a pass."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from peekaboo.eval.correction import (COLUMNS, CONDITIONS, HORIZON, M, ROWS, correction, correction_rows, latency,
                                      plot_curves, plot_latency)
from peekaboo.eval.extract import FRAMES_FORMAT, MISSING
from peekaboo.eval.tables import ModelRun, add_errors, read_runs

STEPS, ONSET, REAPPEAR = 20, 5, 10   # hidden frames 5 to 9, the digit visible again from frame 10
# the sequences: (sequence, condition, role, tuple, surprise type, k, speed)
SEQUENCES = ([(s, "occlusion", "none", -1, "none", 4, 2) for s in range(4)]
             + [(s, "hidden_bounce", "none", -1, "none", 4, 3) for s in range(10, 14)]
             + [(base + i, condition, role, tid, kind, k, speed)
                for tid, base in ((100, 20), (101, 24), (102, 28))
                for i, (condition, role, kind, k, speed) in enumerate((
                    ("occlusion", "A", "none", 4, 2), ("occlusion", "B", "none", 6, 3),
                    ("surprise", "AB", "direction", 4, 2), ("surprise", "BA", "direction", 6, 3)))]
             + [(40, "occlusion", "A", 200, "none", 8, 2), (41, "surprise", "AB", 200, "vanish", 8, 2),
                (42, "surprise", "BA", 200, "appear", 8, 2)])
# the probe error of each model: on every frame, and on the first frames after a surprising reappearance
MODELS = {"M1": (0.5, 2), "M2": (1.0, 3)}


def fake_table() -> pd.DataFrame:
    """Two models on every sequence. The probe is 10 px off for the first frames after a surprising reappearance
    (2 for M1, 3 for M2), the Kalman filter 6 px off on the first one; M2 draws nothing on the reappearance frame."""
    parts = []
    t = np.arange(STEPS)
    j = t - REAPPEAR
    for model, (base_error, frames_off) in MODELS.items():
        for sequence, condition, role, tid, kind, k, speed in SEQUENCES:
            surprising = role in ("AB", "BA")
            vanished = kind == "vanish"
            visible = ((t < ONSET) | (t >= REAPPEAR)) & ~(vanished & (t >= ONSET))
            error = np.where(surprising & (j >= 0) & (j < frames_off), 10.0, base_error)
            y, x = np.full(STEPS, 30.0), 20.0 + 2.0 * t
            modal_y, modal_x = np.where(visible, y, np.nan), np.where(visible, x, np.nan)
            drawn = visible & ~((model == "M2") & (j == 0))
            parts.append(pd.DataFrame({
                "sequence": sequence, "t": t, "y": y, "x": x, "in_window": True, "condition": condition, "k": k,
                "speed": speed, "state": np.where(visible, 0, 2), "since_onset": t - ONSET,
                "since_reappear": j if not vanished else MISSING, "tuple_id": tid, "tuple_role": role,
                "surprise_type": kind, "modal_y": modal_y, "modal_x": modal_x,
                "probe_y": y + error, "probe_x": x, "random_probe_y": y + 9.0, "random_probe_x": x,
                "decoder_corr": 0.8, "random_decoder_corr": 0.1,
                "pred_ink_y": np.where(drawn, y, np.nan), "pred_ink_x": np.where(drawn, x, np.nan),
                "model_mse_digit": 1e-4, "copy_mse_digit": 1e-3, "blank_mse_digit": 2e-3,
                "kalman_y": y + np.where(surprising & (j == 0), 6.0, 0.0), "kalman_x": x,
                "kalman_walls_y": y, "kalman_walls_x": x, "last_seen_y": y, "last_seen_x": x,
                "template_kalman_corr": 0.5, "template_walls_corr": 0.6, "model": model, "seed": 0}))
    return add_errors(pd.concat(parts, ignore_index=True))


def test_latency_on_synthetic_curves() -> None:
    """Immediate, late, a single frame below tau (not enough), never (censored), and a window too short."""
    curves = {0: [1, 1, 1, 1, 1, 1], 1: [9, 9, 9, 1, 1, 1], 2: [9, 1, 9, 1, 1, 9], 3: [9] * 6, 4: [9, 9, 9]}
    rows = pd.concat([pd.DataFrame({"seed": 0, "sequence": s, "since_reappear": range(len(v)), "error": v})
                      for s, v in curves.items()], ignore_index=True)
    out = latency(rows, "error", 2.0).set_index("sequence")
    assert out.loc[0, "latency"] == 0 and out.loc[1, "latency"] == 3 and out.loc[2, "latency"] == 3
    assert out.loc[3, "status"] == "censored" and out.loc[3, "latency"] == HORIZON - M + 1
    assert out.loc[4, "status"] == "short" and np.isnan(out.loc[4, "latency"])
    assert out.loc[[0, 1, 2], "status"].eq("found").all() and np.isclose(out.loc[1, "error"], 5.0)


@pytest.fixture(scope="module")
def tables() -> dict[str, pd.DataFrame]:
    """The analysis of the fake table, with M1 as the reference model."""
    return correction(fake_table(), "M1")


def test_events_and_thresholds() -> None:
    """Tuple members get the kind of their tuple and the k and speed of A; tau has its floor."""
    rows = correction_rows(fake_table())
    b = rows[(rows["tuple_role"] == "B") & (rows["model"] == "M1")]
    assert set(b["event"]) == {"direction"} and set(b["group_k"]) == {4} and set(b["group_speed"]) == {2}
    assert set(rows.loc[rows["sequence"] == 42, "event"]) == {"vanish"}


def test_latency_and_pairs(tables: dict[str, pd.DataFrame]) -> None:
    """The probe corrects at once after an expected reappearance and after 2 (M1) or 3 (M2) frames after a surprising
    one; the pairs give AB - B and BA - A; the Kalman filter takes 1 frame; M2 draws the digit one frame late."""
    taus = tables["taus"].set_index(["readout", "who"])["tau"]
    assert taus[("probe", "M1")] == 3.0 and taus[("probe", "Kalman")] == 3.0
    assert np.isclose(taus[("digit_mse", "M1")], 1e-4) and ("digit_mse", "copy") not in taus.index
    summary = tables["latency"]
    summary = summary[summary["scope"] == "all"].set_index(["event", "role", "readout", "who"])
    assert summary.loc[("occlusion", "none", "probe", "M1"), "latency_mean"] == 0
    assert summary.loc[("direction", "AB", "probe", "M1"), "latency_mean"] == 2
    assert summary.loc[("direction", "AB", "probe", "M2"), "latency_mean"] == 3
    assert summary.loc[("direction", "B", "probe", "M2"), "latency_mean"] == 0
    assert summary.loc[("vanish", "BA", "probe", "M1"), "latency_mean"] == 2
    assert summary.loc[("direction", "AB", "probe", "Kalman"), "latency_mean"] == 1
    assert summary.loc[("occlusion", "none", "readout", "M2"), "latency_mean"] == 1
    assert summary.loc[("occlusion", "none", "readout", "copy"), "latency_mean"] == 1
    assert summary.loc[("occlusion", "none", "digit_mse", "M1"), "latency_mean"] == 0
    assert summary.loc[("direction", "AB", "probe", "M1"), "sequences"] == 3
    assert ("vanish", "AB", "probe", "M1") not in summary.index

    paired = tables["latency_paired"]
    paired = paired[(paired["scope"] == "all") & (paired["measure"] == "latency")]
    pair = paired.set_index(["event", "role", "readout", "who", "against"])["mean"]
    assert pair[("direction", "AB", "probe", "M1", "B")] == 2 and pair[("direction", "AB", "probe", "M2", "B")] == 3
    assert pair[("vanish", "BA", "probe", "M1", "A")] == 2
    assert pair[("direction", "AB", "probe", "M2", "M1")] == 1
    scopes = set(tables["latency"].loc[tables["latency"]["event"] == "direction", "scope"])
    assert scopes == {"all", "k=4", "speed=2"}

    curve = tables["curves"]
    probe = curve[(curve["event"] == "direction") & (curve["role"] == "AB") & (curve["readout"] == "probe")
                  & (curve["who"] == "M1")].set_index("j")["mean"]
    assert probe.index.min() == -3 and probe.loc[0] == 10 and probe.loc[2] == 0.5


def test_figures(tables: dict[str, pd.DataFrame], tmp_path: Path) -> None:
    """Every figure is written."""
    paths = [plot_curves(tables["curves"], tmp_path / "a.png", "t"), plot_latency(tables["latency"],
                                                                                    tmp_path / "b.png", "t")]
    assert all(path.stat().st_size > 0 for path in paths)


def test_rows_read_from_a_pass(tmp_path: Path) -> None:
    """Only the frames around the reappearance and the visible frames before the onset of the cells are read, and
    the vanish member that never reappears is left out."""
    table = fake_table()
    own = table[table["model"] == "M1"]
    folder, baselines = tmp_path / "pass", tmp_path / "baselines"
    for path, frame in ((folder, own[COLUMNS]), (baselines, own[["sequence", "t", "kalman_y", "kalman_x",
                                                                 "kalman_walls_y", "kalman_walls_x"]])):
        path.mkdir()
        frame.to_parquet(path / "part_000000.parquet", index=False)
        (path / "info.json").write_text(json.dumps({"settings": {"format": FRAMES_FORMAT}, "complete": True}))
    read = read_runs([ModelRun("fake", "M1", 0, folder)], baselines, list(CONDITIONS), COLUMNS, rows=ROWS)
    around = read["since_reappear"].between(-3, 7)
    before = (read["tuple_role"] == "none") & (read["since_onset"] < 0) & (read["state"] == 0) & (read["t"] >= 2)
    assert (around | before).all() and before.any() and around.any()
    assert 41 not in set(read["sequence"]) and "kalman_err" in read
