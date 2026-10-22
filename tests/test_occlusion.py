"""Tests for the RQ1 analysis on fake extraction tables with known answers: the curves against the frames since the
onset, the paired differences, the decay, the exit side and the emergence timing, the figures, and the runs found on a
set."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from peekaboo.eval.extract import FRAMES_FORMAT
from peekaboo.eval.occlusion import (curves, decay, emergence, emergence_rows, hidden_summary, plot_curves,
                                     plot_decay, plot_emergence, plot_speeds, timing)
from peekaboo.eval.tables import add_errors, available_runs

ONSET, K, STEPS = 8, 4, 20       # hidden frames 8 to 11, reappearance at frame 12


def fake_sequences(model: str, lag: int, condition: str, first: int, count: int = 6) -> pd.DataFrame:
    """Sequences of one condition read by one model. The probe error on y grows by 0.5 px per hidden frame (and is
    0.5 px elsewhere), the Kalman filter is 2 px off; the true digit comes out on the right of the bar (left for
    hidden bounces) from the reappearance frame on, and the model draws it `lag` frames after it (negative: early)."""
    rows = []
    for sequence in range(first, first + count):
        t = np.arange(STEPS)
        j = t - ONSET
        since = t - (ONSET + K)
        error = np.where((j >= 0) & (j < K), 0.5 * j, 0.5)
        true_out = np.where(since >= 0, 5.0 * (since + 1), 0.0)
        drawn = np.where(since - lag >= 0, 5.0 * (since - lag + 1), 0.0)
        right = condition == "occlusion"
        visible = (j < 0) | (since >= 0)
        rows.append(pd.DataFrame({
            "sequence": sequence, "t": t, "y": 30.0, "x": 40.0 + t, "in_window": True, "condition": condition,
            "k": K, "speed": 2 + sequence % 2, "since_onset": j, "since_reappear": since, "tuple_role": "none",
            "modal_y": np.where(visible, 30.0, np.nan), "modal_x": np.where(visible, 40.0 + t, np.nan),
            "probe_y": 30.0 + error, "probe_x": 40.0 + t, "random_probe_y": 37.0, "random_probe_x": 40.0 + t,
            "decoder_corr": 0.8, "decoder_iou": 0.6, "random_decoder_corr": 0.1, "random_decoder_iou": 0.05,
            "model_mse_bar": 1e-4, "copy_mse_bar": 1e-3, "blank_mse_bar": 2e-3,
            "pred_ink_y": np.where(visible, 30.0, np.nan), "pred_ink_x": np.where(visible, 40.0 + t, np.nan),
            "pred_ink_left": 0.0 if right else drawn, "pred_ink_right": drawn if right else 0.0,
            "true_ink_left": 0.0 if right else true_out, "true_ink_right": true_out if right else 0.0,
            "last_seen_y": 30.0, "last_seen_x": 40.0, "kalman_y": 32.0, "kalman_x": 40.0 + t,
            "kalman_walls_y": 31.0, "kalman_walls_x": 40.0 + t, "bar_center_x": 48.0,
            "template_kalman_corr": 0.5, "template_kalman_iou": 0.3, "template_walls_corr": 0.6,
            "template_walls_iou": 0.4, "model": model, "seed": 0}))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    """Two models on occlusion and hidden bounce sequences: A draws the digit on time, B one frame early."""
    parts = [fake_sequences(model, lag, condition, first)
             for model, lag in (("A", 0), ("B", -1))
             for condition, first in (("occlusion", 0), ("hidden_bounce", 100))]
    return add_errors(pd.concat(parts, ignore_index=True))


def test_curves_and_paired_differences(table: pd.DataFrame) -> None:
    """The probe curve is 0.5 j on the hidden frames, around them is 0.5; its mean over the hidden frames is 0.75,
    1.25 below Kalman's 2 px; the references appear once, the random controls once per model."""
    curve = curves(table)
    probe = curve[(curve["readout"] == "probe_y") & (curve["who"] == "A") & (curve["condition"] == "occlusion")]
    assert probe["j"].tolist() == list(range(-3, K + 3))
    expected = np.where((probe["j"] >= 0) & (probe["j"] < K), 0.5 * probe["j"], 0.5)
    assert np.allclose(probe["mean"], expected) and (probe["n"] == 6).all()
    lines = curve[(curve["readout"] == "probe_y") & (curve["condition"] == "occlusion")].groupby("who")["kind"].first()
    assert lines.to_dict() == {"A": "model", "A, random init": "random", "B": "model", "B, random init": "random",
                               "last seen": "reference", "Kalman": "reference", "Kalman with walls": "reference"}

    summary, paired = hidden_summary(table, reference="A")
    row = summary[(summary["condition"] == "occlusion") & (summary["k"] == "all") & (summary["readout"] == "probe_y")
                  & (summary["who"] == "A")].iloc[0]
    assert np.isclose(row["mean"], 0.75) and row["n"] == 6
    kalman = paired[(paired["k"] == "all") & (paired["readout"] == "probe_y") & (paired["model"] == "A")
                    & (paired["against"] == "Kalman") & (paired["condition"] == "occlusion")].iloc[0]
    assert np.isclose(kalman["mean"], -1.25) and np.isclose(kalman["low"], -1.25)
    versus = paired[(paired["model"] == "B") & (paired["against"] == "A")]
    assert len(versus) > 0 and np.allclose(versus["mean"], 0.0)
    assert not ((paired["model"] == "A") & (paired["against"] == "A")).any()


def test_decay(table: pd.DataFrame) -> None:
    """The slope of the probe error over the hidden frames is 0.5 px per frame, its last hidden value 1.5 px; Kalman's
    error does not change."""
    slopes = decay(table)
    mine = slopes[(slopes["condition"] == "occlusion") & (slopes["readout"] == "probe_y") & (slopes["who"] == "A")]
    assert np.isclose(mine["slope_mean"].item(), 0.5) and np.isclose(mine["last_mean"].item(), 1.5)
    flat = slopes[(slopes["condition"] == "occlusion") & (slopes["readout"] == "probe_y")
                  & (slopes["who"] == "Kalman")]
    assert np.isclose(flat["slope_mean"].item(), 0.0) and np.isclose(flat["last_mean"].item(), 2.0)


def test_emergence_and_timing(table: pd.DataFrame) -> None:
    """The exit side is found on each condition, every drawn ink is on it, the copy shows the true exit ink one frame
    late; A comes out on time, B one frame early, and the copy one frame late."""
    rows = emergence_rows(table, "occlusion")
    at = rows[rows["since_reappear"] == 1]
    assert np.allclose(at["true_exit"], 10.0) and np.allclose(at["copy_exit"], 5.0)
    assert np.allclose(rows.loc[rows["pred_exit"] > 0, "side_share"], 1.0)
    bounce = emergence_rows(table, "hidden_bounce")
    assert np.allclose(bounce.loc[bounce["since_reappear"] == 0, "true_exit"], 5.0)

    when = timing(table)
    overall = when[(when["k"] == "all") & (when["condition"] == "occlusion")].set_index("who")
    assert overall.loc["A", "on_time"] == 1.0 and overall.loc["B", "early"] == 1.0
    assert overall.loc["copy", "late"] == 1.0 and overall.loc["copy", "mean_lag"] == 1.0
    assert (overall["n"] == 6).all()

    curve, summary, paired = emergence(table)
    exit_ink = curve[(curve["measure"] == "exit_ink") & (curve["who"] == "true frame")
                     & (curve["condition"] == "occlusion")]
    assert exit_ink["since_reappear"].tolist() == list(range(-2, 4))
    bar = paired[(paired["measure"] == "bar_mse") & (paired["against"] == "copy") & (paired["k"] == "all")]
    assert np.allclose(bar["mean"], 1e-4 - 1e-3)
    readout = summary[(summary["measure"] == "readout_y") & (summary["who"] == "A") & (summary["k"] == "all")]
    assert np.allclose(readout["mean"], 0.0)


def test_figures(table: pd.DataFrame, tmp_path: Path) -> None:
    """Every figure is written."""
    by_k, by_speed = curves(table), curves(table, by_speed=True)
    paths = [plot_curves(by_k, "occlusion", ("probe_y", "probe_x", "decoder_corr"), "A", tmp_path / "a.png", "t"),
             plot_speeds(by_speed, "occlusion", "probe_y", "A", tmp_path / "b.png", "t"),
             plot_decay(decay(table), "occlusion", "A", tmp_path / "c.png", "t"),
             plot_emergence(emergence(table)[0], tmp_path / "d.png", "t")]
    assert all(path.stat().st_size > 0 for path in paths)


def test_runs_found_on_a_set(tmp_path: Path) -> None:
    """A complete pass of a sweep model is found with its seed, a failed run kept as a record is not, and a pass with
    older columns is refused."""
    def write(key: str, seed: int, version: int) -> None:
        folder = tmp_path / key / f"seed{seed}" / "eval" / "frames_dev_v1"
        folder.mkdir(parents=True)
        (folder / "info.json").write_text(json.dumps({"settings": {"format": version}, "complete": True}))

    write("sweep_prednet5l", 0, FRAMES_FORMAT)
    write("sweep_prednet5l", 2, FRAMES_FORMAT)
    write("sweep_prednet4l", 0, FRAMES_FORMAT)  # the failed seed
    runs = available_runs(tmp_path, "dev_v1")
    assert [(run.key, run.seed) for run in runs] == [("sweep_prednet5l", 0), ("sweep_prednet5l", 2)]
    write("sweep_convlstm", 0, FRAMES_FORMAT - 1)
    with pytest.raises(ValueError):
        available_runs(tmp_path, "dev_v1")
