"""Tests for the position probes: features, ridge and bin probes, saving, the evaluation with its controls, and the
probe's cross in the figures."""

from pathlib import Path

import numpy as np
import pytest
import torch

from peekaboo.config import load_config
from peekaboo.data.conditions import GeneratorSettings, sample_spec
from peekaboo.data.dataset import OnTheFlyDataset, prepare_batch
from peekaboo.data.mnist_pool import DigitPool
from peekaboo.data.render import RenderSettings, render_sequence
from peekaboo.eval.next_frame import MIN_CONTEXT
from peekaboo.models import build_model
from peekaboo.paths import CONFIGS_DIR
from peekaboo.probes.evaluate import (BASELINES, add_errors, baseline_positions, position_table, probe_positions,
                                      summarize_errors)
from peekaboo.probes.features import collect_frames, frame_features, pool_layer
from peekaboo.probes.position import BINS, PositionProbe, fit_position_probe, fit_ridge, position_bins
from peekaboo.viz.predictions import YELLOW, draw_cross, prediction_sheet

RENDER = RenderSettings.from_config(load_config(CONFIGS_DIR / "data" / "base.yaml"))
CPU = torch.device("cpu")
TINY = {"prednet": {"model": "prednet", "stack_sizes": [3, 4, 8], "layer_loss_weights": "L0"},
        "convlstm": {"model": "convlstm", "hidden_sizes": [4], "kernel_size": 3, "patch_size": 4}}


def test_pooling_keeps_at_most_the_limit() -> None:
    """Each layer is pooled by the smallest power of 2 that leaves at most `limit` values."""
    assert pool_layer(torch.rand(2, 3, 64, 96), 2048).shape == (2, 3 * 16 * 24)  # factor 4
    assert pool_layer(torch.rand(2, 128, 4, 6), 2048).shape == (2, 128 * 2 * 3)  # factor 2
    assert pool_layer(torch.rand(2, 8, 4, 6), 2048).shape == (2, 8 * 4 * 6)      # unchanged


def test_ridge_recovers_a_linear_signal() -> None:
    """With a nearly noiseless linear target, the ridge probe predicts held out frames almost exactly, and the
    smallest penalty wins."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 20)).astype(np.float32)
    targets = x @ rng.normal(size=(20, 2)) + np.array([30.0, 50.0]) + 1e-3 * rng.normal(size=(600, 2))
    train = np.arange(600) < 480
    x -= x[train].mean(axis=0)  # the probe works on standardized features, centered on the training frames
    w, intercept, alpha, errors = fit_ridge(x, targets, train, alphas=(1e-3, 1.0, 1e3))
    assert alpha == 1e-3 and errors[1e3] > errors[1e-3]
    assert np.allclose(x[~train] @ w + intercept, targets[~train], atol=0.01)


def test_position_probe_fits_saves_and_loads(tmp_path: Path) -> None:
    """Features that encode the position linearly give a near exact ridge probe and bins well above chance (0.125);
    the saved probe predicts exactly as the fitted one."""
    rng = np.random.default_rng(0)
    n = 3000
    positions = np.column_stack([rng.uniform(5, 59, n), rng.uniform(5, 91, n)])
    features = np.column_stack([positions, rng.normal(size=(n, 10))]).astype(np.float32)
    sequences = np.repeat(np.arange(150), 20)
    probe, validation = fit_position_probe(features, positions, sequences, (64, 96), layer_limit=2048)
    assert np.abs(probe.predict(features) - positions).max() < 0.05
    right = position_bins(positions, probe.bin_edges)
    p = probe.bin_probabilities(features)
    assert np.allclose(p.sum(axis=2), 1.0)
    assert p[np.arange(n), 0, right[:, 0]].mean() > 0.3 and p[np.arange(n), 1, right[:, 1]].mean() > 0.3
    assert set(validation) == {"ridge_mse", "bins_log_loss"}
    loaded = PositionProbe.load(probe.save(tmp_path / "probe.npz"))
    assert np.array_equal(loaded.predict(features), probe.predict(features))
    assert np.array_equal(loaded.bin_probabilities(features), p) and loaded.layer_limit == 2048


def test_bins_cover_the_frame() -> None:
    """Positions map to 8 equal bins per axis, the borders included."""
    edges = np.stack([np.linspace(-0.5, 63.5, BINS + 1), np.linspace(-0.5, 95.5, BINS + 1)])
    bins = position_bins(np.array([[0.0, 0.0], [63.0, 95.0], [8.0, 12.0], [7.4, 11.4]]), edges)
    assert bins.tolist() == [[0, 0], [7, 7], [1, 1], [0, 0]]


@pytest.mark.parametrize("name", list(TINY))
def test_features_of_frame_t_come_from_the_state_that_predicts_it(name: str, pool: DigitPool,
                                                                  settings: GeneratorSettings) -> None:
    """Collected frames start at MIN_CONTEXT, lie in the analysis window when asked, and their features are the
    pooled states R[t] of their own sequence and frame."""
    torch.manual_seed(0)
    model = build_model(TINY[name])
    stream = OnTheFlyDataset(pool, settings, RENDER, "probe", 0)
    frames = collect_frames(model, stream, 4, CPU, batch_size=2, window_only=True)
    rows = frames.rows
    assert len(rows) == len(frames.features) > 0
    assert (rows["t"] >= MIN_CONTEXT).all() and rows["in_window"].all() and rows["y"].notna().all()
    sequence, t = int(rows["sequence"].iloc[0]), int(rows["t"].iloc[0])
    batch = prepare_batch(torch.utils.data.default_collate([stream[sequence]]), CPU)
    with torch.no_grad():
        states = model(batch["frames"], return_states=True)["R"]
    assert np.allclose(frame_features(states)[0, t].numpy(), frames.features[0], atol=1e-6)


def test_evaluation_runs_with_its_controls(pool: DigitPool, settings: GeneratorSettings) -> None:
    """A probe on a trained like model and one on a random model are evaluated next to the trackers and the bar
    center: one row per frame, finite tracker errors, and summaries per state."""
    torch.manual_seed(0)
    stream = OnTheFlyDataset(pool, settings, RENDER, "probe", 0)
    held_out = OnTheFlyDataset(pool, settings, RENDER, "val", 0)
    rows = {}
    for name in ("probe", "random_probe"):
        model = build_model(TINY["prednet"])
        frames = collect_frames(model, stream, 6, CPU, batch_size=3, window_only=True)
        probe, _ = fit_position_probe(frames.features, frames.rows[["y", "x"]].to_numpy(),
                                      frames.rows["sequence"].to_numpy(), (64, 96), layer_limit=2048)
        rows[name] = probe_positions(model, probe, held_out, 3, CPU, batch_size=3, seq_len=20)
    table = add_errors(position_table(rows, baseline_positions(held_out, 3, seq_len=20)),
                       ("probe", "random_probe") + BASELINES)
    assert len(table) == len(rows["probe"]) and table["probe_err_y"].notna().all()
    assert table["bar_center_err_x"].notna().all()
    assert table["kalman_err"].notna().any()
    summary = summarize_errors(table, ("probe", "random_probe") + BASELINES, ["state_name"], "x")
    assert list(summary.columns) == ["frames", "probe", "random_probe", *BASELINES, "bar_center"]


def test_probe_cross_in_the_figure(pool: DigitPool, settings: GeneratorSettings) -> None:
    """The cross is drawn in yellow around the believed position, nothing is drawn for NaN, and a sheet with beliefs
    renders."""
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    draw_cross(image, np.array([10.0, 20.0]), scale=2)
    assert tuple(image[20, 40]) == YELLOW and tuple(image[20, 34]) == YELLOW and tuple(image[26, 40]) == YELLOW
    assert tuple(image[0, 0]) == (0, 0, 0)
    blank = np.zeros_like(image)
    draw_cross(blank, np.array([np.nan, 5.0]), scale=2)
    assert not blank.any()
    spec = sample_spec(pool, settings, "test", 0, 0, condition="occlusion", k=4, speed=3)
    rendered = render_sequence(spec, pool.sprite(spec.digit_index), RENDER, settings.crossing.thresholds)
    prediction = rendered.observed / 255.0
    belief = rendered.truth.center.copy()
    sheet = prediction_sheet(rendered, prediction, [5, 6], "test", belief=belief)
    assert sheet.size[0] > 0
