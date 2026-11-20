"""Tests for the readouts of single layers and the analysis inside the state (step 3.5): the columns of every layer,
probes that read only their layers and the visible probe only its frames, the readout of all the layers equal to the
probe and decoder of the extraction pass, the analysis on a fake table with known answers, the rows read from passes,
and the error maps."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from peekaboo.config import load_config
from peekaboo.data.conditions import GeneratorSettings, sample_spec
from peekaboo.data.dataset import OnTheFlyDataset
from peekaboo.data.mnist_pool import DigitPool
from peekaboo.data.render import RenderSettings, render_sequence
from peekaboo.data.truth import STATE_NAMES
from peekaboo.eval.extract import FRAMES_FORMAT, extract_frames, layer_frames
from peekaboo.eval.layers import COLUMNS, inside, plot_belief, plot_errors, plot_layers, plot_visible, read_layers
from peekaboo.eval.tables import ModelRun
from peekaboo.models import build_model
from peekaboo.paths import CONFIGS_DIR
from peekaboo.probes.features import collect_frames, frame_features, layer_columns, pool_layer
from peekaboo.probes.layers import (VISIBLE, VISIBLE_STATE, LayerProbes, fit_layer_decoders, fit_layer_probes,
                                    layer_subsets, load_readouts, save_readouts, state_shapes)
from peekaboo.probes.position import fit_position_probe
from peekaboo.viz.errors import error_maps, error_sheet, normalize

RENDER = RenderSettings.from_config(load_config(CONFIGS_DIR / "data" / "base.yaml"))
CPU = torch.device("cpu")
TINY = {"model": "prednet", "stack_sizes": [3, 4, 8], "layer_loss_weights": "L0"}
OCCLUDED = STATE_NAMES.index("occluded")
LIMIT = 256  # pooled features per layer in the tests with a model, to keep the ridge fits small


def test_layer_columns_are_the_pooled_layers() -> None:
    """The columns of every layer in the features are that layer's pooled state, and they cover every feature."""
    torch.manual_seed(0)
    model = build_model(TINY)
    shapes = state_shapes(model, (64, 96), CPU)
    with torch.no_grad():
        states = model(torch.rand(2, 3, 3, 64, 96), return_states=True)["R"]
    features = frame_features(states, 64)
    columns = layer_columns(shapes, 64)
    for l, own in enumerate(columns):
        assert torch.equal(features[:, 1][:, torch.as_tensor(own)], pool_layer(states[1][l], 64))
    assert np.array_equal(np.concatenate(columns), np.arange(features.shape[2]))
    assert set(layer_subsets(3)) == {"all", VISIBLE, "only_0", "only_1", "only_2", "without_0", "without_1",
                                     "without_2"}
    assert set(layer_subsets(1)) == {"all", VISIBLE, "only_0"}


def frame_rows(positions: np.ndarray, state: np.ndarray) -> pd.DataFrame:
    """Rows of frames: true centroid, sequence (20 frames each), and state."""
    return pd.DataFrame({"y": positions[:, 0], "x": positions[:, 1], "sequence": np.arange(len(positions)) // 20,
                         "state": state})


def test_layer_probes_read_their_layers(tmp_path: Path) -> None:
    """With the position in layer 1 only, the probes that read layer 1 are exact and the others are not; the probe on
    all the layers is the ridge probe of fit_position_probe; and the saved probes predict as the fitted ones."""
    rng = np.random.default_rng(0)
    n = 2000
    positions = np.column_stack([rng.uniform(5, 59, n), rng.uniform(5, 91, n)])
    features = np.column_stack([rng.normal(size=(n, 5)), positions + 1e-3 * rng.normal(size=(n, 2)),
                                rng.normal(size=(n, 6))]).astype(np.float32)
    columns = [np.arange(5), np.arange(5, 9), np.arange(9, 13)]
    probes, errors = fit_layer_probes(features, frame_rows(positions, np.zeros(n, dtype=int)), columns, 2048)
    error = {name: np.abs(p - positions).mean() for name, p in probes.predict(features).items()}
    for name in ("all", VISIBLE, "only_1", "without_0", "without_2"):
        assert error[name] < 0.05, name
    for name in ("only_0", "only_2", "without_1"):
        assert error[name] > 5, name
    assert set(errors) == set(layer_subsets(3))
    reference, _ = fit_position_probe(features, positions, frame_rows(positions, np.zeros(n))["sequence"].to_numpy(),
                                      (64, 96), layer_limit=2048)
    assert np.allclose(probes.coef["all"], reference.coef) and probes.alpha["all"] == reference.alpha
    loaded = LayerProbes.load(probes.save(tmp_path / "position.npz"))
    for name, position in probes.predict(features).items():
        assert np.array_equal(loaded.predict(features)[name], position)


def test_visible_probe_reads_only_visible_frames() -> None:
    """A state that codes a hidden digit 20 px lower: the probe fitted on visible frames is exact on them and 20 px
    off on hidden frames, while the probe fitted on every frame splits the difference."""
    rng = np.random.default_rng(1)
    n = 2000
    positions = np.column_stack([rng.uniform(5, 59, n), rng.uniform(5, 91, n)])
    state = np.where(rng.random(n) < 0.5, VISIBLE_STATE, OCCLUDED)
    coded = positions + np.where(state == OCCLUDED, 20.0, 0.0)[:, None] * np.array([1.0, 0.0])
    features = np.column_stack([coded, rng.normal(size=(n, 2))]).astype(np.float32)
    probes, _ = fit_layer_probes(features, frame_rows(positions, state), [np.arange(4)], 2048)
    predicted = probes.predict(features)
    hidden = state == OCCLUDED
    off = {name: np.abs(p[:, 0] - positions[:, 0]) for name, p in predicted.items()}
    assert off[VISIBLE][~hidden].mean() < 0.05 and 19 < off[VISIBLE][hidden].mean() < 21
    assert 2 < off["all"][hidden].mean() < 19


def test_decoders_and_the_layer_pass(pool: DigitPool, settings: GeneratorSettings, tmp_path: Path) -> None:
    """Every readout gets a probe and a decoder on its layers; once saved and reloaded, the pass reads them on every
    frame, and its readouts of all the layers are those of the extraction pass with the same probe and decoder."""
    torch.manual_seed(0)
    model = build_model(TINY)
    probe_stream = OnTheFlyDataset(pool, settings, RENDER, "probe", 0)
    decoder_stream = OnTheFlyDataset(pool, settings, RENDER, "decoder", 0, include_amodal=True)
    data = OnTheFlyDataset(pool, settings, RENDER, "val", 0, include_amodal=True)
    frames = collect_frames(model, probe_stream, 4, CPU, batch_size=2, window_only=True, limit=LIMIT)
    shapes = state_shapes(model, (64, 96), CPU)
    probes, _ = fit_layer_probes(frames.features, frames.rows, layer_columns(shapes, LIMIT), LIMIT)
    decoders, losses = fit_layer_decoders(model, decoder_stream, 2, CPU, batch_size=2, log=lambda line: None)
    subsets = layer_subsets(len(shapes))
    assert set(decoders) == set(subsets) == set(probes.columns)
    assert decoders["only_1"].channels == (shapes[1][0],) and len(losses["all"]) == 2
    save_readouts(tmp_path, probes, decoders, {"subsets": {name: list(l) for name, l in subsets.items()}})
    loaded_probes, loaded = load_readouts(tmp_path)
    assert loaded["without_0"][0] == (1, 2)

    table = layer_frames(model, data, 0, 2, CPU, loaded_probes, loaded, batch_size=2)
    assert len(table) == 2 * 40 and {f"decoder_{name}_iou" for name in subsets} <= set(table)
    probe, _ = fit_position_probe(frames.features, frames.rows[["y", "x"]].to_numpy(),
                                  frames.rows["sequence"].to_numpy(), (64, 96), layer_limit=LIMIT)
    reference = extract_frames(model, data, 0, 2, CPU, probe, decoders["all"], batch_size=2)
    assert np.allclose(table["probe_all_y"], reference["probe_y"], atol=1e-3)
    assert np.allclose(table["decoder_all_corr"], reference["decoder_corr"], atol=1e-5, equal_nan=True)


# the analysis, on a fake table

STEPS, ONSET, K = 20, 8, 4
REAPPEAR = ONSET + K
TRUE_Y, TRUE_X = 30.0, 40.0
# the error on y of every readout's probe on the hidden frames (0 on the visible ones)
HIDDEN_Y = {"M1": {"all": 2.0, VISIBLE: 3.0, "only_0": 4.0, "only_1": 1.0, "without_0": 1.0, "without_1": 4.0},
            "M2": {"all": 2.0, VISIBLE: 5.0, "only_0": 2.0}}
SURPRISED = 10.0  # the extra distance of M1's layer 0 probe in AB from the reappearance on
KALMAN = 6.0      # Kalman's error on the hidden frames


def sequences() -> list[dict]:
    """Six occlusion and four hidden bounce cells, then three `direction` tuples."""
    out = [{"sequence": s, "condition": c, "role": "none", "tuple_id": -1, "type": "none"}
           for s, c in enumerate(["occlusion"] * 6 + ["hidden_bounce"] * 4)]
    members = (("A", "occlusion", "none"), ("B", "occlusion", "none"), ("AB", "surprise", "direction"),
               ("BA", "surprise", "direction"))
    for j in range(3):
        out += [{"sequence": 100 + 4 * j + m, "condition": condition, "role": role, "tuple_id": 50 + j, "type": kind}
                for m, (role, condition, kind) in enumerate(members)]
    return out


def fake_table() -> pd.DataFrame:
    """Two models on every sequence, every frame in the analysis window: their probes are off by HIDDEN_Y on the
    hidden frames, M1's layer 0 probe stays SURPRISED px off in AB after the reappearance, and M1's error units are
    at 2 on visible frames, 1 on hidden frames, and 6 on the reappearance frame and the next two (twice as much over
    the bar)."""
    t = np.arange(STEPS)
    hidden = (t >= ONSET) & (t < REAPPEAR)
    back = (t >= REAPPEAR) & (t < REAPPEAR + 3)
    parts = []
    for model, offsets in HIDDEN_Y.items():
        for s in sequences():
            columns = {"sequence": s["sequence"], "t": t, "y": TRUE_Y, "x": TRUE_X, "in_window": True,
                       "condition": s["condition"], "k": K, "speed": 2,
                       "state": np.where(hidden, OCCLUDED, VISIBLE_STATE), "since_onset": t - ONSET,
                       "since_reappear": t - REAPPEAR, "tuple_id": s["tuple_id"], "tuple_role": s["role"],
                       "surprise_type": s["type"], "model": model, "seed": 0,
                       "kalman_err_y": np.where(hidden, KALMAN, 0.0), "kalman_err": np.where(hidden, KALMAN, 0.0)}
            for name, offset in offsets.items():
                y = TRUE_Y + np.where(hidden, offset, 0.0)
                if model == "M1" and name == "only_0" and s["role"] == "AB":
                    y = y + np.where(t >= REAPPEAR, SURPRISED, 0.0)
                columns[f"probe_{name}_y"], columns[f"probe_{name}_x"] = y, np.full(STEPS, TRUE_X)
                columns[f"decoder_{name}_corr"] = np.where(hidden, 0.5, 0.9)
                columns[f"decoder_{name}_iou"] = np.where(hidden, 0.3, 0.6)
            columns["probe_y"] = columns["probe_all_y"]
            if model == "M1":
                level = np.where(hidden, 1.0, np.where(back, 6.0, 2.0))
                for l in range(2):
                    columns[f"e{l}"], columns[f"e{l}_bar"] = level, 2 * level
            parts.append(pd.DataFrame(columns))
    return pd.concat(parts, ignore_index=True)


@pytest.fixture(scope="module")
def tables() -> dict[str, pd.DataFrame]:
    """The analysis of the fake table."""
    return inside(fake_table())


def test_readouts_of_every_layer(tables: dict[str, pd.DataFrame]) -> None:
    """On the hidden frames, every readout's error is its offset, its difference with all the layers is paired, and
    Kalman is there; on the visible frames before the onset, every error is 0; a model without a layer has no
    readout of it."""
    summary = tables["readouts"].set_index(["condition", "frames", "who", "readout", "measure"])
    for model, offsets in HIDDEN_Y.items():
        for name, offset in offsets.items():
            assert np.isclose(summary.loc[("occlusion", "hidden", model, name, "probe_y"), "mean"], offset)
            assert np.isclose(summary.loc[("hidden_bounce", "visible", model, name, "probe_y"), "mean"], 0.0)
            assert np.isclose(summary.loc[("occlusion", "hidden", model, name, "decoder_corr"), "mean"], 0.5)
    assert ("occlusion", "hidden", "M2", "only_1", "probe_y") not in summary.index
    assert np.isclose(summary.loc[("occlusion", "hidden", "Kalman", "", "probe_y"), "mean"], KALMAN)
    assert summary.loc[("occlusion", "hidden", "M1", "all", "probe_y"), "n"] == 6
    paired = tables["readouts_paired"].set_index(["condition", "frames", "who", "readout", "measure"])
    assert np.isclose(paired.loc[("occlusion", "hidden", "M1", "only_0", "probe_y"), "mean"], 2.0)
    assert np.isclose(paired.loc[("occlusion", "hidden", "M2", VISIBLE, "probe_y"), "mean"], 3.0)
    curve = tables["curves"]
    line = curve[(curve["condition"] == "occlusion") & (curve["who"] == "M1") & (curve["readout"] == "only_0")
                 & (curve["measure"] == "probe_y")].set_index("j")["mean"]
    assert line.loc[0] == 4.0 and line.loc[-1] == 0.0 and line.index.max() == K - 1 + 3


def test_belief_after_a_surprise(tables: dict[str, pd.DataFrame]) -> None:
    """Over the frames after the reappearance, only M1's layer 0 probe is farther in AB than in B, by SURPRISED; BA
    against A and Kalman differ by nothing."""
    belief = tables["belief"].set_index(["event", "pair", "who", "readout", "measure"])
    row = belief.loc[("direction", "AB - B", "M1", "only_0", "probe")]
    assert np.isclose(row["mean"], SURPRISED) and np.isclose(row["positive_mean"], SURPRISED) and row["n"] == 3
    assert np.isclose(belief.loc[("direction", "AB - B", "M1", "all", "probe"), "mean"], 0.0)
    assert np.isclose(belief.loc[("direction", "BA - A", "M1", "only_0", "probe"), "mean"], 0.0)
    assert np.isclose(belief.loc[("direction", "AB - B", "Kalman", "", "probe"), "mean"], 0.0)
    assert np.isclose(belief.loc[("direction", "AB - B", "M2", "all", "decoder_corr"), "mean"], 0.0)


def test_error_units_relative_to_the_visible_level(tables: dict[str, pd.DataFrame]) -> None:
    """The error units are at half their visible level while hidden and three times it when the digit comes back,
    on the whole frame and over the bar; a model without error units has none."""
    errors = tables["errors"].set_index(["condition", "who", "layer", "region", "phase"])["mean"]
    for condition in ("occlusion", "hidden_bounce"):
        for layer in range(2):
            for region in ("frame", "bar"):
                assert np.isclose(errors.loc[(condition, "M1", layer, region, "hidden")], 0.5)
                assert np.isclose(errors.loc[(condition, "M1", layer, region, "reappearance")], 3.0)
    assert set(tables["errors"]["who"]) == {"M1"} and set(tables["error_curves"]["who"]) == {"M1"}


def test_figures(tables: dict[str, pd.DataFrame], tmp_path: Path) -> None:
    """Every figure is written."""
    paths = [plot_layers(tables["readouts"], tmp_path / "a.png", "t"),
             plot_visible(tables["curves"], tmp_path / "b.png", "t", k=K),
             plot_belief(tables["belief"], tmp_path / "c.png", "t"),
             plot_errors(tables["error_curves"], tmp_path / "d.png", "t", k=K)]
    assert all(path.stat().st_size > 0 for path in paths)


def test_rows_read_from_passes(tmp_path: Path) -> None:
    """The rows of a model pass and of its layer pass are joined on (sequence, t), with the baselines and their
    errors."""
    table = fake_table()
    own = table[table["model"] == "M1"]
    data = "fake"
    folders = {tmp_path / "M1" / "eval" / f"frames_{data}": own[[c for c in COLUMNS if c in own]],
               tmp_path / "M1" / "eval" / f"layers_{data}": own[["sequence", "t", "condition", "in_window"]
                                                                 + [c for c in own if c.startswith("decoder_")
                                                                    or (c.startswith("probe_") and c != "probe_y")]],
               tmp_path / "baselines": own[["sequence", "t"]].assign(kalman_y=TRUE_Y + 1.0, kalman_x=TRUE_X)}
    for folder, frame in folders.items():
        folder.mkdir(parents=True)
        frame.to_parquet(folder / "part_000000.parquet", index=False)
        (folder / "info.json").write_text(json.dumps({"settings": {"format": FRAMES_FORMAT}, "complete": True}))
    runs = [ModelRun("fake", "M1", 0, tmp_path / "M1" / "eval" / f"frames_{data}")]
    read = read_layers(runs, data, tmp_path / "baselines")
    assert len(read) == len(own)
    assert np.allclose(read["probe_only_0_y"].to_numpy(), own["probe_only_0_y"].to_numpy())
    assert np.allclose(read["kalman_err_y"], 1.0)


def test_error_maps(pool: DigitPool, settings: GeneratorSettings) -> None:
    """The map of a layer is its mean error activity over the channels, enlarged to the frame; normalized maps are in
    [0, 1]; a sheet renders."""
    torch.manual_seed(0)
    model = build_model(TINY)
    spec = sample_spec(pool, settings, "test", 0, 0, condition="occlusion", k=4, speed=3)
    rendered = render_sequence(spec, pool.sprite(spec.digit_index), RENDER, settings.crossing.thresholds)
    frames = torch.from_numpy(rendered.observed).permute(0, 3, 1, 2)[None].float() / 255.0
    with torch.no_grad():
        result = model(frames, return_states=True)
    maps = error_maps(result["E"], (64, 96))
    assert maps.shape == (1, spec.seq_len, 3, 64, 96) and (maps >= 0).all()
    layer = result["E"][5][2].mean(dim=1)[0]  # layer 2 at frame 5, mean over channels
    assert np.isclose(maps[0, 5, 2].mean(), layer.mean().item(), rtol=1e-4)
    normalized = normalize(maps[0])
    assert normalized.min() >= 0.0 and normalized.max() <= 1.0
    prediction = result["prediction"][0].permute(0, 2, 3, 1).numpy()
    assert error_sheet(rendered, prediction, normalized, [5, 6], "test").size[0] > 0
