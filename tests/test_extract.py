"""Tests for the extraction pass of Phase 3: it gives exactly the scores of the step 2 evaluations, its regions and
readouts are right on the true frames, and a stopped pass resumes."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from peekaboo.config import load_config
from peekaboo.data.conditions import GeneratorSettings
from peekaboo.data.dataset import OnTheFlyDataset, make_loader, prepare_batch
from peekaboo.data.mnist_pool import DigitPool
from peekaboo.data.render import RenderSettings
from peekaboo.data.truth import STATE_NAMES
from peekaboo.eval.extract import (baseline_frames, bar_mask, error_activity, extract_frames, ink_readout,
                                   is_complete, part_path, read_frames, region_mse, regions, run_chunks)
from peekaboo.eval.next_frame import evaluate, frame_scores
from peekaboo.models import build_model
from peekaboo.paths import CONFIGS_DIR
from peekaboo.probes.amodal import evaluate_decoders, fit_decoder
from peekaboo.probes.evaluate import baseline_positions, probe_positions
from peekaboo.probes.features import collect_frames
from peekaboo.probes.position import fit_position_probe

RENDER = RenderSettings.from_config(load_config(CONFIGS_DIR / "data" / "base.yaml"))
CPU = torch.device("cpu")
TINY = {"prednet": {"model": "prednet", "stack_sizes": [3, 4, 8], "layer_loss_weights": "L0"},
        "convlstm": {"model": "convlstm", "hidden_sizes": [4], "kernel_size": 3, "patch_size": 4}}


def held_out(pool: DigitPool, settings: GeneratorSettings) -> OnTheFlyDataset:
    """Sequences of the validation stream, with their amodal frames."""
    return OnTheFlyDataset(pool, settings, RENDER, "val", 0, include_amodal=True)


def test_extraction_matches_the_existing_evaluations(pool: DigitPool, settings: GeneratorSettings) -> None:
    """On the same sequences, the extracted next frame scores, probe positions, decoder scores, and baselines are
    those of eval_next_frame, fit_probes, and fit_decoder; the control columns read the control model."""
    torch.manual_seed(0)
    stream = OnTheFlyDataset(pool, settings, RENDER, "probe", 0, include_amodal=True)
    data = held_out(pool, settings)
    model = build_model(TINY["prednet"])
    frames = collect_frames(model, stream, 4, CPU, batch_size=2, window_only=True)
    probe, _ = fit_position_probe(frames.features, frames.rows[["y", "x"]].to_numpy(),
                                  frames.rows["sequence"].to_numpy(), (64, 96), layer_limit=2048)
    decoder, _ = fit_decoder(model, stream, 2, CPU, batch_size=2, log=lambda line: None)
    table = extract_frames(model, data, 0, 3, CPU, probe, decoder, control=(model, probe, decoder), batch_size=2)
    assert len(table) == 3 * 40 and list(table["sequence"].unique()) == [0, 1, 2]

    scored = table[table["t"] >= 1].reset_index(drop=True)
    reference = evaluate(model, data, CPU, batch_size=3, count=3)
    for column in ("model_mse", "model_ssim", "copy_mse", "blank_mae"):
        assert np.allclose(scored[column], reference[column], rtol=1e-4, atol=1e-7), column

    keys = ["sequence", "t"]
    probed = probe_positions(model, probe, data, 3, CPU, batch_size=3).merge(table, on=keys)
    assert len(probed) > 0
    assert np.allclose(probed["probe_y"], probed["pred_y"], atol=1e-3)
    assert np.allclose(probed["probe_p_bin_x"], probed["p_bin_x"], atol=1e-4)
    decoded = evaluate_decoders({"decoder": (model, decoder)}, data, 3, CPU, batch_size=3)
    both = decoded.merge(table, on=keys, suffixes=("_ref", ""))
    assert np.allclose(both["decoder_corr_ref"], both["decoder_corr"], atol=1e-4, equal_nan=True)
    assert np.allclose(table["random_probe_y"], table["probe_y"], equal_nan=True)  # the control read the same model
    assert (table[["e0", "e1", "e2", "e0_bar"]] >= 0).all().all()

    baselines = baseline_frames(data, 0, 3, batch_size=2)
    trackers = baseline_positions(data, 3).merge(baselines, on=keys, suffixes=("_ref", ""))
    assert np.allclose(trackers["kalman_y_ref"], trackers["kalman_y"], equal_nan=True, atol=1e-4)
    assert np.allclose(trackers["bar_center_x_ref"], trackers["bar_center_x"], equal_nan=True)
    templates = decoded.merge(baselines, on=keys, suffixes=("_ref", ""))
    assert np.allclose(templates["template_walls_iou_ref"], templates["template_walls_iou"], equal_nan=True)


def test_regions_and_readouts_on_the_true_frames(pool: DigitPool, settings: GeneratorSettings) -> None:
    """On the true frames, the ink readout finds the visible digit at its visible ink center and on one side of the
    bar; the bar region holds the bar, the digit region holds the digit, and an all frame region gives the MSE."""
    batch = prepare_batch(next(iter(make_loader(held_out(pool, settings), 3, 0, 3))), CPU)
    frames, amodal = batch["frames"], batch["amodal"]
    bar = bar_mask(frames[:, 0])
    assert bar.any(dim=1).all()
    readout = ink_readout(frames, bar)
    visible = batch["state"] == STATE_NAMES.index("visible")
    assert visible.any()
    for axis, name in enumerate(("ink_y", "ink_x")):
        assert torch.allclose(readout[name][visible], batch["modal_center"][..., axis][visible], atol=0.5)
    sides = readout["ink_left"] + readout["ink_right"]
    assert (sides[visible] >= 0.9 * readout["ink"][visible]).all()
    absent = batch["state"] == STATE_NAMES.index("occluded")
    assert torch.isnan(readout["ink_y"][absent]).all()

    areas = regions(bar, amodal)
    assert areas["bar"][bar[:, None, None, :].expand_as(areas["bar"])].all()
    assert areas["digit"][amodal[:, :, 0] > 0].all()
    guess = torch.rand_like(frames)
    everywhere = torch.ones_like(areas["bar"])
    assert torch.allclose(region_mse(guess, frames, everywhere)[:, 1:], frame_scores(guess, frames)["mse"][:, 1:])


def test_error_activity_over_the_bar() -> None:
    """With a bar over the whole width, the error activity over the bar is that of the whole frame, at every layer."""
    torch.manual_seed(0)
    errors = [[torch.rand(2, 4, 8, 48), torch.rand(2, 6, 4, 24)] for _ in range(3)]
    activity = error_activity(errors, torch.ones(2, 48, dtype=torch.bool))
    for l in range(2):
        assert activity[f"e{l}"].shape == (2, 3)
        assert torch.allclose(activity[f"e{l}_bar"], activity[f"e{l}"])
    narrow = torch.zeros(2, 48, dtype=torch.bool)  # the bar region is column 0 and its margin
    narrow[:, 0] = True
    assert not torch.allclose(error_activity(errors, narrow)["e0_bar"], activity["e0"])


def test_a_model_without_error_units(pool: DigitPool, settings: GeneratorSettings) -> None:
    """The ConvLSTM has no error unit columns, and the pass runs without probe, decoder, or control."""
    table = extract_frames(build_model(TINY["convlstm"]), held_out(pool, settings), 0, 2, CPU, batch_size=2)
    assert "e0" not in table and "probe_y" not in table and table["pred_ink"].notna().all()


def test_a_stopped_pass_resumes(tmp_path: Path) -> None:
    """Chunks already written are skipped, a pass stopped by an error is not complete, and a folder written with
    other settings is refused."""
    calls = []

    def make_part(start: int, size: int) -> pd.DataFrame:
        """Two frames per sequence; fails on the chunk at 2 while `fail` is set."""
        if start == 2 and fail:
            raise RuntimeError("stopped")
        calls.append(start)
        return pd.DataFrame({"sequence": np.repeat(np.arange(start, start + size), 2), "t": np.tile([0, 1], size)})

    folder, quiet = tmp_path / "frames", (lambda line: None)
    fail = True
    with pytest.raises(RuntimeError):
        run_chunks(folder, 5, 2, make_part, {"a": 1}, log=quiet)
    assert calls == [0] and not is_complete(folder) and part_path(folder, 0).exists()
    with pytest.raises(FileNotFoundError):
        read_frames(folder)
    fail = False
    run_chunks(folder, 5, 2, make_part, {"a": 1}, log=quiet)
    assert calls == [0, 2, 4] and is_complete(folder)
    assert read_frames(folder)["sequence"].tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    with pytest.raises(ValueError):
        run_chunks(folder, 5, 2, make_part, {"a": 2}, log=quiet)
