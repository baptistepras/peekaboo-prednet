"""Tests for the blackout and closed loop step (3.6): the closed loop mask of both models (an open mask is the open
loop, closed frames are read as the model's own predictions, the old extrapolate_from is a special case), the frames
without the blackout, the closed loop pass, and the analysis on a fake table with known answers."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from peekaboo.config import load_config
from peekaboo.data.conditions import GeneratorSettings
from peekaboo.data.dataset import OnTheFlyDataset, make_loader
from peekaboo.data.mnist_pool import DigitPool
from peekaboo.data.render import RenderSettings
from peekaboo.data.truth import STATE_NAMES
from peekaboo.eval.closed_loop import add_loop_errors, closed_loop, loop_runs, plot_loop
from peekaboo.eval.extract import (LOOP_FORMAT, LOOPS, cell_range, extract_frames, hidden_frames,
                                   loop_frames)
from peekaboo.models import build_model
from peekaboo.models.loop import closed_mask
from peekaboo.paths import CONFIGS_DIR

RENDER = RenderSettings.from_config(load_config(CONFIGS_DIR / "data" / "base.yaml"))
CPU = torch.device("cpu")
MODELS = {"prednet": {"model": "prednet", "stack_sizes": [3, 4, 8], "layer_loss_weights": "L0"},
          "concat": {"model": "prednet", "stack_sizes": [3, 4, 8], "layer_loss_weights": "L0",
                     "error_mode": "concat"},
          "convlstm": {"model": "convlstm", "hidden_sizes": [4], "kernel_size": 3, "patch_size": 4}}
BLACKOUT = STATE_NAMES.index("blackout")


def tiny(name: str) -> torch.nn.Module:
    """A small model with fixed weights."""
    torch.manual_seed(0)
    return build_model(MODELS[name]).eval()


def test_closed_mask() -> None:
    """The mask of extrapolate_from closes every frame from it on, frame 0 is never closed, and a wrong shape or both
    arguments are refused."""
    frames = torch.zeros(2, 5, 3, 8, 8)
    assert closed_mask(frames) is None
    assert closed_mask(frames, extrapolate_from=3).tolist() == [[False, False, False, True, True]] * 2
    assert not closed_mask(frames, closed=torch.ones(2, 5, dtype=torch.bool))[:, 0].any()
    with pytest.raises(ValueError):
        closed_mask(frames, extrapolate_from=2, closed=torch.ones(2, 5, dtype=torch.bool))
    with pytest.raises(ValueError):
        closed_mask(frames, closed=torch.ones(2, 4, dtype=torch.bool))


@pytest.mark.parametrize("name", sorted(MODELS))
def test_an_open_mask_is_the_open_loop(name: str) -> None:
    """A mask with no closed frame gives exactly the open loop outputs."""
    model = tiny(name)
    frames = torch.rand(2, 6, 3, 16, 24)
    with torch.no_grad():
        open_loop = model(frames, return_states=True)
        masked = model(frames, return_states=True, closed=torch.zeros(2, 6, dtype=torch.bool))
    assert torch.equal(open_loop["prediction"], masked["prediction"])
    for a, b in zip(open_loop["R"], masked["R"]):
        assert all(torch.equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("name", sorted(MODELS))
def test_closed_frames_are_the_model_own_predictions(name: str) -> None:
    """In closed loop, every closed frame is read as the model's prediction of that frame: the open loop on frames
    where the closed ones are replaced by those predictions gives the same outputs. The mask is per sequence, the
    closed frames do not change any prediction, and the open sequence of the batch is its open loop."""
    model = tiny(name)
    frames = torch.rand(2, 7, 3, 16, 24)
    mask = torch.zeros(2, 7, dtype=torch.bool)
    mask[0, 2:5] = True
    with torch.no_grad():
        closed = model(frames, closed=mask)
        replaced = torch.where(mask[:, :, None, None, None], closed["prediction"], frames)
        again = model(replaced)
        other = frames.clone()
        other[0, 2:5] = torch.rand(3, 3, 16, 24)
        moved = model(other, closed=mask)
        open_loop = model(frames)
    assert torch.equal(closed["prediction"], again["prediction"])
    assert torch.equal(closed["prediction"], moved["prediction"])
    assert torch.equal(closed["prediction"][1], open_loop["prediction"][1])
    assert not torch.equal(closed["prediction"][0], open_loop["prediction"][0])
    with torch.no_grad():
        later = (torch.arange(7) >= 3).expand(2, 7)
        assert torch.equal(model(frames, extrapolate_from=3)["prediction"], model(frames, closed=later)["prediction"])


def test_prednet_pixel_errors_vanish_in_closed_loop() -> None:
    """In closed loop, PredNet's pixel layer compares its prediction with itself: its errors are exactly 0 there."""
    model = tiny("prednet")
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[:, 3:] = True
    with torch.no_grad():
        errors = model(torch.rand(2, 6, 3, 16, 24), closed=mask)["layer_errors"]
    assert (errors[:, 3:, 0] == 0).all() and (errors[:, 1:3, 0] > 0).all()


def blackouts(pool: DigitPool, settings: GeneratorSettings) -> OnTheFlyDataset:
    """Blackout sequences (k = 4) with their amodal frames and the frames without the blackout."""
    return OnTheFlyDataset(pool, settings, RENDER, "test", 0, condition="blackout", k=4, speed=2,
                           include_amodal=True, include_unblacked=True)


def test_frames_without_the_blackout(pool: DigitPool, settings: GeneratorSettings) -> None:
    """The unblacked frames are the observed ones except on the blackout frames, where they show the bar and the
    digit; hidden_frames gives exactly the blackout frames."""
    batch = next(iter(make_loader(blackouts(pool, settings), 4, 0, 4)))
    blackout = batch["state"] == BLACKOUT
    assert blackout.sum(dim=1).tolist() == [4] * 4
    assert torch.equal(batch["frames"][~blackout], batch["unblacked"][~blackout])
    assert (batch["frames"][blackout] == 0).all()
    unblacked = batch["unblacked"][blackout]
    assert (unblacked[:, 0] > unblacked[:, 1]).flatten(1).any(dim=1).all()  # red ink of the digit on every frame
    assert torch.equal(hidden_frames(batch), blackout)
    plain = next(iter(make_loader(OnTheFlyDataset(pool, settings, RENDER, "test", 0, condition="occlusion", k=4,
                                                  speed=2, include_unblacked=True), 2, 0, 2)))
    assert torch.equal(plain["frames"], plain["unblacked"])
    assert torch.equal(hidden_frames(plain), plain["state"] == STATE_NAMES.index("occluded"))


def test_cell_range() -> None:
    """The consecutive sequences of a cell condition; tuple members are left out, and a split cell is refused."""
    class Spec:
        def __init__(self, condition: str, role: str = "none") -> None:
            self.condition, self.tuple_role = condition, role

    class Stored:
        specs = [Spec("occlusion"), Spec("occlusion", "A"), Spec("blackout"), Spec("blackout"), Spec("control")]

    assert cell_range(Stored(), "blackout") == (2, 2)
    Stored.specs.append(Spec("blackout"))
    with pytest.raises(ValueError):
        cell_range(Stored(), "blackout")


@pytest.mark.parametrize("name", ["prednet", "convlstm"])
def test_loop_pass(name: str, pool: DigitPool, settings: GeneratorSettings) -> None:
    """The open loop columns score the open loop against the frames without the blackout (those of the extraction
    pass off the blackout); both loops are the same before the onset; the closed loop draws on the blackout frames
    where the open loop sees black; and PredNet's pixel errors are 0 on the closed frames."""
    model = tiny(name)
    dataset = blackouts(pool, settings)
    rows = loop_frames(model, dataset, 3, 4, CPU, batch_size=2)
    frames = extract_frames(model, dataset, 3, 4, CPU, batch_size=2)
    assert rows["sequence"].tolist() == frames["sequence"].tolist()
    assert rows["closed"].tolist() == (frames["state"] == BLACKOUT).tolist()
    seen = ~rows["closed"] & (rows["t"] >= 1)
    np.testing.assert_allclose(rows.loc[seen, "open_mse"], frames.loc[seen, "model_mse"], rtol=1e-6)
    before = rows["since_onset"] < 0
    for column in ("mse", "mse_digit", "ink"):
        np.testing.assert_array_equal(rows.loc[before, f"open_{column}"], rows.loc[before, f"closed_{column}"])
    assert rows.loc[rows["closed"], "ref_ink"].gt(0).all()
    if name == "prednet":
        assert (rows.loc[rows["closed"], "closed_e0"] == 0).all() and "open_e1_bar" in rows
    else:
        assert "open_e0" not in rows


def fake_loops() -> pd.DataFrame:
    """Two modes, one model, 6 sequences of 12 frames with k = 4 (frames 4 to 7 closed). The probe is 2 px off in open
    loop and 1 px in closed loop on the closed frames (0.5 px elsewhere); the open loop draws nothing on the closed
    frames, the closed loop draws the digit 0.5 px off; Kalman is 3 px off."""
    parts = []
    for mode in ("blackout", "occlusion"):
        for sequence in range(6):
            t = np.arange(12)
            closed = (t >= 4) & (t < 8)
            y = np.full(12, 30.0)
            parts.append(pd.DataFrame({
                "mode": mode, "model": "PredNet", "seed": 0, "sequence": sequence + (100 if mode == "occlusion" else 0),
                "t": t, "k": 4, "since_onset": t - 4, "since_reappear": t - 8, "closed": closed, "y": y,
                "ref_ink_y": y + 0.1,
                "open_probe_y": y + np.where(closed, 2.0, 0.5), "closed_probe_y": y + np.where(closed, 1.0, 0.5),
                "open_decoder_corr": np.where(closed, 0.2, 0.8), "closed_decoder_corr": np.where(closed, 0.6, 0.8),
                "open_ink_y": np.where(closed, np.nan, y + 0.1), "closed_ink_y": y + np.where(closed, 0.6, 0.1),
                "open_mse_digit": np.where(closed, 0.02, 0.001), "closed_mse_digit": np.where(closed, 0.005, 0.001),
                "kalman_y": y + 3.0, "last_seen_y": y + 4.0}))
    return pd.concat(parts, ignore_index=True)


def test_loop_runs_of_a_model_set(tmp_path: Path) -> None:
    """The complete passes of one mode are found for the models of the set only, and a pass with older columns is
    refused."""
    def write(key: str, seed: int, mode: str, version: int = LOOP_FORMAT) -> None:
        folder = tmp_path / key / f"seed{seed}" / "eval" / f"loop_{mode}_dev_v1"
        folder.mkdir(parents=True)
        (folder / "info.json").write_text(json.dumps({"settings": {"format": version}, "complete": True}))

    write("sweep_prednet5l", 0, "blackout")
    write("blackout_prednet5l", 0, "blackout")
    write("blackout_prednet5l", 2, "blackout")
    write("blackout_prednet5l", 1, "occlusion")
    found = loop_runs(tmp_path, "dev_v1", "blackout", "blackout")
    assert [(run.key, run.seed) for run in found] == [("sweep_prednet5l", 0), ("blackout_prednet5l", 0),
                                                      ("blackout_prednet5l", 2)]
    assert [(run.key, run.seed) for run in loop_runs(tmp_path, "dev_v1", "blackout")] == [("sweep_prednet5l", 0)]
    write("sweep_prednet5l", 1, "blackout", LOOP_FORMAT - 1)
    with pytest.raises(ValueError):
        loop_runs(tmp_path, "dev_v1", "blackout", "blackout")


def test_analysis_with_several_seeds() -> None:
    """A second seed whose closed loop probe is 1 px further off moves the model's mean by 0.5 px; the references are
    read once."""
    table = fake_loops()
    second = table.assign(seed=1, closed_probe_y=table["closed_probe_y"] + 1.0)
    tables = closed_loop(add_loop_errors(pd.concat([table, second], ignore_index=True)))
    hidden = tables["summary"]
    hidden = hidden[(hidden["frames"] == "hidden") & (hidden["k"] == "all") & (hidden["mode"] == "blackout")]
    value = hidden.set_index(["who", "loop", "measure"])["mean"]
    assert value[("PredNet", "closed", "probe_y")] == pytest.approx(1.5)
    assert value[("Kalman", "", "probe_y")] == pytest.approx(3.0)
    curve = tables["curves"]
    kalman = curve[(curve["who"] == "Kalman") & (curve["mode"] == "blackout") & (curve["measure"] == "probe_y")]
    assert (kalman["n"] == 6).all()


def test_analysis_on_a_fake_table(tmp_path: Path) -> None:
    """The summary, the paired differences, and the curves give the known answers, and the figures are written."""
    tables = closed_loop(add_loop_errors(fake_loops()))
    summary, paired = tables["summary"], tables["paired"]
    hidden = summary[(summary["frames"] == "hidden") & (summary["k"] == "all") & (summary["mode"] == "blackout")]
    value = hidden.set_index(["who", "loop", "measure"])["mean"]
    assert value[("PredNet", "open", "probe_y")] == pytest.approx(2.0)
    assert value[("PredNet", "closed", "probe_y")] == pytest.approx(1.0)
    assert value[("PredNet", "open", "drawn")] == pytest.approx(0.0)
    assert value[("PredNet", "closed", "drawn")] == pytest.approx(1.0)
    assert value[("PredNet", "closed", "drawn_y")] == pytest.approx(0.5)
    assert value[("Kalman", "", "probe_y")] == pytest.approx(3.0)
    assert value[("last seen", "", "drawn_y")] == pytest.approx(4.0)
    behind = summary[(summary["frames"] == "hidden") & (summary["mode"] == "occlusion")]
    assert not behind["measure"].isin(["drawn_y", "drawn"]).any() and behind["measure"].eq("probe_y").any()
    difference = paired[(paired["frames"] == "hidden") & (paired["k"] == "all")].set_index(["mode", "measure"])
    assert difference.loc[("occlusion", "probe_y"), "mean"] == pytest.approx(-1.0)
    assert difference.loc[("blackout", "decoder_corr"), "mean"] == pytest.approx(0.4)
    back = summary[(summary["frames"] == "return") & (summary["k"] == "4")]
    back = back.set_index(["mode", "who", "loop", "measure"]).sort_index()["mean"]
    assert back[("blackout", "PredNet", "open", "probe_y")] == pytest.approx(0.5)
    assert back[("blackout", "Kalman", "", "probe_y")] == pytest.approx(3.0)
    curve = tables["curves"]
    at = curve[(curve["mode"] == "blackout") & (curve["measure"] == "probe_y") & (curve["loop"] == "closed")]
    assert at.set_index("j").loc[2, "mean"] == pytest.approx(1.0) and at["j"].min() == -3 and at["j"].max() == 7
    for mode in ("blackout", "occlusion"):
        assert plot_loop(curve, mode, tmp_path / f"{mode}.png", mode, k=4).exists()
    assert set(LOOPS) == {"open", "closed"}
