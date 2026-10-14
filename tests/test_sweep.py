"""Tests for the training sweep: its configs, and the state of a run read from its folder through every stage."""

import csv
import json
import os
import time
from pathlib import Path

import pytest

from peekaboo.config import load_config
from peekaboo.models import build_model
from peekaboo.models.prednet import count_parameters
from peekaboo.paths import PROJECT_ROOT
from peekaboo.train.loop import TrainSettings, learning_rate
from peekaboo.train.metrics import TRAIN_FIELDS, VAL_FIELDS
from peekaboo.train.sweep import FIGURES, SWEEP, check_commands, find_run, next_action, run_status, stage_commands

RUN = SWEEP[0]


def test_sweep_configs_differ_only_in_the_model() -> None:
    """Every sweep config has the same training settings and seed, its own name and model, the weighted loss, and
    the learning rate dropped by 10 exactly at step 10000 of 15000."""
    configs = {run.key: load_config(PROJECT_ROOT / run.config) for run in SWEEP}
    shared = [{k: v for k, v in c.items() if k not in ("name", "model")} for c in configs.values()]
    assert all(s == shared[0] for s in shared)
    assert len({c["model"] for c in configs.values()}) == len(SWEEP)
    for key, config in configs.items():
        assert config["name"] == key and (PROJECT_ROOT / config["model"]).exists()
    settings = TrainSettings.from_config(shared[0])
    assert settings.steps == 15000 and settings.digit_weight == 10 and shared[0]["train"]["seed"] == 0
    assert settings.grad_clip == 1.0
    assert learning_rate(9999, settings) == pytest.approx(1e-3)
    assert learning_rate(10000, settings) == pytest.approx(1e-4)


def test_sweep_models_have_matched_parameters() -> None:
    """Every model of the sweep is within 10% of the parameters of PredNet 5 layers."""
    counts = {run.key: count_parameters(build_model(load_config(PROJECT_ROOT / load_config(
        PROJECT_ROOT / run.config)["model"]))) for run in SWEEP}
    reference = counts["sweep_prednet5l"]
    assert all(abs(n / reference - 1) < 0.1 for n in counts.values()), counts


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    """Write a log as the training loop does."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_training(run_dir: Path, steps: int, mse: dict[int, float], blank: float = 0.0015) -> None:
    """Fake logs of a run trained up to `steps`, validated at the steps of `mse`, at 2 s per step."""
    write_csv(run_dir / "train_metrics.csv", TRAIN_FIELDS,
              [{"step": s, "loss": 0.01, "grad_norm": 1.0, "lr": 1e-3, "seconds": 2.0} for s in range(1, steps + 1)])
    write_csv(run_dir / "val_metrics.csv", VAL_FIELDS,
              [{"step": s, "val_loss": 0.01, "val_l1": 0.001, "val_mse": m, "copy_l1": 0.003, "copy_mse": 0.002,
                "blank_l1": 0.002, "blank_mse": blank, "best_step": s} for s, m in mse.items()])


def age(path: Path, seconds: float) -> None:
    """Make a file look last written `seconds` ago."""
    stamp = time.time() - seconds
    os.utime(path, (stamp, stamp))


def test_a_run_goes_through_every_stage(tmp_path: Path) -> None:
    """The state, the next stage, and the next command follow the files of the run folder, from not started to
    complete, and the next run comes after."""
    run_dir = RUN.run_dir(tmp_path)
    status = run_status(RUN, tmp_path)
    assert status.state == "not started" and status.next_stage == "train"
    sentence, command = next_action([status])
    assert command == stage_commands(RUN)["train"] and "--resume" not in command and "caffeinate" in command

    write_training(run_dir, 3000, {0: 0.06, 500: 0.0012, 1000: 0.0008, 2000: 0.0006, 3000: 0.0005})
    status = run_status(RUN, tmp_path)
    assert status.state == "running" and next_action([status])[1] is None and not status.warnings
    assert status.hours_left() == pytest.approx(1.04 * 2 * 12000 / 3600 + RUN.after_minutes / 60)

    age(run_dir / "train_metrics.csv", 3600)
    status = run_status(RUN, tmp_path)
    assert status.state == "stopped" and "--overwrite" in next_action([status])[1]  # nothing to resume from
    (run_dir / "last.pt").touch()
    status = run_status(RUN, tmp_path)
    assert next_action([status])[1].endswith(f"--resume 2>&1 | tee -a runs/{RUN.key}_seed0.log")
    assert any("resume" in w for w in status.warnings)

    (run_dir / "last.pt").unlink()
    (run_dir / "best.pt").touch()
    write_training(run_dir, 15000, {0: 0.06, 2000: 0.0006, 8000: 0.00014, 10000: 0.00013, 15000: 0.00012})
    status = run_status(RUN, tmp_path)
    assert status.state == "trained" and status.next_stage == "gate" and not status.warnings
    assert status.plateau == pytest.approx(1 - 0.00013 / 0.00014)
    assert next_action([status])[1] == stage_commands(RUN)["gate"]

    eval_dir = run_dir / "eval"
    eval_dir.mkdir()
    gate = {"passed": True, "mse_reduction": 0.95, "model_mse": 0.0001, "blank_mse": 0.002, "baseline": "blank",
            "required_reduction": 0.3}
    (eval_dir / "next_frame_val_v1.json").write_text(json.dumps({"gate": gate}))
    assert run_status(RUN, tmp_path).next_stage == "probes"
    row = {"occlusion / occluded": {"probe": 0.6, "random_probe": 9.7}}
    (eval_dir / "probe_val_v1.json").write_text(json.dumps({"error_y": row}))
    row = {"occlusion / occluded": {"decoder": 0.74, "random_decoder": 0.16}}
    (eval_dir / "amodal_val_v1.json").write_text(json.dumps({"corr": row}))
    status = run_status(RUN, tmp_path)
    assert status.next_stage == "figures" and status.probe["probe"] == 0.6 and status.decoder["decoder"] == 0.74
    (run_dir / "predictions_probe").mkdir()
    for condition in FIGURES:
        (run_dir / "predictions_probe" / f"{condition}.gif").touch()
    statuses = [run_status(run, tmp_path) for run in SWEEP]
    assert statuses[0].next_stage is None and statuses[0].hours_left() == 0
    assert next_action(statuses)[1] == stage_commands(SWEEP[1])["train"]


def test_warnings_from_the_validation_log(tmp_path: Path) -> None:
    """A run still near the blank frame at step 2000, a validation MSE still falling fast before the drop, and a
    failed gate are reported."""
    run_dir = RUN.run_dir(tmp_path)
    write_training(run_dir, 2000, {0: 0.06, 1000: 0.0015, 2000: 0.0014}, blank=0.0015)
    assert any("blank frame" in w for w in run_status(RUN, tmp_path).warnings)

    write_training(run_dir, 15000, {0: 0.06, 2000: 0.0006, 8000: 0.0004, 10000: 0.0002, 15000: 0.00015})
    (run_dir / "best.pt").touch()
    (run_dir / "eval").mkdir()
    gate = {"passed": False, "mse_reduction": -0.5, "model_mse": 0.003, "blank_mse": 0.002, "baseline": "blank",
            "required_reduction": 0.3}
    (run_dir / "eval" / "next_frame_val_v1.json").write_text(json.dumps({"gate": gate}))
    warnings = run_status(RUN, tmp_path).warnings
    assert any("before the learning rate drop" in w for w in warnings)
    assert any("gate D16 failed (50% above the blank frame" in w for w in warnings)

    write_training(run_dir, 15000, {0: 0.06, 1000: 0.003, 15000: 0.0031}, blank=0.0015)
    warning = next(w for w in run_status(RUN, tmp_path).warnings if "blank frame" in w and "validation MSE" in w)
    assert "107% above the blank frame" in warning and "stopping" not in warning  # a finished run is not stopped


def test_a_run_with_its_own_seed(tmp_path: Path) -> None:
    """A run of the sweep with another seed is trained, read, and evaluated in its own folder."""
    run = find_run("sweep_prednet4l")
    assert run.seed == 1 and run.run_dir(tmp_path) == tmp_path / "sweep_prednet4l" / "seed1"
    commands = stage_commands(run)
    assert "--config configs/train/sweep_prednet4l.yaml --seed 1 2>&1 | tee -a runs/sweep_prednet4l_seed1.log" in \
        commands["train"]
    assert commands["gate"].endswith("--run runs/sweep_prednet4l/seed1")
    write_training(run.run_dir(tmp_path), 500, {0: 0.06, 500: 0.0012})
    assert run_status(run, tmp_path).step == 500


def test_check_commands_stay_in_their_own_folder() -> None:
    """The short check of the whole chain writes only to runs/check/ and covers every stage."""
    command = check_commands(RUN)
    assert "--out runs/check" in command and f"--run runs/check/{RUN.key}/seed0" in command
    for script in ("train", "eval_next_frame", "fit_probes", "fit_decoder", "show_predictions"):
        assert f"scripts.{script} " in command
