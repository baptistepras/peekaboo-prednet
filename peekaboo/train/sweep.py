"""The training sweep of step 2.10: its runs in their order, the commands of each stage, and the state of every run,
read from its folder, so that the runs can be launched one at a time and checked in between.

A run goes through five stages: training, the next frame gate D16 (scripts/eval_next_frame.py), the position probes
(scripts/fit_probes.py), the amodal decoder (scripts/fit_decoder.py), and the prediction figures with both readouts
(scripts/show_predictions.py). Each stage is done when its output exists in the run folder. Two checks read the
validation log while training: the model must leave the blank frame early (the first pilot never did, nor did PredNet
4 layers with seed 0, see SWEEP), and the validation MSE should have stopped falling fast before the learning rate drop
(the pilot's sudden change came at step 4500, and a change still under way at the drop would be much slower after it).
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from peekaboo.config import load_config
from peekaboo.paths import PROJECT_ROOT
from peekaboo.train.loop import TrainSettings
from peekaboo.train.metrics import read_csv

DATA = "val_v1"                                  # the set of every evaluation stage
FIGURES = ("occlusion", "hidden_bounce")         # conditions of the prediction figures
STAGES = ("train", "gate", "probes", "decoder", "figures")
HIDDEN = "occlusion / occluded"                  # the summary row shown for the readouts: hidden frames
RUNNING_WITHIN = 600                             # s: a run whose step log changed more recently is taken as running
BLANK_CHECK_FROM = 1000                          # step from which the model must be well below the blank frame
BLANK_REQUIRED = 0.25                            # required MSE reduction against the blank frame from that step on
                                                 # (5 layers and concat were about 40% below at step 1000)
PLATEAU_WINDOW = 2000                            # steps before the drop over which the MSE should be nearly flat
PLATEAU_GAIN = 0.2                               # largest acceptable MSE reduction over that window
MPS = "PYTORCH_ENABLE_MPS_FALLBACK=0"


@dataclass(frozen=True)
class SweepRun:
    """One run of the sweep: its training config (configs/train/<key>.yaml, whose name is the key), its seed, and the
    expected durations on an Apple M5, from the 50 step benchmarks of steps 2.5 and 2.6 and the pilot's evaluation."""

    key: str
    label: str
    train_hours: float      # training, validations included
    after_minutes: float    # gate, probes, decoder, and figures
    seed: int = 0           # overrides train.seed of the config when not 0

    @property
    def config(self) -> str:
        """The training config, relative to the project root."""
        return f"configs/train/{self.key}.yaml"

    def run_dir(self, runs_dir: Path) -> Path:
        """The run folder, as scripts/train.py names it."""
        return runs_dir / self.key / f"seed{self.seed}"


SWEEP = (
    SweepRun("sweep_prednet5l", "PredNet 5 layers", 7.5, 20),
    SweepRun("sweep_prednet5l_concat", "PredNet 5 layers, concat", 7.5, 20),
    SweepRun("sweep_convlstm", "ConvLSTM", 15.0, 35),
    # seed 0 stayed on the blank frame from step 100 on, then its gradient exploded at step 3970 (norm 1.3e4, no
    # clipping then) and its predictions stayed black for 10k steps: kept in runs/sweep_prednet4l/seed0 as a record
    SweepRun("sweep_prednet4l", "PredNet 4 layers", 13.3, 35, seed=1),
    SweepRun("sweep_prednet3l", "PredNet 3 layers", 30.0, 60),
)
FAILED = (SweepRun("sweep_prednet4l", "PredNet 4 layers, seed 0 (failed)", 13.3, 35),)  # records, not in the sweep
REFERENCE = "sweep_prednet5l"  # the reference model (README, Phase 3): best readouts, fastest to train


def find_run(key: str) -> SweepRun:
    """The sweep run with this key."""
    for run in SWEEP:
        if run.key == key:
            return run
    raise KeyError(f"unknown sweep run '{key}': {', '.join(r.key for r in SWEEP)}")


def stage_commands(run: SweepRun, restart: str = "") -> dict[str, str]:
    """The command of every stage, to run from the project root. Training keeps the Mac awake and appends its output
    to runs/<key>_seed<seed>.log; `restart` is "--resume" or "--overwrite" for a run that stopped."""
    seed = run.seed
    run_dir = f"runs/{run.key}/seed{seed}"
    seed_option = f" --seed {seed}" if seed else ""
    resume_option = f" {restart}" if restart else ""
    figures = " && ".join(f"python -m scripts.show_predictions --run {run_dir} --probe --decoder --condition {c}"
                          for c in FIGURES)
    return {
        "train": f"{MPS} caffeinate -i python -u -m scripts.train --config {run.config}{seed_option}{resume_option} "
                 f"2>&1 | tee -a runs/{run.key}_seed{seed}.log",
        "gate": f"python -m scripts.eval_next_frame --run {run_dir}",
        "probes": f"{MPS} python -m scripts.fit_probes --run {run_dir}",
        "decoder": f"{MPS} python -m scripts.fit_decoder --run {run_dir}",
        "figures": figures,
    }


def check_commands(run: SweepRun) -> str:
    """The whole chain of a run on a few steps and sequences, in runs/check/, to check that every stage works with
    this model before the long training. Takes a few minutes."""
    run_dir = f"runs/check/{run.key}/seed0"
    return " && ".join([
        f"{MPS} python -m scripts.train --config {run.config} --stop-at 20 --set validation.sequences=32 "
        f"--out runs/check --overwrite",
        f"python -m scripts.eval_next_frame --run {run_dir} --n 32",
        f"{MPS} python -m scripts.fit_probes --run {run_dir} --train-sequences 40 --n 32",
        f"{MPS} python -m scripts.fit_decoder --run {run_dir} --steps 10 --batch-size 4 --n 32",
        f"python -m scripts.show_predictions --run {run_dir} --probe --decoder --n 1",
    ])


def read_json(path: Path) -> dict[str, Any] | None:
    """A JSON report, or None if it does not exist yet."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def validation_rows(run_dir: Path) -> list[dict[str, float]]:
    """The validation log as numbers (empty fields, from logs older than the blank frame column, are left out)."""
    path = run_dir / "val_metrics.csv"
    if not path.exists():
        return []
    return [{k: float(v) for k, v in row.items() if v not in ("", None)} for row in read_csv(path)]


def relative(value: float, reference: float) -> str:
    """How far below (or above) a reference a value is, in percent."""
    change = 1.0 - value / reference
    return f"{100 * change:.0f}% below" if change >= 0 else f"{-100 * change:.0f}% above"


def blank_warning(rows: list[dict[str, float]], finished: bool = False) -> str | None:
    """A warning when, from step BLANK_CHECK_FROM on, the validation MSE is not at least BLANK_REQUIRED below the
    blank frame: the model may have settled on never drawing the digit, as the first pilot did."""
    rows = [r for r in rows if "blank_mse" in r]
    if not rows or rows[-1]["step"] < BLANK_CHECK_FROM:
        return None
    last = rows[-1]
    reduction = 1.0 - last["val_mse"] / last["blank_mse"]
    if reduction >= BLANK_REQUIRED:
        return None
    return (f"at step {last['step']:.0f} the validation MSE is {relative(last['val_mse'], last['blank_mse'])} the "
            f"blank frame (at least {100 * BLANK_REQUIRED:.0f}% below expected): the model may not draw the digit; "
            f"look at curves.png" + ("" if finished else " and consider stopping the run"))


def plateau_gain(rows: list[dict[str, float]], drop_step: int) -> float | None:
    """The relative fall of the validation MSE over the PLATEAU_WINDOW steps before the learning rate drop, or None
    before the drop."""
    mse = {int(r["step"]): r["val_mse"] for r in rows}
    if drop_step not in mse or drop_step - PLATEAU_WINDOW not in mse:
        return None
    return 1.0 - mse[drop_step] / mse[drop_step - PLATEAU_WINDOW]


@dataclass
class RunStatus:
    """The state of a sweep run, read from its folder."""

    run: SweepRun
    state: str                              # "not started", "running", "stopped", or "trained"
    step: int                               # last training step logged
    resumable: bool                         # last.pt exists, so a stopped run continues from it
    steps: int
    done: dict[str, bool]
    last_validation: dict[str, float] | None = None
    seconds_per_step: float | None = None
    plateau: float | None = None
    gate: dict[str, Any] | None = None
    probe: dict[str, float] | None = None
    decoder: dict[str, float] | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def next_stage(self) -> str | None:
        """The first stage not done, or None when the run is complete."""
        return next((stage for stage in STAGES if not self.done[stage]), None)

    def hours_left(self) -> float:
        """Expected hours until the run is complete."""
        hours = 0.0 if self.done["figures"] else self.run.after_minutes / 60
        if not self.done["train"]:
            if self.seconds_per_step:  # measured speed, plus the validations left (about 4% of the time)
                hours += 1.04 * self.seconds_per_step * (self.steps - self.step) / 3600
            else:
                hours += (1.0 - self.step / self.steps) * self.run.train_hours
        return hours


def run_status(run: SweepRun, runs_dir: Path, now: float | None = None) -> RunStatus:
    """Read a run's folder: training progress, the checks on its validation log, and the outputs of every stage."""
    settings = TrainSettings.from_config(load_config(PROJECT_ROOT / run.config))
    run_dir = run.run_dir(runs_dir)
    log = run_dir / "train_metrics.csv"
    train_rows = read_csv(log) if log.exists() else []
    step = int(train_rows[-1]["step"]) if train_rows else 0
    trained = step == settings.steps and (run_dir / "best.pt").exists() and not (run_dir / "last.pt").exists()
    if trained:
        state = "trained"
    elif not train_rows:
        state = "not started"
    elif (now or time.time()) - log.stat().st_mtime < RUNNING_WITHIN:
        state = "running"
    else:
        state = "stopped"
    gate = read_json(run_dir / "eval" / f"next_frame_{DATA}.json")
    probe = read_json(run_dir / "eval" / f"probe_{DATA}.json")
    decoder = read_json(run_dir / "eval" / f"amodal_{DATA}.json")
    done = {"train": trained, "gate": gate is not None, "probes": probe is not None, "decoder": decoder is not None,
            "figures": all((run_dir / "predictions_probe" / f"{c}.gif").exists() for c in FIGURES)}
    rows = validation_rows(run_dir)
    recent = [float(r["seconds"]) for r in train_rows[-100:]]
    status = RunStatus(run=run, state=state, step=step, resumable=(run_dir / "last.pt").exists(),
                       steps=settings.steps, done=done,
                       last_validation=rows[-1] if rows else None,
                       seconds_per_step=sum(recent) / len(recent) if recent else None,
                       gate=gate["gate"] if gate else None, probe=probe["error_y"].get(HIDDEN) if probe else None,
                       decoder=decoder["corr"].get(HIDDEN) if decoder else None)
    drop_step = round(settings.lr_drop_at * settings.steps) if settings.lr_drop_at is not None else None
    if drop_step is not None:
        status.plateau = plateau_gain(rows, drop_step)
    warning = blank_warning(rows, finished=trained)
    if warning:
        status.warnings.append(warning)
    if status.plateau is not None and status.plateau > PLATEAU_GAIN:
        status.warnings.append(
            f"the validation MSE still fell by {100 * status.plateau:.0f}% over the {PLATEAU_WINDOW} steps before the "
            f"learning rate drop at step {drop_step}: the sudden change may not have been over; compare curves.png "
            f"with the pilot's")
    if gate is not None and not gate["gate"]["passed"]:
        status.warnings.append(f"the gate D16 failed ({relative(gate['gate']['model_mse'], gate['gate']['blank_mse'])} "
                               f"the blank frame, {100 * gate['gate']['required_reduction']:.0f}% below required): "
                               f"document it before going on")
    if state == "stopped":
        status.warnings.append(
            f"training stopped at step {step} of {settings.steps}: " +
            ("resume it (it continues from the last validation)" if status.resumable else
             "no checkpoint was saved before the stop, so the run starts again"))
    return status


def next_action(statuses: list[RunStatus]) -> tuple[str, str | None]:
    """What to do next: a sentence, and the command to run (None while a run is training or when all is done)."""
    for status in statuses:
        stage = status.next_stage
        if stage is None:
            continue
        name = f"{status.run.label} ({status.run.key})"
        if status.state == "running":
            return f"wait: {name} is training, step {status.step} of {status.steps}", None
        restart = ("--resume" if status.resumable else "--overwrite") if status.state == "stopped" else ""
        commands = stage_commands(status.run, restart)
        if stage == "train":
            verb = "resume the training of" if restart == "--resume" else "train"
            return f"{verb} {name}, about {status.hours_left() - status.run.after_minutes / 60:.1f} h", commands[stage]
        return f"{stage} of {name}", commands[stage]
    return "the sweep is complete", None
