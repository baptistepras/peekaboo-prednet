"""Blackout and closed loop (step 3.6), from the closed loop passes (peekaboo/eval/extract.py loop_frames).

Each pass runs a model twice over the sequences of one cell condition: in open loop, as in the other analyses, and in
closed loop, where the model reads its own prediction in place of every frame from the onset to the reappearance.
Both are scored against the frames drawn without the blackout. Three questions, one value per sequence and seed:
- does the state keep the digit when the input disappears (blackout, never seen in training, D13): the probe's error
  on y and the decoder's correlation on the blackout frames in open loop, next to the hidden frames of the occlusion
  cells with the same k;
- does the model keep the digit when it runs on its own predictions (C10): the same readouts in closed loop, against
  open loop on the same sequences; during a blackout, the digit drawn by the model is not behind the bar, so its
  position can be read on the predicted frames (the ink centroid off the bar) against the frame without the blackout,
  a test of tracking on the pixels themselves (H1a); behind the bar of an occlusion cell, the frame without the
  blackout shows no digit, so the drawn digit is read there only from the reappearance on;
- is it ready when the input comes back: the MSE around the digit over the frames EMERGING since the reappearance.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from peekaboo.eval.extract import LOOP_FORMAT, LOOP_MODES, LOOPS, is_complete, pass_format, read_frames
from peekaboo.eval.occlusion import BEFORE, EMERGING, group_mean_se, interval, model_colors, scopes
from peekaboo.eval.stats import bootstrap, paired_difference
from peekaboo.eval.tables import KEYS, ModelRun, loop_folder, per_sequence, reference_rows
from peekaboo.train.sweep import FAILED, SWEEP

K_SHOWN = 8  # the k of the curves in the figures (both cell conditions have it)
AFTER = 4    # frames shown after the last blackout or hidden frame
# the measures: label and the column of a loop ({} is open or closed)
MEASURES = {"probe_y": ("probe |error| on y (px)", "{}_probe_err_y"),
            "decoder_corr": ("decoder correlation", "{}_decoder_corr"),
            "drawn_y": ("drawn digit |error| on y (px)", "{}_drawn_err_y"),
            "drawn": ("share of frames with a digit drawn", "{}_drawn"),
            "mse_digit": ("MSE around the digit", "{}_mse_digit")}
FRAMES = {"hidden": ("probe_y", "decoder_corr", "drawn_y", "drawn", "mse_digit"),  # the blackout or hidden frames
          "return": ("probe_y", "drawn_y", "mse_digit")}                        # the frames EMERGING since the return
BEHIND = ("drawn_y", "drawn")  # not read on the hidden frames of the occlusion cells, where the digit is behind the bar
REFERENCES = {"Kalman": "kalman_err_y", "last seen": "last_seen_err_y"}  # next to the position measures
POSITION = ("probe_y", "drawn_y")
CURVES = ("probe_y", "drawn_y", "decoder_corr", "mse_digit")
BASELINE_COLUMNS = ["sequence", "t", "kalman_y", "last_seen_y"]


def measures_of(mode: str, frames: str) -> tuple[str, ...]:
    """The measures of FRAMES read on these frames of the cells of `mode`: on the hidden frames of the occlusion cells,
    the reference frame shows no digit off the bar (it is behind it), so the drawn digit is left out there."""
    if mode == "occlusion" and frames == "hidden":
        return tuple(m for m in FRAMES[frames] if m not in BEHIND)
    return FRAMES[frames]


def loop_runs(runs_dir: Path, data: str, mode: str) -> list[ModelRun]:
    """Every seed of every sweep model with a complete closed loop pass on the cells of `mode` of the set; a pass with
    older columns is refused."""
    failed = {(run.key, run.seed) for run in FAILED}
    runs = []
    for run in SWEEP:
        seed_dirs = [p for p in (runs_dir / run.key).glob("seed*") if p.name[4:].isdigit()]
        for seed_dir in sorted(seed_dirs, key=lambda p: int(p.name[4:])):
            seed = int(seed_dir.name[4:])
            folder = loop_folder(seed_dir, data, mode)
            if (run.key, seed) in failed or not is_complete(folder):
                continue
            if pass_format(folder) != LOOP_FORMAT:
                raise ValueError(f"{folder} has the columns of format {pass_format(folder)}, not {LOOP_FORMAT}: write "
                                 f"it again with python -m scripts.extract --run {seed_dir} --data {data} "
                                 f"--closed-loop {mode} --overwrite")
            runs.append(ModelRun(run.key, run.label, seed, folder))
    return runs


def add_loop_errors(table: pd.DataFrame) -> pd.DataFrame:
    """For each loop: the probe's error on y against the true center, the drawn digit's error on y against the frame
    without the blackout (both ink centroids off the bar), and whether a digit is drawn where that frame shows one,
    both left undefined on the hidden frames of the occlusion cells (BEHIND); and the errors on y of the trackers
    against the true center."""
    columns = {}
    behind = (table["mode"] == "occlusion") & table["closed"] if "mode" in table else pd.Series(False, table.index)
    shown = table["ref_ink_y"].notna() & ~behind
    for loop in LOOPS:
        columns[f"{loop}_probe_err_y"] = (table[f"{loop}_probe_y"] - table["y"]).abs()
        columns[f"{loop}_drawn_err_y"] = (table[f"{loop}_ink_y"] - table["ref_ink_y"]).abs().where(shown)
        columns[f"{loop}_drawn"] = np.where(shown, table[f"{loop}_ink_y"].notna().astype(float), np.nan)
    for name, column in (("kalman", "kalman_y"), ("last_seen", "last_seen_y")):
        if column in table:
            columns[f"{name}_err_y"] = (table[column] - table["y"]).abs()
    return table.assign(**columns)


def read_loops(runs: dict[str, list[ModelRun]], baselines: Path) -> pd.DataFrame:
    """The analysis window rows of the closed loop passes of every mode and run, with `mode`, `model`, and `seed`,
    joined with the trackers of the set, and with the errors of add_loop_errors."""
    parts = [read_frames(run.folder, filters=[("in_window", "==", True)]).assign(mode=mode, model=run.label,
                                                                                 seed=run.seed)
             for mode, own in runs.items() for run in own]
    table = pd.concat(parts, ignore_index=True)
    trackers = read_frames(baselines, BASELINE_COLUMNS)
    table = table.merge(trackers[trackers["sequence"].isin(table["sequence"].unique())], on=KEYS, how="left")
    return add_loop_errors(table)


def frame_rows(rows: pd.DataFrame, frames: str) -> pd.DataFrame:
    """The blackout or hidden frames (those read in closed loop), or the frames EMERGING since the reappearance."""
    if frames == "hidden":
        return rows[rows["closed"]]
    return rows[rows["since_reappear"].between(EMERGING[0], EMERGING[1])]


def summary(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per mode, frames, k (all and each), model, and measure: the mean over sequences of each loop with its bootstrap
    interval, and the trackers for the position measures; and closed minus open loop on the same sequences."""
    lines, paired = [], []
    for mode in LOOP_MODES:
        for frames in FRAMES:
            measures = measures_of(mode, frames)
            rows = frame_rows(table[table["mode"] == mode], frames)
            for scope, own in scopes(rows):
                where = {"mode": mode, "frames": frames, "k": scope}
                for model, mine in own.groupby("model", sort=False):
                    for measure in measures:
                        values = {loop: per_sequence(mine, MEASURES[measure][1].format(loop)) for loop in LOOPS}
                        for loop in LOOPS:
                            lines.append({**where, "who": model, "kind": "model", "loop": loop, "measure": measure,
                                          **interval(bootstrap(values[loop]))})
                        paired.append({**where, "who": model, "measure": measure, "difference": "closed - open",
                                       **interval(paired_difference(values["closed"], values["open"]))})
                if len(own):
                    first = reference_rows(own)
                    for name, column in REFERENCES.items():
                        if column in first:
                            for measure in (m for m in POSITION if m in measures):
                                lines.append({**where, "who": name, "kind": "reference", "loop": "", "measure": measure,
                                              **interval(bootstrap(per_sequence(first, column)))})
    return pd.DataFrame(lines), pd.DataFrame(paired)


def curves(table: pd.DataFrame) -> pd.DataFrame:
    """Every measure of each loop against j, the frames since the onset, from BEFORE frames before it to AFTER frames
    after the last blackout or hidden frame, per mode, k, and model; Kalman for the position measures."""
    rows = table[(table["since_onset"] >= -BEFORE) & (table["since_onset"] <= table["k"] - 1 + AFTER)]
    parts = []
    for mode, own in rows.groupby("mode", sort=False):
        for model, mine in own.groupby("model", sort=False):
            for measure in CURVES:
                for loop in LOOPS:
                    parts.append(group_mean_se(mine, MEASURES[measure][1].format(loop), ["k", "since_onset"]).assign(
                        mode=mode, measure=measure, who=model, kind="model", loop=loop))
        first = reference_rows(own)
        if "kalman_err_y" in first:
            for measure in POSITION:
                parts.append(group_mean_se(first, "kalman_err_y", ["k", "since_onset"]).assign(
                    mode=mode, measure=measure, who="Kalman", kind="reference", loop=""))
    out = pd.concat(parts, ignore_index=True).rename(columns={"since_onset": "j"})
    return out[["mode", "measure", "who", "kind", "loop", "k", "j", "mean", "se", "n"]]


def closed_loop(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Every table of the analysis: summary, paired, and curves."""
    lines, paired = summary(table)
    return {"summary": lines, "paired": paired, "curves": curves(table)}


def plot_loop(curve: pd.DataFrame, mode: str, path: Path, title: str, k: int = K_SHOWN) -> Path:
    """One column per model, one row per measure: open (dotted) and closed loop (solid) against the frames since the
    onset, for one k, with Kalman on the position measures; the blackout or hidden frames shaded."""
    rows = curve[(curve["mode"] == mode) & (curve["k"] == k)]
    models = list(rows.loc[rows["kind"] == "model", "who"].unique())
    colors = model_colors(rows)
    fig = Figure(figsize=(3.1 * max(len(models), 1), 2.4 * len(CURVES)))
    axes = fig.subplots(len(CURVES), max(len(models), 1), squeeze=False, sharex=True)
    for column, model in enumerate(models):
        for row, measure in enumerate(CURVES):
            ax = axes[row, column]
            own = rows[rows["measure"] == measure]
            for loop, style in (("open", "s:"), ("closed", "o-")):
                line = own[(own["who"] == model) & (own["loop"] == loop)].sort_values("j")
                ax.plot(line["j"], line["mean"], style, ms=2.5, lw=1.3, color=colors[model], label=f"{loop} loop")
                ax.fill_between(line["j"], line["mean"] - line["se"], line["mean"] + line["se"], color=colors[model],
                                alpha=0.2, lw=0)
            kalman = own[own["who"] == "Kalman"].sort_values("j")
            if len(kalman):
                ax.plot(kalman["j"], kalman["mean"], "--", lw=1.1, color="0.3", label="Kalman")
            ax.axvspan(-0.5, k - 0.5, color="0.92", zorder=0)
            if measure == "decoder_corr":
                ax.set_ylim(min(0.0, ax.get_ylim()[0]), 1)
            elif measure == "mse_digit":
                ax.set_yscale("log")
            else:
                ax.set_ylim(bottom=0)
            if row == 0:
                ax.set_title(model, fontsize=9)
            if row == len(CURVES) - 1:
                ax.set_xlabel("frames since the onset")
            if column == 0:
                ax.set_ylabel(MEASURES[measure][0])
    if models:
        axes[0, 0].legend(fontsize=6)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return path
