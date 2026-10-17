"""Comparison of the models of the training sweep (step 2.10), read from the outputs that every run already has: the
validation log, the gate D16 (eval/next_frame_<set>.json), the position probes (eval/probe_<set>.json and .parquet),
and the amodal decoder (eval/amodal_<set>.json and .parquet). Nothing is recomputed.

The readouts are compared on the frames of the analysis window, overall per condition and state, and against the
frames since the onset of the occlusion. Later frames of these curves mix values of k: for short occlusions the digit
is already reappearing there.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from peekaboo.train.sweep import DATA, SweepRun, read_json, validation_rows

SINCE_ONSET = range(-3, 16)
ROWS = {"occlusion, hidden": "occlusion / occluded", "hidden bounce, hidden": "hidden_bounce / occluded",
        "occlusion, visible": "occlusion / visible"}


def summary_row(run: SweepRun, run_dir: Path) -> dict[str, float | str]:
    """The main numbers of one run: training, next frame quality, and both readouts with their controls, on hidden
    frames of occlusion and hidden bounce sequences and on visible frames of occlusion sequences."""
    rows = validation_rows(run_dir)
    best = next((r for r in rows if r["step"] == rows[-1]["best_step"]), {}) if rows else {}  # the row of best.pt
    gate = read_json(run_dir / "eval" / f"next_frame_{DATA}.json")
    probe = read_json(run_dir / "eval" / f"probe_{DATA}.json")
    decoder = read_json(run_dir / "eval" / f"amodal_{DATA}.json")
    out: dict[str, float | str] = {"model": run.label, "run": f"{run.key}/seed{run.seed}",
                                   "best step": best.get("step", np.nan), "val loss": best.get("val_loss", np.nan),
                                   "val MSE": best.get("val_mse", np.nan)}
    if gate:
        out["gate"] = gate["gate"]["mse_reduction"]
        out["visible MSE"] = gate["gate"]["model_mse"]
    if probe:
        out["probe features"] = probe["probes"]["model"]["features"]
        for name, key in ROWS.items():
            out[f"probe y, {name}"] = probe["error_y"][key]["probe"]
            out[f"probe x, {name}"] = probe["error_x"][key]["probe"]
        hidden = ROWS["occlusion, hidden"]
        out["random probe y, occlusion, hidden"] = probe["error_y"][hidden]["random_probe"]
        out["p true bin y, occlusion, hidden"] = probe["p_true_bin"][hidden]["probe_p_bin_y"]
    if decoder:
        for name, key in ROWS.items():
            out[f"decoder corr, {name}"] = decoder["corr"][key]["decoder"]
            out[f"decoder IoU, {name}"] = decoder["iou"][key]["decoder"]
        out["random decoder corr, occlusion, hidden"] = decoder["corr"][ROWS["occlusion, hidden"]]["random_decoder"]
    return out


def summary_table(runs: list[SweepRun], runs_dir: Path) -> pd.DataFrame:
    """One row per run, from its folder."""
    return pd.DataFrame([summary_row(run, run.run_dir(runs_dir)) for run in runs])


def since_onset_curve(table: pd.DataFrame, column: str, condition: str) -> pd.Series:
    """Mean of a per frame column against the frames since the onset, on the analysis window of one condition."""
    rows = table[(table["condition"] == condition) & table["in_window"]
                 & table["since_onset"].isin(list(SINCE_ONSET))]
    return rows.groupby("since_onset")[column].mean()


def plot_since_onset(tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]], path: Path) -> Path:
    """Probe error on y and x, and decoder correlation, against the frames since the onset, one line per model, with
    the constant velocity references (Kalman filter for the position, the template it moves for the digit). `tables`
    maps a model label to its probe and decoder tables."""
    fig = Figure(figsize=(10, 9))
    axes = fig.subplots(3, 2, sharex=True)
    panels = (("probe", "probe_err_y", "kalman_err_y", "mean |error| on y (px)"),
              ("probe", "probe_err_x", "kalman_err_x", "mean |error| on x (px)"),
              ("decoder", "decoder_corr", "template_kalman_corr", "decoder correlation"))
    for column, condition in enumerate(("occlusion", "hidden_bounce")):
        for row, (kind, name, reference, label) in enumerate(panels):
            ax = axes[row, column]
            for i, (model, (probe, decoder)) in enumerate(tables.items()):
                table = probe if kind == "probe" else decoder
                curve = since_onset_curve(table, name, condition)
                ax.plot(curve.index, curve.values, "o-", ms=3, lw=1.5, color=f"C{i}", label=model)
            first = next(iter(tables.values()))
            curve = since_onset_curve(first[0] if kind == "probe" else first[1], reference, condition)
            ax.plot(curve.index, curve.values, "--", lw=1, color="0.4",
                    label="Kalman, constant velocity" if kind == "probe" else "template moved by it")
            ax.axvline(0, color="0.6", lw=0.8)
            if kind == "decoder":
                ax.set_ylim(0, 1)
            else:
                ax.set_ylim(bottom=0)
            ax.set_ylabel(label)
            if row == 0:
                ax.set_title(condition.replace("_", " "), fontsize=9)
            if row == len(panels) - 1:
                ax.set_xlabel("frames since the onset of the occlusion")
    axes[0, 0].legend(fontsize=7)
    axes[2, 0].legend(fontsize=7)
    fig.suptitle(f"readouts of the sweep models on the analysis windows of {DATA}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return path


def plot_validation(runs: dict[str, Path], path: Path, failed: dict[str, Path] | None = None) -> Path:
    """Validation MSE against the training step, one line per run (failed runs dotted), with the blank frame."""
    fig = Figure(figsize=(8, 4))
    ax = fig.subplots()
    blank = None
    for i, (label, run_dir) in enumerate(runs.items()):
        rows = validation_rows(run_dir)
        ax.plot([r["step"] for r in rows], [r["val_mse"] for r in rows], "-", lw=1.2, color=f"C{i}", label=label)
        blank = next((r["blank_mse"] for r in rows if "blank_mse" in r), blank)
    for label, run_dir in (failed or {}).items():
        rows = validation_rows(run_dir)
        ax.plot([r["step"] for r in rows], [r["val_mse"] for r in rows], ":", lw=1.2, color="0.5", label=label)
    if blank is not None:
        ax.axhline(blank, color="C3", ls="--", lw=0.8, label="blank frame")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel(f"validation MSE per pixel ({DATA})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return path
