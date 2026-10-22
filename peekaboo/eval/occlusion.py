"""Tracking through occlusion (RQ1, step 3.2), from the extraction tables of step 3.1 (peekaboo/eval/tables.py).

Two sides of the question:
- the state (H1b): the position probe and the amodal decoder against j, the frames since the onset of the occlusion,
  for the hidden frames (j from 0 to k - 1) and a few frames around them, per k and speed, per model, next to the
  programmed trackers and templates and the random initialization control; the decay of the probe error over the
  hidden frames (its slope against j) and its value at the last hidden frame against k;
- the predicted frames (H1a, C1): a hidden digit is not to be drawn, so the informative frames are those where it
  comes out. On these: the ink drawn on the side of the bar where the digit comes out (the exit side), the readout of
  the drawn digit against its visible ink center, the error over the bar region, and the frame at which the drawn
  digit first comes out against the true one, next to the copy of the last frame and the blank frame.

The sequences are those of the occlusion cells (not the members of surprise tuples, left to RQ2 and RQ3) and of the
hidden bounce cells, where the digit bounces off a wall behind the bar and comes back out on its entry side: constant
velocity fails there, and following the bounce takes the wall into account. Every statistic is over sequences, with
the seeds as the outer level when a model has several (peekaboo/eval/stats.py); comparisons are paired on the same
sequences.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from peekaboo.eval.extract import MIN_INK
from peekaboo.eval.stats import bootstrap, paired_difference
from peekaboo.eval.tables import per_sequence
from peekaboo.train.sweep import SWEEP

CONDITIONS = ("occlusion", "hidden_bounce")
BEFORE, AFTER = 3, 3       # frames shown before the onset and after the last hidden frame
EMERGENCE = (-2, 3)        # frames since the reappearance read for the emergence
EMERGING = (0, 2)          # the frames summarized: the reappearance frame and the next two, the digit partly out
EMERGE_INK = MIN_INK       # ink on the exit side from which the digit counts as coming out
SEQUENCE = ["model", "seed", "sequence"]
COLUMNS = ["sequence", "t", "y", "x", "in_window", "condition", "k", "speed", "since_onset", "modal_y", "modal_x",
           "since_reappear", "tuple_role", "probe_y", "probe_x", "random_probe_y", "random_probe_x", "decoder_corr",
           "decoder_iou", "random_decoder_corr", "random_decoder_iou", "model_mse_bar", "copy_mse_bar",
           "blank_mse_bar", "pred_ink_y", "pred_ink_x", "pred_ink_left", "pred_ink_right", "true_ink_left",
           "true_ink_right"]  # the columns of a model pass read here


@dataclass(frozen=True)
class Readout:
    """A per frame readout of the model, with its random initialization control and its programmed references."""

    name: str
    column: str
    label: str
    random: str
    references: dict[str, str] = field(default_factory=dict)  # label: column, the same for every model

    def lines(self, rows: pd.DataFrame) -> list[tuple[str, str, str, pd.DataFrame]]:
        """(who, kind, column, rows) of every line: each model, its random initialization control, then each
        reference once (read on the rows of the first model, since they do not depend on the model)."""
        out = []
        for model, own in rows.groupby("model", sort=False):
            out.append((model, "model", self.column, own))
            out.append((f"{model}, random init", "random", self.random, own))
        if len(rows):
            first = rows[rows["model"] == rows["model"].iloc[0]]
            out += [(label, "reference", column, first) for label, column in self.references.items()]
        return out


TRACKERS = {"last seen": "last_seen", "Kalman": "kalman", "Kalman with walls": "kalman_walls"}
READOUTS = (
    Readout("probe_y", "probe_err_y", "probe |error| on y (px)", "random_probe_err_y",
            {label: f"{name}_err_y" for label, name in TRACKERS.items()}),
    Readout("probe_x", "probe_err_x", "probe |error| on x (px)", "random_probe_err_x",
            {**{label: f"{name}_err_x" for label, name in TRACKERS.items()}, "bar center": "bar_center_err_x"}),
    Readout("probe", "probe_err", "probe distance to the center (px)", "random_probe_err",
            {label: f"{name}_err" for label, name in TRACKERS.items()}),
    Readout("decoder_corr", "decoder_corr", "decoder correlation", "random_decoder_corr",
            {"template, Kalman": "template_kalman_corr", "template, Kalman with walls": "template_walls_corr"}),
    Readout("decoder_iou", "decoder_iou", "decoder IoU", "random_decoder_iou",
            {"template, Kalman": "template_kalman_iou", "template, Kalman with walls": "template_walls_iou"}),
)
# the emergence measures: label, the model's column, and the references' columns
MEASURES = {
    "exit_ink": ("ink on the exit side", "pred_exit", {"true frame": "true_exit", "copy": "copy_exit"}),
    "side_share": ("share of the ink on the exit side", "side_share", {}),
    "readout_y": ("|readout - visible center| on y (px)", "pred_modal_err_y", {"copy": "copy_modal_err_y"}),
    "readout_x": ("|readout - visible center| on x (px)", "pred_modal_err_x", {"copy": "copy_modal_err_x"}),
    "drawn": ("share of frames with a drawn digit", "pred_drawn", {}),
    "bar_mse": ("MSE over the bar region", "model_mse_bar", {"copy": "copy_mse_bar", "blank": "blank_mse_bar"}),
}


def find_readout(name: str) -> Readout:
    """The readout with this name."""
    return next(r for r in READOUTS if r.name == name)


def event_rows(table: pd.DataFrame, condition: str) -> pd.DataFrame:
    """The analysis window rows of the sequences of one condition's cells (no surprise tuple member)."""
    return table[(table["condition"] == condition) & (table["tuple_role"] == "none") & table["in_window"]]


def around_hidden(rows: pd.DataFrame) -> pd.DataFrame:
    """The rows from BEFORE frames before the onset to AFTER frames after the last hidden frame."""
    return rows[(rows["since_onset"] >= -BEFORE) & (rows["since_onset"] <= rows["k"] - 1 + AFTER)]


def hidden(rows: pd.DataFrame) -> pd.DataFrame:
    """The hidden frames, j = 0 to k - 1."""
    return rows[(rows["since_onset"] >= 0) & (rows["since_onset"] <= rows["k"] - 1)]


def group_mean_se(rows: pd.DataFrame, column: str, by: list[str]) -> pd.DataFrame:
    """Mean, standard error, and number of sequences of a column per group: over sequences for one seed, over the
    seed means for several (the outer level)."""
    per_seed = rows.groupby([*by, "seed"])[column].agg(["mean", "std", "count"])
    seeds = per_seed.groupby(level=by)
    n_seeds = seeds["mean"].count()
    single = (per_seed["std"] / np.sqrt(per_seed["count"])).groupby(level=by).first()
    se = np.where(n_seeds > 1, seeds["mean"].std() / np.sqrt(n_seeds.clip(lower=1)), single)
    return pd.DataFrame({"mean": seeds["mean"].mean(), "se": se, "n": seeds["count"].max()}).reset_index()


def curves(table: pd.DataFrame, by_speed: bool = False) -> pd.DataFrame:
    """Every readout against j around the hidden frames, per condition, k (and speed), and line."""
    by = ["k", "speed", "since_onset"] if by_speed else ["k", "since_onset"]
    parts = []
    for condition in CONDITIONS:
        rows = around_hidden(event_rows(table, condition))
        for readout in READOUTS:
            for who, kind, column, own in readout.lines(rows):
                parts.append(group_mean_se(own, column, by).assign(
                    condition=condition, readout=readout.name, who=who, kind=kind))
    out = pd.concat(parts, ignore_index=True).rename(columns={"since_onset": "j"})
    first = ["condition", "readout", "who", "kind", "k"] + (["speed"] if by_speed else [])
    return out[first + ["j", "mean", "se", "n"]]


def scopes(rows: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """The rows of every k, and all of them together."""
    return [("all", rows)] + [(str(k), rows[rows["k"] == k]) for k in sorted(rows["k"].unique())]


def interval(stats: dict[str, float], prefix: str = "") -> dict[str, float]:
    """A bootstrap result as table columns."""
    return {f"{prefix}{name}": stats[name] for name in ("mean", "se", "low", "high", "n")}


def hidden_summary(table: pd.DataFrame, reference: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every readout on the hidden frames, one value per sequence (its mean over the hidden frames), per condition
    and k, with bootstrap intervals; and the paired differences of each model against its random initialization
    control, the references, and the reference model (on the seed means of each sequence)."""
    summary, paired = [], []
    for condition in CONDITIONS:
        for k, rows in scopes(hidden(event_rows(table, condition))):
            for readout in READOUTS:
                for who, kind, column, own in readout.lines(rows):
                    summary.append({"condition": condition, "k": k, "readout": readout.name, "who": who,
                                    "kind": kind, **interval(bootstrap(per_sequence(own, column)))})
                models = dict(tuple(rows.groupby("model", sort=False)))
                for model, own in models.items():
                    values = per_sequence(own, readout.column)
                    against = {"random init": per_sequence(own, readout.random)}
                    against.update({label: per_sequence(own, column).mean() for label, column
                                    in readout.references.items()})
                    if reference in models and model != reference:
                        against[reference] = per_sequence(models[reference], readout.column).mean()
                    for label, other in against.items():
                        paired.append({"condition": condition, "k": k, "readout": readout.name, "model": model,
                                       "against": label, **interval(paired_difference(values, other))})
    return pd.DataFrame(summary), pd.DataFrame(paired)


def slopes(rows: pd.DataFrame, column: str) -> pd.DataFrame:
    """Least squares slope of a column against j over the rows of every seed and sequence (one row per seed, one
    column per sequence), NaN with fewer than two frames."""
    keys = [rows["seed"], rows["sequence"]]
    j = rows["since_onset"].astype(float)
    value = rows[column].astype(float)
    dj = j - j.groupby(keys).transform("mean")
    dv = value - value.groupby(keys).transform("mean")
    slope = (dj * dv).groupby(keys).sum() / (dj * dj).groupby(keys).sum().replace(0.0, np.nan)
    return slope.unstack("sequence")


def decay(table: pd.DataFrame) -> pd.DataFrame:
    """Per condition, k, readout, and line: the slope of the readout over the hidden frames, and its value at the last
    hidden frame (j = k - 1), each with its bootstrap interval."""
    out = []
    for condition in CONDITIONS:
        rows = hidden(event_rows(table, condition))
        for k, own_k in scopes(rows)[1:]:
            last = own_k[own_k["since_onset"] == int(k) - 1]
            for readout in READOUTS:
                for who, kind, column, own in readout.lines(own_k):
                    out.append({"condition": condition, "k": int(k), "readout": readout.name, "who": who,
                                "kind": kind, **interval(bootstrap(slopes(own, column)), "slope_"),
                                **interval(bootstrap(per_sequence(last[last["model"] == own["model"].iloc[0]],
                                                                  column)), "last_")})
    return pd.DataFrame(out)


def emergence_rows(table: pd.DataFrame, condition: str) -> pd.DataFrame:
    """The rows of EMERGENCE frames since the reappearance, with the emergence measures (MEASURES):
    - the exit side, where the digit comes out: the side of the bar with the more true ink on the reappearance frame;
    - the ink on the exit side drawn by the model (pred_exit), on the other side (pred_entry), on the true frame
      (true_exit), and on the copy of the last frame (copy_exit, the true exit ink of the previous frame), and the
      share of the drawn ink off the bar on the exit side (side_share, when there is at least EMERGE_INK of it);
    - the readout of the drawn digit (its ink centroid off the bar) against the visible ink center, for the model
      and for the copy, whose readout is the visible center of the previous frame; and whether a digit is drawn at
      all where one is visible (pred_drawn)."""
    rows = event_rows(table, condition).sort_values(["sequence", "t"], kind="stable")  # keeps the models in order
    previous = rows.groupby(SEQUENCE)[["true_ink_left", "true_ink_right", "modal_y", "modal_x"]].shift(1)
    at = rows[rows["since_reappear"] == 0].set_index(SEQUENCE)
    side = pd.Series(np.where(at["true_ink_left"] > at["true_ink_right"], "left", "right"), index=at.index)
    side[(at["true_ink_left"] == 0) & (at["true_ink_right"] == 0)] = None
    exit_side = side.reindex(pd.MultiIndex.from_frame(rows[SEQUENCE])).to_numpy()
    left = exit_side == "left"
    known = pd.notna(exit_side)
    pick = lambda a, b: np.where(known, np.where(left, a, b), np.nan)  # noqa: E731
    pred_exit = pick(rows["pred_ink_left"], rows["pred_ink_right"])
    pred_entry = pick(rows["pred_ink_right"], rows["pred_ink_left"])
    off_bar = pred_exit + pred_entry
    visible = rows["modal_y"].notna()
    rows = rows.assign(
        pred_exit=pred_exit, pred_entry=pred_entry,
        true_exit=pick(rows["true_ink_left"], rows["true_ink_right"]),
        copy_exit=pick(previous["true_ink_left"], previous["true_ink_right"]),
        side_share=np.where(off_bar >= EMERGE_INK, pred_exit / np.maximum(off_bar, 1e-12), np.nan),
        pred_modal_err_y=(rows["pred_ink_y"] - rows["modal_y"]).abs(),
        pred_modal_err_x=(rows["pred_ink_x"] - rows["modal_x"]).abs(),
        copy_modal_err_y=(previous["modal_y"] - rows["modal_y"]).abs(),
        copy_modal_err_x=(previous["modal_x"] - rows["modal_x"]).abs(),
        pred_drawn=np.where(visible, rows["pred_ink_y"].notna().astype(float), np.nan))
    return rows[(rows["since_reappear"] >= EMERGENCE[0]) & (rows["since_reappear"] <= EMERGENCE[1])]


def emergence_lines(rows: pd.DataFrame, measure: str) -> list[tuple[str, str, str, pd.DataFrame]]:
    """(who, kind, column, rows) of every line of an emergence measure: each model, then each reference once."""
    _, column, references = MEASURES[measure]
    out = [(model, "model", column, own) for model, own in rows.groupby("model", sort=False)]
    if len(rows):
        first = rows[rows["model"] == rows["model"].iloc[0]]
        out += [(label, "reference", ref, first) for label, ref in references.items()]
    return out


def emergence(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The emergence measures against the frames since the reappearance (curves), their means over the EMERGING
    frames with bootstrap intervals (summary), and the paired differences of each model against its references
    on these frames (paired)."""
    curve_parts, summary, paired = [], [], []
    for condition in CONDITIONS:
        rows = emergence_rows(table, condition)
        emerging = rows[(rows["since_reappear"] >= EMERGING[0]) & (rows["since_reappear"] <= EMERGING[1])]
        for measure in MEASURES:
            for who, kind, column, own in emergence_lines(rows, measure):
                curve_parts.append(group_mean_se(own, column, ["since_reappear"]).assign(
                    condition=condition, measure=measure, who=who, kind=kind))
            for k, scope in scopes(emerging):
                for who, kind, column, own in emergence_lines(scope, measure):
                    summary.append({"condition": condition, "k": k, "measure": measure, "who": who, "kind": kind,
                                    **interval(bootstrap(per_sequence(own, column)))})
                _, column, references = MEASURES[measure]
                for model, own in scope.groupby("model", sort=False):
                    for label, ref in references.items():
                        if label == "true frame":
                            continue
                        paired.append({"condition": condition, "k": k, "measure": measure, "model": model,
                                       "against": label, **interval(paired_difference(
                                           per_sequence(own, column), per_sequence(own, ref).mean()))})
    curve = pd.concat(curve_parts, ignore_index=True)
    curve = curve[["condition", "measure", "who", "kind", "since_reappear", "mean", "se", "n"]]
    return curve, pd.DataFrame(summary), pd.DataFrame(paired)


def first_emergence(rows: pd.DataFrame, column: str) -> pd.Series:
    """Per seed and sequence, the first frame since the reappearance (within EMERGENCE) with at least EMERGE_INK of
    ink on the exit side, NaN if none."""
    found = rows[rows[column] >= EMERGE_INK]
    first = found.groupby(["seed", "sequence"])["since_reappear"].min()
    every = rows.groupby(["seed", "sequence"]).size().index
    return first.reindex(every).astype(float)


def timing(table: pd.DataFrame) -> pd.DataFrame:
    """When the drawn digit comes out against the true one: per condition, k, and model (and the copy), the shares of
    sequences where the first frame with ink on the exit side is early, on time, late, or never within EMERGENCE,
    and the mean lag in frames of those that come out. Sequences whose true digit does not come out within EMERGENCE
    are left out (n counts the others)."""
    out = []
    for condition in CONDITIONS:
        rows = emergence_rows(table, condition)
        for k, scope in scopes(rows):
            lines = [(model, "model", own, "pred_exit") for model, own in scope.groupby("model", sort=False)]
            if len(scope):
                lines.append(("copy", "reference", scope[scope["model"] == scope["model"].iloc[0]], "copy_exit"))
            for who, kind, own, column in lines:
                true = first_emergence(own, "true_exit")
                lag = (first_emergence(own, column) - true)[true.notna()]
                n = len(lag)
                share = lambda mask: float(mask.sum() / n) if n else np.nan  # noqa: E731
                out.append({"condition": condition, "k": k, "who": who, "kind": kind, "n": n,
                            "early": share(lag < 0), "on_time": share(lag == 0), "late": share(lag > 0),
                            "never": share(lag.isna()), "mean_lag": float(lag.mean()) if lag.notna().any() else np.nan})
    return pd.DataFrame(out)


# figures

REFERENCE_STYLE = {"last seen": (":", "0.55"), "Kalman": ("--", "0.3"), "Kalman with walls": ("-.", "0.3"),
                   "bar center": (":", "tab:brown"), "template, Kalman": ("--", "0.3"),
                   "template, Kalman with walls": ("-.", "0.3"), "true frame": ("-", "black"),
                   "copy": (":", "0.45"), "blank": ("--", "0.65")}


def draw_line(ax, curve: pd.DataFrame, x: str, who: str, kind: str, color: str, band: bool = True) -> None:
    """One line of a curve table (its `x`, mean, and se), with a standard error band for the models."""
    curve = curve.sort_values(x)
    if kind == "model":
        ax.plot(curve[x], curve["mean"], "o-", ms=2.5, lw=1.4, color=color, label=who)
        if band:
            ax.fill_between(curve[x], curve["mean"] - curve["se"], curve["mean"] + curve["se"], color=color,
                            alpha=0.2, lw=0)
    elif kind == "random":
        ax.plot(curve[x], curve["mean"], ":", lw=1.2, color=color, label=who)
    else:
        style, gray = REFERENCE_STYLE.get(who, ("--", "0.5"))
        ax.plot(curve[x], curve["mean"], style, lw=1.1, color=gray, label=who)


def model_colors(table: pd.DataFrame) -> dict[str, str]:
    """A color per model, the same in every figure: the sweep models in the order of the sweep, then any other."""
    order = [run.label for run in SWEEP]
    order += [model for model in table.loc[table["kind"] == "model", "who"].unique() if model not in order]
    return {model: f"C{i}" for i, model in enumerate(order)}


def draw_readout(ax, curve: pd.DataFrame, readout: Readout, colors: dict[str, str], reference: str, k: int,
                 shade: bool = True) -> None:
    """The lines of one readout against j: every model, the reference model's random initialization control, and
    the programmed references, with the hidden frames shaded."""
    for who, line in curve.groupby("who", sort=False):
        kind = line["kind"].iloc[0]
        if kind == "random" and who != f"{reference}, random init":
            continue
        model = who.removesuffix(", random init")
        draw_line(ax, line, "j", who, kind, colors.get(model, "0.5"))
    if shade:
        ax.axvspan(-0.5, k - 0.5, color="0.92", zorder=0)
    if readout.name.startswith("decoder"):
        ax.set_ylim(0, 1)
    else:
        ax.set_ylim(bottom=0)


def plot_curves(curve: pd.DataFrame, condition: str, names: tuple[str, ...], reference: str, path: Path,
                title: str) -> Path:
    """One column per k, one row per readout: every model against j, with the references; hidden frames shaded."""
    rows = curve[curve["condition"] == condition]
    ks = sorted(rows["k"].unique())
    colors = model_colors(rows)
    fig = Figure(figsize=(3.0 * len(ks) + 1, 2.6 * len(names) + 0.6))
    axes = np.atleast_2d(fig.subplots(len(names), len(ks), squeeze=False))
    for r, name in enumerate(names):
        readout = find_readout(name)
        for c, k in enumerate(ks):
            ax = axes[r, c]
            draw_readout(ax, rows[(rows["readout"] == name) & (rows["k"] == k)], readout, colors, reference, k)
            if r == 0:
                ax.set_title(f"k = {k}", fontsize=9)
            if c == 0:
                ax.set_ylabel(readout.label, fontsize=8)
            if r == len(names) - 1:
                ax.set_xlabel("j, frames since the onset", fontsize=8)
            ax.tick_params(labelsize=7)
    for r in range(len(names)):
        axes[r, -1].legend(fontsize=6, loc="best")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return path


def plot_speeds(curve: pd.DataFrame, condition: str, name: str, reference: str, path: Path, title: str) -> Path:
    """One readout per k (columns) and speed (rows)."""
    rows = curve[(curve["condition"] == condition) & (curve["readout"] == name)]
    ks, speeds = sorted(rows["k"].unique()), sorted(rows["speed"].unique())
    colors = model_colors(rows)
    readout = find_readout(name)
    fig = Figure(figsize=(3.0 * len(ks) + 1, 2.4 * len(speeds) + 0.6))
    axes = fig.subplots(len(speeds), len(ks), squeeze=False)
    for r, speed in enumerate(speeds):
        for c, k in enumerate(ks):
            ax = axes[r, c]
            cell = rows[(rows["k"] == k) & (rows["speed"] == speed)]
            if cell.empty:
                ax.axis("off")
                continue
            draw_readout(ax, cell, readout, colors, reference, k)
            ax.set_title(f"k = {k}, speed {speed} px/frame", fontsize=8)
            if c == 0:
                ax.set_ylabel(readout.label, fontsize=8)
            if r == len(speeds) - 1:
                ax.set_xlabel("j, frames since the onset", fontsize=8)
            ax.tick_params(labelsize=7)
    axes[0, -1].legend(fontsize=6)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return path


def plot_decay(table: pd.DataFrame, condition: str, reference: str, path: Path, title: str) -> Path:
    """Against k: the probe error on y at the last hidden frame, its slope over the hidden frames, and the decoder
    correlation at the last hidden frame, with 95% intervals."""
    rows = table[table["condition"] == condition]
    colors = model_colors(rows)
    panels = (("probe_y", "last_", "probe |error| on y, last hidden frame (px)"),
              ("probe_y", "slope_", "slope of the probe |error| on y (px/frame)"),
              ("decoder_corr", "last_", "decoder correlation, last hidden frame"))
    fig = Figure(figsize=(13, 3.8))
    axes = fig.subplots(1, len(panels))
    for ax, (name, prefix, label) in zip(axes, panels):
        own = rows[rows["readout"] == name]
        for who, line in own.groupby("who", sort=False):
            kind = line["kind"].iloc[0]
            if kind == "random" and who != f"{reference}, random init":
                continue
            line = line.sort_values("k")
            curve = pd.DataFrame({"k": line["k"], "mean": line[f"{prefix}mean"], "se": 0.0})
            draw_line(ax, curve, "k", who, kind, colors.get(who.removesuffix(", random init"), "0.5"), band=False)
            if kind == "model":
                ax.errorbar(line["k"], line[f"{prefix}mean"],
                            yerr=[line[f"{prefix}mean"] - line[f"{prefix}low"],
                                  line[f"{prefix}high"] - line[f"{prefix}mean"]],
                            fmt="none", ecolor=colors[who], lw=1, capsize=2)
        if prefix == "slope_":
            ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xlabel("k, hidden frames", fontsize=8)
        ax.set_ylabel(label, fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=6)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return path


def plot_emergence(curve: pd.DataFrame, path: Path, title: str) -> Path:
    """One column per condition: the emergence measures against the frames since the reappearance, for every model,
    the true frame, the copy, and the blank frame. The share on the exit side is left to the tables: it is 1 for every
    model from the reappearance on."""
    names = ("exit_ink", "readout_y", "bar_mse")
    colors = model_colors(curve)
    fig = Figure(figsize=(10, 2.6 * len(names) + 0.6))
    axes = fig.subplots(len(names), len(CONDITIONS), squeeze=False)
    for c, condition in enumerate(CONDITIONS):
        for r, name in enumerate(names):
            ax = axes[r, c]
            own = curve[(curve["condition"] == condition) & (curve["measure"] == name)]
            for who, line in own.groupby("who", sort=False):
                draw_line(ax, line, "since_reappear", who, line["kind"].iloc[0], colors.get(who, "0.5"))
            ax.axvline(-0.5, color="0.6", lw=0.8)
            if name == "bar_mse":
                ax.set_yscale("log")
            else:
                ax.set_ylim(bottom=0)
            if r == 0:
                ax.set_title(condition.replace("_", " "), fontsize=9)
            if c == 0:
                ax.set_ylabel(MEASURES[name][0], fontsize=8)
            if r == len(names) - 1:
                ax.set_xlabel("frames since the reappearance", fontsize=8)
            ax.tick_params(labelsize=7)
            if c == len(CONDITIONS) - 1:
                ax.legend(fontsize=6)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return path
