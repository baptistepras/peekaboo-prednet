"""Inside the state (step 3.5), from the extraction tables of step 3.1 and the layer passes of the readouts of single
layers (peekaboo/probes/layers.py, read by peekaboo/eval/extract.py layer_frames).

Four questions, each on the sequences of the test set, one value per sequence (or per tuple) and seed:
- which layers hold the hidden digit: the probe's error on y and the decoder's correlation and IoU on the hidden
  frames of the occlusion and hidden bounce cells, for every layer alone, every layer but one, and all the layers
  (and on the visible frames before the onset, for comparison), each against all the layers on the same sequences;
- is a hidden digit coded as a visible one: the readouts fitted on the fully visible frames only, on the same hidden
  frames, against those fitted on every frame of the analysis windows;
- where is the belief after a surprise (left open by step 3.3, where the probe stayed on the old trajectory while
  the drawn digit was corrected): the distance of every readout's probe to the true center, and its decoder's
  correlation, over the HORIZON frames since the reappearance, AB against B and BA against A on the same tuples;
- what the error units of every PredNet layer do (from the extraction pass): their activity relative to its mean on
  the fully visible frames before the onset, while the digit is hidden and when it comes back (the frames EMERGING
  since the reappearance).
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from peekaboo.eval.correction import HORIZON, matrix
from peekaboo.eval.extract import MISSING, frame_columns, is_complete, read_frames
from peekaboo.eval.next_frame import MIN_CONTEXT
from peekaboo.eval.occlusion import (EMERGING, around_hidden, draw_line, event_rows, group_mean_se, hidden, interval,
                                     model_colors)
from peekaboo.eval.stats import bootstrap, paired_difference
from peekaboo.eval.surprise import KINDS, LAYERS, PAIRS, tuple_rows
from peekaboo.eval.tables import ModelRun, layers_folder, per_sequence, read_runs
from peekaboo.probes.layers import VISIBLE, VISIBLE_STATE

CONDITIONS = ("occlusion", "hidden_bounce", "surprise", "empty")
CELLS = ("occlusion", "hidden_bounce")
K_SHOWN = 8                # the k of the curves in the figures (both cell conditions have it)
ERRORS = [f"e{l}{region}" for l in range(LAYERS) for region in ("", "_bar")]
COLUMNS = (["sequence", "t", "y", "x", "in_window", "condition", "k", "speed", "state", "since_onset",
            "since_reappear", "tuple_id", "tuple_role", "surprise_type", "probe_y"] + ERRORS)  # of the model passes
ROWS = [[("in_window", "==", True)]]  # the rows read: the analysis windows, around the hidden frames
# the measures: label and the column of a readout ({} is its name)
MEASURES = {"probe_y": ("probe |error| on y (px)", "probe_{}_err_y"),
            "probe": ("probe distance to the center (px)", "probe_{}_err"),
            "decoder_corr": ("decoder correlation", "decoder_{}_corr"),
            "decoder_iou": ("decoder IoU", "decoder_{}_iou")}
HIDDEN_MEASURES = ("probe_y", "decoder_corr", "decoder_iou")  # on the hidden frames (y: the main hidden axis, C4)
BELIEF_MEASURES = ("probe", "decoder_corr")                    # after the reappearance, as in step 3.3
REFERENCES = {"probe_y": "kalman_err_y", "probe": "kalman_err"}  # the Kalman filter, next to the probes
NAME = re.compile(r"^probe_(all|visible|only_\d+|without_\d+)_y$")


def layer_runs(runs: list[ModelRun], data: str) -> list[ModelRun]:
    """The runs whose layer pass on the set is complete."""
    return [run for run in runs if is_complete(layers_folder(run.folder.parent.parent, data))]


def read_layers(runs: list[ModelRun], data: str, baselines: Path) -> pd.DataFrame:
    """The rows of the analysis windows of the cells and tuples, with the columns of the model passes (COLUMNS), the
    baselines and their errors (peekaboo/eval/tables.py read_runs), and the readouts of the layer pass of each run."""
    table = read_runs(runs, baselines, list(CONDITIONS), COLUMNS, rows=ROWS)
    parts = []
    for run in runs:
        folder = layers_folder(run.folder.parent.parent, data)
        readouts = [c for c in frame_columns(folder) if c.startswith(("probe_", "decoder_"))]
        filters = [("condition", "in", list(CONDITIONS)), ("in_window", "==", True)]
        rows = read_frames(folder, ["sequence", "t", "condition", "in_window"] + readouts, filters)
        parts.append(rows[["sequence", "t"] + readouts].assign(model=run.label, seed=run.seed))
    keys = ["model", "seed", "sequence", "t"]
    return table.merge(pd.concat(parts, ignore_index=True), on=keys, how="inner", validate="one_to_one")


def order(name: str) -> tuple[int, int]:
    """Sort key of the readouts: all, visible, every layer alone, every layer but one, then anything else (a
    reference)."""
    kind, _, layer = name.partition("_")
    return {"all": 0, VISIBLE: 1, "only": 2, "without": 3}.get(kind, 4), int(layer) if layer.isdigit() else 0


def readout_names(table: pd.DataFrame) -> list[str]:
    """The readouts of the layer passes in the table, in order."""
    return sorted({match.group(1) for column in table if (match := NAME.match(column))}, key=order)


def add_layer_errors(table: pd.DataFrame) -> pd.DataFrame:
    """The error on y (probe_<name>_err_y) and the distance (probe_<name>_err) of every readout's probe to the true
    center."""
    columns = {}
    for name in readout_names(table):
        dy, dx = table[f"probe_{name}_y"] - table["y"], table[f"probe_{name}_x"] - table["x"]
        columns[f"probe_{name}_err_y"], columns[f"probe_{name}_err"] = dy.abs(), np.hypot(dy, dx)
    return table.assign(**columns)


def visible_before(rows: pd.DataFrame) -> pd.DataFrame:
    """The fully visible frames before the onset, from frame MIN_CONTEXT on."""
    return rows[(rows["since_onset"] > MISSING) & (rows["since_onset"] < 0) & (rows["state"] == VISIBLE_STATE)
                & (rows["t"] >= MIN_CONTEXT)]


def present(rows: pd.DataFrame, column: str) -> bool:
    """Whether a column has values in these rows (a model with fewer layers has no readout of the missing ones)."""
    return column in rows and bool(rows[column].notna().any())


def readouts(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per cell condition, frames (hidden, visible before the onset), model, readout, and measure: the mean over
    sequences with its bootstrap interval, and the Kalman filter for the probe; and the paired difference of every
    readout against all the layers, on the same sequences."""
    summary, paired = [], []
    names = readout_names(table)
    for condition in CELLS:
        cells = event_rows(table, condition)
        for frames, rows in (("hidden", hidden(cells)), ("visible", visible_before(cells))):
            where = {"condition": condition, "frames": frames}
            for model, own in rows.groupby("model", sort=False):
                for measure in HIDDEN_MEASURES:
                    template = MEASURES[measure][1]
                    base = per_sequence(own, template.format("all"))
                    for name in names:
                        if not present(own, template.format(name)):
                            continue
                        values = per_sequence(own, template.format(name))
                        line = {**where, "who": model, "kind": "model", "readout": name, "measure": measure}
                        summary.append({**line, **interval(bootstrap(values))})
                        if name != "all":
                            paired.append({**line, "against": "all", **interval(paired_difference(values, base))})
            if len(rows):
                first = rows[rows["model"] == rows["model"].iloc[0]]
                summary.append({**where, "who": "Kalman", "kind": "reference", "readout": "", "measure": "probe_y",
                                **interval(bootstrap(per_sequence(first, REFERENCES["probe_y"])))})
    return pd.DataFrame(summary), pd.DataFrame(paired)


def curves(table: pd.DataFrame) -> pd.DataFrame:
    """The probe's error on y and the decoder's correlation of every readout against j, the frames since the onset,
    around the hidden frames of the cells, per condition, k, and model."""
    parts = []
    names = readout_names(table)
    for condition in CELLS:
        rows = around_hidden(event_rows(table, condition))
        for model, own in rows.groupby("model", sort=False):
            for measure in ("probe_y", "decoder_corr"):
                for name in names:
                    column = MEASURES[measure][1].format(name)
                    if present(own, column):
                        parts.append(group_mean_se(own, column, ["k", "since_onset"]).assign(
                            condition=condition, who=model, kind="model", readout=name, measure=measure))
    out = pd.concat(parts, ignore_index=True).rename(columns={"since_onset": "j"})
    return out[["condition", "measure", "who", "kind", "readout", "k", "j", "mean", "se", "n"]]


def belief(table: pd.DataFrame) -> pd.DataFrame:
    """Per surprise kind, pair (AB - B, BA - A), model, readout, and measure: the mean over the HORIZON frames since
    each member's own reappearance, the impossible member minus the possible one on the same tuples (bootstrap over
    tuples), with the mean of each member; the Kalman filter for the probe."""
    rows = tuple_rows(table)
    rows = rows[(rows["since_reappear"] >= 0) & (rows["since_reappear"] < HORIZON)]
    attributes = rows.groupby("sequence")[["event", "role", "tuple_id"]].first()
    names = readout_names(table)
    lines = []
    for model, own in rows.groupby("model", sort=False):
        for measure in BELIEF_MEASURES:
            template = MEASURES[measure][1]
            lines += [(model, "model", name, measure, own, template.format(name)) for name in names
                      if present(own, template.format(name))]
    if len(rows) and "probe" in BELIEF_MEASURES:
        first = rows[rows["model"] == rows["model"].iloc[0]]
        lines.append(("Kalman", "reference", "", "probe", first, REFERENCES["probe"]))
    out = []
    for who, kind, name, measure, own, column in lines:
        values = own.groupby(["seed", "sequence"])[column].mean().rename("value").reset_index()
        values = values.join(attributes, on="sequence")
        for event in KINDS:
            for role, control in PAIRS.items():
                a = values[(values["event"] == event) & (values["role"] == role)]
                b = values[(values["event"] == event) & (values["role"] == control)]
                if len(a) == 0 or len(b) == 0:  # AB of `vanish` never reappears
                    continue
                a, b = matrix(a, "value", "tuple_id"), matrix(b, "value", "tuple_id")
                common = a.columns.intersection(b.columns)
                if len(common) == 0:
                    continue
                out.append({"event": event, "pair": f"{role} - {control}", "who": who, "kind": kind, "readout": name,
                            "measure": measure, "positive_mean": float(a[common].mean().mean()),
                            "negative_mean": float(b[common].mean().mean()),
                            **interval(paired_difference(a[common], b[common]))})
    return pd.DataFrame(out)


def relative_errors(table: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """The table with the activity of the error units of every layer and region relative to its mean, per model and
    seed, on the fully visible frames before the onset of the cells (<column>_rel), and the columns."""
    columns = [c for c in ERRORS if present(table, c)]
    cells = table[table["condition"].isin(CELLS) & (table["tuple_role"] == "none") & table["in_window"]]
    baseline = visible_before(cells).groupby(["model", "seed"])[columns].mean()
    scale = baseline.reindex(pd.MultiIndex.from_frame(table[["model", "seed"]]))
    return table.assign(**{f"{c}_rel": table[c].to_numpy() / scale[c].to_numpy() for c in columns}), columns


def split_error(column: str) -> tuple[int, str]:
    """The layer and region of an error column."""
    layer, _, region = column[1:].partition("_")
    return int(layer), region or "frame"


def errors(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The relative activity of the error units of every layer and region (frame, bar): its curve against j around
    the hidden frames of the cells, per condition, k, and model; and per condition and model its mean over the
    hidden frames and over the frames EMERGING since the reappearance, with bootstrap intervals over sequences."""
    table, columns = relative_errors(table)
    curve, summary = [], []
    for condition in CELLS:
        cells = event_rows(table, condition)
        phases = {"hidden": hidden(cells),
                  "reappearance": cells[cells["since_reappear"].between(EMERGING[0], EMERGING[1])]}
        around = around_hidden(cells)
        for model, own in around.groupby("model", sort=False):
            for column in columns:
                if not present(own, column):
                    continue
                layer, region = split_error(column)
                line = {"condition": condition, "who": model, "layer": layer, "region": region}
                curve.append(group_mean_se(own, f"{column}_rel", ["k", "since_onset"]).assign(**line))
                for phase, rows in phases.items():
                    values = per_sequence(rows[rows["model"] == model], f"{column}_rel")
                    summary.append({**line, "phase": phase, **interval(bootstrap(values))})
    out = pd.concat(curve, ignore_index=True).rename(columns={"since_onset": "j"}) if curve else pd.DataFrame()
    return out, pd.DataFrame(summary)


def inside(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Every table of the analysis: readouts, readouts_paired, curves, belief, errors, and error_curves."""
    table = add_layer_errors(table)
    summary, paired = readouts(table)
    error_curves, error_summary = errors(table)
    return {"readouts": summary, "readouts_paired": paired, "curves": curves(table), "belief": belief(table),
            "errors": error_summary, "error_curves": error_curves}


# figures

def layer_count(names: pd.Series) -> int:
    """The number of layers of a model, from the names of its readouts."""
    return sum(1 for name in names.unique() if name.startswith("only_"))


def errorbar(ax, x: list[float], rows: pd.DataFrame, style: str, color: str, label: str) -> None:
    """Means with their 95% intervals."""
    ax.errorbar(x, rows["mean"], yerr=[rows["mean"] - rows["low"], rows["high"] - rows["mean"]], fmt=style,
                color=color, ms=4, lw=1.2, capsize=2, label=label)


def plot_layers(summary: pd.DataFrame, path: Path, title: str, condition: str = "occlusion") -> Path:
    """One column per model, one row per measure: on the hidden frames, the readout of every layer alone and of all
    the layers but one against the layer, with all the layers, the readout fitted on visible frames, and Kalman."""
    rows = summary[(summary["condition"] == condition) & (summary["frames"] == "hidden")]
    models = [m for m in rows.loc[rows["kind"] == "model", "who"].unique()]
    colors = model_colors(rows)
    measures = ("probe_y", "decoder_corr")
    fig = Figure(figsize=(3.1 * max(len(models), 1), 5.4))
    axes = fig.subplots(len(measures), max(len(models), 1), squeeze=False)
    kalman = rows[(rows["who"] == "Kalman") & (rows["measure"] == "probe_y")]
    for column, model in enumerate(models):
        for row, measure in enumerate(measures):
            ax = axes[row, column]
            own = rows[(rows["who"] == model) & (rows["measure"] == measure)].set_index("readout")
            n = layer_count(own.index.to_series())
            x = list(range(n))
            errorbar(ax, x, own.reindex([f"only_{l}" for l in x]), "o-", colors[model], "this layer alone")
            if n > 1:
                errorbar(ax, x, own.reindex([f"without_{l}" for l in x]), "x--", "0.45", "all but this layer")
            ax.axhline(own.loc["all", "mean"], color=colors[model], lw=1.0, label="all layers")
            if VISIBLE in own.index:
                ax.axhline(own.loc[VISIBLE, "mean"], color=colors[model], lw=1.0, ls=":",
                           label="all layers, fitted on visible frames")
            if measure == "probe_y" and len(kalman):
                ax.axhline(kalman["mean"].iloc[0], color="0.3", lw=1.0, ls="-.", label="Kalman")
            ax.set_xticks(x)
            # a decoder below chance (a correlation under 0) stays in view
            ax.set_ylim(*((min(0.0, ax.get_ylim()[0]), 1) if measure.startswith("decoder") else (0, None)))
            if row == 0:
                ax.set_title(model, fontsize=9)
            else:
                ax.set_xlabel("layer")
            if column == 0:
                ax.set_ylabel(MEASURES[measure][0])
    if models:
        axes[0, 0].legend(fontsize=6)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return path


def plot_visible(curve: pd.DataFrame, path: Path, title: str, condition: str = "occlusion", k: int = K_SHOWN) -> Path:
    """One column per model, one row per measure: the readouts of all the layers fitted on every frame (solid) and on
    visible frames only (dotted) against j, for one k; hidden frames shaded."""
    rows = curve[(curve["condition"] == condition) & (curve["k"] == k) & curve["readout"].isin(["all", VISIBLE])]
    models = list(rows["who"].unique())
    colors = model_colors(rows)
    measures = ("probe_y", "decoder_corr")
    fig = Figure(figsize=(3.1 * max(len(models), 1), 5.4))
    axes = fig.subplots(len(measures), max(len(models), 1), squeeze=False, sharex=True)
    for column, model in enumerate(models):
        for row, measure in enumerate(measures):
            ax = axes[row, column]
            for name, style, label in (("all", "o-", "every frame"), (VISIBLE, "s:", "visible frames only")):
                line = rows[(rows["who"] == model) & (rows["measure"] == measure) & (rows["readout"] == name)]
                line = line.sort_values("j")
                ax.plot(line["j"], line["mean"], style, ms=2.5, lw=1.3, color=colors[model], label=label)
                ax.fill_between(line["j"], line["mean"] - line["se"], line["mean"] + line["se"],
                                color=colors[model], alpha=0.2, lw=0)
            ax.axvspan(-0.5, k - 0.5, color="0.92", zorder=0)
            # a decoder below chance (a correlation under 0) stays in view
            ax.set_ylim(*((min(0.0, ax.get_ylim()[0]), 1) if measure.startswith("decoder") else (0, None)))
            if row == 0:
                ax.set_title(model, fontsize=9)
            else:
                ax.set_xlabel("frames since the onset")
            if column == 0:
                ax.set_ylabel(MEASURES[measure][0])
    if models:
        axes[0, 0].legend(fontsize=6, title="readout fitted on", title_fontsize=6)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return path


def plot_belief(table: pd.DataFrame, path: Path, title: str, measure: str = "probe") -> Path:
    """One panel per surprise kind: the impossible member minus the possible one (AB - B, BA - A for `vanish`) of
    every layer alone and of all the layers, per model, with Kalman."""
    rows = table[table["measure"] == measure]
    colors = model_colors(rows)
    fig = Figure(figsize=(12, 6.2))
    axes = fig.subplots(2, 3, squeeze=False).ravel()
    for index, (ax, event) in enumerate(zip(axes, KINDS)):
        pair = "BA - A" if event == "vanish" else "AB - B"
        own = rows[(rows["event"] == event) & (rows["pair"] == pair)]
        for model, line in own[own["kind"] == "model"].groupby("who", sort=False):
            line = line.set_index("readout")
            x = list(range(layer_count(line.index.to_series())))
            errorbar(ax, [-1] + x, line.reindex(["all"] + [f"only_{l}" for l in x]), "o-", colors[model], model)
        kalman = own[own["who"] == "Kalman"]
        if len(kalman):
            ax.axhline(kalman["mean"].iloc[0], color="0.3", lw=1.0, ls="-.", label="Kalman")
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xticks(range(-1, LAYERS), ["all"] + [str(l) for l in range(LAYERS)])
        ax.set_title(f"{event.replace('_', ' ')}, {pair}", fontsize=9)
        ax.set_xlabel("layer read")
        if index % 3 == 0:
            ax.set_ylabel(f"{MEASURES[measure][0]},\nimpossible - possible")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=max(len(labels), 1), fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=120)
    return path


def plot_errors(curve: pd.DataFrame, path: Path, title: str, k: int = K_SHOWN) -> Path:
    """One row per PredNet model, one column per cell condition: the relative activity of the error units of every
    layer against j, for one k, on the whole frame (solid) and over the bar (dashed); hidden frames shaded."""
    rows = curve[curve["k"] == k] if len(curve) else curve
    models = list(rows["who"].unique()) if len(rows) else []
    fig = Figure(figsize=(9, 2.6 * max(len(models), 1)))
    axes = fig.subplots(max(len(models), 1), len(CELLS), squeeze=False, sharex=True)
    for row, model in enumerate(models):
        for column, condition in enumerate(CELLS):
            ax = axes[row, column]
            own = rows[(rows["who"] == model) & (rows["condition"] == condition)]
            for (layer, region), line in own.groupby(["layer", "region"]):
                line = line.sort_values("j")
                ax.plot(line["j"], line["mean"], "-" if region == "frame" else "--", lw=1.2, color=f"C{layer}",
                        label=f"layer {layer}" if region == "frame" else None)
            ax.axhline(1, color="0.6", lw=0.8)
            ax.axvspan(-0.5, k - 0.5, color="0.92", zorder=0)
            ax.set_ylim(bottom=0)
            ax.set_title(f"{model}, {condition.replace('_', ' ')}, k = {k}", fontsize=9)
            if column == 0:
                ax.set_ylabel("error units / visible level")
            if row == len(models) - 1:
                ax.set_xlabel("frames since the onset")
    if models:
        axes[0, 0].legend(fontsize=6, title="solid: frame, dashed: bar", title_fontsize=6)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return path
