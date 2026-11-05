"""Correction after the reappearance (RQ2, step 3.3), from the extraction tables of step 3.1 (peekaboo/eval/tables.py).

When the digit comes back into view, how many frames does each model need to be right again, and is it slower when
the reappearance is surprising (H2)? Everything is aligned on the actual reappearance frame (j = 0, the first frame
with visible ink):
- expected reappearances: the sequences of the occlusion and hidden bounce cells;
- surprising ones: the impossible member of each surprise tuple, against the possible member that ends with the
  same frames (peekaboo/data/splicing.py): AB against B, and BA against A as the mirror. For `vanish`, AB never
  reappears; its BA, where the digit appears from nowhere, is compared with A.

Three readouts are corrected, each against its references:
- the state: the distance of the position probe to the true center, next to the Kalman filters, which correct at
  the first detection;
- the predicted frame: the distance of the drawn digit (its ink centroid off the bar) to the visible ink center
  (critique C9: the modal readout is compared with the modal truth), next to the copy of the last frame;
- the pixels: the MSE in the box around the digit, next to the copy and the blank frame.

The latency (decision D14) is the first frame since the reappearance from which the error stays at or below tau for
M frames in a row, within the first HORIZON frames. Tau is per model and readout: the TAU_QUANTILE quantile of its
error on the fully visible frames before the onset of the occlusion and hidden bounce cells (from frame MIN_CONTEXT
on), at least the readout's floor (3 px for positions); the copy and the blank frame get no latency on the MSE. A
sequence that never gets there is censored, and counts as HORIZON - M + 1 frames (a restricted mean); one whose
window ends too early to tell is left out. The full curves come with every latency.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from peekaboo.data.truth import STATE_NAMES
from peekaboo.eval.extract import MISSING
from peekaboo.eval.next_frame import MIN_CONTEXT
from peekaboo.eval.occlusion import Readout, draw_line, group_mean_se, interval, model_colors
from peekaboo.eval.stats import bootstrap, paired_difference

CONDITIONS = ("occlusion", "hidden_bounce", "surprise")
BEFORE, AFTER = 3, 7       # frames since the reappearance shown: -BEFORE to AFTER
HORIZON = 6                # frames since the reappearance read for the latency, j = 0 to HORIZON - 1
M = 2                      # frames in a row at or below tau
TAU_QUANTILE = 0.9
VISIBLE = STATE_NAMES.index("visible")
SURPRISING = {"AB": "B", "BA": "A"}  # each impossible member and the possible one that ends with the same frames
# the panels of the figures: the event, the role shown, and its matched control
PANELS = (("occlusion", "none", None), ("hidden_bounce", "none", None), ("direction", "AB", "B"),
          ("offset", "AB", "B"), ("early", "AB", "B"), ("speed_fast", "AB", "B"), ("speed_slow", "AB", "B"),
          ("vanish", "BA", "A"))
COLUMNS = ["sequence", "t", "y", "x", "in_window", "condition", "k", "speed", "state", "since_onset",
           "since_reappear", "tuple_id", "tuple_role", "surprise_type", "modal_y", "modal_x", "probe_y", "probe_x",
           "random_probe_y", "random_probe_x", "decoder_corr", "random_decoder_corr", "pred_ink_y", "pred_ink_x",
           "model_mse_digit", "copy_mse_digit", "blank_mse_digit"]  # the columns of a model pass read here
# the rows read: around the reappearance, and the visible frames before the onset of the cells (for tau)
ROWS = [[("in_window", "==", True), ("since_reappear", ">=", -BEFORE), ("since_reappear", "<=", AFTER)],
        [("in_window", "==", True), ("tuple_role", "==", "none"), ("since_onset", ">", MISSING),
         ("since_onset", "<", 0), ("state", "==", VISIBLE), ("t", ">=", MIN_CONTEXT)]]

READOUTS = (
    Readout("probe", "probe_err", "probe distance to the center (px)", "random_probe_err",
            {"Kalman": "kalman_err", "Kalman with walls": "kalman_walls_err"}),
    Readout("readout", "pred_modal_err", "drawn digit, distance to the visible center (px)", None,
            {"copy": "copy_modal_err"}),
    Readout("digit_mse", "model_mse_digit", "MSE around the digit", None,
            {"copy": "copy_mse_digit", "blank": "blank_mse_digit"}),
    Readout("decoder_corr", "decoder_corr", "decoder correlation", "random_decoder_corr",
            {"template, Kalman": "template_kalman_corr", "template, Kalman with walls": "template_walls_corr"}),
)
FLOORS = {"probe": 3.0, "readout": 3.0, "digit_mse": 0.0}  # the readouts with a latency, and the least tau of each
# readouts whose references get no latency: the copy and the blank frame never correct, so against a threshold taken
# from their own (large) error on visible frames, their latency is 0 and means nothing
NO_REFERENCE_LATENCY = ("digit_mse",)


def latency_readouts() -> list[Readout]:
    """The readouts with a latency (the decoder is a similarity, shown in the curves only)."""
    return [r for r in READOUTS if r.name in FLOORS]


def correction_rows(table: pd.DataFrame) -> pd.DataFrame:
    """The rows with what the analysis needs: the event (the condition of a cell, or the kind of a tuple), the role,
    the k and speed of the event (those of A for every member of a tuple), the distance of the drawn digit to the
    visible center (pred_modal_err, NaN when either is undefined), and that of the copy (copy_modal_err: the visible
    center of the previous frame)."""
    rows = table.sort_values(["sequence", "t"], kind="stable")
    role = rows["tuple_role"]
    members = rows[role.isin(["AB", "BA"])]
    kind = members.groupby("tuple_id")["surprise_type"].first().replace({"appear": "vanish"})
    first = rows[role == "A"].groupby("tuple_id")[["k", "speed"]].first()
    member = (role != "none").to_numpy()
    tuple_id = rows["tuple_id"]
    previous = rows.groupby(["model", "seed", "sequence"])[["t", "modal_y", "modal_x"]].shift(1)
    follows = (rows["t"] - previous["t"]) == 1
    return rows.assign(
        event=np.where(member, tuple_id.map(kind), rows["condition"]),
        role=role,
        group_k=np.where(member, tuple_id.map(first["k"]), rows["k"]),
        group_speed=np.where(member, tuple_id.map(first["speed"]), rows["speed"]),
        pred_modal_err=np.hypot(rows["pred_ink_y"] - rows["modal_y"], rows["pred_ink_x"] - rows["modal_x"]),
        copy_modal_err=np.hypot(previous["modal_y"] - rows["modal_y"], previous["modal_x"] - rows["modal_x"])
        .where(follows))


def tau_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """The fully visible frames before the onset, in the sequences of the occlusion and hidden bounce cells, from
    frame MIN_CONTEXT on (the first frames are predicted with no motion seen, and the probes are fitted without
    them)."""
    return rows[rows["condition"].isin(["occlusion", "hidden_bounce"]) & (rows["role"] == "none")
                & rows["in_window"] & (rows["since_onset"] > MISSING) & (rows["since_onset"] < 0)
                & (rows["state"] == VISIBLE) & (rows["t"] >= MIN_CONTEXT)]


def around(rows: pd.DataFrame) -> pd.DataFrame:
    """The rows from BEFORE frames before the reappearance to AFTER frames after it."""
    return rows[rows["in_window"] & (rows["since_reappear"] >= -BEFORE) & (rows["since_reappear"] <= AFTER)]


def thresholds(rows: pd.DataFrame) -> pd.DataFrame:
    """Tau of every line (model or reference) of every readout with a latency: the TAU_QUANTILE quantile of its error
    on the visible frames before the onset, at least the readout's floor."""
    out = []
    visible = tau_rows(rows)
    for readout in latency_readouts():
        for who, kind, column, own in readout.lines(visible):
            if kind == "random" or (kind == "reference" and readout.name in NO_REFERENCE_LATENCY):
                continue
            quantile = float(own[column].quantile(TAU_QUANTILE))
            out.append({"readout": readout.name, "who": who, "kind": kind, "quantile": quantile,
                        "tau": max(quantile, FLOORS[readout.name]), "n": int(own[column].notna().sum())})
    return pd.DataFrame(out)


def latency(rows: pd.DataFrame, column: str, tau: float) -> pd.DataFrame:
    """Per seed and sequence: the first frame j (0 to HORIZON - M) from which `column` stays at or below tau for M
    frames (a NaN value is not), the status (found, censored when it never does, short when frames are missing
    and it was not found), the latency counted as HORIZON - M + 1 when censored (NaN when short), and the mean of
    the column over the HORIZON frames."""
    rows = rows[(rows["since_reappear"] >= 0) & (rows["since_reappear"] < HORIZON)]
    keys = [rows["seed"], rows["sequence"], rows["since_reappear"]]
    below = (rows[column] <= tau).astype(float).groupby(keys).first().unstack("since_reappear")
    values = below.reindex(columns=range(HORIZON)).to_numpy()
    ok = values == 1
    span = HORIZON - M + 1
    run = np.ones((len(values), span), dtype=bool)
    for i in range(M):
        run &= ok[:, i:i + span]
    found = run.any(axis=1)
    complete = ~np.isnan(values).any(axis=1)
    error = rows.groupby([rows["seed"], rows["sequence"]])[column].mean().reindex(below.index)
    return pd.DataFrame({"latency": np.where(found, run.argmax(axis=1), np.where(complete, span, np.nan)),
                         "status": np.select([found, complete], ["found", "censored"], "short"),
                         "error": error.to_numpy()}, index=below.index).reset_index()


def sequence_latencies(rows: pd.DataFrame, taus: pd.DataFrame) -> pd.DataFrame:
    """The latency of every line of every readout on every sequence around its reappearance, with the event, role,
    tuple, k, and speed of the sequence."""
    window = around(rows)
    attributes = window.groupby("sequence")[["event", "role", "tuple_id", "group_k", "group_speed"]].first()
    tau = taus.set_index(["readout", "who"])["tau"]
    parts = []
    for readout in latency_readouts():
        for who, kind, column, own in readout.lines(window):
            if (readout.name, who) in tau.index:
                parts.append(latency(own, column, tau[(readout.name, who)]).assign(
                    readout=readout.name, who=who, kind=kind))
    return pd.concat(parts, ignore_index=True).join(attributes, on="sequence")


def scopes(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """All the rows, then those of every k, then those of every speed."""
    ks, speeds = sorted(frame["group_k"].dropna().unique()), sorted(frame["group_speed"].dropna().unique())
    return ([("all", frame)] + [(f"k={int(k)}", frame[frame["group_k"] == k]) for k in ks]
            + [(f"speed={int(v)}", frame[frame["group_speed"] == v]) for v in speeds])


def matrix(frame: pd.DataFrame, value: str, key: str = "sequence") -> pd.DataFrame:
    """One row per seed, one column per sequence (or tuple)."""
    return frame.set_index(["seed", key])[value].unstack(key)


def summary(latencies: pd.DataFrame, reference: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per event, role, scope, readout, and line: the shares of sequences found, censored, and short, the restricted
    mean latency and the mean error over the HORIZON frames, with bootstrap intervals. And the paired differences:
    of each surprising role against its control on the same tuples (AB - B, BA - A), and of each model against the
    reference model on the same sequences."""
    rows, paired = [], []
    by_line = dict(tuple(latencies.groupby(["event", "role", "readout", "who"], sort=False)))
    for (event, role), group in latencies.groupby(["event", "role"], sort=False):
        for scope, own in scopes(group):
            for (readout, who), line in own.groupby(["readout", "who"], sort=False):
                n = line["sequence"].nunique()
                shares = line["status"].value_counts(normalize=True)
                rows.append({"event": event, "role": role, "scope": scope, "readout": readout, "who": who,
                             "kind": line["kind"].iloc[0], "sequences": n,
                             **{status: float(shares.get(status, 0.0)) for status in ("found", "censored", "short")},
                             **interval(bootstrap(matrix(line, "latency")), "latency_"),
                             **interval(bootstrap(matrix(line, "error")), "error_")})
                if role in SURPRISING:
                    control = by_line.get((event, SURPRISING[role], readout, who), line.iloc[:0])
                    control = control[control["tuple_id"].isin(line["tuple_id"])]
                    for value in ("latency", "error"):
                        paired.append({"event": event, "role": role, "scope": scope, "readout": readout, "who": who,
                                       "against": SURPRISING[role], "measure": value, **interval(paired_difference(
                                           matrix(line, value, "tuple_id"), matrix(control, value, "tuple_id")))})
                if line["kind"].iloc[0] == "model" and who != reference:
                    base = own[(own["readout"] == readout) & (own["who"] == reference)]
                    if len(base):
                        for value in ("latency", "error"):
                            paired.append({"event": event, "role": role, "scope": scope, "readout": readout,
                                           "who": who, "against": reference, "measure": value,
                                           **interval(paired_difference(matrix(line, value),
                                                                        matrix(base, value).mean()))})
    return pd.DataFrame(rows), pd.DataFrame(paired)


def curves(rows: pd.DataFrame) -> pd.DataFrame:
    """Every readout against j, the frames since the reappearance, per event, role, and line."""
    window = around(rows)
    parts = []
    for (event, role), group in window.groupby(["event", "role"], sort=False):
        for readout in READOUTS:
            for who, kind, column, own in readout.lines(group):
                parts.append(group_mean_se(own, column, ["since_reappear"]).assign(
                    event=event, role=role, readout=readout.name, who=who, kind=kind))
    out = pd.concat(parts, ignore_index=True).rename(columns={"since_reappear": "j"})
    return out[["event", "role", "readout", "who", "kind", "j", "mean", "se", "n"]]


def correction(table: pd.DataFrame, reference: str) -> dict[str, pd.DataFrame]:
    """Every table of the analysis: taus, curves, latency (summary), and latency_paired."""
    rows = correction_rows(table)
    taus = thresholds(rows)
    latency_summary, paired = summary(sequence_latencies(rows, taus), reference)
    return {"taus": taus, "curves": curves(rows), "latency": latency_summary, "latency_paired": paired}


# figures

def panel_title(event: str, role: str, control: str | None) -> str:
    """The title of a panel: the event, and for a tuple which member is drawn and which is dashed."""
    name = "appear (vanish)" if event == "vanish" else event.replace("_", " ")
    return name if control is None else f"{name}: {role}, {control} dashed"


def plot_curves(curve: pd.DataFrame, path: Path, title: str) -> Path:
    """One row per readout, one column per event: every model against j, with the references in gray; for a tuple,
    the surprising member, and its matched control dashed."""
    names = [r.name for r in READOUTS]
    panels = [p for p in PANELS if ((curve["event"] == p[0]) & (curve["role"] == p[1])).any()]
    colors = model_colors(curve)
    fig = Figure(figsize=(2.7 * len(panels) + 1, 2.4 * len(names) + 0.6))
    axes = fig.subplots(len(names), len(panels), squeeze=False)
    for c, (event, role, control) in enumerate(panels):
        for r, name in enumerate(names):
            ax = axes[r, c]
            own = curve[(curve["event"] == event) & (curve["readout"] == name)]
            for who, line in own[own["role"] == role].groupby("who", sort=False):
                kind = line["kind"].iloc[0]
                if kind != "random":
                    draw_line(ax, line, "j", who, kind, colors.get(who, "0.5"))
            if control is not None:
                matched = own[(own["role"] == control) & (own["kind"] == "model")]
                for who, line in matched.groupby("who", sort=False):
                    line = line.sort_values("j")
                    ax.plot(line["j"], line["mean"], "--", lw=1.0, color=colors.get(who, "0.5"))
            ax.axvline(-0.5, color="0.6", lw=0.8)
            if name == "digit_mse":
                ax.set_yscale("log")
            elif name == "decoder_corr":
                ax.set_ylim(0, 1)
            else:
                ax.set_ylim(bottom=0)
            if r == 0:
                ax.set_title(panel_title(event, role, control), fontsize=8)
            if c == 0:
                ax.set_ylabel(next(x.label for x in READOUTS if x.name == name), fontsize=8)
            if r == len(names) - 1:
                ax.set_xlabel("j, frames since the reappearance", fontsize=8)
            ax.tick_params(labelsize=7)
    for r in range(len(names)):
        axes[r, -1].legend(fontsize=6)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return path


def plot_latency(table: pd.DataFrame, path: Path, title: str) -> Path:
    """One panel per readout: the restricted mean latency of every model on every event (filled: the reappearance of
    the cell or the surprising member; open: its matched control), with 95% intervals, and the references in gray."""
    rows = table[table["scope"] == "all"]
    panels = [p for p in PANELS if ((rows["event"] == p[0]) & (rows["role"] == p[1])).any()]
    colors = model_colors(rows)
    fig = Figure(figsize=(13, 3.6 * len(FLOORS) + 0.4))
    axes = fig.subplots(len(FLOORS), 1, squeeze=False)[:, 0]
    for ax, readout in zip(axes, latency_readouts()):
        own = rows[rows["readout"] == readout.name]
        lines = list(own["who"].unique())
        for i, who in enumerate(lines):
            offset = (i - (len(lines) - 1) / 2) * 0.09
            kind = own.loc[own["who"] == who, "kind"].iloc[0]
            color = colors.get(who, "0.4") if kind == "model" else "0.45"
            marker = "o" if kind == "model" else {"Kalman": "s", "Kalman with walls": "D", "copy": "x"}.get(who, "^")
            for x, (event, role, control) in enumerate(panels):
                for member, filled in ((role, True), (control, False)):
                    match = own[(own["event"] == event) & (own["role"] == member) & (own["who"] == who)]
                    if member is None or match.empty:
                        continue
                    value = match.iloc[0]
                    ax.errorbar(x + offset + (0.0 if filled else 0.03), value["latency_mean"],
                                yerr=[[value["latency_mean"] - value["latency_low"]],
                                      [value["latency_high"] - value["latency_mean"]]],
                                fmt=marker, ms=4, color=color, mfc=color if filled else "white", lw=0.8,
                                label=who if (x == 0 and filled) else None)
        ax.set_xticks(range(len(panels)))
        ax.set_xticklabels([panel_title(e, r, c) for e, r, c in panels], fontsize=7)
        ax.axhline(HORIZON - M + 1, color="0.7", lw=0.8, ls=":")
        ax.set_ylim(-0.2, HORIZON - M + 1.4)
        ax.set_ylabel(f"latency, {readout.name} (frames)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, ncol=2)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return path
