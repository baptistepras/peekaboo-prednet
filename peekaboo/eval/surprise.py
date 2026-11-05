"""Surprise (RQ3, step 3.4), from the extraction tables of step 3.1 (peekaboo/eval/tables.py).

Are the prediction errors larger after a surprising reappearance than after a matched plausible one? Each surprise
tuple (peekaboo/data/splicing.py) gives the controlled comparison of PLATO (critique C3): the impossible member AB and
the possible member B show exactly the same frames after the splice, and differ only in their history; BA against A is
the mirror. For `vanish`, AB is the digit that never comes back and B the same scene without a digit; its BA is the
digit that appears from nowhere.

The surprise is read in a window of WINDOW frames starting at the first frame where the impossible member departs
from what its own history announces: the earlier of the two reappearances (that of A, expected by AB's history, and
that of B, which AB shows). This is the actual reappearance when it comes early or on time, and the expected one when
the digit comes late or never. The splice is hidden in both members, so the window starts after it and every frame
in it is the same in AB and B (the copy of the last frame is too, which gives an exact zero check).

Signals: the next frame MSE on the whole frame and over the bar region (every model, next to the copy and the blank
frame), and for PredNet the mean activity of the error units of every layer, on the whole frame and over the bar.
Per tuple, the surprise index is the signal of AB minus that of B, averaged over the window; the AUROC says how well
the signal tells AB from B, with a paired bootstrap interval over tuples. With several seeds, the AUROC is taken on
the seed means of every tuple. The AUROC ranks all the AB against all the B, so it also falls when the signal varies
between tuples more than AB and B differ; the share of tuples where AB is higher is the paired reading.

The key contrast (critique C13): a digit that comes back out on its entry side is impossible without a wall
(`direction`) and expected with one (the hidden bounce cells). The same window, at the reappearance, compares the
hidden bounce sequences with the occlusion sequences and with the AB members of `direction` (unpaired AUROC).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from peekaboo.eval.correction import matrix, scopes
from peekaboo.eval.extract import MISSING
from peekaboo.eval.occlusion import draw_line, group_mean_se, interval, model_colors
from peekaboo.eval.stats import bootstrap_auroc, paired_difference

CONDITIONS = ("occlusion", "hidden_bounce", "surprise", "empty")
WINDOW = 5                 # frames read for the surprise, from the first frame where AB departs from its history
BEFORE, AFTER = 3, WINDOW + 2  # frames shown around the window start
LAYERS = 5                 # most PredNet layers in the sweep
PAIRS = {"AB": "B", "BA": "A"}  # each impossible member and the possible one that ends with the same frames
KINDS = ("direction", "offset", "early", "speed_fast", "speed_slow", "vanish")
CONTRASTS = (("hidden_bounce", "occlusion"), ("direction AB", "hidden_bounce"))  # C13, unpaired
# the signals: label, the model's column, and the references' columns
SIGNALS = {"mse": ("next frame MSE", "model_mse", {"copy": "copy_mse", "blank": "blank_mse"}),
           "mse_bar": ("MSE over the bar region", "model_mse_bar", {"copy": "copy_mse_bar", "blank": "blank_mse_bar"}),
           **{f"e{l}": (f"error units, layer {l}", f"e{l}", {}) for l in range(LAYERS)},
           **{f"e{l}_bar": (f"error units, layer {l}, bar", f"e{l}_bar", {}) for l in range(LAYERS)}}
MAIN = ("mse", "mse_bar")  # the signals of every model, read per k and speed too; the error units on all tuples only
COLUMNS = (["sequence", "t", "condition", "k", "speed", "since_reappear", "tuple_id", "tuple_role", "surprise_type"]
           + [column for _, column, _ in SIGNALS.values()]
           + [column for _, _, refs in SIGNALS.values() for column in refs.values()])
# the rows read: every member of a tuple, and the cells around the reappearance
ROWS = [[("tuple_role", "in", list(PAIRS) + list(PAIRS.values()))],
        [("tuple_role", "==", "none"), ("since_reappear", ">=", -BEFORE), ("since_reappear", "<=", AFTER)]]


def lines(rows: pd.DataFrame, signal: str) -> list[tuple[str, str, str, pd.DataFrame]]:
    """(who, kind, column, rows) of every line of a signal: each model that has it, then each reference once (read
    on the rows of the first model, since it does not depend on the model)."""
    _, column, references = SIGNALS[signal]
    out = [(model, "model", column, own) for model, own in rows.groupby("model", sort=False)
           if column in own and own[column].notna().any()]
    if len(rows):
        first = rows[rows["model"] == rows["model"].iloc[0]]
        out += [(label, "reference", ref, first) for label, ref in references.items()]
    return out


def reappearance(rows: pd.DataFrame) -> pd.Series:
    """The reappearance frame of every sequence that has one."""
    known = rows[rows["since_reappear"] > MISSING]
    return (known["t"] - known["since_reappear"]).groupby(known["sequence"]).first()


def tuple_rows(table: pd.DataFrame) -> pd.DataFrame:
    """The rows of the tuple members with their event (the kind of the tuple), role, k and speed (those of A), the
    window start of the tuple (the earlier reappearance of A and B), and f, the frames since the window start."""
    rows = table[table["tuple_role"] != "none"].sort_values(["sequence", "t"], kind="stable")
    role = rows["tuple_role"]
    kind = rows[role.isin(["AB", "BA"])].groupby("tuple_id")["surprise_type"].first().replace({"appear": "vanish"})
    first = rows[role == "A"].groupby("tuple_id")[["k", "speed"]].first()
    back = reappearance(rows)
    members = rows.groupby("sequence")[["tuple_id", "tuple_role"]].first()
    members["reappear"] = back.reindex(members.index)
    by_role = members.pivot_table(index="tuple_id", columns="tuple_role", values="reappear", aggfunc="first")
    start = np.fmin(by_role.get("A"), by_role.get("B")) if "B" in by_role else by_role["A"]
    tuple_id = rows["tuple_id"]
    return rows.assign(event=tuple_id.map(kind), role=role, group_k=tuple_id.map(first["k"]),
                       group_speed=tuple_id.map(first["speed"]), f=rows["t"] - tuple_id.map(start))


def cell_rows(table: pd.DataFrame) -> pd.DataFrame:
    """The rows of the occlusion and hidden bounce cells, with f, the frames since the reappearance."""
    rows = table[table["tuple_role"] == "none"]
    return rows.assign(event=rows["condition"], role="none", group_k=rows["k"], group_speed=rows["speed"],
                       f=rows["since_reappear"])


def window_values(rows: pd.DataFrame, column: str) -> pd.DataFrame:
    """Per seed and sequence, the mean of a column over the window, with the event, role, tuple, k, and speed."""
    window = rows[(rows["f"] >= 0) & (rows["f"] < WINDOW)]
    values = window.groupby(["seed", "sequence"])[column].mean().rename("value").reset_index()
    attributes = window.groupby("sequence")[["event", "role", "tuple_id", "group_k", "group_speed"]].first()
    return values.join(attributes, on="sequence")


def pair_stats(positive: pd.DataFrame, negative: pd.DataFrame) -> dict[str, float]:
    """The surprise index (positive minus negative on the same tuples, bootstrap over tuples), the AUROC with its
    paired bootstrap interval (on the seed means of every tuple), and the share of tuples where the positive is
    higher."""
    a, b = matrix(positive, "value", "tuple_id"), matrix(negative, "value", "tuple_id")
    common = a.columns.intersection(b.columns)
    mean_a, mean_b = a[common].mean().to_numpy(), b[common].mean().to_numpy()
    auc = bootstrap_auroc(mean_a, mean_b, paired=True)
    finite = np.isfinite(mean_a) & np.isfinite(mean_b)
    return {**interval(paired_difference(a[common], b[common]), "index_"), "auroc": auc["auroc"],
            "auroc_low": auc["low"], "auroc_high": auc["high"], "tuples": auc["n"],
            "higher": float((mean_a[finite] > mean_b[finite]).mean()) if finite.any() else np.nan}


def surprise_index(table: pd.DataFrame) -> pd.DataFrame:
    """Per kind, pair (AB - B, BA - A), scope (all, per k and speed for the main signals), signal, and line: the
    surprise index, the AUROC, and the share of tuples where the impossible member is higher."""
    rows = tuple_rows(table)
    out = []
    for signal in SIGNALS:
        for who, kind, column, own in lines(rows, signal):
            values = window_values(own, column)
            for event, group in values.groupby("event", sort=False):
                for role, control in PAIRS.items():
                    for scope, part in (scopes(group) if signal in MAIN else [("all", group)]):
                        positive, negative = part[part["role"] == role], part[part["role"] == control]
                        if len(positive) and len(negative):
                            out.append({"event": event, "pair": f"{role} - {control}", "scope": scope,
                                        "signal": signal, "who": who, "kind": kind,
                                        **pair_stats(positive, negative)})
    return pd.DataFrame(out)


def contrast(table: pd.DataFrame) -> pd.DataFrame:
    """The key contrast (C13), per signal and line: the AUROC of the window signal, hidden bounce against occlusion
    and `direction` AB against hidden bounce (unpaired, on the seed means of every sequence), with the mean of each."""
    rows = pd.concat([cell_rows(table), tuple_rows(table)], ignore_index=True)
    rows = rows[(rows["role"] == "none") | ((rows["event"] == "direction") & (rows["role"] == "AB"))]
    names = rows["event"].where(rows["role"] == "none", "direction AB")
    rows = rows.assign(event=names)
    out = []
    for signal in SIGNALS:
        for who, kind, column, own in lines(rows, signal):
            values = window_values(own, column)
            means = values.groupby(["event", "sequence"])["value"].mean()
            for positive, negative in CONTRASTS:
                if positive not in means.index.get_level_values(0) or negative not in means.index.get_level_values(0):
                    continue
                auc = bootstrap_auroc(means[positive].to_numpy(), means[negative].to_numpy(), paired=False)
                out.append({"positive": positive, "negative": negative, "signal": signal, "who": who, "kind": kind,
                            "positive_mean": float(means[positive].mean()),
                            "negative_mean": float(means[negative].mean()), "auroc": auc["auroc"],
                            "auroc_low": auc["low"], "auroc_high": auc["high"], "n": auc["n"]})
    return pd.DataFrame(out)


def curves(table: pd.DataFrame) -> pd.DataFrame:
    """Every signal against f, the frames since the window start, per kind, role, and line."""
    rows = tuple_rows(table)
    rows = rows[(rows["f"] >= -BEFORE) & (rows["f"] <= AFTER)]
    parts = []
    for (event, role), group in rows.groupby(["event", "role"], sort=False):
        for signal in SIGNALS:
            for who, kind, column, own in lines(group, signal):
                parts.append(group_mean_se(own, column, ["f"]).assign(event=event, role=role, signal=signal,
                                                                     who=who, kind=kind))
    out = pd.concat(parts, ignore_index=True)
    return out[["event", "role", "signal", "who", "kind", "f", "mean", "se", "n"]]


def surprise(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Every table of the analysis: index, contrast, and curves."""
    return {"index": surprise_index(table), "contrast": contrast(table), "curves": curves(table)}


# figures

def kind_name(kind: str) -> str:
    """The name of a surprise kind in the figures."""
    return kind.replace("_", " ")


def spread(rows: pd.DataFrame) -> np.ndarray:
    """The error bars (2, n) of AUROC rows from their interval, never negative (the estimate on all the tuples can
    fall just outside the bootstrap interval)."""
    return np.clip(np.stack([rows["auroc"] - rows["auroc_low"], rows["auroc_high"] - rows["auroc"]]), 0.0, None)


def plot_auroc(index: pd.DataFrame, contrasts: pd.DataFrame, path: Path, title: str) -> Path:
    """The AUROC of every model per surprise kind, for the MSE on the whole frame and over the bar (filled: AB
    against B, open: BA against A), and the key contrast on the whole frame MSE."""
    rows = index[(index["scope"] == "all") & (index["kind"] == "model")]
    colors = model_colors(rows)
    models = list(rows["who"].unique())
    fig = Figure(figsize=(14, 3.4 * (len(MAIN) + 1) + 0.4))
    axes = fig.subplots(len(MAIN) + 1, 1, squeeze=False)[:, 0]
    for ax, signal in zip(axes, MAIN):
        own = rows[rows["signal"] == signal]
        for i, who in enumerate(models):
            offset = (i - (len(models) - 1) / 2) * 0.1
            for x, kind in enumerate(KINDS):
                for pair, filled in (("AB - B", True), ("BA - A", False)):
                    match = own[(own["event"] == kind) & (own["pair"] == pair) & (own["who"] == who)]
                    if match.empty:
                        continue
                    value = match.iloc[0]
                    ax.errorbar(x + offset + (0.0 if filled else 0.04), value["auroc"], yerr=spread(match),
                                fmt="o", ms=4, color=colors[who], mfc=colors[who] if filled else "white", lw=0.8,
                                label=who if (x == 0 and filled) else None)
        ax.set_xticks(range(len(KINDS)))
        ax.set_xticklabels([f"{kind_name(k)}\nAB - B (filled), BA - A (open)" for k in KINDS], fontsize=7)
        ax.set_ylabel(f"AUROC, {SIGNALS[signal][0]}", fontsize=8)
    contrast_rows = contrasts[(contrasts["signal"] == "mse") & (contrasts["kind"] == "model")]
    ax = axes[-1]
    for i, who in enumerate(models):
        offset = (i - (len(models) - 1) / 2) * 0.1
        for x, (positive, negative) in enumerate(CONTRASTS):
            match = contrast_rows[(contrast_rows["positive"] == positive) & (contrast_rows["negative"] == negative)
                                  & (contrast_rows["who"] == who)]
            if len(match):
                ax.errorbar(x + offset, match.iloc[0]["auroc"], yerr=spread(match),
                            fmt="o", ms=4, color=colors[who], lw=0.8, label=who if x == 0 else None)
    ax.set_xticks(range(len(CONTRASTS)))
    ax.set_xticklabels([f"{kind_name(p)} against {kind_name(n)}" for p, n in CONTRASTS], fontsize=7)
    ax.set_ylabel("AUROC, next frame MSE", fontsize=8)
    for ax in axes:
        ax.axhline(0.5, color="0.6", lw=0.8, ls=":")
        ax.set_ylim(0.0, 1.02)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, ncol=3)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return path


def plot_layers(index: pd.DataFrame, path: Path, title: str) -> Path:
    """For every model with error units, per layer and surprise kind, AB against B: the share of tuples where AB is
    higher, on the whole frame and over the bar (paired, on the same frames), and the AUROC on the whole frame, which
    also falls when the activity of the layer varies between tuples more than AB and B differ."""
    rows = index[(index["scope"] == "all") & (index["pair"] == "AB - B") & (index["kind"] == "model")
                 & index["signal"].str.match(r"e\d")]
    models = list(rows["who"].unique())
    panels = (("higher", "", "share of tuples with AB higher"), ("higher", "_bar", "the same, over the bar"),
              ("auroc", "", "AUROC"))
    fig = Figure(figsize=(3.2 * max(len(models), 1) + 1, 9.6))
    axes = fig.subplots(len(panels), max(len(models), 1), squeeze=False)
    for c, who in enumerate(models):
        for r, (measure, suffix, label) in enumerate(panels):
            ax = axes[r, c]
            own = rows[(rows["who"] == who) & rows["signal"].str.fullmatch(r"e\d" + suffix)]
            for i, kind in enumerate(KINDS):
                line = own[own["event"] == kind].copy()
                line["layer"] = line["signal"].str[1].astype(int)
                line = line.sort_values("layer")
                ax.errorbar(line["layer"], line[measure], yerr=spread(line) if measure == "auroc" else None,
                            fmt="o-", ms=3, lw=1.0, color=f"C{i}", label=kind_name(kind))
            ax.axhline(0.5, color="0.6", lw=0.8, ls=":")
            ax.set_ylim(0.0, 1.02)
            ax.set_xticks(range(own["signal"].nunique()))
            ax.set_xlabel("layer", fontsize=8)
            ax.tick_params(labelsize=7)
            if r == 0:
                ax.set_title(who, fontsize=8)
            if c == 0:
                ax.set_ylabel(label, fontsize=8)
    axes[0, -1].legend(fontsize=6)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return path


def plot_curves(curve: pd.DataFrame, path: Path, title: str) -> Path:
    """One column per surprise kind, one row per main signal: every model against f, the frames since the window
    start, the impossible member AB solid and its control B dashed, the copy of the last frame in gray."""
    colors = model_colors(curve)
    fig = Figure(figsize=(2.8 * len(KINDS) + 1, 2.6 * len(MAIN) + 0.6))
    axes = fig.subplots(len(MAIN), len(KINDS), squeeze=False)
    for c, kind in enumerate(KINDS):
        for r, signal in enumerate(MAIN):
            ax = axes[r, c]
            own = curve[(curve["event"] == kind) & (curve["signal"] == signal)]
            for who, line in own[own["role"] == "AB"].groupby("who", sort=False):
                kind_of = line["kind"].iloc[0]
                if kind_of == "model" or who == "copy":
                    draw_line(ax, line, "f", who, kind_of, colors.get(who, "0.5"))
            for who, line in own[(own["role"] == "B") & (own["kind"] == "model")].groupby("who", sort=False):
                line = line.sort_values("f")
                ax.plot(line["f"], line["mean"], "--", lw=1.0, color=colors.get(who, "0.5"))
            ax.axvspan(-0.5, WINDOW - 0.5, color="0.92", zorder=0)
            ax.set_yscale("log")
            if r == 0:
                ax.set_title(f"{kind_name(kind)}: AB, B dashed", fontsize=8)
            if c == 0:
                ax.set_ylabel(SIGNALS[signal][0], fontsize=8)
            if r == len(MAIN) - 1:
                ax.set_xlabel("frames since the window start", fontsize=8)
            ax.tick_params(labelsize=7)
    axes[0, -1].legend(fontsize=6)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return path
