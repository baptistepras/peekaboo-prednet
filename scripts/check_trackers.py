"""Check the programmed baselines on a stored set: the detector against the exact ground truth, then the error of
every tracker while the digit is hidden and when it reappears.

Prints, per visibility state, how often the digit is detected, reported at the bar, or reported absent, and the largest
centroid error of a detection (0 up to rounding: a detected digit is fully visible). Then, per condition, the mean
distance in pixels between each tracker's prediction and the true centroid on the visible frames of the analysis
window, on the hidden frames of the main event, and at the reappearance frame. Saves figures/trackers.png: the mean
error against the frames since the onset of the occlusion, per tracker, for occlusion and hidden bounce sequences.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from peekaboo.data.dataset import CONDITION_NAMES
from peekaboo.data.store import StoredDataset
from peekaboo.data.truth import EPISODE_MAIN, STATE_NAMES
from peekaboo.paths import DATASETS_DIR, FIGURES_DIR
from peekaboo.trackers.baselines import TRACKER_NAMES, run_trackers
from peekaboo.trackers.detector import STATUS_NAMES, detect

SINCE_ONSET = range(-3, 16)


def main() -> int:
    """Run the detector and the trackers on every sequence, print the reports, and save the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="val_v1", help="a set name in data/datasets/ or a folder")
    parser.add_argument("--n", type=int, default=None, help="first sequences only (default: all)")
    parser.add_argument("--out", type=Path, default=FIGURES_DIR / "trackers.png")
    args = parser.parse_args()

    directory = Path(args.data) if Path(args.data).exists() else DATASETS_DIR / args.data
    dataset = StoredDataset(directory)
    count = min(args.n or len(dataset), len(dataset))
    detector_rows, tracker_rows = [], []
    for i in range(count):
        sample = {key: value.numpy() for key, value in dataset[i].items()}
        frames, center, state = sample["frames"], sample["center"], sample["state"]
        condition = CONDITION_NAMES[int(sample["condition"])]
        found = detect(frames)
        for t in range(len(frames)):
            detector_rows.append({"state": STATE_NAMES[state[t]], "status": STATUS_NAMES[found.status[t]],
                                  "error": float(np.linalg.norm(found.center[t] - center[t])),
                                  "fraction": float(sample["visible_fraction"][t])})
        onset, back = int(sample["onset_frame"]), int(sample["actual_reappear_frame"])
        window = np.zeros(len(frames), dtype=bool)
        window[int(sample["window_start"]):int(sample["window_end"]) + 1] = True
        main_event = sample["episode"] == EPISODE_MAIN
        for name, track in run_trackers(frames, center, state).items():
            error = np.linalg.norm(track.prediction - center, axis=1)
            for t in range(len(frames)):
                if not window[t]:
                    continue
                phase = ("visible" if STATE_NAMES[state[t]] == "visible" else
                         "hidden" if STATE_NAMES[state[t]] in ("occluded", "blackout") and main_event[t] else None)
                tracker_rows.append({"tracker": name, "condition": condition, "phase": phase, "error": error[t],
                                     "since_onset": t - onset if onset >= 0 else None, "reappearance": t == back})

    detector = pd.DataFrame(detector_rows)
    table = pd.crosstab(detector["state"], detector["status"], normalize="index").reindex(columns=list(STATUS_NAMES),
                                                                                          fill_value=0.0)
    table.insert(0, "frames", detector["state"].value_counts())
    print(f"detector on {count} sequences of {directory.name}: share of frames per status")
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))
    hits = detector[detector["status"] == "detected"]
    print(f"detections: {len(hits)}, largest centroid error {hits['error'].max():.2e} px, "
          f"{int((hits['fraction'] < 1).sum())} on a partly hidden digit")

    trackers = pd.DataFrame(tracker_rows)
    report = {
        "visible": trackers[trackers["phase"] == "visible"],
        "hidden": trackers[trackers["phase"] == "hidden"],
        "reappearance": trackers[trackers["reappearance"]],
    }
    for title, rows in report.items():
        means = rows.groupby(["condition", "tracker"])["error"].mean().unstack().reindex(columns=list(TRACKER_NAMES))
        print(f"\nmean prediction error in px, {title} frames (analysis window)")
        print(means.to_string(float_format=lambda v: f"{v:.3f}"))

    fig = Figure(figsize=(10, 3.6))
    for ax, condition in zip(fig.subplots(1, 2), ("occlusion", "hidden_bounce")):
        rows = trackers[(trackers["condition"] == condition) & trackers["since_onset"].isin(list(SINCE_ONSET))]
        for name in TRACKER_NAMES:
            curve = rows[rows["tracker"] == name].groupby("since_onset")["error"].mean()
            ax.plot(curve.index, curve.values, "o-", ms=3, label=name)
        ax.axvline(0, color="0.6", lw=0.8)
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_xlabel("frames since the onset of the occlusion")
        ax.set_ylabel("mean prediction error (px)")
        ax.set_title(condition.replace("_", " "), fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"\nsaved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
