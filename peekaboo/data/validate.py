"""Dataset validation: every sequence is re-rendered and checked against the rules the generator promises.

The checks follow section 6.6 of the plan (center against amodal centroid, exactly k hidden frames, surprises only
while hidden, occluded frame shares, determinism) plus identical splice frames, clean motion around the event,
held out digits, and the integrity of stored sets. A Validator takes sequences one by one, so any dataset size fits
in memory, and surprise tuples are checked as soon as their four sequences have been seen.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from peekaboo.data.conditions import GeneratorSettings
from peekaboo.data.dataset import TRUTH_KEYS, truth_arrays
from peekaboo.data.mnist_pool import split_indices
from peekaboo.data.occluder import FULLY_VISIBLE, STATE_OCCLUDED, column_ink, visible_fraction
from peekaboo.data.render import RenderSettings, bar_columns, measure_from_pixels, render_sequence
from peekaboo.data.spec import SequenceSpec
from peekaboo.data.store import StoredDataset, frames_checksum
from peekaboo.data.truth import EPISODE_MAIN, STATE_ABSENT, runs, sequence_summary

MNIST_TRAIN_SIZE = 60_000  # images in the MNIST train file, split into train and val

CHECKS = {
    "render_matches_truth": "visible fraction, true center, and visible ink center measured on pixels equal the truth",
    "colors": "red digit only, bar at its gray level, blackout frames all zero",
    "window_clean": "fully visible context and post frames, no other contact with the bar in the analysis window",
    "exact_k": "the main event has exactly k consecutive occluded frames, from the onset frame",
    "clean_motion": "no wall bounce near entry and exit, and exactly one hidden bounce for hidden_bounce",
    "splice_hidden": "surprises change the trajectory strictly while the digit is hidden",
    "splice_identical": "AB and BA are pixel exact splices of A and B",
    "holdout": "no digit of the training split",
    "stored_truth": "stored ground truth equals the truth recomputed from the spec",
    "checksums": "frames render exactly as when the dataset was written",
    "determinism": "the same seed and index give the same sequence",
}
TOLERANCE_PX = 1e-4  # stored truth is float32


def _max_error(a: np.ndarray, b: np.ndarray) -> float:
    """Largest absolute difference, with matching NaNs equal and mismatching NaNs infinite."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return float("inf")
    mask = ~np.isnan(a)
    return float(np.abs(a[mask] - b[mask]).max()) if mask.any() else 0.0


def _sign_flips(values: np.ndarray, first: int, last: int, skip: int) -> list[int]:
    """Frames t in (first, last] where a nonzero value changes sign from t - 1, ignoring frame `skip`."""
    first, last = max(first, 0), min(last, len(values) - 1)
    flips = []
    for t in range(first + 1, last + 1):
        if t != skip and values[t] * values[t - 1] < 0:
            flips.append(t)
    return flips


@dataclass
class ValidationReport:
    """Per check counts of sequences checked and failed, plus dataset statistics."""

    checked: Counter = field(default_factory=Counter)
    failed: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    conditions: Counter = field(default_factory=Counter)
    cells: Counter = field(default_factory=Counter)
    occluded_share: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    other_occlusions: dict[str, int] = field(default_factory=Counter)

    def record(self, check: str, position: int, passed: bool) -> None:
        """Count one check of one sequence, keeping the positions of failures."""
        self.checked[check] += 1
        if not passed:
            self.failed[check].append(position)

    @property
    def ok(self) -> bool:
        """True if every check passed on every sequence."""
        return not any(self.failed.values())

    def to_dict(self) -> dict[str, Any]:
        """A JSON friendly summary, with at most 20 failing positions per check."""
        return {
            "ok": self.ok,
            "checks": {name: {"checked": self.checked[name], "failed": len(self.failed.get(name, [])),
                              "first_failures": self.failed.get(name, [])[:20], "rule": CHECKS[name]}
                       for name in CHECKS if self.checked[name]},
            "conditions": dict(self.conditions),
            "cells": {f"{c} k={k} v={v}": n for (c, k, v), n in sorted(self.cells.items())},
            "occluded_frame_share": {c: float(np.mean(v)) for c, v in self.occluded_share.items()},
            "sequences_with_other_occlusions": dict(self.other_occlusions),
        }


class Validator:
    """Checks sequences one at a time and accumulates a ValidationReport."""

    def __init__(self, settings: GeneratorSettings, render_settings: RenderSettings,
                 forbidden_digits: set[int] | None = None) -> None:
        """Keep the rules to check against; forbidden_digits are MNIST indices that must not appear."""
        self.settings = settings
        self.render_settings = render_settings
        self.forbidden = forbidden_digits
        self.report = ValidationReport()
        self._tuples: dict[tuple[int, int], dict[str, tuple[int, SequenceSpec, np.ndarray]]] = {}

    def add(self, position: int, spec: SequenceSpec, sprite: np.ndarray,
            stored_truth: dict[str, np.ndarray] | None = None, stored_checksums: tuple[int, int] | None = None) -> None:
        """Render one sequence and run every check that applies to it."""
        cs = self.settings.crossing
        rep = self.report
        rendered = render_sequence(spec, sprite, self.render_settings, cs.thresholds)
        truth = rendered.truth

        # pixels against the exact truth, which includes the center against the amodal centroid
        measured = measure_from_pixels(rendered)
        rep.record("render_matches_truth", position,
                   all(_max_error(measured[key], getattr(truth, key)) < 1e-9
                       for key in ("visible_fraction", "center", "modal_center")))

        # colors
        obs, under_bar = rendered.observed, bar_columns(spec)
        shown = np.ones(spec.seq_len, dtype=bool)
        shown[list(spec.blackout_frames)] = False
        rep.record("colors", position,
                   bool((obs[~shown] == 0).all() and (obs[shown][:, :, under_bar] == self.render_settings.bar_value).all()
                        and (obs[shown][:, :, ~under_bar, 1:] == 0).all()))

        # analysis window, judged on the bar alone (blackout frames may fall in the post frames)
        bar_visible = visible_fraction(column_ink(sprite), spec.positions[:, 1], spec.bar_left,
                                       spec.bar_width) >= FULLY_VISIBLE
        summary = sequence_summary(truth)
        if spec.window_start >= 0:
            clean = summary["other_frames_in_window"] == 0
            if spec.entry_frame >= 0:
                clean &= bool(bar_visible[spec.window_start:spec.entry_frame].all())
            if spec.exit_frame >= 0:
                clean &= bool(bar_visible[spec.exit_frame + 1:spec.exit_frame + cs.n_post + 1].all())
            rep.record("window_clean", position, clean)

        # exactly k occluded frames in the main event
        if spec.condition in ("occlusion", "hidden_bounce"):
            hidden = runs((truth.state == STATE_OCCLUDED) & (truth.episode == EPISODE_MAIN))
            rep.record("exact_k", position, hidden == [(spec.onset_frame, spec.onset_frame + spec.k_target - 1)])

        # clean motion around entry and exit; a surprise's scripted change at the splice is not a bounce
        if spec.entry_frame >= 0 and spec.exit_frame >= 0:
            c, skip = cs.n_clean, spec.surprise_frame
            x_flips = _sign_flips(spec.velocities[:, 1], spec.entry_frame - c, spec.exit_frame + c, skip)
            y_flips = (_sign_flips(spec.velocities[:, 0], spec.entry_frame - c, spec.entry_frame, skip)
                       + _sign_flips(spec.velocities[:, 0], spec.exit_frame, spec.exit_frame + c, skip))
            if spec.condition == "hidden_bounce":
                x_ok = len(x_flips) == 1 and all(truth.state[t - 1] == truth.state[t] == STATE_OCCLUDED
                                                 for t in x_flips)
            else:
                x_ok = not x_flips
            rep.record("clean_motion", position, x_ok and not y_flips)

        # surprises: the change happens while hidden, and the four sequences splice exactly
        if spec.tuple_role in ("AB", "BA"):
            ts = spec.surprise_frame
            not_visible = (STATE_OCCLUDED, STATE_ABSENT)
            rep.record("splice_hidden", position,
                       ts >= 1 and truth.state[ts - 1] in not_visible and truth.state[ts] in not_visible)
        if spec.tuple_role != "none":
            self._add_to_tuple(position, spec, obs)

        if self.forbidden is not None:
            rep.record("holdout", position, spec.mnist_index not in self.forbidden)
        if stored_truth is not None:
            recomputed = truth_arrays(truth)
            rep.record("stored_truth", position,
                       all(_max_error(stored_truth[key], recomputed[key]) <= TOLERANCE_PX for key in TRUTH_KEYS))
        if stored_checksums is not None:
            rep.record("checksums", position,
                       (frames_checksum(rendered.amodal), frames_checksum(obs)) == tuple(stored_checksums))

        # statistics
        rep.conditions[spec.condition] += 1
        rep.cells[(spec.condition, spec.k_target, spec.speed)] += 1
        rep.occluded_share[spec.condition].append(float(np.mean(truth.state == STATE_OCCLUDED)))
        if summary["other_occlusions"] > 0:
            rep.other_occlusions[spec.condition] += 1

    def _add_to_tuple(self, position: int, spec: SequenceSpec, observed: np.ndarray) -> None:
        """Collect the members of a surprise tuple and check the splices once all four are there."""
        key = (spec.seed, spec.tuple_id)
        members = self._tuples.setdefault(key, {})
        members[spec.tuple_role] = (position, spec, observed)
        if len(members) < 4:
            return
        ts = spec.surprise_frame
        a, b, ab, ba = (members[role][2] for role in ("A", "B", "AB", "BA"))
        identical = (np.array_equal(ab[:ts], a[:ts]) and np.array_equal(ab[ts:], b[ts:])
                     and np.array_equal(ba[:ts], b[:ts]) and np.array_equal(ba[ts:], a[ts:]))
        self.report.record("splice_identical", members["AB"][0], identical)
        del self._tuples[key]

    def finish(self) -> ValidationReport:
        """Close the report: tuples still missing members count as failed splices."""
        for members in self._tuples.values():
            position = next(iter(members.values()))[0]
            self.report.record("splice_identical", position, False)
        self._tuples.clear()
        return self.report


def format_report(report: ValidationReport) -> str:
    """Return the report as text: one line per check, then the dataset statistics."""
    summary = report.to_dict()
    lines = [f"{'check':>22} {'checked':>8} {'failed':>7}  rule"]
    for name in CHECKS:
        entry = summary["checks"].get(name)
        if entry is None:
            lines.append(f"{name:>22} {'-':>8} {'-':>7}  not applicable here")
            continue
        flag = "" if entry["failed"] == 0 else f"   first failures at {entry['first_failures'][:5]}"
        lines.append(f"{name:>22} {entry['checked']:>8} {entry['failed']:>7}  {entry['rule']}{flag}")
    lines.append("")
    lines.append("sequences per condition: " + ", ".join(f"{c} {n}" for c, n in summary["conditions"].items()))
    lines.append("occluded frame share: "
                 + ", ".join(f"{c} {s:.0%}" for c, s in summary["occluded_frame_share"].items()))
    others = summary["sequences_with_other_occlusions"]
    if others:
        lines.append("sequences with another full occlusion outside the window: "
                     + ", ".join(f"{c} {n}/{summary['conditions'][c]}" for c, n in others.items()))
    lines.append("")
    lines.append("all checks passed" if report.ok else "SOME CHECKS FAILED")
    return "\n".join(lines)


def training_digits(val_size: int, holdout_seed: int) -> set[int]:
    """MNIST train file indices of the training split, which validation sequences must never use."""
    return set(split_indices("train", MNIST_TRAIN_SIZE, val_size, holdout_seed).tolist())


def validate_stored(dataset: StoredDataset, settings: GeneratorSettings,
                    forbidden_digits: set[int] | None = None) -> ValidationReport:
    """Validate every sequence of a stored dataset, including its stored truth and checksums."""
    validator = Validator(settings, dataset.render_settings, forbidden_digits)
    for i, spec in enumerate(dataset.specs):
        row = dataset.metadata.iloc[i]
        validator.add(i, spec, dataset.sprites[spec.digit_index],
                      stored_truth={key: values[i] for key, values in dataset.truth.items()},
                      stored_checksums=(int(row["amodal_crc32"]), int(row["observed_crc32"])))
    return validator.finish()
