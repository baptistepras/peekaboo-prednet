"""Render sequences of every condition and two surprises, check pixels against the exact ground truth, save a preview."""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only, no window
import matplotlib.pyplot as plt
import numpy as np

from peekaboo.config import load_config
from peekaboo.data.conditions import CONDITIONS, GeneratorSettings, sample_spec
from peekaboo.data.mnist_pool import build_digit_pool
from peekaboo.data.occluder import STATE_OCCLUDED
from peekaboo.data.render import RenderedSequence, RenderSettings, measure_from_pixels, render_sequence
from peekaboo.data.spec import SequenceSpec
from peekaboo.data.splicing import SURPRISES, sample_surprise_tuple
from peekaboo.data.truth import STATE_BLACKOUT, sequence_summary
from peekaboo.paths import CONFIGS_DIR, FIGURES_DIR

GAP = 2  # white pixels between frames in the strips


def strip(frames: np.ndarray) -> np.ndarray:
    """Place RGB frames side by side, separated by white gaps."""
    t, h, w, _ = frames.shape
    out = np.full((h, t * (w + GAP) - GAP, 3), 255, dtype=np.uint8)
    for i, frame in enumerate(frames):
        out[:, i * (w + GAP):i * (w + GAP) + w] = frame
    return out


def max_error(a: np.ndarray, b: np.ndarray) -> float:
    """Largest absolute difference, treating matching NaNs as equal and mismatching NaNs as infinite."""
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return float("inf")
    mask = ~np.isnan(a)
    return float(np.abs(a[mask] - b[mask]).max()) if mask.any() else 0.0


def main() -> int:
    """Check many rendered sequences against their ground truth, then draw a few of them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "data" / "base.yaml")
    parser.add_argument("--split", default="val")
    parser.add_argument("--n", type=int, default=300, help="training mix sequences for the consistency check")
    parser.add_argument("--step", type=int, default=2, help="show one frame out of this many")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES_DIR / "render_preview.png")
    args = parser.parse_args()

    config = load_config(args.config)
    settings = GeneratorSettings.from_config(config)
    render_settings = RenderSettings.from_config(config)
    thresholds = settings.crossing.thresholds
    digits = config["digits"]
    pool = build_digit_pool(args.split, digits["box_size"], digits["ink_threshold"], digits["val_size"],
                            digits["holdout_seed"], download=False)

    def render(spec: SequenceSpec) -> RenderedSequence:
        """Render one spec with the configured colors."""
        return render_sequence(spec, pool.sprite(spec.digit_index), render_settings, thresholds)

    # 1. consistency of pixels and exact truth, on the training mix, blackouts, and all surprise tuples
    specs = [sample_spec(pool, settings, "preview_mix", i, args.seed) for i in range(args.n)]
    specs += [sample_spec(pool, settings, "preview_blackout", i, args.seed, condition="blackout") for i in range(50)]
    for kind in SURPRISES:
        speed = 4 if kind == "speed_slow" else 3
        for i in range(10):
            specs += sample_surprise_tuple(pool, settings, "preview_tuple", i, args.seed, kind, 6, speed).specs()
    errors = {"visible_fraction": 0.0, "center": 0.0, "modal_center": 0.0}
    k_mismatch, window_intrusions, other_occ, occlusion_count = 0, 0, 0, 0
    for spec in specs:
        r = render(spec)
        measured = measure_from_pixels(r)
        for key in errors:
            errors[key] = max(errors[key], max_error(measured[key], getattr(r.truth, key)))
        summary = sequence_summary(r.truth)
        window_intrusions += summary["other_frames_in_window"] > 0
        if spec.condition in ("occlusion", "hidden_bounce") and spec.tuple_role == "none":
            k_mismatch += summary["measured_k"] != spec.k_target
        if spec.condition == "occlusion" and spec.tuple_role == "none":
            occlusion_count += 1
            other_occ += summary["other_occlusions"] > 0
    ok = max(errors.values()) < 1e-9 and k_mismatch == 0 and window_intrusions == 0
    print(f"checked {len(specs)} sequences")
    print("largest pixel vs truth difference: " + ", ".join(f"{k} {v:.1e}" for k, v in errors.items()))
    print(f"measured k different from target: {k_mismatch}")
    print(f"sequences with another contact inside the analysis window: {window_intrusions}")
    print(f"occlusion sequences with another full occlusion outside the window: {other_occ}/{occlusion_count} "
          f"(labeled as 'other' episodes in the ground truth)")

    # 2. preview: one sequence per condition and two surprises, observed and amodal strips
    examples = [(f"{c}", sample_spec(pool, settings, "preview", 0, args.seed, condition=c)) for c in CONDITIONS]
    for kind in ("direction", "vanish"):
        t = sample_surprise_tuple(pool, settings, "preview", 0, args.seed, kind, 6, 3)
        examples.append((f"surprise {kind} (impossible AB)", t.impossible_ab))
    frames = np.arange(0, settings.crossing.seq_len, args.step)
    fig, axes = plt.subplots(2 * len(examples), 1, figsize=(18, 2.1 * len(examples)), squeeze=False)
    for row, (name, spec) in enumerate(examples):
        r = render(spec)
        w = spec.frame_width + GAP
        amodal_rgb = np.repeat(r.amodal[frames][..., None], 3, axis=3)
        for sub, (image, label) in enumerate(((strip(r.observed[frames]), "observed"),
                                              (strip(amodal_rgb), "amodal"))):
            ax = axes[2 * row + sub, 0]
            ax.imshow(image, interpolation="nearest")
            ax.set_yticks([])
            ax.set_xticks([(i + 0.5) * w for i in range(len(frames))], [str(t) for t in frames], fontsize=6)
            ax.set_ylabel(label, fontsize=7)
            if sub == 0:
                ax.set_title(f"{name}: k={spec.k_target}, v={spec.speed}, bar {spec.bar_width} px, "
                             f"frames {frames[0]} to {frames[-1]} (cross = true center, cyan when hidden)",
                             fontsize=8, loc="left")
                # the true center, white when some ink is visible, cyan when hidden or blacked out
                for i, t in enumerate(frames):
                    cy, cx = r.truth.center[t]
                    if np.isnan(cy):
                        continue
                    hidden = r.truth.state[t] in (STATE_OCCLUDED, STATE_BLACKOUT)
                    ax.plot(i * w + cx, cy, "+", color="cyan" if hidden else "white", ms=5, mew=1)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"saved {args.out}")
    print("all checks passed" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
