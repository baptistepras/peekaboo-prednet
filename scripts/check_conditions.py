"""Generate sequences of every condition and every surprise tuple on real digits, report statistics, and draw them."""

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only, no window
import matplotlib.pyplot as plt
import numpy as np

from peekaboo.config import load_config
from peekaboo.data.conditions import CONDITIONS, GeneratorSettings, sample_spec
from peekaboo.data.mnist_pool import DigitPool, build_digit_pool
from peekaboo.data.occluder import column_ink, visible_fraction
from peekaboo.data.spec import SequenceSpec
from peekaboo.data.splicing import SURPRISES, sample_surprise_tuple
from peekaboo.paths import CONFIGS_DIR, FIGURES_DIR
from peekaboo.viz.space_time import draw_spec

TUPLE_CELLS = [(4, 2), (4, 4), (8, 2), (8, 4)]  # (k, speed) cells tried for each surprise


def bar_fractions(spec: SequenceSpec, pool: DigitPool) -> np.ndarray:
    """Visible ink fraction per frame due to the bar alone (ignores vanish and blackout)."""
    return visible_fraction(column_ink(pool.sprite(spec.digit_index)), spec.positions[:, 1], spec.bar_left,
                            spec.bar_width)


def occluded_outside_window(spec: SequenceSpec, fractions: np.ndarray, occluded_max: float) -> bool:
    """True if the digit is fully hidden by the bar somewhere outside the analysis window."""
    hidden = fractions <= occluded_max
    outside = np.ones(len(hidden), dtype=bool)
    outside[spec.window_start:spec.window_end + 1] = False
    return bool((hidden & outside).any())


def main() -> int:
    """Run the three reports (conditions, training mix, surprise tuples) and save the two figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "data" / "base.yaml")
    parser.add_argument("--split", default="val")
    parser.add_argument("--n", type=int, default=100, help="sequences per condition")
    parser.add_argument("--n-mix", type=int, default=1000, help="sequences drawn from the training mix")
    parser.add_argument("--n-tuples", type=int, default=20, help="tuples per surprise and cell")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()

    config = load_config(args.config)
    settings = GeneratorSettings.from_config(config)
    occ_max = settings.crossing.thresholds.occluded_max
    digits = config["digits"]
    pool = build_digit_pool(args.split, digits["box_size"], digits["ink_threshold"], digits["val_size"],
                            digits["holdout_seed"], download=False)

    # 1. each condition on its own
    print("conditions")
    print(f"{'condition':>14} {'built':>6} {'k values':>16} {'bar width':>12} {'hidden frames':>14} {'other hidden':>13}")
    examples: dict[str, list[SequenceSpec]] = {}
    for condition in CONDITIONS:
        specs, failures = [], 0
        for i in range(args.n):
            try:
                specs.append(sample_spec(pool, settings, f"check_{condition}", i, args.seed, condition=condition))
            except RuntimeError:
                failures += 1
        examples[condition] = specs[:2]
        fractions = [bar_fractions(s, pool) for s in specs]
        hidden = np.mean([np.mean(f <= occ_max) for f in fractions])
        other = np.mean([occluded_outside_window(s, f, occ_max) for s, f in zip(specs, fractions)])
        ks = sorted(Counter(s.k_target for s in specs))
        widths = [s.bar_width for s in specs]
        print(f"{condition:>14} {len(specs) / args.n:>5.0%} {str(ks):>16} {min(widths):>4}-{int(np.median(widths)):>3}-"
              f"{max(widths):<3} {hidden:>13.0%} {other:>12.0%}")

    # 2. the training mix
    mix = [sample_spec(pool, settings, "check_mix", i, args.seed) for i in range(args.n_mix)]
    shares = Counter(s.condition for s in mix)
    occlusion = [s for s in mix if s.condition == "occlusion"]
    hidden_occ = np.mean([np.mean(bar_fractions(s, pool) <= occ_max) for s in occlusion])
    hidden_all = np.mean([np.mean(bar_fractions(s, pool) <= occ_max) for s in mix])
    print("\ntraining mix: " + ", ".join(f"{c} {shares[c] / len(mix):.0%}" for c in settings.train_mix))
    print(f"fully hidden frames: {hidden_occ:.0%} of occlusion sequence frames, {hidden_all:.0%} of all frames")

    # 3. surprise tuples
    print("\nsurprise tuples")
    print(f"{'surprise':>12} {'cells built':>12} {'splice hidden':>14} {'reappear shift (frames)':>24}")
    tuple_examples = {}
    all_hidden = True
    for kind in SURPRISES:
        built, tried, shifts, hidden_ok = 0, 0, [], True
        for k, speed in TUPLE_CELLS:
            if kind == "early" and k < 4:
                continue
            for i in range(args.n_tuples):
                tried += 1
                try:
                    t = sample_surprise_tuple(pool, settings, f"check_{kind}", k * 1000 + speed * 100 + i,
                                              args.seed, kind, k, speed)
                except RuntimeError:
                    continue
                built += 1
                ab = t.impossible_ab
                f = bar_fractions(ab, pool)
                hidden_ok &= bool(f[t.splice_frame - 1] <= occ_max and f[t.splice_frame] <= occ_max)
                if ab.actual_reappear_frame >= 0:
                    shifts.append(ab.actual_reappear_frame - ab.expected_reappear_frame)
                tuple_examples.setdefault(kind, t)
        all_hidden &= hidden_ok
        shift = f"{min(shifts)} to {max(shifts)}" if shifts else "never reappears"
        print(f"{kind:>12} {built / tried:>11.0%} {'yes' if hidden_ok else 'NO':>14} {shift:>24}")

    # figures
    args.out.mkdir(parents=True, exist_ok=True)
    thresholds = settings.crossing.thresholds
    fig, axes = plt.subplots(2, len(CONDITIONS), figsize=(4.2 * len(CONDITIONS), 7.2), squeeze=False)
    for col, condition in enumerate(CONDITIONS):
        for row, spec in enumerate(examples[condition]):
            draw_spec(axes[row, col], spec, pool.sprite(spec.digit_index), thresholds,
                      f"{condition}, k={spec.k_target}, v={spec.speed}, bar {spec.bar_width} px")
    fig.suptitle("Conditions: green visible, orange partial, red hidden, black blackout, gray band = bar", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out / "conditions_examples.png", dpi=110)

    kinds = [k for k in SURPRISES if k in tuple_examples]
    fig, axes = plt.subplots(len(kinds), 4, figsize=(16.8, 3.4 * len(kinds)), squeeze=False)
    for row, kind in enumerate(kinds):
        t = tuple_examples[kind]
        for col, (spec, role) in enumerate(zip(t.specs(), ("possible A", "possible B", "impossible AB",
                                                            "impossible BA"))):
            draw_spec(axes[row, col], spec, pool.sprite(spec.digit_index), thresholds,
                      f"{kind}: {role} (k={spec.k_target}, v={spec.speed})")
    fig.suptitle("Surprise tuples: AB = start of A + end of B, BA = start of B + end of A, spliced while hidden",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out / "surprise_tuples.png", dpi=100)
    print(f"\nsaved {args.out / 'conditions_examples.png'} and {args.out / 'surprise_tuples.png'}")
    return 0 if all_hidden else 1


if __name__ == "__main__":
    sys.exit(main())
