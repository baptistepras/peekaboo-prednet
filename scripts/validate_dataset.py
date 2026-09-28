"""Validate a stored dataset (--data) or a sample of the on the fly stream (--stream), and print a check report."""

import argparse
import json
import sys
from pathlib import Path

from peekaboo.config import load_config
from peekaboo.data.conditions import GeneratorSettings
from peekaboo.data.dataset import OnTheFlyDataset
from peekaboo.data.mnist_pool import build_digit_pool
from peekaboo.data.render import RenderSettings
from peekaboo.data.splicing import SURPRISES, sample_surprise_tuple, surprise_speeds
from peekaboo.data.store import StoredDataset
from peekaboo.data.validate import CHECKS, ValidationReport, Validator, training_digits, validate_stored
from peekaboo.paths import CONFIGS_DIR


def print_report(report: ValidationReport) -> None:
    """Print one line per check, then the dataset statistics."""
    summary = report.to_dict()
    print(f"{'check':>22} {'checked':>8} {'failed':>7}  rule")
    for name in CHECKS:
        entry = summary["checks"].get(name)
        if entry is None:
            print(f"{name:>22} {'-':>8} {'-':>7}  not applicable here")
            continue
        flag = "" if entry["failed"] == 0 else f"   first failures at {entry['first_failures'][:5]}"
        print(f"{name:>22} {entry['checked']:>8} {entry['failed']:>7}  {entry['rule']}{flag}")
    print("\nsequences per condition: " + ", ".join(f"{c} {n}" for c, n in summary["conditions"].items()))
    print("occluded frame share: " + ", ".join(f"{c} {s:.0%}" for c, s in summary["occluded_frame_share"].items()))
    others = summary["sequences_with_other_occlusions"]
    if others:
        print("sequences with another full occlusion outside the window: "
              + ", ".join(f"{c} {n}/{summary['conditions'][c]}" for c, n in others.items()))


def main() -> int:
    """Run the validation and return a nonzero exit code if any check fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="folder of a stored dataset")
    source.add_argument("--stream", action="store_true", help="validate sequences generated on the fly")
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "data" / "base.yaml",
                        help="generator config (for --stream, or a stored set without one)")
    parser.add_argument("--split", default="val", help="digit split of the stream")
    parser.add_argument("--n", type=int, default=300, help="stream sequences from the training mix")
    parser.add_argument("--tuples", type=int, default=5, help="stream tuples per surprise type")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.data is not None:
        dataset = StoredDataset(args.data)
        config = dataset.info.get("generator_config") or load_config(args.config)
        settings = GeneratorSettings.from_config(config)
        split = dataset.info.get("split")
        digits = config["digits"]
        forbidden = training_digits(digits["val_size"], digits["holdout_seed"]) if split == "val" else None
        print(f"validating {len(dataset)} sequences of {args.data} (split {split})")
        report = validate_stored(dataset, settings, forbidden)
        (args.data / "validation.json").write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        print(f"report saved to {args.data / 'validation.json'}\n")
    else:
        config = load_config(args.config)
        settings = GeneratorSettings.from_config(config)
        render_settings = RenderSettings.from_config(config)
        digits = config["digits"]
        pool = build_digit_pool(args.split, digits["box_size"], digits["ink_threshold"], digits["val_size"],
                                digits["holdout_seed"], download=False)
        forbidden = training_digits(digits["val_size"], digits["holdout_seed"]) if args.split == "val" else None
        validator = Validator(settings, render_settings, forbidden)
        stream = OnTheFlyDataset(pool, settings, render_settings, "validate", args.seed)
        position = 0
        for i in range(args.n):
            spec = stream.spec(i)
            validator.report.record("determinism", position, stream.spec(i) == spec)
            validator.add(position, spec, pool.sprite(spec.digit_index))
            position += 1
        for kind in SURPRISES:
            allowed = surprise_speeds(kind, settings)
            speed = 3 if 3 in allowed else allowed[0]
            for i in range(args.tuples):
                for spec in sample_surprise_tuple(pool, settings, "validate", i, args.seed, kind, 6, speed).specs():
                    validator.add(position, spec, pool.sprite(spec.digit_index))
                    position += 1
        print(f"validating {position} stream sequences (split {args.split})\n")
        report = validator.finish()

    print_report(report)
    print("\nall checks passed" if report.ok else "\nSOME CHECKS FAILED")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
