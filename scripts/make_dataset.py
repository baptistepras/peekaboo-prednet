"""Build a validation or test set from a set config, write it to data/datasets/<name>/, and validate it."""

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

from peekaboo.config import load_config
from peekaboo.data.build import build_specs, cells, count_sequences
from peekaboo.data.conditions import GeneratorSettings
from peekaboo.data.mnist_pool import build_digit_pool
from peekaboo.data.render import RenderSettings
from peekaboo.data.store import StoredDataset, write_dataset
from peekaboo.data.validate import format_report, training_digits, validate_stored
from peekaboo.paths import DATASETS_DIR, PROJECT_ROOT


def main() -> int:
    """Build, write, and validate one dataset; return a nonzero exit code if validation fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="set config, e.g. configs/data/test_v1.yaml")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing dataset of the same name")
    parser.add_argument("--limit-per-cell", type=int, default=None,
                        help="cap every cell at this size, for a quick trial (written as <name>_limit<N>)")
    parser.add_argument("--no-validate", action="store_true", help="skip the validation after writing")
    args = parser.parse_args()

    set_config = load_config(args.config)
    generator_path = PROJECT_ROOT / set_config["generator"]
    config = load_config(generator_path)
    settings = GeneratorSettings.from_config(config)
    render_settings = RenderSettings.from_config(config)
    digits = config["digits"]
    pool = build_digit_pool(set_config["split"], digits["box_size"], digits["ink_threshold"], digits["val_size"],
                            digits["holdout_seed"], download=False)

    name = set_config["name"] + (f"_limit{args.limit_per_cell}" if args.limit_per_cell else "")
    total = (sum(min(n, args.limit_per_cell) * (4 if family == "tuple" else 1)
                 for family, _, _, _, n in cells(set_config)) if args.limit_per_cell else count_sequences(set_config))
    print(f"building {name}: {len(list(cells(set_config)))} cells, {total} sequences, "
          f"split '{set_config['split']}', base seed {set_config['base_seed']}")

    start = time.perf_counter()
    with tqdm(total=total, unit="seq") as bar:
        specs = build_specs(set_config, pool, settings, args.limit_per_cell, progress=bar.update)
    built = time.perf_counter() - start

    start = time.perf_counter()
    folder = write_dataset(DATASETS_DIR / name, specs, pool, render_settings, settings.crossing.thresholds,
                           info={"name": name, "split": set_config["split"], "base_seed": set_config["base_seed"],
                                 "set_config": set_config, "generator_config": config,
                                 "limit_per_cell": args.limit_per_cell},
                           overwrite=args.overwrite)
    written = time.perf_counter() - start
    size = sum(f.stat().st_size for f in folder.iterdir()) / 1e6
    print(f"built in {built:.0f} s, written in {written:.0f} s: {folder} ({size:.1f} MB)")

    if args.no_validate:
        return 0
    start = time.perf_counter()
    forbidden = training_digits(digits["val_size"], digits["holdout_seed"]) if set_config["split"] == "val" else None
    report = validate_stored(StoredDataset(folder), settings, forbidden)
    (folder / "validation.json").write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(f"validated in {time.perf_counter() - start:.0f} s, report saved to {folder / 'validation.json'}\n")
    print(format_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
