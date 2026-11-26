"""Every result of Phase 3 from the saved runs and passes (step 3.8): no figure or number is made by hand. With
--models blackout, the same results for the models of step 4.1 (PredNet 5 layers and the same model trained with
blackouts), in runs/analysis/<set>_blackout/ in place of runs/analysis/<set>/.

In order, each step a script of its own, run here one after the other:
1. the comparison of the training sweep (scripts/compare_sweep.py, runs/sweep_summary/), for the sweep only;
2. the five analyses on the set: tracking through occlusion, correction, surprise, inside the state, and blackout and
   closed loop (scripts/analyze_*.py, runs/analysis/<set>/<analysis>/);
3. examples on the set: for every model of the set, its predictions with the probe and the decoder on the first
   occlusion, hidden bounce, and blackout sequence with k = 8, and for the sweep, the error maps of every layer of
   the reference model on an occlusion and on the first `direction` tuple with k = 8 (runs/analysis/<set>/examples/);
4. the summary (runs/analysis/<set>/summary/): headline.csv, the main number of every research question for every
   model, read from the tables of step 2, and seeds.csv, every seed of each model of the set trained with several
   seeds (peekaboo/eval/results.py).

Only steps 3 and 4 run models: step 3 on a few sequences, step 4 not at all (it reads the passes). Every analysis
choice was fixed on dev_v1; this script is meant for test_v1, whose outputs are tracked.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from peekaboo.eval.results import headline, seed_table
from peekaboo.eval.tables import analysis_folder
from peekaboo.paths import RUNS_DIR
from peekaboo.train.sweep import MODEL_SETS, REFERENCE, RUNS, find_run

ANALYSES = ("occlusion", "correction", "surprise", "layers", "closed_loop")
EXAMPLES = ("occlusion", "hidden_bounce", "blackout")  # the conditions drawn for every model
EXAMPLE_K = 8
STEPS = ("sweep", "analyses", "examples", "summary")


def commands(data: str, runs_dir: Path, models: str = "sweep") -> dict[str, list[list[str]]]:
    """The commands of steps 1 to 3, as argument lists for `python -m`."""
    folder = analysis_folder(runs_dir, data, models) / "examples"
    examples = []
    for run in MODEL_SETS[models]:
        run_dir = str(run.run_dir(runs_dir))
        for condition in EXAMPLES:
            examples.append(["scripts.show_predictions", "--run", run_dir, "--data", data, "--condition", condition,
                             "--k", str(EXAMPLE_K), "--n", "1", "--probe", "--decoder",
                             "--out", str(folder / run.key)])
    sweep = models == "sweep"
    if sweep:
        reference = str(find_run(REFERENCE).run_dir(runs_dir))
        errors = ["scripts.show_errors", "--run", reference, "--data", data, "--k", str(EXAMPLE_K),
                  "--out", str(folder / "errors")]
        examples += [errors + ["--condition", "occlusion", "--n", "1"], errors + ["--surprise", "direction"]]
    return {"sweep": [["scripts.compare_sweep", "--runs", str(runs_dir)]] if sweep else [],
            "analyses": [[f"scripts.analyze_{name}", "--data", data, "--runs", str(runs_dir), "--models", models]
                         for name in ANALYSES],
            "examples": examples}


def seeded(models: str) -> list[str]:
    """The keys of the models of the set trained with several seeds, whose seeds are listed in seeds.csv."""
    return [run.key for run in MODEL_SETS[models] if sum(other.key == run.key for other in RUNS) > 1]


def summary(data: str, runs_dir: Path, models: str = "sweep") -> None:
    """Write and print the headline table and the seed table."""
    folder = analysis_folder(runs_dir, data, models)
    out = folder / "summary"
    out.mkdir(parents=True, exist_ok=True)
    main = headline(folder, models)
    main.to_csv(out / "headline.csv", index=False)
    seeds = pd.concat([seed_table(runs_dir, data, key) for key in seeded(models)], ignore_index=True)
    seeds.to_csv(out / "seeds.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.max_colwidth", 60):
        print(f"\nheadline, {data} (95% bootstrap intervals; seeds as the outer level)")
        print(main.set_index("model").T.to_string())
        if len(seeds):
            print(f"\nevery seed of {', '.join(seeds['model'].unique())}, {data}")
            print(seeds.set_index(["model", "seed"]).T.to_string(
                float_format=lambda v: f"{v:.5f}" if abs(v) < 0.01 else f"{v:.3f}"))
    print(f"\nsaved in {out}: headline.csv, seeds.csv")


def main() -> int:
    """Run the chosen steps in order, stopping at the first that fails."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="test_v1", help="the set of the analyses and the examples")
    parser.add_argument("--runs", type=Path, default=RUNS_DIR, help="folder of the runs")
    parser.add_argument("--models", choices=MODEL_SETS, default="sweep",
                        help="the model set: the sweep (Phase 3), or blackout (step 4.1, in "
                             "runs/analysis/<set>_blackout/)")
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS), help="the steps to run")
    parser.add_argument("--dry-run", action="store_true", help="print the commands of steps 1 to 3 and stop")
    args = parser.parse_args()

    plan = commands(args.data, args.runs, args.models)
    if args.dry_run:
        for step in STEPS[:3]:
            for command in plan[step]:
                print(f"python -m {' '.join(command)}")
        return 0
    env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "0"}
    started = time.perf_counter()
    for step in args.steps:
        if step == "summary":
            summary(args.data, args.runs, args.models)
            continue
        for command in plan[step]:
            print(f"\n$ python -m {' '.join(command)}", flush=True)
            if subprocess.run([sys.executable, "-m", *command], env=env).returncode != 0:
                print(f"stopped: python -m {' '.join(command)} failed")
                return 1
    print(f"\ndone ({(time.perf_counter() - started) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
