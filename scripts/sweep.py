"""State of the training sweep (step 2.10): what is done, what is running, the checks, and the next command.

The sweep trains every model with the same settings and seed (configs/train/sweep_*.yaml), one run at a time, then
evaluates each one: the gate D16, the position probes, the amodal decoder, and the prediction figures. This script only
reads the run folders; it launches nothing. Run it between two commands, or while a run is training.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from peekaboo.paths import RUNS_DIR
from peekaboo.train.sweep import (HIDDEN, STAGES, SWEEP, RunStatus, check_commands, find_run, next_action, relative,
                                  run_status, stage_commands)


def summary_row(status: RunStatus) -> dict[str, str]:
    """One line of the table: progress, the training checks, and the main result of every stage."""
    validation = status.last_validation
    row = {"run": f"{status.run.key}/seed{status.run.seed}", "state": status.state,
           "step": f"{status.step}/{status.steps}",
           "val MSE vs blank": "", "drop check": "", "gate": "", "probe y err": "", "decoder corr": "",
           "stages done": " ".join(s for s in STAGES if status.done[s]) or "-",
           "hours left": f"{status.hours_left():.1f}"}
    if validation and "blank_mse" in validation:
        row["val MSE vs blank"] = relative(validation["val_mse"], validation["blank_mse"])
    if status.plateau is not None:
        row["drop check"] = f"{-100 * status.plateau:+.0f}% in 2k"  # change of the MSE: negative when it fell
    if status.gate:
        row["gate"] = f"{'PASSED' if status.gate['passed'] else 'FAILED'} {100 * status.gate['mse_reduction']:.0f}%"
    if status.probe:
        row["probe y err"] = f"{status.probe['probe']:.2f} (random {status.probe['random_probe']:.1f})"
    if status.decoder:
        row["decoder corr"] = f"{status.decoder['decoder']:.2f} (random {status.decoder['random_decoder']:.2f})"
    return row


def main() -> int:
    """Print the state of every run, the warnings, and the next command; or the commands of one run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=RUNS_DIR, help="folder of the runs")
    parser.add_argument("--commands", metavar="RUN", default=None,
                        help="print every command of this run (a key such as sweep_prednet5l), in order")
    parser.add_argument("--check", action="store_true",
                        help="print the commands that run the whole chain of every model on a few steps, in "
                             "runs/check/, before the long trainings")
    args = parser.parse_args()

    if args.check:
        for run in SWEEP:
            print(f"# {run.label}\n{check_commands(run)}\n")
        return 0
    if args.commands:
        run = find_run(args.commands)
        for stage, command in stage_commands(run).items():
            print(f"# {stage}\n{command}\n")
        return 0

    statuses = [run_status(run, args.runs) for run in SWEEP]
    table = pd.DataFrame([summary_row(s) for s in statuses])
    print(f"training sweep (probe y error in px and decoder correlation on hidden frames, "
          f"{HIDDEN}; drop check: fall of the validation MSE over the 2000 steps before the learning rate drop)\n")
    print(table.to_string(index=False))
    left = sum(s.hours_left() for s in statuses)
    print(f"\nabout {left:.0f} h of computation left")
    warnings = [(s.run.key, w) for s in statuses for w in s.warnings]
    if warnings:
        print("\nwarnings:")
        for key, warning in warnings:
            print(f"  {key}: {warning}")
    sentence, command = next_action(statuses)
    print(f"\nnext: {sentence}")
    if command:
        print(command)
    return 0


if __name__ == "__main__":
    sys.exit(main())
