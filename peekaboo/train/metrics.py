"""CSV logs of a training run: one row per training step, one row per validation."""

import csv
from pathlib import Path

TRAIN_FIELDS = ("step", "loss", "grad_norm", "lr", "seconds")
VAL_FIELDS = ("step", "val_l1", "val_mse", "copy_l1", "copy_mse", "best_step")


class CsvLog:
    """An append only CSV file with a fixed header.

    A new log starts empty. A resumed log keeps only the rows up to the resumed step, since rows written after the
    last checkpoint belong to steps that will be trained again.
    """

    def __init__(self, path: str | Path, fields: tuple[str, ...], keep_until: int | None = None) -> None:
        """Create the file with its header, keeping the rows up to step `keep_until` when resuming."""
        self.path = Path(path)
        self.fields = fields
        rows = []
        if keep_until is not None and self.path.exists():
            rows = [row for row in read_csv(self.path) if int(row["step"]) <= keep_until]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields)
            writer.writeheader()
            writer.writerows(rows)

    def write(self, **row: float | int) -> None:
        """Append one row; every field of the header must be given."""
        with open(self.path, "a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self.fields).writerow(row)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    """Read a CSV log as a list of rows, values as strings."""
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
