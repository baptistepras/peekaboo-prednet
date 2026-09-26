"""Standard project locations, resolved from the package location so scripts work from any directory."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MNIST_DIR = DATA_DIR / "mnist"
DATASETS_DIR = DATA_DIR / "datasets"
RUNS_DIR = PROJECT_ROOT / "runs"
FIGURES_DIR = PROJECT_ROOT / "figures"
CONFIGS_DIR = PROJECT_ROOT / "configs"
