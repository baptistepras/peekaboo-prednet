"""Check the project environment: packages, selected device, and a forward and backward pass of the model ops."""

import argparse
import importlib
import os
import platform
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from peekaboo.device import describe_device, get_device, synchronize

# import name -> reason it is needed
REQUIRED_PACKAGES = {
    "torch": "models",
    "torchvision": "MNIST download",
    "numpy": "generator",
    "scipy": "statistics",
    "pandas": "metadata tables",
    "pyarrow": "parquet files",
    "yaml": "configs",
    "cv2": "connected components, resize",
    "PIL": "GIF export",
    "matplotlib": "figures",
    "sklearn": "probes, AUROC",
    "tqdm": "progress bars",
    "pytest": "tests",
}


def check_packages() -> bool:
    """Print the version of each required package and return True if all of them import."""
    all_ok = True
    for name, reason in REQUIRED_PACKAGES.items():
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "?")
            print(f"  ok       {name:<12} {version:<12} ({reason})")
        except ImportError:
            all_ok = False
            print(f"  MISSING  {name:<12} {'':<12} ({reason})")
    return all_ok


def run_op_check(device: torch.device) -> float:
    """Run the ops the models need forward and backward: conv, hard sigmoid, pooling, upsampling, and split errors for
    PredNet; space to depth, sigmoid, and depth to space for the ConvLSTM."""
    x = torch.randn(2, 41, 64, 64, device=device, requires_grad=True)
    conv = nn.Conv2d(41, 12, 3, padding=1).to(device)
    y = conv(x)
    gate = torch.clamp(0.2 * y + 0.5, 0.0, 1.0)  # Keras 2 hard sigmoid
    h = gate * torch.tanh(y)
    up = F.interpolate(F.max_pool2d(h, 2), scale_factor=2, mode="nearest")
    error = torch.cat([F.relu(up - h), F.relu(h - up)], dim=1)
    patches = torch.sigmoid(F.pixel_unshuffle(h, 4))  # ConvLSTM gates on a grid of 4 x 4 patches
    frame = F.pixel_shuffle(patches, 4)
    loss = error.mean() + torch.clamp(F.relu(y), max=1.0).mean() + frame.mean()  # SatLU at pixel_max = 1
    loss.backward()
    synchronize(device)
    return float(loss.item())


def main() -> int:
    """Print the environment report and return a nonzero exit code if anything fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    args = parser.parse_args()

    print(f"python   {sys.version.split()[0]} on {platform.platform()}")
    print(f"torch    {torch.__version__}  cuda available: {torch.cuda.is_available()}  "
          f"mps built: {torch.backends.mps.is_built()}  mps available: {torch.backends.mps.is_available()}")
    print(f"PYTORCH_ENABLE_MPS_FALLBACK = {os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '(unset)')}")

    print("packages")
    packages_ok = check_packages()

    device = get_device(args.device)
    print(f"device   {describe_device(device)}")
    try:
        loss = run_op_check(device)
        print(f"op check ok (loss {loss:.4f})")
        ops_ok = True
    except Exception as error:  # report any backend failure instead of a traceback
        print(f"op check FAILED: {type(error).__name__}: {error}")
        ops_ok = False

    return 0 if packages_ok and ops_ok else 1


if __name__ == "__main__":
    sys.exit(main())
