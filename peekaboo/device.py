"""Single place where the compute device is chosen, plus small device agnostic helpers."""

import numpy as np
import torch

_VALID_TYPES = ("cuda", "mps", "cpu")


def get_device(preference: str = "auto") -> torch.device:
    """Return CUDA if available, else MPS, else CPU. A specific device can be requested instead of "auto"."""
    pref = preference.lower()
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(pref)
    if device.type not in _VALID_TYPES:
        raise ValueError(f"Unsupported device '{preference}', expected auto, cuda, mps, or cpu.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return device


def describe_device(device: torch.device) -> str:
    """Return a short human readable name for the device."""
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    if device.type == "mps":
        return "mps (Apple GPU)"
    return "cpu"


def synchronize(device: torch.device) -> None:
    """Wait for all queued work on the device, so wall clock timings are meaningful."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def to_tensor(array: np.ndarray | torch.Tensor, device: torch.device,
              dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert an array to a tensor on the device, casting on the CPU first so float64 never reaches MPS."""
    tensor = torch.as_tensor(array)
    if tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    return tensor.to(device)
