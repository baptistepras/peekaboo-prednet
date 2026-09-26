"""Tests for the shared device helper."""

import numpy as np
import pytest
import torch

from peekaboo.device import describe_device, get_device, synchronize, to_tensor


def test_auto_follows_priority() -> None:
    """Auto selection picks CUDA first, then MPS, then CPU."""
    device = get_device("auto")
    if torch.cuda.is_available():
        assert device.type == "cuda"
    elif torch.backends.mps.is_available():
        assert device.type == "mps"
    else:
        assert device.type == "cpu"


def test_cpu_always_available() -> None:
    """An explicit CPU request always works."""
    assert get_device("cpu") == torch.device("cpu")
    assert describe_device(torch.device("cpu")) == "cpu"


def test_invalid_device_raises() -> None:
    """Unknown device types are rejected."""
    with pytest.raises((ValueError, RuntimeError)):
        get_device("xla")


def test_unavailable_backend_raises() -> None:
    """Requesting a backend that is not present raises instead of silently falling back."""
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            get_device("cuda")
    if not torch.backends.mps.is_available():
        with pytest.raises(RuntimeError):
            get_device("mps")


def test_to_tensor_casts_float64() -> None:
    """Float64 numpy arrays arrive on the device as float32."""
    device = get_device("auto")
    tensor = to_tensor(np.zeros((2, 3), dtype=np.float64), device)
    assert tensor.dtype == torch.float32
    assert tensor.device.type == device.type
    synchronize(device)
