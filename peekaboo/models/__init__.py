"""Video prediction models: PredNet, its ablations, and baselines."""

from typing import Any

import torch

from peekaboo.models.convlstm import ConvLSTM, ConvLSTMConfig
from peekaboo.models.prednet import PredNet, PredNetConfig


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    """Build a model from a model config, chosen by its `model` key."""
    name = config.get("model")
    if name == "prednet":
        return PredNet(PredNetConfig.from_config(config))
    if name == "convlstm":
        return ConvLSTM(ConvLSTMConfig.from_config(config))
    raise ValueError(f"Unknown model '{name}'.")
