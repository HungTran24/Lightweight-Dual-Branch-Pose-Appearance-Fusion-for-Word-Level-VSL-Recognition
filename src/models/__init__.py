from __future__ import annotations

import importlib
from typing import Any

from torch import nn


def build_model(config: dict[str, Any], *, nodes: int, num_classes: int) -> nn.Module:
    """Build a classifier from its module name under src.models."""
    module = importlib.import_module(f"{__name__}.{config['name']}")
    return module.Model(nodes=nodes, classes=num_classes, config=config)
