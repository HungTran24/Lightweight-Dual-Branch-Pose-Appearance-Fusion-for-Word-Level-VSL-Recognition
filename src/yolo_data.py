"""Compatibility shim for src.yolo."""

from __future__ import annotations

from src.yolo.materialize import materialize
from src.yolo.selection import _identity, canonical_selection
from src.yolo.transforms import (
    _clip_box,
    _hand_box,
    _labels,
    _letterbox,
    _open_arrays,
    _visible,
)

__all__ = [
    "_clip_box",
    "_hand_box",
    "_identity",
    "_labels",
    "_letterbox",
    "_open_arrays",
    "_visible",
    "canonical_selection",
    "materialize",
]
