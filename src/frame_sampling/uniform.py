from __future__ import annotations

import numpy as np


def uniform_indices(frame_count: int, num_frames: int) -> np.ndarray:
    """Return uniformly spaced frame indices, padding short sequences with -1."""
    if frame_count >= num_frames:
        return np.linspace(0, frame_count - 1, num_frames).round().astype(np.int32)

    indices = np.full(num_frames, -1, dtype=np.int32)
    indices[:frame_count] = np.arange(frame_count, dtype=np.int32)
    return indices
