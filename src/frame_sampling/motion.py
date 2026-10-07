from __future__ import annotations

import numpy as np


def motion_indices(keypoints: np.ndarray, num_frames: int = 24) -> np.ndarray:
    """Select frame indices with peak motion per temporal bin on raw keypoints [T, K, C]."""
    frame_count = len(keypoints)
    if frame_count <= 0:
        return np.full(num_frames, -1, dtype=np.int32)
    if frame_count <= num_frames:
        indices = np.full(num_frames, -1, dtype=np.int32)
        indices[:frame_count] = np.arange(frame_count, dtype=np.int32)
        return indices

    motion = np.zeros(frame_count, dtype=np.float32)
    if frame_count > 1:
        diffs = np.abs(keypoints[1:, :, :2] - keypoints[:-1, :, :2]).mean(axis=(1, 2))
        motion[1:] = diffs

    bin_edges = np.linspace(0, frame_count, num_frames + 1)
    selected: list[int] = []
    for i in range(num_frames):
        s = int(np.floor(bin_edges[i]))
        e = int(np.floor(bin_edges[i + 1]))
        if e <= s:
            e = min(s + 1, frame_count)
        b_motion = motion[s:e]
        best = s + int(np.argmax(b_motion)) if len(b_motion) > 0 else s
        selected.append(best)
    return np.array(selected, dtype=np.int32)
