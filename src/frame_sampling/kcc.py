from __future__ import annotations

import numpy as np


def kcc_indices(keypoints: np.ndarray, num_frames: int = 24, thresh: float = 4.0) -> np.ndarray:
    """Select keyframes using reference difference thresholding downsampled/interpolated to num_frames."""
    frame_count = len(keypoints)
    if frame_count <= 0:
        return np.full(num_frames, -1, dtype=np.int32)
    if frame_count <= num_frames:
        indices = np.full(num_frames, -1, dtype=np.int32)
        indices[:frame_count] = np.arange(frame_count, dtype=np.int32)
        return indices

    keyframes = [0]
    ref = keypoints[0, :, :2]
    for t in range(1, frame_count):
        d = np.mean(np.abs(keypoints[t, :, :2] - ref))
        if d >= thresh:
            keyframes.append(t)
            ref = keypoints[t, :, :2]
    if keyframes[-1] != frame_count - 1:
        keyframes.append(frame_count - 1)

    k_count = len(keyframes)
    if k_count == num_frames:
        return np.array(keyframes, dtype=np.int32)
    elif k_count > num_frames:
        sub = np.linspace(0, k_count - 1, num_frames).round().astype(np.int32)
        return np.array([keyframes[i] for i in sub], dtype=np.int32)
    else:
        all_f = set(range(frame_count))
        remaining = sorted(all_f - set(keyframes))
        needed = num_frames - k_count
        if len(remaining) <= needed:
            res = sorted(keyframes + remaining)
            if len(res) < num_frames:
                padded = np.full(num_frames, -1, dtype=np.int32)
                padded[:len(res)] = res
                return padded
            return np.array(res[:num_frames], dtype=np.int32)
        step = len(remaining) / float(needed)
        picked_extra = [remaining[int(i * step)] for i in range(needed)]
        return np.array(sorted(keyframes + picked_extra), dtype=np.int32)
