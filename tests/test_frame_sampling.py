from __future__ import annotations

import numpy as np

from src.frame_sampling import kcc_indices, motion_indices, uniform_indices


def test_uniform_indices_basic():
    idx = uniform_indices(frame_count=48, num_frames=24)
    assert len(idx) == 24
    assert idx[0] == 0
    assert idx[-1] == 47
    assert np.all(np.diff(idx) >= 0)


def test_uniform_indices_short():
    idx = uniform_indices(frame_count=10, num_frames=24)
    assert len(idx) == 24
    assert list(idx[:10]) == list(range(10))
    assert list(idx[10:]) == [-1] * 14


def test_motion_indices_basic():
    np.random.seed(42)
    kpts = np.random.randn(60, 53, 3).astype(np.float32)
    idx = motion_indices(kpts, num_frames=24)
    assert len(idx) == 24
    assert np.all(idx >= 0)
    assert np.all(idx < 60)
    assert np.all(np.diff(idx) >= 0)


def test_motion_indices_short():
    kpts = np.random.randn(12, 53, 3).astype(np.float32)
    idx = motion_indices(kpts, num_frames=24)
    assert len(idx) == 24
    assert list(idx[:12]) == list(range(12))
    assert list(idx[12:]) == [-1] * 12


def test_kcc_indices_basic():
    np.random.seed(42)
    kpts = np.random.randn(60, 53, 3).astype(np.float32)
    idx = kcc_indices(kpts, num_frames=24, thresh=1.0)
    assert len(idx) == 24
    assert np.all(idx >= 0)
    assert np.all(idx < 60)
    assert np.all(np.diff(idx) >= 0)


def test_kcc_indices_short():
    kpts = np.random.randn(15, 53, 3).astype(np.float32)
    idx = kcc_indices(kpts, num_frames=24, thresh=1.0)
    assert len(idx) == 24
    assert list(idx[:15]) == list(range(15))
    assert list(idx[15:]) == [-1] * 9


if __name__ == "__main__":
    test_uniform_indices_basic()
    test_uniform_indices_short()
    test_motion_indices_basic()
    test_motion_indices_short()
    test_kcc_indices_basic()
    test_kcc_indices_short()
    print("ALL SAMPLING TESTS PASSED!")
