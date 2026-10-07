from pathlib import Path

import numpy as np
import pytest

from src.config import load_config, output_path
from src.data import FeatureSplit
from src.signer_diagnostic import _select_rows, _validate_selected_labels


def _split() -> FeatureSplit:
    return FeatureSplit(
        X=np.zeros((3, 2, 53, 3), dtype=np.float32),
        y=np.array([4, 7, 4], dtype=np.int64),
        video_id=np.array(["000001", "000002", "000003"], dtype="U6"),
        signer_id=np.array(["024", "025", "024"], dtype="U3"),
        frame_indices=np.zeros((3, 2), dtype=np.int32),
        valid_frame_mask=np.ones((3, 2), dtype=np.uint8),
        keypoint_valid_mask=np.ones((3, 2, 53), dtype=np.uint8),
    )


def test_signer_diagnostic_config_and_clean_label_contract(tmp_path: Path) -> None:
    path = (
        Path(__file__).parents[1]
        / "configs/experiments/e025_stgcn_pose53_train_hearing_diagnostic_s42.yaml"
    )
    config = load_config(path)
    assert config["experiment"]["operation"] == "signer_diagnostic"
    assert config["diagnostic"]["signers"] == ["024", "025", "026", "027"]
    assert output_path(config, tmp_path).name == "e025_stgcn_pose53_train_hearing_diagnostic_s42"

    selected = _select_rows(_split(), np.array([0, 2], dtype=np.int64))
    manifest = {
        "000001": {"video_id": "000001", "class_id": "4", "signer_id": "024", "gloss": "A"},
        "000003": {"video_id": "000003", "class_id": "4", "signer_id": "024", "gloss": "A"},
    }
    _validate_selected_labels(selected, manifest)

    manifest["000003"]["class_id"] = "7"
    with pytest.raises(ValueError, match="Class label mismatch"):
        _validate_selected_labels(selected, manifest)
