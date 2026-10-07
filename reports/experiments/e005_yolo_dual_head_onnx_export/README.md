# E005 dual-head YOLO ONNX export proof

The pinned E005 shared-trunk Person17/Hand21 dual-head model exported successfully to a static FP32 ONNX graph. This is an export and parity proof only.

## Locked source and export contract

- Config: `configs/experiments/e005_yolo_pose59_extract.yaml`
- Checkpoint: `hungchill24/vsl400-kp-selection-models@8f3c52134d8499741d5394e240d05e127f7a2d84/runs/e005_yolo_person17_hand21_discriminative_max30_s42/best.pt`
- Checkpoint SHA-256: `0ee2a0e349f79550271550274aa4e5e17fa09ab1c8b6de5c141392dbebb4a505`
- Static input: `images`, FP32, shape `[1, 3, 640, 640]`
- Outputs: `person_raw` `[1, 56, 8400]` and `hands_raw` `[1, 69, 8400]`
- ONNX: opset 18; ONNX 1.22.0; ONNX Runtime 1.30.0, `CPUExecutionProvider`
- Artifact SHA-256: `b9d477485901ba79cab5c73dc831f4623bf116ce0a0b72daec27bcded8bf1736`

## Parity result

The ONNX checker and ONNX Runtime CPU session both passed on the same deterministic synthetic FP32 input. PyTorch-to-ONNX raw-output maximum absolute error was `0.0003662109375` for `person_raw` and `0.00030517578125` for `hands_raw`, both within the recorded `0.001` tolerance. Existing independent NMS and Pose59 selection then produced identical Pose59 masks and `0.0` maximum pose-coordinate difference.

The tolerance covers small FP32 kernel/export rounding differences. It is not an accuracy evaluation.

## Boundary

NMS, Pose59 selection, letterbox inversion, video decoding, crop handling, and the canonical CUDA extractor remain outside the ONNX graph. This result does not measure CPU end-to-end latency, dataset-wide detection accuracy equivalence, full extraction, mobile deployment, edge deployment, or low-power operation.
