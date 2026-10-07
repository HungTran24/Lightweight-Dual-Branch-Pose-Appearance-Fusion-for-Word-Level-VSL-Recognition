# Frontend runtime and complexity benchmark

## Locked runtime protocol

- Population: clean validation only, `manifests/audit/val_clean.csv`.
- Selection: seed 42, signer allocation `006/020/028 = 4/3/3`, 10 deterministic videos, all with at least 30 decoded frames.
- Frames: the same 300 real Uniform-30 frames for every backend. Manifest SHA-256: `1bd92e5de767b60a1ba86dd907c6db3c54c37252884b98638a8f19e33a4a73f1`.
- Warm-up: one complete pass over all 300 frames.
- Measurement: one B=1 latency per selected frame, then median, P95 and throughput.
- Timing boundary: native-frame preprocessing, host/device transfer where applicable, backend inference and output postprocessing. Video decode, archive download, model construction/checkpoint download, artifact writing and classifier forward are excluded.
- This is an evaluated frontend/runtime configuration benchmark, not an end-to-end recognition benchmark or an architecture-only ranking.

The selected video IDs are `003570`, `004011`, `004144`, `004206`, `015360`, `015555`, `015676`, `024150`, `024448` and `024641`.

## Runtime results

### A100

Hardware: NVIDIA A100-SXM4-40GB. Runtime: PyTorch 2.0.1+cu118, FP32, B=1.

| Frontend | Median | P95 | Throughput | Output presence |
|---|---:|---:|---:|---:|
| RTMDet-nano + RTMW-m | 56.42 ms | 58.60 ms | 17.67 FPS | 100 % |
| E005 dual-head YOLO | **26.80 ms** | **27.73 ms** | **37.24 FPS** | 100 % |

The E005 row includes letterbox preprocessing, transfer, shared-trunk dual-head inference, independent NMS and Pose59 selection. The RTMW row includes RTMDet-nano, person selection and RTMW-m top-down inference. E005 median latency is approximately 2.1× lower in this evaluated A100 configuration.

### Ryzen 7 7735HS CPU

Runtime: native PyTorch FP32 for MediaPipe, RTMW and E005; ONNX Runtime 1.23.2 with `CPUExecutionProvider` for the ONNX row. CPU thread setting: 8 intra-op and 1 inter-op.

| Frontend/runtime | Median | P95 | Throughput | Output presence |
|---|---:|---:|---:|---:|
| MediaPipe Holistic, TFLite/XNNPACK | **27.75 ms** | **30.21 ms** | **35.74 FPS** | 100 % |
| RTMDet-nano + RTMW-m, PyTorch | 103.52 ms | 119.86 ms | 9.43 FPS | 100 % |
| E005 dual-head YOLO, PyTorch | 255.50 ms | 339.26 ms | 3.70 FPS | 100 % |
| E005 dual-head YOLO, ONNX Runtime | 277.66 ms | 314.26 ms | 3.75 FPS | 100 % |

The ONNX row uses the parity-verified static artifact `artifacts/yolo/onnx/e005_dual_head_b1_640/model.onnx`, SHA-256 `b9d477485901ba79cab5c73dc831f4623bf116ce0a0b72daec27bcded8bf1736`. ONNX export improves packaging and portability, but did not reduce CPU median latency in this run.

## Complexity profile

### Frontend models

Measured locally at B=1 and 640×640 using the Ultralytics THOP-backed FLOPs helper. FLOPs use the convention `2 × MACs`.

| Model graph | Parameters | Estimated FLOPs |
|---|---:|---:|
| Stock YOLO26m-pose, current unfused graph | 24.344 M | 86.135 G |
| E005 custom dual-head, current unfused graph | **24.622 M** | **87.034 G** |

Relative to the same current stock graph, E005 adds approximately 1.14 % parameters and 1.04 % estimated FLOPs. These values are not directly comparable to the official YOLO26 table's fused/pruned `20.4 M / 68.4 G` entry.

### Classifiers

Measured locally at B=1, Uniform-30 and the exact project input topology. Parameters are exact; THOP FLOPs are estimates and may omit some functional graph/attention operations such as custom `einsum` paths.

| Classifier | Parameters | Estimated FLOPs/forward |
|---|---:|---:|
| BiLSTM Pose53 | 0.345 M | 0.0149 G |
| TCN Pose53 | 2.101 M | 0.1202 G |
| SPOTER controlled Pose54 | 5.950 M | 0.1932 G |
| SL-GCN bone controlled Pose27 | 3.883 M | 1.4529 G |
| ST-GCN Pose53 | 3.252 M | 3.5763 G |

These are classifier-only complexity figures and do not include video decoding, pose extraction, transfer or metrics. They should be reported as estimated GFLOPs, not as exact hardware work counts.

## Interpretation boundaries

- E005 is GPU-favorable in the evaluated runtime: 26.80 ms / 37.24 FPS on A100 versus 56.42 ms / 17.67 FPS for RTMDet-nano + RTMW-m.
- MediaPipe is the CPU-oriented option in this table. E005's native PyTorch and ONNX CPU rows are not CPU-real-time configurations under the 25 FPS, 40 ms/frame budget.
- No claim is made for complete video-pipeline real time, mobile readiness, edge readiness, low-power operation or universal architecture efficiency.
- RTMW paper FLOPs/latency values are pose-network-only ONNX measurements and are not substituted for the project's full detector-plus-pose runtime row.

## Sources

- Durable committed summary: `reports/experiments/frontend_pose_benchmark/summary.json`
- CPU summary: `artifacts/benchmarks/frontend_pose_cpu_ryzen7_7735hs/summary.json`
- A100 summary: `artifacts/benchmarks/frontend_pose_a100/summary.json`
- CPU benchmark config: `configs/experiments/benchmark_frontends_pose_cpu_ryzen7_7735hs.yaml`
- A100 benchmark config: `configs/experiments/benchmark_frontends_pose_a100.yaml`
- ONNX parity report: `reports/experiments/e005_yolo_dual_head_onnx_export/`
- Classifier efficiency report: `reports/experiments/efficiency_benchmark/README.md`
