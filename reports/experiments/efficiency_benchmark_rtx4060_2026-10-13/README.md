# RTX 4060 efficiency benchmark (2026-10-13)

Final efficiency benchmark. It replaces the A100 tables (`../efficiency_benchmark/`, `../frontend_pose_benchmark/`) and adds GFLOPs to `../efficiency_benchmark_rtx4060/`.

## Why the benchmark was repeated
- The earlier A100 numbers were slower than this laptop GPU (TCN U30 1.680 vs 0.823 ms), i.e. they mostly measured overhead of a shared Colab machine; they used a single round and did not fix TF32.
- The earlier ST-GCN GFLOPs (3.58) were wrong because THOP counts `einsum` as 0; `FlopCounterMode` gives 3.949.
- All measurements use one controlled machine (on mains power, performance power profile, GPU clocks logged during the run). Absolute numbers apply to this configuration only and are mainly used for relative comparison.

## Hardware and software
NVIDIA GeForce RTX 4060 Laptop GPU 8 GB, AMD Ryzen 7 7735HS. Details in `provenance.json`.

## Classifiers (`classifier_benchmark.csv`)
- Config: `configs/experiments/benchmark_pose_classifiers_rtx4060_gflops.yaml`. PyTorch 2.14.0+cu130, FP32, TF32 matmul/cuDNN off, eager `inference_mode`.
- Input: synthetic zero tensor, all frames valid, resident on the GPU. 50 warm-ups; 200 B=1 forwards (one CUDA event pair per forward); 100 B=32 forwards in one block; 5 rounds, order shuffled with seed 42.
- GFLOPs: `torch.utils.flop_counter.FlopCounterMode` at B=1, 2 × MACs. SPOTER only traces under grad mode (counts `addmm` only, so a small part of the attention matmuls may be missed).
- Checkpoints: BiLSTM e010 s42, TCN e008 s42, ST-GCN e013 s42, SPOTER e034 s43, SL-GCN e035 s43 (`best.pt`). U24/U60 reuse the U30 weights, since timing depends only on input shape.
- GPU clock under load: median 2460 MHz, maximum temperature 68°C (`classifier_gpu_during.csv`).

| Model | T | Params | GFLOPs | B1 median (ms) | B32 clips/s | VRAM (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| BiLSTM | 24/30/60 | 0.345M | 0.012/0.015/0.029 | 0.71/0.79/1.17 | 46,285/40,161/27,529 | 33/35/43 |
| TCN | 24/30/60 | 2.101M | 0.096/0.120/0.239 | 0.80/0.80/0.79 | 36,734/30,817/17,436 | 21/22/28 |
| ST-GCN | 24/30/60 | 3.252M | 3.063/3.949/7.657 | 3.34/3.14/3.11 | 1,474/1,130/501 | 182/223/423 |
| SPOTER | 30 | 5.950M | 0.193 | 4.00 | 7,209 | 47 |
| SL-GCN | 30 | 3.883M | 1.957 | 8.33 | 1,366 | 124 |

B=1 latency of TCN and ST-GCN is flat across T because it is dominated by overhead, so models are compared by GFLOPs and B=32 throughput. At U30, ST-GCN needs about 33 times the GFLOPs of TCN and has about 27 times lower throughput.

## Front-end (`frontend_benchmark.csv`, `frontend_rounds.csv`)
- Config: `configs/experiments/benchmark_frontends_pose_rtx4060.yaml`. Separate environment (Python 3.10, PyTorch 2.0.1+cu118, mmcv 2.0.1, mmdet 3.2.0, mmpose 1.3.1, ultralytics 8.4.146), built as in `src/runtime/colab.py`.
- 300 U30 frames from 10 clean validation videos (seed 42; signers 006/020/028 = 4/3/3), read from `artifacts/benchmarks/frontend_pose_videos/`. Per-video IDs, hashes and frame indices are not distributed; the selection is deterministic in code and config and requires the licensed manifest.
- YOLO E005 local checkpoint `artifacts/yolo/e005_yolo_person17_hand21_discriminative_max30/run/best.pt`, sha `0ee2a0e3…` (the same weights used for E021 Pose59 extraction and ONNX export). The earlier Hub revision `8f3c5213` is no longer available.
- Timing boundary: preprocessing, transfer, inference, NMS/postprocessing; excludes video decoding, model loading and the classifier. One warm-up pass, then per-frame B=1 timing. Five independent runs.

| Front-end | Median (ms), median of 5 runs | SD across runs | Range | p95 (ms) | FPS |
|---|---:|---:|---|---:|---:|
| RTMDet-nano + RTMW-m | 44.54 | 3.15 | 40.72-48.58 | 57.38 | 21.9 |
| YOLO dual-head E005 | 18.72 | 1.12 | 18.43-21.13 | 24.88 | 51.7 |

Per-run RTMW/YOLO ratios: 2.30, 2.50, 2.19, 2.35, 2.32. The front-ends use only about 35-40% of the GPU (CPU-side work dominates), so p95 varies; the median is reported. An earlier three-run trial had one abnormally slow run caused by a GPU clock drop and was discarded.

## Limitations
- Classifiers (PyTorch 2.14) and front-ends (PyTorch 2.0.1, required by mmcv) use different PyTorch versions.
- No end-to-end or edge-device measurement. The earlier CPU front-end numbers (Ryzen, `../frontend_pose_benchmark/`) are still used for the CPU comparison.

## U16 addendum (2026-10-14)

- Config: `configs/experiments/benchmark_pose_classifiers_rtx4060_u16.yaml`, same protocol (FP32, TF32 off, 50 warm-ups, 200 B1 / 100 B32 iterations, 5 rounds), separate session; raw summary `classifier_u16_summary.json`.
- U16: BiLSTM 0.0079 GFLOPs, 57,921 clips/s; TCN 0.0640 G, 40,327 clips/s; ST-GCN 2.0419 G, 2,443 clips/s.
- U30 anchors rerun in the same session: GFLOPs identical; B32 throughput BiLSTM 44,299 (+10.3% vs main run), TCN 29,142 (-5.4%), ST-GCN 1,128 (-0.2%). Cross-session throughput drift for the small models is up to ~10%; GFLOPs is deterministic.
