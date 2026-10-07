# Pose-classifier efficiency benchmark

Controlled forward-only benchmark on one NVIDIA A100-SXM4-40GB with PyTorch 2.11.0+cu128, CUDA 12.8 and FP32. Inputs were already resident on GPU; each row used 50 warm-up iterations. Batch-1 latency reports median and p95 over 200 forwards. Batch-32 throughput uses 100 forwards. Peak VRAM is `torch.cuda.max_memory_allocated` during the batch-32 measurement. Validation accuracy is the canonical three-seed mean ± sample SD; runtime, throughput, and VRAM use the seed-42 checkpoint for each model/frame setting.

| Model | Frames | Val Top-1 | Params | FP32 state | B=1 median | B=1 p95 | B=32 throughput | B=32 peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BiLSTM | 24 | 65.92 ± 0.86% | 0.34M | 1.4 MB | 1.299 ms | 1.356 ms | 22,541 samples/s | 35.6 MiB |
| BiLSTM | 30 | 63.53 ± 0.80% | 0.34M | 1.4 MB | 1.408 ms | 1.442 ms | 20,975 samples/s | 36.8 MiB |
| BiLSTM | 60 | 56.24 ± 3.17% | 0.34M | 1.4 MB | 1.987 ms | 2.022 ms | 14,546 samples/s | 44.5 MiB |
| TCN | 24 | 83.21 ± 0.10% | 2.10M | 8.4 MB | 1.690 ms | 1.735 ms | 19,015 samples/s | 31.5 MiB |
| TCN | 30 | 83.30 ± 0.83% | 2.10M | 8.4 MB | 1.680 ms | 1.738 ms | 19,373 samples/s | 33.5 MiB |
| TCN | 60 | 84.76 ± 0.14% | 2.10M | 8.4 MB | 1.682 ms | 2.286 ms | 19,324 samples/s | 37.6 MiB |
| ST-GCN | 24 | 83.97 ± 0.34% | 3.25M | 13.0 MB | 5.431 ms | 5.644 ms | 5,714 samples/s | 195.6 MiB |
| ST-GCN | 30 | 84.46 ± 1.22% | 3.25M | 13.0 MB | 5.601 ms | 5.866 ms | 5,272 samples/s | 237.2 MiB |
| ST-GCN | 60 | 85.67 ± 0.67% | 3.25M | 13.0 MB | 5.468 ms | 5.662 ms | 2,723 samples/s | 435.9 MiB |

## U30 complexity profile

The following local profile uses `B=1`, the Uniform-30 classifier input topology, and THOP-style FLOPs reported as `2 × MACs`. Parameters are exact; FLOPs are estimates because custom functional graph/attention operations may not all be counted by module hooks.

| Model | Parameters | Estimated FLOPs/forward |
|---|---:|---:|
| BiLSTM Pose53 | 0.345 M | 0.0149 G |
| TCN Pose53 | 2.101 M | 0.1202 G |
| SPOTER controlled Pose54 | 5.950 M | 0.1932 G |
| SL-GCN bone controlled Pose27 | 3.883 M | 1.4529 G |
| ST-GCN Pose53 | 3.252 M | 3.5763 G |

These are classifier-only complexity estimates. They exclude video decoding, pose extraction, feature transfer and metric computation, matching the forward-only benchmark boundary above.

## Interpretation boundaries

- BiLSTM is the smallest model and is fastest at U24/U30, but its validation accuracy is substantially lower than TCN and ST-GCN. At U60 its throughput falls to 14,546 samples/s while validation accuracy also declines.
- TCN is approximately 3.2× faster than ST-GCN at batch 1 under this GPU-only protocol and uses substantially less batch-32 memory.
- Increasing TCN from U24 through U60 has little effect on median latency or throughput in this protocol. ST-GCN U60 approximately halves batch-32 throughput and increases peak VRAM by about 1.84× relative to U30.
- These measurements support relative classifier-efficiency claims on A100 only. They exclude feature extraction, video decoding, CPU transfer and metric computation, so they are not end-to-end latency or mobile/edge measurements.

Durable committed ledger: `benchmark_summary.csv`. Local machine-readable sources: `artifacts/benchmarks/pose_classifiers_u24_u30_u60_a100/summary.json` and `artifacts/benchmarks/bilstm_u60_a100/summary.json`; `artifacts/` is not tracked by Git.
