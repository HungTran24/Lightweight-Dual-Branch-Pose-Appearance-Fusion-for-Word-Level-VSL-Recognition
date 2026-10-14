# RTX 4060 controlled-length classifier benchmark

Date: 2026-10-11. All 9 configurations, 45 round cells and 9,000 latency samples were recorded; checkpoint hashes, input shapes, precision flags and the runner source hash match the protocol.

## Goal and scope

Measure inference latency, throughput and memory when the Pose53 input length changes from 24 to 30 and 60 frames. This is a classifier-only benchmark, not an accuracy, pose-extraction or end-to-end measurement. The final GFLOPs benchmark is in `../efficiency_benchmark_rtx4060_2026-10-13/`.

Each classifier uses its U30 seed-42 checkpoint for all three lengths. Keeping the weights fixed removes checkpoint differences from the length comparison; it is not a reproduction of the U60-trained E030/E031/E032 models.

## Protocol (fixed before the run)

Source: `configs/experiments/benchmark_pose_classifiers_rtx4060_controlled_lengths.yaml`.

- Hardware: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB VRAM.
- Input: synthetic zero tensor, Pose53, two channels `(x,y)`, all frames valid. No dataset split is read.
- Shape: B × T × 53 × 2, T ∈ {24, 30, 60}.
- Models: BiLSTM, TCN and ST-GCN, 400-class classifier.
- Precision: FP32 tensors, CUDA matmul TF32 and cuDNN TF32 off.
- Execution: eager `.eval()` and `torch.inference_mode()`, no compilation or CUDA graphs.
- Warm-up: 50 forwards before each batch-size measurement.
- Latency: 200 batch-1 forwards per round, CUDA events around each forward.
- Throughput: 100 batch-32 forwards in one contiguous timing block per round.
- Repeats: 5 rounds in the same process, entry order shuffled with seed 42.
- Inputs are created on the GPU beforehand; allocation, checkpoint loading and output serialization are outside the timing.
- Model-internal dispatch, synchronization and transfers stay inside the forward boundary, e.g. BiLSTM moves sequence lengths to the CPU at `src/models/bilstm.py:33`.
- VRAM: peak allocated during the batch-32 block; checkpoints are loaded through the CPU and released before measurement.

## Checkpoints

| Model | Checkpoint U30 seed 42 | Epoch | SHA256 |
|---|---|---:|---|
| BiLSTM | `artifacts/runs/e010_bilstm_pose53_uniform30_s42/best.pt` | 96 | `cb0723a84a43b53c8536964469030655aded4a9f1739a5abf083c5e63bc11de7` |
| TCN | `artifacts/runs/e008_tcn_pose53_uniform30_s42/best.pt` | 55 | `e6d899f28324877d7cb230c74976cc31593e05b214afce6e065cc4c857e57f21` |
| ST-GCN | `artifacts/runs/e013_stgcn_pose53_uniform30_s42/best.pt` | 74 | `25a99efc95034c0d79cc6fa6ec2faf8118f08788e245356de958d3458098cadc` |

Parameter counts are identical at all three lengths: BiLSTM 344,676; TCN 2,101,092; ST-GCN 3,251,602.

## Command

```bash
uv run --extra ml python scripts/run.py configs/experiments/benchmark_pose_classifiers_rtx4060_controlled_lengths.yaml --device cuda
```

## Results

Runtime: PyTorch 2.14.0+cu130, CUDA 13.0, cuDNN 92400, NVIDIA driver 615.71.09, AMD Ryzen 7 7735HS, 8 intra-op and 8 inter-op threads. TF32 matmul/cuDNN false; cuDNN benchmark/deterministic false.

| Model | Frames | B1 median (ms) | B1 p95 (ms) | SD of round medians (ms) | B32 samples/s | Peak VRAM (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| BiLSTM | 24 | 0.742 | 0.808 | 0.015 | 43,508 | 33.3 |
| BiLSTM | 30 | 0.820 | 0.954 | 0.012 | 39,264 | 34.5 |
| BiLSTM | 60 | 1.212 | 1.338 | 0.029 | 26,137 | 42.2 |
| TCN | 24 | 0.827 | 0.923 | 0.048 | 36,024 | 21.4 |
| TCN | 30 | 0.823 | 0.908 | 0.018 | 30,849 | 22.4 |
| TCN | 60 | 0.811 | 0.892 | 0.021 | 17,458 | 27.5 |
| ST-GCN | 24 | 3.350 | 3.962 | 0.307 | 1,470 | 181.6 |
| ST-GCN | 30 | 3.137 | 3.543 | 0.201 | 1,129 | 222.1 |
| ST-GCN | 60 | 3.212 | 4.196 | 0.270 | 500 | 422.4 |

Median and p95 use 1,000 B1 forwards per configuration. SD is the sample SD of the five round medians, not a recognition-seed SD. Throughput is 16,000 samples divided by the total time of the five B32 blocks. Peak VRAM is the maximum allocated memory across rounds, not device or reserved memory.

### TCN U24 vs U60

The pooled U60 median is 0.016032 ms (1.94%) lower than U24, but the ordering is not stable across rounds:

| Round | U24 median (ms) | U60 median (ms) | U60 - U24 (ms) |
|---:|---:|---:|---:|
| 1 | 0.739328 | 0.784384 | +0.045056 |
| 2 | 0.794752 | 0.792816 | -0.001936 |
| 3 | 0.827392 | 0.819200 | -0.008192 |
| 4 | 0.856784 | 0.825792 | -0.030992 |
| 5 | 0.848800 | 0.831488 | -0.017312 |

The mean paired round difference is -0.002675 ms (sample SD 0.028833 ms). U24 is faster in the first round and U60 in the other four, by varying margins. These data do not rank the two frame budgets by B1 latency and show drift across rounds within one process.

At B=32 the effect of length is clear: TCN U60 throughput is 51.54% lower than U24 and peak VRAM is 28.60% higher. Longer inputs add processing cost in the large-batch regime even though the B1 median is nearly flat.

BiLSTM U60 is slower than U24 in all five rounds; the pooled B1 median increases by 63.28%. ST-GCN also has an unstable B1 ordering, but U60 B32 throughput drops by 65.95% and VRAM increases by 132.52% relative to U24.

### Interpretation

- The non-monotonic B1 latency appears without different checkpoints or wrong input shapes.
- A plausible explanation is that resident-input eager forwards are dominated by host dispatch, many small kernels and runtime variability. No profiling was done to separate these factors.
- Round-to-round variation is not attributed to thermal throttling: the before/after snapshots only record 41°C/56°C, without continuous telemetry or locked clocks.

## Files

- `summary.json`: raw timings with full checkpoint and runtime metadata.
- `benchmark_summary.csv`: 9 aggregate rows.
- `round_summary.csv`: 45 round rows with measurement order.
- `provenance.json`: config, summary and source hashes, git revision, GPU snapshots and U24/U60 paired comparisons.
- `gpu_before.csv`, `gpu_after.csv`: observed GPU state before and after the run.

## Limitations

- The five rounds are repeated measurements in one process and session, not independent sessions.
- Median forward latency is not total kernel time; a CUDA-event span can include GPU idle time caused by host dispatch.
- A larger frame budget does not necessarily give a larger median latency on the GPU.
- Accuracy of the U24/U60-trained models should not be combined with this fixed-checkpoint comparison into a Pareto plot.
- These numbers are not directly comparable with the earlier A100 numbers: GPU, software, precision flags, checkpoint protocol and throughput boundary differ.
- The machine was on mains power; power mode and GPU clocks were not locked by the driver.
