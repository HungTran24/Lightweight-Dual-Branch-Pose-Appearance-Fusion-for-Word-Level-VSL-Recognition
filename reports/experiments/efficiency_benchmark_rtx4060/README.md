# RTX 4060 controlled-length classifier benchmark

Ngày: 2026-10-11.

Trạng thái: run hoàn tất với exit code 0; đã xác minh đủ 9 cấu hình, 45 cells theo round và 9,000 latency samples. Checkpoint hashes, input shapes, precision flags và runner source hash khớp protocol.

## Mục tiêu và phạm vi

Kiểm tra latency, throughput và bộ nhớ inference khi thay chiều dài input Pose53 từ 24 lên 30 và 60 frame. Đây là benchmark classifier-only, không phải đo accuracy, pose extraction hoặc end-to-end video recognition. Kết quả RTX 4060 được lưu riêng, không thay thế bảng A100 đã công bố trong project.

Ba checkpoint U60 trong config A100 chưa có ở local. Vì vậy, mỗi classifier dùng duy nhất checkpoint U30 seed 42 cho cả ba chiều dài. Việc giữ cùng trọng số loại bỏ khác biệt checkpoint khỏi comparison theo chiều dài. Đây không phải phép tái lập chính xác các hàng E030/E031/E032 đã huấn luyện trên U60.

## Protocol khóa trước run

Source: `configs/experiments/benchmark_pose_classifiers_rtx4060_controlled_lengths.yaml`.

- Experiment ID: `benchmark_pose_classifiers_rtx4060_controlled_lengths`.
- Hardware: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB VRAM.
- Input: synthetic zero tensor, Pose53, hai channel `(x,y)`, mọi frame đều valid. Không đọc dataset hoặc split train/val/test.
- Shape: B × T × 53 × 2; T thuộc {24, 30, 60}.
- Model: BiLSTM, TCN và ST-GCN hiện có, classifier 400 lớp.
- Precision: tensor FP32, CUDA matmul TF32 và cuDNN TF32 tắt.
- Execution: eager `.eval()` và `torch.inference_mode()`, không compile hoặc CUDA graph.
- Warm-up: 50 forwards trước mỗi batch-size measurement.
- Latency: 200 batch-1 forwards mỗi round, CUDA events quanh từng forward.
- Throughput: 100 batch-32 forwards trong một contiguous timing block mỗi round.
- Repeats: 5 rounds trong cùng process, thứ tự entry được shuffle với seed 42.
- Resident input: input tạo sẵn trên GPU; allocation, checkpoint load và output serialization nằm ngoài timing.
- Model-internal dispatch, synchronization và transfer vẫn nằm trong forward boundary. Ví dụ BiLSTM chuyển sequence lengths về CPU tại `src/models/bilstm.py:33`.
- VRAM: peak allocated trong batch-32 block; checkpoint load qua CPU và được giải phóng trước measurement.
- Output runtime: `artifacts/benchmarks/pose_classifiers_rtx4060_controlled_lengths/summary.json`.
- Durable report: thư mục chứa tài liệu này; raw timing và checkpoint metadata được lưu cùng report sau run.

## Model revision

| Model | Checkpoint U30 seed 42 | Epoch | SHA256 |
|---|---|---:|---|
| BiLSTM | `artifacts/runs/e010_bilstm_pose53_uniform30_s42/best.pt` | 96 | `cb0723a84a43b53c8536964469030655aded4a9f1739a5abf083c5e63bc11de7` |
| TCN | `artifacts/runs/e008_tcn_pose53_uniform30_s42/best.pt` | 55 | `e6d899f28324877d7cb230c74976cc31593e05b214afce6e065cc4c857e57f21` |
| ST-GCN | `artifacts/runs/e013_stgcn_pose53_uniform30_s42/best.pt` | 74 | `25a99efc95034c0d79cc6fa6ec2faf8118f08788e245356de958d3458098cadc` |

Checkpoint hashes và strict model loading đã được kiểm tra trước run. Mỗi classifier có cùng parameter count tại cả ba chiều dài: BiLSTM 344,676; TCN 2,101,092; ST-GCN 3,251,602.

## Lệnh chạy

```bash
.venv/bin/python scripts/run.py configs/experiments/benchmark_pose_classifiers_rtx4060_controlled_lengths.yaml --device cuda
```

## Kết quả

Runtime: PyTorch 2.14.0+cu130, CUDA 13.0, cuDNN 92400, driver NVIDIA 615.71.09, CPU AMD Ryzen 7 7735HS, intra-op/inter-op đều 8 threads. TF32 matmul/cuDNN đều false; cuDNN benchmark/deterministic đều false. Driver được lấy từ GPU snapshots vì API driver version của PyTorch không có ở runtime này.

| Model | Frames | B1 median (ms) | B1 p95 (ms) | SD round median (ms) | B32 samples/s | Peak VRAM (MiB) |
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

Median/p95 được tính từ 1,000 B1 forwards mỗi cấu hình. SD là sample SD của năm round medians, không phải recognition-seed SD. Throughput được tính từ tổng 16,000 samples chia tổng thời gian năm B32 blocks. Peak VRAM là max allocated qua năm rounds, không phải toàn bộ device memory hoặc reserved memory.

### Finding chính: TCN U24 so với U60

Pooled median U60 thấp hơn U24 0.016032 ms (1.94%), nhưng ordering không ổn định theo round:

| Round | U24 median (ms) | U60 median (ms) | U60 - U24 (ms) |
|---:|---:|---:|---:|
| 1 | 0.739328 | 0.784384 | +0.045056 |
| 2 | 0.794752 | 0.792816 | -0.001936 |
| 3 | 0.827392 | 0.819200 | -0.008192 |
| 4 | 0.856784 | 0.825792 | -0.030992 |
| 5 | 0.848800 | 0.831488 | -0.017312 |

Mean của paired round differences là -0.002675 ms, sample SD 0.028833 ms. Vòng đầu U24 nhanh hơn, bốn vòng còn lại U60 nhanh hơn với các chênh lệch khác nhau. Dữ liệu này chưa đủ để xếp hạng hai frame budgets theo B1 latency; nó cũng cho thấy drift giữa các rounds trong cùng process.

Khi B=32, ảnh hưởng của chiều dài rõ hơn: TCN U60 throughput thấp hơn U24 51.54%, trong khi peak allocated VRAM cao hơn 28.60%. Đây là evidence rằng input dài hơn tạo thêm chi phí xử lý trong regime batch lớn, dù B1 median gần như phẳng.

BiLSTM U60 chậm hơn U24 ở cả năm rounds; pooled B1 median tăng 63.28%. ST-GCN cũng có B1 ordering đảo giữa rounds, nhưng U60 B32 throughput giảm 65.95% và VRAM tăng 132.52% so với U24.

### Diễn giải có giới hạn

- Kết quả local tái hiện hiện tượng non-monotonic B1 latency mà không dùng checkpoint khác nhau hoặc sai input shape.
- Hypothesis phù hợp là resident-input eager forward bị ảnh hưởng bởi host dispatch, nhiều kernel nhỏ, backend execution efficiency và runtime variability. Chưa profile để tách tỷ lệ đóng góp của từng yếu tố.
- Không kết luận bảng A100 cũ đúng hoặc sai từ run này. GPU/software/precision/protocol khác nhau và raw A100 chưa có local.
- Không quy biến thiên giữa rounds cho thermal throttling: snapshot trước/sau chỉ ghi GPU 41°C/56°C, không phải continuous telemetry hoặc locked clocks.

### Artifact và verification

- `summary.json`: durable raw timing và đầy đủ checkpoint/runtime metadata, SHA256 giống summary runtime.
- `benchmark_summary.csv`: 9 aggregate rows.
- `round_summary.csv`: 45 round rows, có measurement order.
- `provenance.json`: config/summary/source hashes, git revision, GPU snapshots và U24/U60 paired comparisons.
- `gpu_before.csv`, `gpu_after.csv`: điều kiện GPU quan sát trước/sau.
- Runtime log: `artifacts/benchmarks/pose_classifiers_rtx4060_controlled_lengths/run.log`.

Checks: `pytest tests/test_classifier_benchmark.py -q` có 11 passed gồm CUDA tiny; Ruff và `git diff --check` pass; run exit 0; output parser xác minh 9 rows, 45 cells, 9,000 positive raw timing samples và mọi checkpoint SHA256/input shape/TF32 flag. Không sửa ledger A100 hoặc manuscript.

## Giới hạn diễn giải

- Năm rounds là repeated measurements trong cùng process và session, không phải năm GPU/session độc lập.
- Median forward latency không tương đương tổng kernel execution time; CUDA-event span có thể chứa GPU idle do host dispatch.
- Một frame budget lớn hơn không bắt buộc có median latency lớn hơn trên GPU.
- Không dùng accuracy của các model U24/U60 khác checkpoint để ghép thành Pareto với comparison này.
- Không so trực tiếp tốc độ RTX 4060 với số A100 cũ: GPU, software, precision flags, checkpoint protocol và throughput boundary khác nhau.
- Mains adapter đã được kiểm tra online trước run; power mode và GPU clocks chưa được khóa bằng driver. Snapshot GPU trước/sau run chỉ mô tả điều kiện quan sát.
- Không training, extraction, HF/W&B upload hoặc Colab trong task này.
