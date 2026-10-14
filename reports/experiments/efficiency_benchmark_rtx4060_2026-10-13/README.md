# RTX 4060 efficiency benchmark (2026-10-13)

Bản chốt cho paper IMCOM 2027. Thay thế bảng A100 (`../efficiency_benchmark/`, `../frontend_pose_benchmark/`) và bổ sung GFLOPs cho `../efficiency_benchmark_rtx4060/`.

## Vì sao đo lại
- Số A100 cũ chậm hơn laptop (TCN U30 1.680 so với 0.823 ms), tức là đo overhead của máy Colab dùng chung; chỉ 1 vòng; TF32 không khóa.
- GFLOPs ST-GCN cũ (3.58) sai vì THOP đếm `einsum` = 0. `FlopCounterMode` cho 3.949.
- Đo trên một máy kiểm soát được (cắm điện, `powerprofilesctl performance`, ghi xung GPU trong lúc chạy). Số tuyệt đối chỉ đúng cho cấu hình này; dùng chủ yếu để so tương đối.

## Phần cứng và môi trường
NVIDIA GeForce RTX 4060 Laptop GPU 8 GB, AMD Ryzen 7 7735HS. Chi tiết trong `provenance.json`.

## Classifier (`classifier_benchmark.csv`)
- Config: `configs/experiments/benchmark_pose_classifiers_rtx4060_gflops.yaml`. PyTorch 2.14.0+cu130, FP32, TF32 matmul/cuDNN tắt, eager `inference_mode`.
- Input: tensor 0 tổng hợp, mọi frame valid, đặt sẵn trên GPU. 50 warm-up; 200 forward B=1 (CUDA event từng forward); 100 forward B=32 trong một khối; 5 vòng, thứ tự xáo với seed 42.
- GFLOPs: `torch.utils.flop_counter.FlopCounterMode` ở B=1, quy ước 2 × MACs. SPOTER chỉ trace được dưới grad mode (chỉ đếm `addmm`, có thể sót một phần nhỏ matmul attention).
- Checkpoint: BiLSTM e010 s42, TCN e008 s42, ST-GCN e013 s42, SPOTER e034 s43, SL-GCN e035 s43 (`best.pt`). U24/U60 dùng trọng số U30: latency chỉ phụ thuộc shape.
- Xung GPU khi tải: trung vị 2460 MHz, nhiệt độ tối đa 68°C (`classifier_gpu_during.csv`).

| Model | T | Params | GFLOPs | B1 median (ms) | B32 clip/s | VRAM (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| BiLSTM | 24/30/60 | 0.345M | 0.012/0.015/0.029 | 0.71/0.79/1.17 | 46,285/40,161/27,529 | 33/35/43 |
| TCN | 24/30/60 | 2.101M | 0.096/0.120/0.239 | 0.80/0.80/0.79 | 36,734/30,817/17,436 | 21/22/28 |
| ST-GCN | 24/30/60 | 3.252M | 3.063/3.949/7.657 | 3.34/3.14/3.11 | 1,474/1,130/501 | 182/223/423 |
| SPOTER | 30 | 5.950M | 0.193 | 4.00 | 7,209 | 47 |
| SL-GCN | 30 | 3.883M | 1.957 | 8.33 | 1,366 | 124 |

Đọc kết quả: latency B=1 của TCN và ST-GCN phẳng theo T (bị overhead chi phối), nên so sánh model dựa vào GFLOPs và throughput B=32. U30: ST-GCN tốn GFLOPs gấp ~33 lần TCN, throughput thấp hơn ~27 lần.

## Front-end (`frontend_benchmark.csv`, `frontend_rounds.csv`)
- Config: `configs/experiments/benchmark_frontends_pose_rtx4060.yaml`. Venv riêng `~/.venvs/frontend-rtmw` (Python 3.10, PyTorch 2.0.1+cu118, mmcv 2.0.1, mmdet 3.2.0, mmpose 1.3.1, ultralytics 8.4.146), dựng theo `src/runtime/colab.py`.
- 300 frame U30 từ 10 video clean validation (seed 42; signer 006/020/028 = 4/3/3). Đầu vào local: `artifacts/benchmarks/frontend_pose_videos/`. Bản public chỉ giữ số lượng và fingerprint của nhóm bản ghi đã chọn; ID/hash/frame-index cấp từng video không được phân phối kèm report. Thuật toán chọn cố định nằm trong code/config và cần manifest được cấp phép.
- YOLO E005 checkpoint local `artifacts/yolo/e005_yolo_person17_hand21_discriminative_max30/run/best.pt`, sha `0ee2a0e3…` (cùng bản dùng trích Pose59 E021 và export ONNX). Revision HF `8f3c5213` cũ đã không còn (404).
- Ranh giới đo: tiền xử lý, chuyển dữ liệu, inference, NMS/hậu xử lý; không gồm decode video, nạp model, classifier. 1 warm-up pass, rồi đo B=1 từng frame. Chạy 5 lần độc lập.

| Front-end | Median (ms), median 5 lần | SD giữa lần | Dải | p95 (ms) | FPS |
|---|---:|---:|---|---:|---:|
| RTMDet-nano + RTMW-m | 44.54 | 3.15 | 40.72-48.58 | 57.38 | 21.9 |
| YOLO dual-head E005 | 18.72 | 1.12 | 18.43-21.13 | 24.88 | 51.7 |

Tỉ lệ RTMW/YOLO từng lần: 2.30, 2.50, 2.19, 2.35, 2.32. Front-end chỉ dùng GPU khoảng 35-40% (phần CPU chi phối), nên p95 dao động; báo median. Một lần đo thử trước đó (3 lượt) có một lượt chậm bất thường do xung GPU tụt, đã bỏ, không dùng.

## Giới hạn
- Classifier (PyTorch 2.14) và front-end (PyTorch 2.0.1, do mmcv) khác phiên bản PyTorch.
- Không đo end-to-end, không đo thiết bị biên. Số CPU front-end cũ (Ryzen, `../frontend_pose_benchmark/`) vẫn dùng cho Limitations.

## U16 addendum (2026-10-14)

- Config: `configs/experiments/benchmark_pose_classifiers_rtx4060_u16.yaml`, same protocol (FP32, TF32 off, 50 warm-ups, 200 B1 / 100 B32 iterations, 5 rounds), separate session; raw summary `classifier_u16_summary.json`.
- U16: BiLSTM 0.0079 GFLOPs, 57,921 clips/s; TCN 0.0640 G, 40,327 clips/s; ST-GCN 2.0419 G, 2,443 clips/s.
- U30 anchors rerun in the same session: GFLOPs identical; B32 throughput BiLSTM 44,299 (+10.3% vs main run), TCN 29,142 (-5.4%), ST-GCN 1,128 (-0.2%). Cross-session throughput drift for the small models is up to ~10%; GFLOPs is deterministic.
