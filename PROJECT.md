# Chạy Project AI

Chạy các lệnh dưới đây tại thư mục gốc project.

## 1. Cài môi trường

```bash
uv sync --locked --extra ml
cp .env.example .env
```

Điền `HF_TOKEN` và `WANDB_API_KEY` trong `.env` theo các integration đang bật trong YAML. HF token cần quyền truy cập dataset và model repo của experiment; nếu experiment upload checkpoint, token cần quyền ghi.

## 2. Chạy local

```bash
CONFIG=configs/experiments/e008_tcn_pose53_uniform30_s42.yaml
uv run --extra ml python scripts/run.py "$CONFIG"
```

Config trên huấn luyện TCN Pose53 Uniform-30, seed 42. Runner tải feature từ HF theo revision/path trong YAML khi chưa có cache; không cần trích xuất lại YOLO hoặc RTMW.

Tiếp tục run đã có checkpoint:

```bash
uv run --extra ml python scripts/run.py "$CONFIG" --resume
```

Xem tùy chọn:

```bash
uv run --extra ml python scripts/run.py --help
```

## 3. Chạy Colab

Cần CLI `colab` hoặc `colab-acc2`, quyền tạo session và SSH identity. Identity mặc định là `~/.ssh/id_ed25519_colab_cli`.

```bash
CONFIG=configs/experiments/e008_tcn_pose53_uniform30_s42.yaml
uv run --extra ml python scripts/colab_run.py "$CONFIG" --gpu A100
```

Tiếp tục run:

```bash
uv run --extra ml python scripts/colab_run.py "$CONFIG" --gpu A100 --resume
```

Dùng `scripts/colab_run.py` cho mọi run Colab. Launcher cài môi trường remote, sync source, chạy cùng runner và lấy output về.

## 4. Chuẩn bị đầu vào theo tác vụ

| Tác vụ | Đầu vào cần có |
|---|---|
| Train pose classifier | Feature HF theo `data.repo_id`, `data.revision`, `data.artifact_path`; runner tự tải |
| Train RGB | Feature local tại `rgb_data.root` và receipt theo YAML |
| Fusion/test | Checkpoint, feature và prediction của các nhánh theo YAML |
| Benchmark classifier | Checkpoint local tại từng `entries[].checkpoint`; phần cứng khớp `hardware` |
| Benchmark front-end | Dataset/checkpoint theo YAML và runtime cho các backend đã chọn |

Khôi phục đúng file và đường dẫn trước khi chạy các tác vụ cần đầu vào local. Dùng revision đã khóa trong YAML.

ONNX export/runtime cần nhóm dependency riêng:

```bash
uv sync --locked --extra ml --group export
```

Nhánh RTMW cần OpenMMLab; MediaPipe cần runtime MediaPipe. Launcher Colab đã có môi trường riêng cho `frontend_benchmark` trong `src/runtime/colab.py`.

## 5. Dữ liệu và output

- Train/validation recognition dùng `manifests/audit/train_clean.csv` và `manifests/audit/val_clean.csv`.
- Test dùng `manifests/vsl400_front_400gloss/test.csv`.
- Đọc YAML trước khi chạy: experiment, seed, revision, feature view, số frame, model, phần cứng và output.
- Output ghi theo `output` trong YAML. Dùng `--resume` khi tiếp tục run cũ; giữ nguyên protocol.

## 6. Kiểm tra

```bash
uv lock --check
uv run --extra ml pytest -q tests
```
