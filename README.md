# Project AI

## Cài môi trường

```bash
uv sync --locked --extra ml
cp .env.example .env
```

Điền token trong `.env` theo các integration đang bật trong config.

## Chạy local

```bash
CONFIG=configs/experiments/e008_tcn_pose53_uniform30_s42.yaml
uv run --extra ml python scripts/run.py "$CONFIG"
```

## Chạy Colab

Cần CLI `colab`, SSH identity và quyền tạo session.

```bash
CONFIG=configs/experiments/e008_tcn_pose53_uniform30_s42.yaml
uv run --extra ml python scripts/colab_run.py "$CONFIG" --gpu A100
```

Thêm `--resume` để tiếp tục run. Pose training tự tải feature HF đã khóa; RGB, fusion và benchmark cần chuẩn bị các đầu vào ghi trong YAML.

Xem [PROJECT.md](PROJECT.md) để chuẩn bị đầu vào và chạy kiểm tra.
