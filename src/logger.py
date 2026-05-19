import csv
import json
import os
from datetime import datetime
from pathlib import Path


def log_run(
    # ── Định danh ──────────────────────────────────────────
    model_name: str,                        # "NeuralNetwork" | "RandomForest"
    run_note: str = "",                     # ghi chú tự do, vd: "thử log target"

    # ── Hyperparameters ────────────────────────────────────
    hyperparams: dict = None,               # {"lr": 0.01, "batch_size": 64, ...}

    # ── Metrics ────────────────────────────────────────────
    metrics: dict = None,                   # {"rmse": 12.3, "mae": 5.1, "pct_rmse": 8.2, "r2": 0.91}

    # ── Config ─────────────────────────────────────────────
    log_dir: str = None,                    # mặc định: thư mục Results/ gần nhất
    is_active: bool = True,                 # False → bỏ qua, không ghi gì
):
    """
    Ghi kết quả 1 lần chạy model vào file CSV để tiện so sánh.

    Ví dụ:
        log_run(
            model_name="NeuralNetwork",
            run_note="thêm dropout 0.3",
            hyperparams={"lr": 0.003, "batch_size": 64, "epochs": 200},
            metrics={"rmse": 45.2, "mae": 8.1, "pct_rmse": 22.1, "r2": 0.85},
        )
    """
    if not is_active:
        return

    # ── Tìm thư mục lưu ──────────────────────────────────
    if log_dir is None:
        # tự động tìm thư mục Results/ từ vị trí file này
        src_dir = Path(__file__).resolve().parent
        log_dir = src_dir.parent / "Results"

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "experiment_log.csv"

    # ── Chuẩn bị dữ liệu ─────────────────────────────────
    hyperparams = hyperparams or {}
    metrics     = metrics     or {}
    timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = {
        "timestamp"  : timestamp,
        "model_name" : model_name,
        "run_note"   : run_note,
        # flatten hyperparams vào cột riêng với prefix "hp_"
        **{f"hp_{k}": v for k, v in hyperparams.items()},
        # flatten metrics vào cột riêng
        **metrics,
    }

    # ── Ghi CSV ──────────────────────────────────────────
    file_exists = log_file.exists()

    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        # Ghi header nếu file mới
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(row)

    # ── In ra console ─────────────────────────────────────
    print(f"✅ Logged → {log_file}")
    print(f"   [{timestamp}] {model_name} | {run_note}")
    if metrics:
        metrics_str = " | ".join(f"{k}={v}" for k, v in metrics.items())
        print(f"   Metrics: {metrics_str}")
