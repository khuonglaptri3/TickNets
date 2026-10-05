#!/usr/bin/env python3
"""Generate 4 focused Kaggle notebooks for the Final Exam Grid Search."""
import json
from pathlib import Path

phases = [
    {
        "filename": "Phase1_CIFAR10_SGD.ipynb",
        "title": "Phase 1: CIFAR-10 with SGD (lr=0.10, lr=0.15)",
        "dataset": "cifar10",
        "dataset_name": "CIFAR-10",
        "optimizer_name": "SGD",
        "runs": [
            ("configs/final/cifar10_sgd_lr010.json", "cifar10_sgd_lr010", "SGD lr=0.10 (Momentum 0.9, Weight Decay 1e-4)"),
            ("configs/final/cifar10_sgd_lr015.json", "cifar10_sgd_lr015", "SGD lr=0.15 (Momentum 0.9, Weight Decay 1e-4)"),
        ],
        "zip_name": "phase1_cifar10_sgd_results.zip",
    },
    {
        "filename": "Phase2_CIFAR10_Adam.ipynb",
        "title": "Phase 2: CIFAR-10 with Adam (lr=0.001, lr=0.0003)",
        "dataset": "cifar10",
        "dataset_name": "CIFAR-10",
        "optimizer_name": "Adam",
        "runs": [
            ("configs/final/cifar10_adam_lr0001.json", "cifar10_adam_lr0001", "Adam lr=0.001 (Betas 0.9/0.999, Weight Decay 1e-4)"),
            ("configs/final/cifar10_adam_lr00003.json", "cifar10_adam_lr00003", "Adam lr=0.0003 (Betas 0.9/0.999, Weight Decay 1e-4)"),
        ],
        "zip_name": "phase2_cifar10_adam_results.zip",
    },
    {
        "filename": "Phase3_CIFAR100_SGD.ipynb",
        "title": "Phase 3: CIFAR-100 with SGD (lr=0.10, lr=0.15)",
        "dataset": "cifar100",
        "dataset_name": "CIFAR-100",
        "optimizer_name": "SGD",
        "runs": [
            ("configs/final/cifar100_sgd_lr010.json", "cifar100_sgd_lr010", "SGD lr=0.10 (Momentum 0.9, Weight Decay 1e-4)"),
            ("configs/final/cifar100_sgd_lr015.json", "cifar100_sgd_lr015", "SGD lr=0.15 (Momentum 0.9, Weight Decay 1e-4)"),
        ],
        "zip_name": "phase3_cifar100_sgd_results.zip",
    },
    {
        "filename": "Phase4_CIFAR100_Adam.ipynb",
        "title": "Phase 4: CIFAR-100 with Adam (lr=0.001, lr=0.0003)",
        "dataset": "cifar100",
        "dataset_name": "CIFAR-100",
        "optimizer_name": "Adam",
        "runs": [
            ("configs/final/cifar100_adam_lr0001.json", "cifar100_adam_lr0001", "Adam lr=0.001 (Betas 0.9/0.999, Weight Decay 1e-4)"),
            ("configs/final/cifar100_adam_lr00003.json", "cifar100_adam_lr00003", "Adam lr=0.0003 (Betas 0.9/0.999, Weight Decay 1e-4)"),
        ],
        "zip_name": "phase4_cifar100_adam_results.zip",
    },
]

out_dir = Path("docs/kaggle")
out_dir.mkdir(parents=True, exist_ok=True)

for p in phases:
    cells = []
    title = p["title"]
    dataset_name = p["dataset_name"]
    optimizer_name = p["optimizer_name"]
    dataset_key = p["dataset"]
    zip_name = p["zip_name"]
    zip_stem = Path(zip_name).stem
    target_configs = [r[1] for r in p["runs"]]

    # Title Markdown
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# Final Exam: {title}\n",
            f"Tập trung thực nghiệm trên **{dataset_name}** sử dụng thuật toán **{optimizer_name}** với kiến trúc **TickNet-L v1** (200 epochs).\n",
            "\n",
            "### Hướng dẫn chạy nhanh trên Kaggle:\n",
            "1. Chọn **Accelerator**: GPU T4 x2 hoặc GPU P100.\n",
            "2. Bật **Internet**: Always on (để clone repo và tải dataset).\n",
            "3. Bấm **Save Version** -> **Save & Run All (Commit)** hoặc chạy từng cell.\n",
            f"4. Sau khi hoàn thành, file `{zip_name}` sẽ tự động được tạo ở `/kaggle/working/` để tải về máy.\n",
        ],
    })

    # Cell 1: Preflight & Clone
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Environment & GPU Preflight Check\n",
            "import os\n",
            "import torch\n",
            "\n",
            "# Auto-clone repository from branch feature/final-exam-model-l\n",
            "if not os.path.exists('train_cifar.py'):\n",
            "    !test -d /kaggle/working/TickNets/.git || git clone --depth 1 --branch feature/final-exam-model-l https://github.com/khuonglaptri3/TickNets.git /kaggle/working/TickNets\n",
            "    %cd /kaggle/working/TickNets\n",
            "\n",
            "print(f'PyTorch Version: {torch.__version__}')\n",
            "print(f'CUDA Available: {torch.cuda.is_available()}')\n",
            "if torch.cuda.is_available():\n",
            "    print(f'GPU Device: {torch.cuda.get_device_name(0)}')\n",
            "!nvidia-smi\n",
        ],
    })

    # Cell 2: Download dataset
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            f"# 2. Download and Verify {dataset_name} Dataset\n",
            f"!python download_cifar.py --data-root /kaggle/working/data --dataset {dataset_key}\n",
        ],
    })

    # Run experiments
    for idx, (cfg_path, exp_name, desc) in enumerate(p["runs"], start=1):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"### Thực nghiệm {idx}: {desc}",
            ],
        })
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"# Run {exp_name}\n",
                f"!python train_cifar.py --config {cfg_path} --data-root /kaggle/working/data --output-dir /kaggle/working/runs/{exp_name}\n",
            ],
        })

    # Summary & Zip
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Tổng kết Kết Quả và Đóng Gói Artifacts",
        ],
    })

    summary_source = [
        "import json\n",
        "import shutil\n",
        "from pathlib import Path\n",
        "import pandas as pd\n",
        "\n",
        "runs_dir = Path('/kaggle/working/runs')\n",
        f"target_configs = {target_configs!r}\n",
        "records = []\n",
        "\n",
        "for exp_name in target_configs:\n",
        "    p = runs_dir / exp_name\n",
        "    metric_file = p / 'test_metrics.json'\n",
        "    cfg_file = p / 'config.json'\n",
        "    if metric_file.is_file() and cfg_file.is_file():\n",
        "        metrics = json.loads(metric_file.read_text())\n",
        "        cfg = json.loads(cfg_file.read_text())\n",
        "        records.append({\n",
        "            'Experiment': exp_name,\n",
        "            'Dataset': cfg['dataset'].upper(),\n",
        "            'Optimizer': cfg['optimizer'].upper(),\n",
        "            'LR': cfg['learning_rate'],\n",
        "            'Test Top-1 (%)': f\"{metrics['top1']:.2f}%\",\n",
        "            'Test Loss': f\"{metrics['loss']:.4f}\",\n",
        "            'Macro F1': f\"{metrics['macro_f1']:.4f}\",\n",
        "            'Params': f\"{cfg['learnable_parameters']:,}\",\n",
        "            'FLOPs (G)': f\"{cfg['gflops_forward']:.4f}G\",\n",
        "        })\n",
        "\n",
        "df = pd.DataFrame(records)\n",
        "print('=== KẾT QUẢ THỰC NGHIỆM ===')\n",
        "display(df)\n",
        "\n",
        "# Đóng gói kết quả thành file zip để download\n",
        f"zip_base = Path('/kaggle/working') / '{zip_stem}'\n",
        "stage_dir = Path('/kaggle/working/stage_results')\n",
        "if stage_dir.exists():\n",
        "    shutil.rmtree(stage_dir)\n",
        "stage_dir.mkdir(parents=True, exist_ok=True)\n",
        "for exp in target_configs:\n",
        "    src = runs_dir / exp\n",
        "    if src.exists():\n",
        "        shutil.copytree(src, stage_dir / exp)\n",
        "\n",
        "shutil.make_archive(str(zip_base), 'zip', stage_dir)\n",
        "print(f'[✓] Đã tạo file kết quả tải về tại: {zip_base}.zip')\n",
    ]

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": summary_source,
    })

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11.14",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    file_path = out_dir / p["filename"]
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Created: {file_path}")

print("All 4 phase notebooks created successfully.")
