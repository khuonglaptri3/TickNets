# -*- coding: utf-8 -*-
"""Script chạy kiểm tra và hiển thị kết quả tốt nhất trên tập Mid224 (224x224) cho báo cáo Giữa kỳ."""
from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.mid_data import CLASSES, RequireImageSize
from models.mid_models import MODEL_REVISIONS, build_mid_model
from models.model_profile import profile_model

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "l_mid224_smoothing010" / "best_val.pt"
DEFAULT_DATA_ROOT = ROOT / "data"
VARIANT = "Mid224"
IMAGE_SIZE = 224


def ensure_test_data(data_root: Path, variant: str) -> Path:
    test_dir = data_root / variant / "test"
    if test_dir.is_dir() and all((test_dir / c).is_dir() for c in CLASSES):
        return test_dir
    archive = data_root / f"{variant}.tar.xz"
    if archive.is_file():
        print(f"[*] Đang giải nén {archive.name} vào {data_root} ...")
        with tarfile.open(archive, "r:xz") as tar:
            tar.extractall(data_root)
        if test_dir.is_dir():
            return test_dir
    raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu test: {test_dir}")


def run_checkpoint_224(checkpoint_path: Path, data_root: Path, batch_size: int = 64) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file checkpoint: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg = payload.get("config", {})
    model_name = cfg.get("model", "l")

    model_cpu = build_mid_model(model_name, num_classes=len(CLASSES), variant=VARIANT)
    profile = profile_model(model_cpu, IMAGE_SIZE, cross_check=True)
    params = profile["learnable_parameters"]
    flops = profile["flops"]
    gflops = flops / 1e9

    model = model_cpu.to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    test_dir = ensure_test_data(data_root, VARIANT)
    eval_tf = transforms.Compose([RequireImageSize(IMAGE_SIZE), transforms.ToTensor()])
    test_ds = datasets.ImageFolder(str(test_dir), transform=eval_tf)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    loss_sum = 0.0
    cls_correct = [0] * len(CLASSES)
    cls_total = [0] * len(CLASSES)
    conf_matrix = [[0] * len(CLASSES) for _ in range(len(CLASSES))]

    with torch.inference_mode():
        for images, labels in test_loader:
            images_dev = images.to(device)
            labels_dev = labels.to(device)
            logits = model(images_dev)
            loss = criterion(logits, labels_dev)
            loss_sum += loss.item() * labels.numel()
            preds = logits.argmax(dim=1).cpu().tolist()
            for t, p in zip(labels.tolist(), preds):
                cls_total[t] += 1
                conf_matrix[t][p] += 1
                if t == p:
                    cls_correct[t] += 1

    total_samples = sum(cls_total)
    total_correct = sum(cls_correct)
    test_top1 = 100.0 * total_correct / total_samples
    test_loss = loss_sum / total_samples
    macro_f1 = 100.0 * sum(
        2 * conf_matrix[i][i] / max(1, sum(conf_matrix[i]) + sum(row[i] for row in conf_matrix))
        for i in range(len(CLASSES))
    ) / len(CLASSES)

    best_val = payload.get("best_validation", {})
    archive_path = data_root / f"{VARIANT}.tar.xz"
    archive_mb = archive_path.stat().st_size / 1e6 if archive_path.is_file() else None

    print("=" * 88)
    print("  KIỂM TRA CHECKPOINT GIỮA KỲ — MID224 (224x224)")
    print("=" * 88)
    print(f"  - Thiết bị chạy         : {device}")
    print(f"  - Đường dẫn Checkpoint  : {checkpoint_path.relative_to(ROOT) if checkpoint_path.is_relative_to(ROOT) else checkpoint_path}")
    print(f"  - Kiến trúc mô hình     : TickNet-{model_name.upper()} ({MODEL_REVISIONS.get(model_name, 'unknown')})")
    print(f"  - Số lớp phân loại      : {len(CLASSES)} ({', '.join(CLASSES)})")
    print(f"  - Độ phân giải đầu vào  : (1, 3, {IMAGE_SIZE}, {IMAGE_SIZE})")
    if archive_mb is not None:
        print(f"  - Dung lượng gói nén    : {archive_mb:.4f} MB (Trần <= 25.0 MB: {'ĐẠT' if archive_mb <= 25.0 else 'VƯỢT TRẦN'})")
    print("-" * 88)
    print("  [1] KIỂM TRA RÀNG BUỘC ĐỘ PHỨC TẠP MÔ HÌNH:")
    print(f"  - Tổng tham số học được : {params:,} params (Trần <= 6,000,000: {'ĐẠT' if params <= 6_000_000 else 'VƯỢT TRẦN'})")
    print(f"  - Chi phí tính toán     : {gflops:.6f} GFLOPs ({flops:,} FLOPs) (Trần < 1.0G: {'ĐẠT' if gflops < 1.0 else 'VƯỢT TRẦN'})")
    print("-" * 88)
    print("  [2] CẤU HÌNH HUẤN LUYỆN (TRAINING RECIPE):")
    print(f"  - Optimizer             : SGD (lr={cfg.get('learning_rate', 0.1)}, momentum={cfg.get('momentum', 0.9)}, weight_decay={cfg.get('weight_decay', 1e-4)})")
    print(f"  - Label Smoothing       : {cfg.get('label_smoothing', 0.0)}")
    print(f"  - Lịch giảm LR          : CosineAnnealingLR (epochs={cfg.get('epochs', 200)})")
    print(f"  - Epoch đạt đỉnh Val    : Epoch {payload.get('epoch', 'N/A')} / {cfg.get('epochs', 200)} (Val Top-1: {best_val.get('top1', 0.0):.2f}%, Val Loss: {best_val.get('loss', 0.0):.4f})")
    print("-" * 88)
    print("  [3] KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST (250 ẢNH):")
    print(f"  >>> Test Top-1 Accuracy : {test_top1:.2f}% ({total_correct}/{total_samples} ảnh đúng)")
    print(f"  >>> Test Macro-F1       : {macro_f1:.2f}%")
    print(f"  >>> Test Cross-Entropy  : {test_loss:.4f}")
    print("-" * 88)
    print("  [4] ĐỘ CHÍNH XÁC THEO TỪNG LỚP & MA TRẬN NHẦM LẪN:")
    print(f"  {'Lớp (Class)':<12} | {'Đúng/Tổng':>10} | {'Accuracy':>10} | Dự đoán [bird, cat, dog, frog, horse]")
    print("  " + "-" * 84)
    for idx, cname in enumerate(CLASSES):
        acc = 100.0 * cls_correct[idx] / cls_total[idx]
        print(f"  {cname:<12} | {f'{cls_correct[idx]}/{cls_total[idx]}':>10} | {acc:>9.2f}% | {conf_matrix[idx]}")
    print("=" * 88)

    return {
        "variant": VARIANT,
        "test_top1": test_top1,
        "test_macro_f1": macro_f1,
        "test_loss": test_loss,
        "parameters": params,
        "gflops": gflops,
    }


def main():
    parser = argparse.ArgumentParser(description="Chạy kiểm tra checkpoint tốt nhất trên tập Mid224 (224x224)")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Đường dẫn tới file checkpoint (.pt)")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Thư mục gốc chứa dữ liệu Mid224")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size khi đánh giá")
    args = parser.parse_args()
    run_checkpoint_224(args.checkpoint, args.data_root, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
