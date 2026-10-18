"""Shared evaluation for the two final submission entry points."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from .checkpoint_utils import (NUM_CLASSES, atomic_json, dataset_fingerprint, evaluate,
                               file_sha256, get_test_transform, seed_everything,
                               seed_worker, source_evidence)
from .model_profile import profile_model
from .ticknet_l import ARCHITECTURE_REVISION, build_ticknet_l

ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "docs/05_results/selected_checkpoints.json"


def run_checkpoint(dataset_name, args, *, test_dataset=None):
    device = torch.device("cuda:0" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable in this Python environment. "
            "Activate an environment with CUDA-enabled PyTorch; "
            "checkpoint evaluation will not fall back to CPU."
        )
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))[dataset_name.upper()]
    default_checkpoint = ROOT / selection["checkpoint"]
    checkpoint_path = (args.checkpoint or default_checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")
    digest = file_sha256(checkpoint_path)
    is_submitted = checkpoint_path == default_checkpoint.resolve()
    if is_submitted and digest != selection["checkpoint_sha256"]:
        raise ValueError("SHA-256 checkpoint nộp bài không khớp manifest")

    # best_val.pt contains tensors and primitive metadata, not a pickled model.
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = payload["config"]
    num_classes = NUM_CLASSES[dataset_name]
    if (cfg.get("dataset") != dataset_name or cfg.get("num_classes") != num_classes
            or cfg.get("model") != "l_large_v2"
            or cfg.get("architecture_revision") != ARCHITECTURE_REVISION):
        raise ValueError("Checkpoint không khớp dataset hoặc kiến trúc TickNet-L Large v2")

    output_dir = args.output_dir.resolve()
    for folder in ("checkpoints", "docs", "data"):
        protected = ROOT / folder
        if output_dir == protected or protected in output_dir.parents:
            raise ValueError("Hãy lưu kết quả đánh giá vào runs/, ngoài dữ liệu và bằng chứng gốc")
    if output_dir == checkpoint_path.parent:
        raise ValueError("Thư mục output phải khác thư mục checkpoint gốc")

    seed_everything(cfg["seed"])
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = build_ticknet_l(num_classes=num_classes, cifar=True)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model = model.to(device)
    profile = profile_model(model, 32, cross_check=True)
    if not profile["within_exam_limits"]:
        raise ValueError("Mô hình vượt giới hạn 6M tham số hoặc 1G FLOPs")

    print(f"\nĐÁNH GIÁ CHECKPOINT FINAL — CIFAR-{num_classes} (32×32)", flush=True)
    print(f"Checkpoint: {checkpoint_path}\nSHA-256: {digest}")
    print(f"Thiết bị: {device}; batch size: {args.batch_size}; epoch: {payload['epoch']}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
    print(f"TickNet-L Large v2: 14 khối PDP; {profile['learnable_parameters']:,} tham số; "
          f"{profile['flops'] / 1e9:.9f} GFLOPs — ĐẠT giới hạn đề bài")
    print(f"FLOPs: {profile['counting_convention']}")
    print(f"Train: {cfg['optimizer'].upper()}, lr={cfg['learning_rate']}, "
          f"weight_decay={cfg['weight_decay']}, epochs={cfg['epochs']}, seed={cfg['seed']}")
    best = payload["best_validation"]
    print(f"Validation đã lưu: Top-1={best['top1']:.2f}%, loss={best['loss']:.6f}")
    if test_dataset is None:
        print("Đang chuẩn bị tập test chính thức (lần đầu sẽ tải dữ liệu nếu chưa có)...", flush=True)
        dataset_cls = datasets.CIFAR10 if num_classes == 10 else datasets.CIFAR100
        test_set = dataset_cls(root=str(args.data_root), train=False, download=not args.no_download,
                               transform=get_test_transform(dataset_name))
    else:
        # Optional verification adapter; the same pixel/label/order hash is mandatory.
        test_set = test_dataset
    evidence = dataset_fingerprint(test_set)
    if evidence != cfg["data_evidence"]["test"] or len(test_set) != 10_000:
        raise ValueError("Ảnh, nhãn hoặc thứ tự tập test không khớp dữ liệu trong báo cáo")
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda",
                        worker_init_fn=seed_worker)
    print(f"Đang chạy inference trên {len(test_set):,} ảnh test...", flush=True)
    result = evaluate(model, loader, device, output_dir, num_classes)
    recorded = json.loads((checkpoint_path.parent / "test_metrics.json").read_text(encoding="utf-8")) \
        if is_submitted else None
    result.update(dataset=dataset_name, model="l_large_v2", checkpoint_epoch=payload["epoch"],
                  checkpoint_sha256=digest, test_data_sha256=evidence["sha256"],
                  parameters=profile["learnable_parameters"], flops=profile["flops"],
                  counting_convention=profile["counting_convention"],
                  excluded_operations=profile["excluded_operations"],
                  device=str(device), batch_size=args.batch_size,
                  device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                  torch_version=str(torch.__version__),
                  training_source_sha256=cfg["source_sha256"],
                  evaluation_source_sha256=source_evidence()["source_sha256"])
    if recorded:
        result["reported_top1"] = selection["test_top1"]
        result["top1_difference"] = result["top1"] - selection["test_top1"]
        result["matches_report"] = (
            result["correct"] == recorded["correct"]
            and math.isclose(result["macro_f1"], recorded["macro_f1"], abs_tol=1e-6)
            and math.isclose(result["loss"], recorded["loss"], abs_tol=1e-4)
        )
    atomic_json(output_dir / "test_metrics.json", result)

    with (output_dir / "confusion_matrix.csv").open(encoding="utf-8", newline="") as handle:
        matrix = [[int(value) for value in row] for row in csv.reader(handle)]
    rows = []
    for idx, name in enumerate(test_set.classes):
        support = sum(matrix[idx])
        predicted = sum(row[idx] for row in matrix)
        correct = matrix[idx][idx]
        rows.append({"class_id": idx, "class": name, "correct": correct, "support": support,
                     "recall": correct / max(1, support),
                     "precision": correct / max(1, predicted),
                     "f1": 2 * correct / max(1, support + predicted)})
    with (output_dir / "per_class_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\nKẾT QUẢ ĐÁNH GIÁ TỪ TRỌNG SỐ")
    print(f"Test Top-1: {result['top1']:.2f}% ({result['correct']}/{result['samples']})")
    print(f"Macro-F1: {result['macro_f1']:.6f} ({100 * result['macro_f1']:.4f}%)")
    print(f"Cross-entropy loss: {result['loss']:.6f}")
    if recorded:
        print(f"Top-1 báo cáo: {selection['test_top1']:.2f}%; "
              f"chênh lệch: {result['top1_difference']:+.2f} điểm phần trăm")
        print("Đối chiếu accuracy, Macro-F1, loss: "
              + ("KHỚP BÁO CÁO" if result["matches_report"] else "KHÁC BÁO CÁO — xem JSON kết quả"))
    print("\nĐộ chính xác theo lớp (recall):")
    for row in rows:
        print(f"  {row['class_id']:3d} {row['class']:<22} "
              f"{row['correct']:4d}/{row['support']:<4d} {100 * row['recall']:6.2f}%")
    if num_classes == 10:
        print("\nConfusion matrix (hàng = nhãn thật, cột = dự đoán; cùng thứ tự lớp ở trên):")
        for row in matrix:
            print("  " + " ".join(f"{value:4d}" for value in row))
    print(f"\nKết quả JSON, dự đoán, confusion matrix và chỉ số từng lớp: {output_dir}")
    return result


def main(dataset_name, argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    num_classes = NUM_CLASSES[dataset_name]
    parser = argparse.ArgumentParser(description=f"Kiểm tra checkpoint final CIFAR-{num_classes}")
    parser.add_argument("--checkpoint", type=Path, help="Mặc định: best_val.pt được chọn trong báo cáo")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / f"runs/checkpoint{num_classes}")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cuda:0",
                        help="Default: cuda:0; auto also requires CUDA. CPU requires --device cpu.")
    parser.add_argument("--no-download", action="store_true", help="Chỉ dùng dữ liệu đã có")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0 or args.threads < 1:
        parser.error("batch-size, threads phải dương; num-workers phải không âm")
    return run_checkpoint(dataset_name, args)
