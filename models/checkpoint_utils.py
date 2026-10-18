"""Test preprocessing, reproducibility, metrics and evidence for checkpoint inference."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
from typing import Any, Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
NUM_CLASSES = {"cifar10": 10, "cifar100": 100}
CIFAR_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}


def get_test_transform(dataset_name):
    mean, std = CIFAR_STATS[dataset_name]
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])


def source_evidence():
    paths = [ROOT / "checkpoint10.py", ROOT / "checkpoint100.py",
             *sorted((ROOT / "models").glob("*.py")), ROOT / "author/models/common.py"]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(
        p.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")).hexdigest() for p in paths}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"source_sha256": hashes, "git_commit": commit}



def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        writer(Path(temporary))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    _atomic(path, lambda p: p.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                                       encoding="utf-8"))


def dataset_fingerprint(dataset):
    """Hash raw pixels, targets and membership without invoking random transforms."""
    digest = hashlib.sha256()
    digest.update(str(len(dataset)).encode())
    if hasattr(dataset, "data") and hasattr(dataset, "targets"):
        arrays = [np.asarray(dataset.data), np.asarray(dataset.targets, dtype=np.int64)]
        if hasattr(dataset, "indices"):
            arrays.append(np.asarray(dataset.indices, dtype=np.int64))
    elif hasattr(dataset, "tensors"):
        arrays = [tensor.detach().cpu().numpy() for tensor in dataset.tensors]
    else:
        raise TypeError("Dataset needs raw data/targets or tensors for provenance verification")
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(memoryview(array).cast("B"))
    return {"samples": len(dataset), "sha256": digest.hexdigest()}


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    num_classes: int,
) -> Dict[str, Any]:
    """Comprehensive test evaluation: metrics, confusion matrix, macro F1."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    matrix = [[0] * num_classes for _ in range(num_classes)]
    loss_sum = 0.0
    correct = 0
    total = 0
    predictions_rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            if logits.shape != (labels.numel(), num_classes) or not torch.isfinite(logits).all():
                raise FloatingPointError("Invalid or non-finite evaluation logits")
            sample_losses = nn.functional.cross_entropy(logits, labels, reduction="none")
            loss = sample_losses.mean()
            preds = logits.argmax(dim=1)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite evaluation loss")

            loss_sum += sample_losses.double().sum().item()
            correct += (preds == labels).sum().item()
            total += labels.numel()

            for target, pred, sample_loss in zip(labels.cpu().tolist(), preds.cpu().tolist(), sample_losses.cpu().tolist()):
                matrix[target][pred] += 1
                predictions_rows.append({"sample_index": len(predictions_rows), "target": target,
                                         "prediction": pred, "negative_log_likelihood": sample_loss})

    if total == 0:
        raise ValueError("Cannot evaluate an empty loader")
    top1 = 100.0 * correct / total
    loss_avg = loss_sum / total

    # Macro F1 computation
    f1_sum = 0.0
    for i in range(num_classes):
        tp = matrix[i][i]
        fp_plus_fn = sum(matrix[i]) + sum(matrix[r][i] for r in range(num_classes))
        f1_sum += (2.0 * tp) / max(1, fp_plus_fn)
    macro_f1 = f1_sum / num_classes

    result = {
        "top1": top1,
        "loss": loss_avg,
        "macro_f1": macro_f1,
        "samples": total,
        "correct": correct,
    }

    # Save artifacts
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "test_metrics.json", result)

    with (output_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(matrix)

    with (output_dir / "test_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("sample_index", "target", "prediction", "negative_log_likelihood"))
        writer.writeheader()
        writer.writerows(predictions_rows)

    return result
