"""Evidence and atomic artifacts for CIFAR experiments (no training side effects)."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
EPOCH_FIELDS = ("epoch", "learning_rate", "train_loss", "train_top1", "train_samples",
                "val_loss", "val_top1", "val_samples")
RECIPE_FIELDS = ("model", "dataset", "optimizer", "learning_rate", "momentum", "nesterov",
                 "adam_beta1", "adam_beta2", "adam_eps", "weight_decay", "eta_min", "epochs",
                 "batch_size", "val_fraction", "seed", "num_workers", "cutout", "cutout_length")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_evidence():
    paths = [ROOT / "train_cifar.py", *sorted((ROOT / "models").glob("*.py"))]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(
        p.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")).hexdigest() for p in paths}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"source_sha256": hashes, "git_commit": commit}


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


def atomic_checkpoint(path, value):
    _atomic(path, lambda p: torch.save(value, p))


def export_history(path, history):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=EPOCH_FIELDS)
    writer.writeheader()
    writer.writerows(history)
    _atomic(path, lambda p: p.write_text(buffer.getvalue(), encoding="utf-8", newline=""))


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


def data_evidence(loaders):
    return {name: dataset_fingerprint(loader.dataset) if loader is not None else None
            for name, loader in zip(("train", "validation", "test"), loaders)}


def capture_rng(loaders):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "generators": [loader.generator.get_state() if loader is not None and
                           loader.generator is not None else None for loader in loaders]}


def restore_rng(state, loaders):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count changed since the checkpoint")
        torch.cuda.set_rng_state_all(state["cuda"])
    for loader, generator_state in zip(loaders, state["generators"]):
        if generator_state is not None:
            if loader is None or loader.generator is None:
                raise ValueError("DataLoader generator missing during resume")
            loader.generator.set_state(generator_state)


def cpu_weights(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def write_completion(output_dir, metrics):
    names = ("config.json", "epochs.csv", "last.pt", "best_val.pt", "test_metrics.json",
             "confusion_matrix.csv", "test_predictions.csv")
    atomic_json(Path(output_dir) / "completion.json", {
        "status": "complete", "checkpoint_epoch": metrics["checkpoint_epoch"],
        "artifact_sha256": {name: file_sha256(Path(output_dir) / name) for name in names},
    })


def verify_artifact_hashes(output_dir):
    output_dir = Path(output_dir)
    completed = json.loads((output_dir / "completion.json").read_text(encoding="utf-8"))
    names = {"config.json", "epochs.csv", "last.pt", "best_val.pt", "test_metrics.json",
             "confusion_matrix.csv", "test_predictions.csv"}
    if completed.get("status") != "complete" or set(completed.get("artifact_sha256", {})) != names:
        raise ValueError("Missing completion evidence")
    for name, expected in completed["artifact_sha256"].items():
        if not (output_dir / name).is_file() or file_sha256(output_dir / name) != expected:
            raise ValueError(f"Artifact missing or changed: {name}")
    return completed
