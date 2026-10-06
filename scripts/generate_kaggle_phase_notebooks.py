#!/usr/bin/env python3
"""Generate 4 focused Kaggle notebooks for the Final Exam Grid Search."""
import json
import base64
import hashlib
import io
import zipfile
from pathlib import Path

phases = [
    {
        "filename": "Phase1_CIFAR10_SGD.ipynb",
        "title": "Phase 1: CIFAR-10 with SGD (lr=0.10, lr=0.15)",
        "dataset": "cifar10",
        "dataset_name": "CIFAR-10",
        "optimizer_name": "SGD",
        "runs": [
            ("configs/final/cifar10_sgd_lr010.json", "cifar10_sgd_lr010", "SGD lr=0.10 + Nesterov 0.9, Cutout 16x16 (Weight Decay 1e-4)"),
            ("configs/final/cifar10_sgd_lr015.json", "cifar10_sgd_lr015", "SGD lr=0.15 + Nesterov 0.9, Cutout 16x16 (Weight Decay 1e-4)"),
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
            ("configs/final/cifar10_adam_lr0001.json", "cifar10_adam_lr0001", "Adam lr=0.001, Cutout 16x16 (Betas 0.9/0.999, Weight Decay 1e-4)"),
            ("configs/final/cifar10_adam_lr00003.json", "cifar10_adam_lr00003", "Adam lr=0.0003, Cutout 16x16 (Betas 0.9/0.999, Weight Decay 1e-4)"),
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
            ("configs/final/cifar100_sgd_lr010.json", "cifar100_sgd_lr010", "SGD lr=0.10 + Nesterov 0.9, Cutout 16x16 (Weight Decay 1e-4)"),
            ("configs/final/cifar100_sgd_lr015.json", "cifar100_sgd_lr015", "SGD lr=0.15 + Nesterov 0.9, Cutout 16x16 (Weight Decay 1e-4)"),
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
            ("configs/final/cifar100_adam_lr0001.json", "cifar100_adam_lr0001", "Adam lr=0.001, Cutout 16x16 (Betas 0.9/0.999, Weight Decay 1e-4)"),
            ("configs/final/cifar100_adam_lr00003.json", "cifar100_adam_lr00003", "Adam lr=0.0003, Cutout 16x16 (Betas 0.9/0.999, Weight Decay 1e-4)"),
        ],
        "zip_name": "phase4_cifar100_adam_results.zip",
    },
]

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_TESTS = ["test_train_cifar.py", "test_cifar_data.py", "test_download_cifar.py",
                "test_cifar_recovery.py", "test_cifar_contracts.py",
                "test_cifar_download_integrity.py", "test_cifar_results.py"]


def build_source_bundle():
    paths = [ROOT / "train_cifar.py", ROOT / "download_cifar.py"]
    paths += sorted((ROOT / "models").glob("*.py"))
    paths += sorted((ROOT / "configs/final").glob("*.json"))
    paths += [ROOT / "scripts" / name for name in
              ("download_cifar.py", "verify_cifar_run.py", "aggregate_grid_search.py", "run_kaggle_phase.py")]
    paths += [ROOT / "tests" / name for name in ["conftest.py", *BUNDLE_TESTS]]
    buffer = io.BytesIO()
    manifest = {}
    # Stored entries avoid platform/zlib-dependent bytes in the embedded source snapshot.
    def entry(name):
        info = zipfile.ZipInfo(name, date_time=(2026, 10, 5, 0, 0, 0))
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        info.compress_type = zipfile.ZIP_STORED
        return info
    with zipfile.ZipFile(buffer, "w") as archive:
        for path in sorted(paths):
            name = path.relative_to(ROOT).as_posix()
            # Canonical newlines ensure source fingerprints survive Windows -> Kaggle.
            content = path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
            manifest[name] = hashlib.sha256(content).hexdigest()
            archive.writestr(entry(name), content)
        archive.writestr(entry("source_manifest.json"), json.dumps(manifest, sort_keys=True))
    return buffer.getvalue()


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def generate_notebooks(output_dir=None):
    output_dir = ROOT / "docs/kaggle" if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = build_source_bundle()
    encoded = base64.b64encode(bundle).decode("ascii")
    digest = hashlib.sha256(bundle).hexdigest()
    for phase_index, phase in enumerate(phases, 1):
        intro = f"""# Final Exam: {phase['title']}
Each notebook bundles audited source code and unit tests; no git cloning of changing branches required.
Select a GPU accelerator, enable Internet access to download CIFAR, and ensure dependencies (torch, torchvision, numpy, pandas, pillow, pytest) are installed.
Fixed configuration: 200 epochs, seed 42, 45,000 train / 5,000 validation split; checkpoints selected via validation metrics.
Official test evaluation occurs strictly after completing each experiment; never select learning rates using test set outcomes.

If splitting execution across multiple sessions, set EPOCHS_PER_SESSION (e.g. 50); download recovery.zip before the session terminates.
Next session: extract recovery.zip into a Kaggle Dataset, attach the dataset, and set RESUME_ROOT to the path containing run folders.
Resuming requires matching torch/torchvision/NumPy/Pillow versions, identical GPU architecture, source code, and configurations.
Results are packaged as results.zip only after both runs complete all 200 epochs and pass full artifact validation.
"""
        bootstrap = f"""import base64, hashlib, io, json, os, sys, zipfile
from pathlib import Path
import subprocess, importlib.util

BUNDLE_SHA256 = {digest!r}
SOURCE_BUNDLE = {encoded!r}
raw = base64.b64decode(SOURCE_BUNDLE)
assert hashlib.sha256(raw).hexdigest() == BUNDLE_SHA256
PROJECT = Path('/kaggle/working') / ('TickNets-' + BUNDLE_SHA256[:16])
PROJECT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(raw)) as archive:
    manifest = json.loads(archive.read('source_manifest.json'))
    for name, expected in manifest.items():
        payload = archive.read(name)
        assert hashlib.sha256(payload).hexdigest() == expected
        target = (PROJECT / name).resolve()
        assert target.is_relative_to(PROJECT.resolve())
        if target.exists() and target.read_bytes() != payload:
            raise RuntimeError('Extracted source was modified: ' + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    (PROJECT / 'source_manifest.json').write_bytes(archive.read('source_manifest.json'))
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
# Install only missing test/report dependencies; preserve Kaggle's torch/CUDA build.
requirements = {{'pytest': 'pytest>=8,<10', 'pandas': 'pandas', 'numpy': 'numpy', 'PIL': 'pillow'}}
missing = [package for module, package in requirements.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.run([sys.executable, '-m', 'pip', 'install', *missing], check=True)
import torch, torchvision
from torch.utils.flop_counter import FlopCounterMode
assert torch.cuda.is_available(), 'Select a Kaggle GPU accelerator before running.'
print('Source bundle:', BUNDLE_SHA256)
print('Runtime:', torch.__version__, torchvision.__version__, torch.cuda.get_device_name(0))
"""
        preflight = """# Validate the embedded code before spending GPU time.
environment = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
subprocess.run([sys.executable, '-m', 'pytest', *""" + repr(['tests/' + n for n in BUNDLE_TESTS]) + """,
                '-q', '-p', 'no:cacheprovider'], cwd=PROJECT, env=environment, check=True)
"""
        settings = """# None: run all remaining epochs. Integer: epochs per experiment for this session.
EPOCHS_PER_SESSION = None
# Path to the attached recovery dataset containing the experiment folders (not the zip itself).
RESUME_ROOT = None
"""
        launch = f"""command = [sys.executable, 'scripts/run_kaggle_phase.py', '--phase', '{phase_index}']
if EPOCHS_PER_SESSION is not None:
    command += ['--epochs-per-session', str(EPOCHS_PER_SESSION)]
if RESUME_ROOT is not None:
    command += ['--resume-root', str(RESUME_ROOT)]
subprocess.run(command, cwd=PROJECT, check=True)
"""
        notebook = {"nbformat": 4, "nbformat_minor": 4,
                    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                 "ticknets_source_sha256": digest, "ticknets_phase": phase_index},
                    "cells": [cell("markdown", intro), cell("code", bootstrap), cell("code", preflight),
                              cell("code", settings), cell("code", launch)]}
        path = output_dir / phase["filename"]
        path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Created {path} (source {digest[:16]})")
    return digest


if __name__ == "__main__":
    generate_notebooks()
