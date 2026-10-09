"""Execute frozen phase recipes and export verified results or recovery checkpoints."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.cifar_experiment import atomic_json, file_sha256
from scripts.aggregate_grid_search import aggregate_results
from scripts.verify_cifar_run import verify_run

PHASES = {
    1: ("cifar10_sgd_lr010", "cifar10_sgd_lr015"),
    2: ("cifar10_adam_lr0001", "cifar10_adam_lr00003"),
    3: ("cifar100_sgd_lr010", "cifar100_sgd_lr015"),
    4: ("cifar100_adam_lr0001", "cifar100_adam_lr00003"),
}


def phase_recipe(name):
    config = json.loads((ROOT / "configs" / "final" / (name + ".json")).read_text(encoding="utf-8"))
    config.setdefault("model", "l")
    config.update(adam_eps=1e-8, eta_min=0.0)
    return config


def package_phase(phase, runs_dir, output_dir, *, official=True):
    """A partial run is always called recovery, never a completed results bundle."""
    runs_dir, output_dir = Path(runs_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    complete = all((runs_dir / name / "completion.json").is_file() for name in PHASES[phase])
    verified = []
    for name in PHASES[phase]:
        if (runs_dir / name / "completion.json").is_file():
            verify_run(runs_dir / name, official=official, expected_config=phase_recipe(name))
            verified.append(name)
    dataset, optimizer = PHASES[phase][0].split("_")[:2]
    stem = f"phase{phase}_{dataset}_{optimizer}_" + ("results" if complete else "recovery")
    if complete:
        aggregate_results(runs_dir, output_dir / (stem + ".csv"), output_dir / (stem + ".md"),
                          expected_experiments=PHASES[phase], official=official)
    files = {f"{name}/{p.relative_to(runs_dir / name).as_posix()}": p
             for name in PHASES[phase] if (runs_dir / name).is_dir()
             for p in (runs_dir / name).rglob("*") if p.is_file() and p.suffix != ".tmp"}
    manifest = {"phase": phase, "status": "complete" if complete else "paused",
                "expected_experiments": list(PHASES[phase]), "verified_experiments": verified,
                "artifact_sha256": {name: file_sha256(path) for name, path in sorted(files.items())}}
    target = output_dir / (stem + ".zip")
    temporary = target.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
        archive.writestr("phase_manifest.json", json.dumps(manifest, indent=2))
        if complete:
            for extension in ("csv", "md"):
                archive.write(output_dir / (stem + "." + extension), "summary." + extension)
    temporary.replace(target)
    atomic_json(output_dir / (stem + "_manifest.json"), manifest)
    return target, complete


def restore_run(source, target):
    source, target = Path(source), Path(target)
    if not (source / "last.pt").is_file():
        raise ValueError(f"Recovery folder has no last.pt: {source}")
    if target.exists():
        raise FileExistsError(f"Recovery destination already exists: {target}")
    shutil.copytree(source, target)


def run_phase(phase, runs_dir, data_root, output_dir, *, epochs_per_session=None,
              resume_root=None, device="cuda:0"):
    import torch
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU is unavailable; select a GPU accelerator")
    if epochs_per_session is not None and epochs_per_session < 1:
        raise ValueError("epochs-per-session must be positive")
    runs_dir, data_root = Path(runs_dir), Path(data_root)
    dataset = PHASES[phase][0].split("_")[0]
    # subprocess.check=True ensures notebook execution stops on any failure.
    subprocess.run([sys.executable, str(ROOT / "download_cifar.py"), "--dataset", dataset,
                    "--data-root", str(data_root)], cwd=ROOT, check=True)
    for name in PHASES[phase]:
        target = runs_dir / name
        if not target.exists() and resume_root and (Path(resume_root) / name).exists():
            restore_run(Path(resume_root) / name, target)
        cfg = phase_recipe(name)
        command = [sys.executable, str(ROOT / "train_cifar.py"), "--config",
                   str(ROOT / "configs" / "final" / (name + ".json")), "--data-root", str(data_root),
                   "--output-dir", str(target), "--device", device, "--no-download"]
        saved_epoch = 0
        if (target / "last.pt").is_file():
            saved = torch.load(target / "last.pt", map_location="cpu", weights_only=False)
            saved_epoch = saved["epoch"]
            for key, value in cfg.items():
                if saved["config"].get(key) != value:
                    raise ValueError(f"Recovery recipe differs: {name}/{key}")
            command += ["--resume", str(target / "last.pt")]
        elif target.exists():
            raise ValueError(f"Incomplete folder without recovery checkpoint: {target}")
        if epochs_per_session and saved_epoch < cfg["epochs"]:
            command += ["--stop-after-epoch", str(min(cfg["epochs"], saved_epoch + epochs_per_session))]
        try:
            subprocess.run(command, cwd=ROOT, check=True)
        except BaseException:
            package_phase(phase, runs_dir, output_dir)
            raise
    target, complete = package_phase(phase, runs_dir, output_dir)
    print(f"{'Verified results' if complete else 'Recovery checkpoints'}: {target}", flush=True)
    return target, complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, choices=PHASES, required=True)
    parser.add_argument("--runs-dir", type=Path, default=Path("/kaggle/working/runs"))
    parser.add_argument("--data-root", type=Path, default=Path("/kaggle/working/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--epochs-per-session", type=int)
    parser.add_argument("--resume-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run_phase(**vars(args))


if __name__ == "__main__":
    main()
