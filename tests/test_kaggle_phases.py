"""Four notebooks must be executable Python, carry current source, and stop on errors."""
import ast
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from scripts.generate_kaggle_phase_notebooks import ROOT, build_source_bundle, generate_notebooks, phases
from scripts.run_kaggle_phase import PHASES, phase_recipe, restore_run, run_phase


def test_notebooks_contain_current_source_and_fail_fast_commands(tmp_path):
    digest = generate_notebooks(tmp_path)
    for i, phase in enumerate(phases, 1):
        notebook = json.loads((tmp_path / phase["filename"]).read_text(encoding="utf-8"))
        assert notebook["metadata"]["ticknets_phase"] == i
        assert notebook["metadata"]["ticknets_source_sha256"] == digest
        source = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        for code in source:
            compile(code, phase["filename"], "exec")
        assignments = {node.targets[0].id: ast.literal_eval(node.value)
                       for node in ast.parse(source[0]).body
                       if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                       and node.targets[0].id in ("SOURCE_BUNDLE", "BUNDLE_SHA256")}
        payload = base64.b64decode(assignments["SOURCE_BUNDLE"])
        assert hashlib.sha256(payload).hexdigest() == assignments["BUNDLE_SHA256"] == digest
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            manifest = json.loads(archive.read("source_manifest.json"))
            for name, expected in manifest.items():
                raw = archive.read(name)
                assert hashlib.sha256(raw).hexdigest() == expected
                assert raw == (ROOT / name).read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
                if name.endswith(".py"):
                    compile(raw, name, "exec")
        assert "git clone" not in "\n".join(source)
        assert "check=True" in source[1] and "check=True" in source[-1]
        assert "CUDA_VISIBLE_DEVICES='-1'" in source[1]
        assert "Each notebook bundles audited source code" in "".join(notebook["cells"][0]["source"])
        assert "'--phase', '" + str(i) + "'" in source[-1]


def test_source_bundle_is_byte_identical_across_regeneration():
    assert build_source_bundle() == build_source_bundle()


def test_bootstrap_extracts_verified_source_and_rejects_modified_code(tmp_path, monkeypatch):
    import sys
    import torch
    generate_notebooks(tmp_path / "notebooks")
    notebook = json.loads((tmp_path / "notebooks" / phases[0]["filename"]).read_text(encoding="utf-8"))
    bootstrap = "".join(notebook["cells"][1]["source"])
    bootstrap = bootstrap.replace("Path('/kaggle/working')", "Path(" + repr(str(tmp_path / "working")) + ")")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "test GPU")
    scope = {}
    exec(compile(bootstrap, "bootstrap", "exec"), scope)
    project = scope["PROJECT"]
    assert (project / "train_cifar.py").is_file()
    assert (project / "source_manifest.json").is_file()
    (project / "train_cifar.py").write_text("modified")
    with pytest.raises(RuntimeError, match="modified"):
        exec(compile(bootstrap, "bootstrap", "exec"), {})


@pytest.mark.parametrize("phase", (1, 2, 3, 4))
def test_frozen_phase_matrix_matches_exam_protocol(phase):
    configs = [phase_recipe(name) for name in PHASES[phase]]
    expected_dataset = "cifar10" if phase <= 2 else "cifar100"
    expected_optimizer = "sgd" if phase in (1, 3) else "adam"
    assert all(c["dataset"] == expected_dataset and c["optimizer"] == expected_optimizer for c in configs)
    assert {c["learning_rate"] for c in configs} == ({0.1, 0.15} if expected_optimizer == "sgd" else {0.001, 0.0003})
    assert all(c["model"] == "l" and c["epochs"] == 200 and c["seed"] == 42 and
               c["val_fraction"] == 0.1 and c["cutout"] for c in configs)


def test_runner_stops_if_download_subprocess_fails(monkeypatch, tmp_path):
    import subprocess
    calls = []
    def failed(command, **kwargs):
        calls.append(command)
        assert kwargs["check"] is True
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(subprocess, "run", failed)
    with pytest.raises(subprocess.CalledProcessError):
        run_phase(1, tmp_path / "runs", tmp_path / "data", tmp_path, device="cpu")
    assert len(calls) == 1 and "download_cifar.py" in calls[0][1]


def test_recovery_copy_preserves_source_and_rejects_existing_destination(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    (source / "last.pt").write_bytes(b"checkpoint")
    restore_run(source, destination)
    assert (source / "last.pt").read_bytes() == (destination / "last.pt").read_bytes()
    with pytest.raises(FileExistsError):
        restore_run(source, destination)
