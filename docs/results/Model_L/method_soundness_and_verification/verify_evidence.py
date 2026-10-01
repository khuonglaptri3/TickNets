"""Read-only model/data audit; writes evidence beside this script, never trains.

Run from any directory with the project's PyTorch environment.
"""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
import torchvision
from models.mid_data import build_mid_loaders
from models.mid_models import build_mid_model
from models.model_profile import profile_model


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(correct, n):
    z = 1.959963984540054
    p = correct / n
    center = (p + z*z / (2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    return [100*(center-half), 100*(center+half)]


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    manifest = read_csv(ROOT / "data/split_manifest.csv")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=("variant", "split", "class_name", "filename"))
    writer.writeheader()
    for variant in ("Mid32", "Mid224"):
        for split in ("train", "test"):
            for cls in ("bird", "cat", "dog", "frog", "horse"):
                names = sorted(Path(r[f"{variant.lower()}_path"]).name for r in manifest
                               if r["split"] == split and r["class_name"] == cls)
                actual = sorted(p.name for p in (ROOT / "data" / variant / split / cls).glob("*.jpeg"))
                assert names == actual, (variant, split, cls)
                for name in names:
                    writer.writerow(dict(variant=variant, split=split, class_name=cls, filename=name))
    reconstructed_hash = hashlib.sha256(stream.getvalue().encode("utf-8")).hexdigest()
    byte_mismatches = []
    for row in manifest:
        for prefix in ("mid32", "mid224"):
            if digest(ROOT / "data" / row[f"{prefix}_path"]) != row[f"{prefix}_sha256"]:
                byte_mismatches.append(row[f"{prefix}_path"])
    assert not byte_mismatches, byte_mismatches[:5]
    result = {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
        "device": "cpu", "evaluation_batch_size": 16,
        "canonical_manifest_sha256": digest(ROOT / "data/split_manifest.csv"),
        "canonical_membership_in_l_notebook_schema_sha256": reconstructed_hash,
        "local_image_hashes_verified": 2*len(manifest),
        "scope": "Checkpoint inference on local canonical test; no retraining; historical Kaggle image bytes unavailable.",
        "runs": [], "paired_tests": [],
    }
    predictions = {}
    reports = {"basic": "Model_Basic", "l": "Model_L"}
    for record in read_csv(ROOT / "docs/results/checkpoints/checkpoint_manifest.csv"):
        checkpoint_path = ROOT / record["relative_path"]
        assert digest(checkpoint_path) == record["sha256"]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config = json.loads((checkpoint_path.parent / "config.json").read_text(encoding="utf-8"))
        assert checkpoint["config"] == config
        assert checkpoint["epoch"] == 200
        if config["model"] == "l":
            assert config["split_manifest_sha256"] == reconstructed_hash
        else:
            assert config["split_manifest_sha256"] == result["canonical_manifest_sha256"]
        model = build_mid_model(config["model"], variant=config["variant"]).eval()
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        assert all(torch.isfinite(v).all() for v in model.state_dict().values())
        size = 32 if config["variant"] == "Mid32" else 224
        profile = profile_model(model, size, cross_check=True)
        assert profile["flops"] == config["complexity"]["flops"]
        assert profile["learnable_parameters"] == config["learnable_parameters"]
        _, loader = build_mid_loaders(ROOT / "data", config["variant"], batch_size=16, augment=False)
        assert loader.dataset.class_to_idx == checkpoint["class_to_idx"]
        labels, preds, loss_sum = [], [], 0.0
        with torch.inference_mode():
            for images, target in loader:
                logits = model(images)
                loss_sum += torch.nn.functional.cross_entropy(logits, target, reduction="sum").item()
                labels.extend(target.tolist())
                preds.extend(logits.argmax(1).tolist())
        cm = [[0]*5 for _ in range(5)]
        for a, b in zip(labels, preds):
            cm[a][b] += 1
        correct = sum(cm[i][i] for i in range(5))
        macro_f1 = sum(2*cm[i][i] / (sum(cm[i])+sum(row[i] for row in cm)) for i in range(5))/5
        assets = ROOT / "docs/results" / reports[config["model"]] / "midterm_report_assets"
        with (assets / f"{config['variant'].lower()}_confusion_matrix.csv").open() as handle:
            reported_cm = [[int(x) for x in row] for row in csv.reader(handle)]
        history = read_csv(checkpoint_path.parent / "epochs.csv")
        assert [int(r["epoch"]) for r in history] == list(range(1, 201))
        lr_error = max(abs(float(r["learning_rate"]) - .1*(1+math.cos(math.pi*(int(r["epoch"])-1)/200))/2) for r in history)
        metrics = json.loads((checkpoint_path.parent / "test_metrics.json").read_text())
        item = {"model": config["model"], "variant": config["variant"],
                "checkpoint_sha256_verified": True, "state_dict_strict_load": True,
                "checkpoint_keys": sorted(checkpoint),
                "epoch": checkpoint["epoch"], "scheduler_last_epoch": checkpoint["scheduler_state_dict"]["last_epoch"],
                "scheduler_T_max": checkpoint["scheduler_state_dict"]["T_max"],
                "epochs_csv_rows": len(history), "max_cosine_lr_error": lr_error,
                "final_train_loss": float(history[-1]["train_loss"]),
                "final_train_top1": float(history[-1]["train_top1"]),
                "reported_test": metrics, "reevaluated_top1": 100*correct/len(labels),
                "reevaluated_loss": loss_sum/len(labels), "reevaluated_macro_f1": macro_f1,
                "test_correct": correct, "test_samples": len(labels),
                "wilson_95_percent": wilson(correct, len(labels)),
                "confusion_matrix": cm, "reported_confusion_matrix_exact_match": cm == reported_cm,
                "learnable_parameters": profile["learnable_parameters"],
                "flops": profile["flops"], "independent_operator_flops": profile["pytorch_cross_check_flops"],
                "stage_flops": profile["stage_flops"],
                "batch_size": config["batch_size"], "num_workers": config["num_workers"]}
        predictions[(config["model"], config["variant"])] = [a == b for a,b in zip(labels,preds)]
        pred_path = OUT / f"{config['model']}_{config['variant'].lower()}_predictions.csv"
        with pred_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("path", "target", "prediction"))
            for (path, _), target, pred in zip(loader.dataset.samples, labels, preds):
                writer.writerow((Path(path).relative_to(ROOT / "data").as_posix(), target, pred))
        result["runs"].append(item)
        print(json.dumps(item), flush=True)
    for variant in ("Mid32", "Mid224"):
        for first, second in [("basic", "l")]:
            a, b = predictions[(first,variant)], predictions[(second,variant)]
            losses = sum(x and not y for x,y in zip(a,b))
            gains = sum(y and not x for x,y in zip(a,b))
            n = losses+gains
            p = min(1.0, 2*sum(math.comb(n,k) for k in range(min(losses,gains)+1))/2**n) if n else 1.0
            result["paired_tests"].append(dict(variant=variant, first=first, second=second,
                first_only_correct=losses, second_only_correct=gains, mcnemar_exact_two_sided_p=p))
    (OUT / "verification.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result["paired_tests"], indent=2), flush=True)


if __name__ == "__main__":
    main()
