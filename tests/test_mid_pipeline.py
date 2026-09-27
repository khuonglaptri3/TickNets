"""Exercise the project loader and training entry point on real image fixtures."""
import csv
import importlib
import json

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from test_mid_dataset import source_dataset


@pytest.fixture
def prepared(source_dataset, tmp_path):
    prepare = importlib.import_module("prepare_mid_dataset").prepare_dataset
    root = tmp_path / "data"
    prepare(source_dataset, root, seed=42, train_per_class=4, test_per_class=2, workers=2)
    return root


def test_loader_keeps_native_resolution_classes_and_matching_sample_order(prepared):
    module = importlib.import_module("models.mid_data")
    orders = []
    for variant, size in (("Mid32", 32), ("Mid224", 224)):
        train, test = module.build_mid_loaders(prepared, variant, batch_size=4, seed=42, augment=False)
        assert train.dataset.class_to_idx == {"bird": 0, "cat": 1, "dog": 2, "frog": 3, "horse": 4}
        assert train.dataset.class_to_idx == test.dataset.class_to_idx
        assert len(train.dataset) == 20
        assert len(test.dataset) == 10
        orders.append(list(iter(train.sampler)))
        images, labels = next(iter(train))
        assert images.shape == (4, 3, size, size)
        assert images.dtype == torch.float32
        assert images.min() >= 0 and images.max() <= 1
        assert labels.min() >= 0 and labels.max() <= 4
    assert orders[0] == orders[1]


def test_loader_seed_controls_shuffle_and_test_is_sequential(prepared):
    module = importlib.import_module("models.mid_data")
    orders = []
    for seed in (42, 42, 43):
        train, test = module.build_mid_loaders(prepared, "Mid32", seed=seed, augment=False)
        orders.append(list(iter(train.sampler)))
        assert list(iter(test.sampler)) == list(range(10))
    assert orders[0] == orders[1]
    assert orders[0] != orders[2]


def test_loader_rejects_class_mapping_drift(prepared):
    directory = prepared / "Mid32" / "test" / "bird"
    for path in directory.iterdir():
        path.unlink()
    directory.rmdir()
    module = importlib.import_module("models.mid_data")
    with pytest.raises(ValueError, match="class"):
        module.build_mid_loaders(prepared, "Mid32")


def test_epoch_metrics_weight_partial_batches_by_sample_count():
    module = importlib.import_module("train_mid")
    logits = torch.tensor([[8., 0.], [8., 0.], [8., 0.]])
    labels = torch.tensor([0, 0, 1])
    loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
    result = module.run_epoch(torch.nn.Identity(), loader, torch.nn.CrossEntropyLoss(), torch.device("cpu"))
    assert result["samples"] == 3
    assert result["top1"] == pytest.approx(200 / 3)
    assert result["loss"] == pytest.approx(torch.nn.functional.cross_entropy(logits, labels).item())


def test_training_writes_epoch_log_checkpoint_and_final_test_metrics(prepared, tmp_path):
    module = importlib.import_module("train_mid")
    output = tmp_path / "run"
    module.main(["--data-root", str(prepared), "--variant", "Mid32", "--output-dir", str(output),
                 "--epochs", "1", "--batch-size", "4", "--device", "cpu", "--threads", "2", "--no-augment"])
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["seed"] == 42 and config["num_classes"] == 5
    with (output / "epochs.csv").open(encoding="utf-8", newline="") as handle:
        epochs = list(csv.DictReader(handle))
    assert len(epochs) == 1 and int(epochs[0]["train_samples"]) == 20
    assert "test_top1" not in epochs[0]
    checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == 1
    assert checkpoint["class_to_idx"] == {"bird": 0, "cat": 1, "dog": 2, "frog": 3, "horse": 4}
    results = json.loads((output / "test_metrics.json").read_text(encoding="utf-8"))
    assert results["samples"] == 10 and 0 <= results["top1"] <= 100
    evaluated = module.main(["--data-root", str(prepared), "--variant", "Mid32", "--device", "cpu",
                             "--evaluate", str(output / "last.pt"), "--output-dir", str(tmp_path / "matching_eval")])
    assert evaluated["samples"] == 10
    assert evaluated["top1"] == results["top1"]
    assert evaluated["loss"] == pytest.approx(results["loss"], rel=1e-5)


@pytest.mark.parametrize("recorded_hash", ["another-split-hash", None])
def test_evaluation_rejects_changed_or_missing_split_provenance(prepared, tmp_path, recorded_hash):
    module = importlib.import_module("train_mid")
    model = module.build_TickNet(num_classes=5, typesize="basic", cifar=True)
    checkpoint = tmp_path / "from-another-split.pt"
    config = {"variant": "Mid32", "model": "basic", "split_manifest_sha256": recorded_hash}
    torch.save({"epoch": 1, "model_state_dict": model.state_dict(), "config": config,
                "class_to_idx": {"bird": 0, "cat": 1, "dog": 2, "frog": 3, "horse": 4}}, checkpoint)
    with pytest.raises(ValueError, match="manifest"):
        module.main(["--data-root", str(prepared), "--variant", "Mid32", "--evaluate", str(checkpoint),
                     "--output-dir", str(tmp_path / "evaluation"), "--device", "cpu"])
    assert not (tmp_path / "evaluation").exists()
