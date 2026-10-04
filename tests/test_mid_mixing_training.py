"""Mixed training loss, explicit clean evaluation, and recipe resume."""
import importlib
import json

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from test_mid_dataset import source_dataset
from test_mid_pipeline import prepared
from test_mid_experiment_training import arguments

@pytest.mark.parametrize("method", ["mixup", "cutmix"])
def test_branch_registers_its_mixing_method(method):
    module = importlib.import_module("train_mid_experiment")
    assert method in module.MIXING_METHODS


def test_epoch_uses_weighted_loss_and_never_mixes_validation():
    module = importlib.import_module("train_mid_experiment")
    inputs = torch.tensor([[2., 0.], [0., 3.], [1., 2.]])
    labels = torch.tensor([0, 1, 0])
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=3)
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    calls = []
    def mix(images, target, alpha):
        calls.append(True)
        return images, target, target.flip(0), 0.25
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    criterion = torch.nn.CrossEntropyLoss()
    result = module.run_epoch(model, loader, criterion, torch.device("cpu"), optimizer, mix_fn=mix)
    expected = 0.25 * criterion(inputs, labels) + 0.75 * criterion(inputs, labels.flip(0))
    assert result["loss"] == pytest.approx(expected.item())
    calls.clear()
    validation = module.run_epoch(model, loader, criterion, torch.device("cpu"), mix_fn=mix)
    assert not calls
    assert validation["loss"] == pytest.approx(criterion(inputs, labels).item())
    module.run_epoch(model, loader, criterion, torch.device("cpu"), optimizer, mix_fn=mix, probability=0)
    assert not calls


@pytest.mark.parametrize("method", ["mixup", "cutmix"])
def test_mixed_run_and_epoch_boundary_resume(prepared, tmp_path, method):
    module = importlib.import_module("train_mid_experiment")
    full, resumed = tmp_path / f"full_{method}", tmp_path / f"resumed_{method}"
    flags = ["--mixing", method, "--mixing-alpha", "0.2", "--mixing-probability", "0.5"]
    module.main(arguments(prepared, full, *flags))
    module.main(arguments(prepared, resumed, *flags, "--stop-after-epoch", "1"))
    module.main(arguments(prepared, resumed, *flags, "--resume", str(resumed / "last.pt")))
    a, b = (torch.load(path / "last.pt", weights_only=True) for path in (full, resumed))
    for name in a["model_state_dict"]:
        assert torch.equal(a["model_state_dict"][name], b["model_state_dict"][name]), name
    assert (full / "epochs.csv").read_bytes() == (resumed / "epochs.csv").read_bytes()
    assert json.loads((full / "config.json").read_text())["mixing"] == method
    assert not (full / "test_metrics.json").exists()
