"""Verify that C retains the teacher's TickNet code and meets the exam budget."""
import pytest
import torch
from torch import nn

from models.TickNet import FR_PDP_block, TickNet
from models.mid_models import build_mid_model
from models.model_profile import profile_model


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_c_uses_original_ticknet_backbone_and_fr_pdp_forward():
    model = build_mid_model("c", variant="Mid224")
    assert type(model) is TickNet
    assert model.forward.__func__ is TickNet.forward
    blocks = [block for name, stage in model.backbone.named_children()
              if name.startswith("stage") for block in stage.children()]
    assert len(blocks) == 9
    for block in blocks:
        assert type(block) is FR_PDP_block
        assert block.forward.__func__ is FR_PDP_block.forward
        assert block.Dw.conv.kernel_size == (3, 3)
    assert model.backbone.init_conv.conv.out_channels == 32
    assert model.final_conv_channels == 1024


@pytest.mark.parametrize("size", (32, 224))
def test_c_backward_uses_all_counted_parameters_at_native_resolution(size):
    torch.manual_seed(42)
    model = build_mid_model("c", variant=f"Mid{size}").train()
    logits = model(torch.randn(2, 3, size, size))
    assert logits.shape == (2, 5)
    loss = nn.functional.cross_entropy(logits, torch.tensor([0, 4]))
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("size", (32, 224))
def test_c_uses_parameter_budget_without_exceeding_flop_budget(size):
    result = profile_model(build_mid_model("c", variant=f"Mid{size}"), size, cross_check=True)
    assert 5_000_000 <= result["learnable_parameters"] <= 6_000_000
    assert result["flops"] <= 850_000_000
    assert result["pytorch_cross_check_flops"] == result["flops"]
    assert result["within_exam_limits"]
