"""Count Conv2d/Linear MACs for one evaluation forward pass, with explicit scope."""
import torch
from torch import nn


COUNTING_CONVENTION = "1 MAC = 2 FLOPs; Conv2d and Linear only; batch size 1; eval forward"
EXCLUDED_OPERATIONS = (
    "bias additions", "batch normalization", "activations", "pooling",
    "residual additions", "SE elementwise scaling", "data movement",
)


def profile_model(model, input_size, *, cross_check=False):
    """Profile without changing weights, running statistics, or training flags.

The optional independent check uses PyTorch's operator-level FlopCounterMode.
FLOPs here are an operation-count convention, not measured device latency.
"""
    if input_size not in (32, 224):
        raise ValueError("input_size must be 32 or 224")
    parameter = next(model.parameters())
    sample = torch.zeros(1, 3, input_size, input_size,
                         device=parameter.device, dtype=parameter.dtype)
    flags = {module: module.training for module in model.modules()}
    layers = []
    handles = []

    def record(name):
        def hook(layer, inputs, output):
            if isinstance(layer, nn.Conv2d):
                products = (layer.in_channels // layer.groups) * layer.kernel_size[0] * layer.kernel_size[1]
            else:
                products = layer.in_features
            macs = output.numel() * products
            layers.append({"name": name, "type": type(layer).__name__,
                           "input_shape": list(inputs[0].shape), "output_shape": list(output.shape),
                           "macs": macs, "flops": 2 * macs})
        return hook

    operator_flops = None
    try:
        model.eval()
        for name, layer in model.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                handles.append(layer.register_forward_hook(record(name)))
        with torch.inference_mode():
            output = model(sample)
        for handle in handles:
            handle.remove()
        handles.clear()
        if cross_check:
            from torch.utils.flop_counter import FlopCounterMode
            with torch.inference_mode(), FlopCounterMode(display=False) as counter:
                model(sample)
            operator_flops = counter.get_total_flops()
            if operator_flops != sum(layer["flops"] for layer in layers):
                raise RuntimeError("Module MAC count disagrees with PyTorch operator FLOP count")
    finally:
        for handle in handles:
            handle.remove()
        for module, training in flags.items():
            module.training = training

    stages = {}
    for layer in layers:
        stage = ".".join(layer["name"].split(".")[:2])
        stages[stage] = stages.get(stage, 0) + layer["flops"]
    learnable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    macs = sum(layer["macs"] for layer in layers)
    return {
        "input_shape": list(sample.shape), "output_shape": list(output.shape),
        "learnable_parameters": learnable_parameters,
        "macs": macs, "flops": 2 * macs,
        "counting_convention": COUNTING_CONVENTION,
        "excluded_operations": list(EXCLUDED_OPERATIONS),
        "within_exam_limits": learnable_parameters <= 6_000_000 and 2 * macs < 1_000_000_000,
        "pytorch_cross_check_flops": operator_flops,
        "stage_flops": stages, "layers": layers,
    }
