# -*- coding: utf-8 -*-
"""Script to inspect and compare architectures, parameters, and FLOPs for TickNet models.

Supports both:
1. Author original TickNet (TickNet-Basic)
2. Proposed TickNet-L v1
"""
import argparse
import torch

from models.TickNet import build_TickNet
from models.ticknet_l import build_ticknet_l
from models.model_profile import profile_model

try:
    from torchsummary import summary as torch_summary
except ImportError:
    torch_summary = None


def pytorch_summary_by_stage(model, input_size=(1, 3, 32, 32), name="Model"):
    """Detailed breakdown of parameters and FLOPs stage-by-stage using pure PyTorch hooks."""
    device = next(model.parameters()).device
    x = torch.zeros(input_size, device=device)
    
    stages = []
    if hasattr(model, "backbone"):
        for sub_name, module in model.backbone.named_children():
            stages.append((f"backbone.{sub_name}", module))
    if hasattr(model, "classifier"):
        stages.append(("classifier", model.classifier))
        
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    profile = profile_model(model.cpu(), input_size[2], cross_check=True)
    total_flops = profile["flops"]
    stage_flops_map = profile["stage_flops"]
    
    stage_outputs = {}
    handles = []
    for s_name, module in stages:
        def make_hook(n):
            def hook(mod, inp, out):
                if isinstance(out, torch.Tensor):
                    stage_outputs[n] = tuple(out.shape)
                else:
                    stage_outputs[n] = "N/A"
            return hook
        handles.append(module.register_forward_hook(make_hook(s_name)))
        
    model.eval()
    with torch.no_grad():
        _ = model.to(device)(x)
        
    for h in handles:
        h.remove()
        
    print("\n[Parameter and FLOP Breakdown by Stage (PyTorch Summary)]:")
    header = f"{'Stage / Submodule':<24} | {'Output Shape':<18} | {'Params':>10} | {'Param %':>8} | {'FLOPs':>12} | {'FLOP %':>8}"
    print("-" * 88)
    print(header)
    print("-" * 88)
    
    for s_name, module in stages:
        p_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
        p_ratio = (p_count / total_params * 100) if total_params > 0 else 0
        shape_str = str(stage_outputs.get(s_name, "N/A"))
        s_flops = stage_flops_map.get(s_name, 0)
        f_ratio = (s_flops / total_flops * 100) if total_flops > 0 else 0
        print(f"{s_name:<24} | {shape_str:<18} | {p_count:>10,} | {p_ratio:>7.2f}% | {s_flops:>12,} | {f_ratio:>7.2f}%")
        
    print("-" * 88)
    print(f"{'TOTAL':<24} | {str(tuple(x.shape)) + ' -> out':<18} | {total_params:>10,} | {100.0:>7.2f}% | {total_flops:>12,} | {100.0:>7.2f}%")
    print("-" * 88)


def inspect_model(name, model, input_size=32, show_layers=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print("=" * 88)
    print(f"  MODEL INSPECTION: {name}")
    print("=" * 88)
    
    if show_layers:
        print("\n[Layer Hierarchy Details]:")
        print(model)
    
    # Accurate parameter and FLOP profiling
    model_cpu = model.cpu()
    profile = profile_model(model_cpu, input_size, cross_check=True)
    params = profile["learnable_parameters"]
    flops = profile["flops"]
    gflops = flops / 1e9
    pass_limits = profile["within_exam_limits"]
    
    print("\n[Examination Budget Compliance]:")
    print(f"  - Device                : {device}")
    print(f"  - Input Tensor Shape    : (1, 3, {input_size}, {input_size})")
    print(f"  - Learnable Parameters  : {params:,} (Budget <= 6,000,000: {'PASSED' if params <= 6_000_000 else 'EXCEEDED'})")
    print(f"  - Forward FLOPs         : {gflops:.4f} GFLOPs ({flops:,} FLOPs) (Budget < 1.0G: {'PASSED' if gflops < 1.0 else 'EXCEEDED'})")
    print(f"  - Compliance Status     : {'PASSED' if pass_limits else 'FAILED'}")

    # Print breakdown by stage
    pytorch_summary_by_stage(model, input_size=(1, 3, input_size, input_size), name=name)

    # Print summary if torchsummary is available
    if torch_summary is not None and torch.cuda.is_available():
        print("\n[torchsummary Table]:")
        try:
            torch_summary(model.to(device), (3, input_size, input_size))
        except Exception as e:
            print(f"  (Unable to export torchsummary: {e})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Inspect and profile TickNet model architectures")
    parser.add_argument("--model", choices=("both", "basic", "l"), default="both",
                        help="Choose model: 'basic' (Author), 'l' (Proposed), or 'both'")
    parser.add_argument("--classes", type=int, default=10,
                        help="Number of target classes (default: 10 for CIFAR-10, 100 for CIFAR-100)")
    parser.add_argument("--show-layers", action="store_true",
                        help="Print detailed hierarchical network layer structure")
    args = parser.parse_args()

    models_to_check = []
    if args.model in ("both", "basic"):
        models_to_check.append((
            f"TickNet-Basic (Author Baseline) - CIFAR-{args.classes}",
            build_TickNet(num_classes=args.classes, typesize="basic", cifar=True)
        ))
    if args.model in ("both", "l"):
        models_to_check.append((
            f"TickNet-L v1 (Proposed) - CIFAR-{args.classes}",
            build_ticknet_l(num_classes=args.classes, cifar=True)
        ))

    for name, model in models_to_check:
        inspect_model(name, model, input_size=32, show_layers=args.show_layers)


if __name__ == "__main__":
    main()