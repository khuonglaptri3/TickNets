"""Quick, dataset-free check of the project's current final-exam Model L.

Put this file beside train_cifar.py, then run:
    python checkmodel_l.py
    python checkmodel_l.py --num-classes 100
    python checkmodel_l.py --details

Imports the real model; does not copy its architecture, download data, load
checkpoints, save files, or update weights. Only PyTorch is required.
"""
import argparse
from pathlib import Path
import sys
import time
import traceback

# Keep this check from creating __pycache__ files in the project.
sys.dont_write_bytecode = True


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-classes", type=int, default=10,
                        help="Classifier outputs: 10 for CIFAR-10, 100 for CIFAR-100")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                        help="Default: cpu; CUDA is optional")
    parser.add_argument("--details", action="store_true",
                        help="Also print every executed Conv2d/Linear with shapes and FLOPs")
    args = parser.parse_args()
    if args.num_classes < 1:
        parser.error("--num-classes must be positive")
    return args


def check_model(args):
    # Resolve models from the directory containing this file, even if the
    # command was launched from a different working directory.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    from torch import nn
    from models.ticknet_l import build_ticknet_l
    from models.model_profile import profile_model

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run again without --device cuda.")
    if args.device == "cpu":
        torch.set_num_threads(2)
    torch.manual_seed(42)
    model = build_ticknet_l(num_classes=args.num_classes, cifar=True).to(args.device)

    print(f"Model source: {Path(sys.modules[build_ticknet_l.__module__].__file__).resolve()}")
    print(f"Device: {args.device} | Input: (N, 3, 32, 32) | Classes: {args.num_classes}")
    print("\nMODEL\n")
    print(model)

    # Observe stage shapes during the same eval forward used for profiling.
    # No hardcoded stage channels or depths: edits are reflected immediately.
    stage_rows = []
    handles = []

    def record_stage(name):
        def hook(layer, inputs, output):
            parameters = sum(p.numel() for p in layer.parameters() if p.requires_grad)
            stage_rows.append((name, str(tuple(output.shape)), parameters))
        return hook

    try:
        for name, layer in model.backbone.named_children():
            handles.append(layer.register_forward_hook(record_stage("backbone." + name)))
        handles.append(model.classifier.register_forward_hook(record_stage("classifier")))
        profile = profile_model(model, 32)
    finally:
        for handle in handles:
            handle.remove()

    expected_eval_shape = [1, args.num_classes]
    if profile["output_shape"] != expected_eval_shape:
        raise RuntimeError(f"Eval output shape {profile['output_shape']}; expected {expected_eval_shape}")

    print("\nSTAGE SUMMARY (eval, batch size 1)")
    print(f"{'Stage':<30} {'Output shape (N,C,H,W)':<25} {'Trainable params':>18}")
    for name, shape, parameters in stage_rows:
        print(f"{name:<30} {shape:<25} {parameters:>18,}")

    if args.details:
        modules = dict(model.named_modules())
        print("\nCONV2D / LINEAR DETAILS (execution order, batch size 1)")
        print(f"{'Layer':<58} {'Input':<23} {'Output':<23} {'Params':>12} {'FLOPs':>14}")
        print(f"{row['name']:<58} {str(tuple(row['input_shape'])):<23} "
              f"{str(tuple(row['output_shape'])):<23} {parameters:>12,} "
              f"{row['flops'] / 1e9:>14.9f}")
        print("Detail rows count Conv2d/Linear parameters; overall totals also include BN parameters.")

    total_parameters = sum(p.numel() for p in model.parameters())
    print("\nTOTALS")
    print(f"Total parameters:     {total_parameters:,}")
    print(f"Trainable parameters: {profile['learnable_parameters']:,}")
    print(f"MACs / image:         {profile['macs']:,}")
    print(f"GFLOPs / image:       {profile['flops'] / 1e9:.6f}")
    print(f"Convention: {profile['counting_convention']}")
    print("Excluded: " + ", ".join(profile["excluded_operations"]))
    within_limits = profile["within_exam_limits"]
    print(f"[{'PASS' if within_limits else 'FAIL'}] Exam budget: trainable params <= 6,000,000; GFLOPs < 1.0")
    sys.stdout.flush()

    # Random inputs only. Batch size 2 also permits train-mode BatchNorm at 1x1.
    model.train()
    model.zero_grad(set_to_none=True)
    images = torch.randn(2, 3, 32, 32, device=args.device)
    targets = torch.tensor([0, args.num_classes - 1], device=args.device)
    logits = model(images)
    expected_shape = (2, args.num_classes)
    if tuple(logits.shape) != expected_shape:
        raise RuntimeError(f"Train output shape {tuple(logits.shape)}; expected {expected_shape}")
    if not torch.isfinite(logits).all().item():
        raise RuntimeError("Forward produced NaN/Inf logits")
    loss = nn.functional.cross_entropy(logits, targets)
    if not torch.isfinite(loss).item():
        raise RuntimeError("Cross-entropy loss is NaN/Inf")
    loss.backward()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            raise RuntimeError(f"Missing gradient: {name} (trainable parameter disconnected from loss)")
        if not torch.isfinite(parameter.grad).all().item():
            raise RuntimeError(f"NaN/Inf gradient: {name}")
    print(f"[PASS] Forward/backward: logits {tuple(logits.shape)}, loss={loss.item():.6f}; all trainable gradients finite")
    print("Synthetic data only. This checks execution and gradients, not trained accuracy.")
    return 0 if within_limits else 1


def main():
    args = parse_args()
    started = time.perf_counter()
    try:
        result = check_model(args)
    except ModuleNotFoundError as error:
        print(f"\n[FAIL] Missing module: {error.name}", file=sys.stderr)
        print("Use the Python environment containing PyTorch and put checkmodel_l.py beside train_cifar.py.",
              file=sys.stderr)
        return 1
    except Exception as error:
        print(f"\n[FAIL] {type(error).__name__}: {error}", file=sys.stderr)
        traceback.print_exc()
        return 1
    print(f"Completed in {time.perf_counter() - started:.2f}s")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
