"""Train TickNet baselines or TickNet-L and evaluate the held-out Mid test once."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import torch
import torchvision

from models.mid_models import MODEL_NAMES, MODEL_REVISIONS, build_mid_model
from models.mid_data import build_mid_loaders, seed_everything
from models.model_profile import profile_model


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; training/evaluation aborted")
            if training:
                loss.backward()
                optimizer.step()
            batch_count = labels.numel()
            loss_sum += loss.item() * batch_count
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += batch_count
    if not count:
        raise ValueError("Cannot evaluate an empty data loader")
    return {"loss": loss_sum / count, "top1": 100.0 * correct / count, "samples": count}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--variant", choices=("Mid32", "Mid224"), required=True)
    parser.add_argument("--model", choices=MODEL_NAMES, default="basic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-data", action="store_true", help="Read a train/test batch and report shapes; no training")
    parser.add_argument("--evaluate", type=Path, metavar="CHECKPOINT", help="Evaluate a trusted local last.pt checkpoint")
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.num_workers < 0:
        parser.error("epochs, batch-size and threads must be positive; num-workers must be nonnegative")
    if args.learning_rate <= 0 or args.momentum < 0 or args.weight_decay < 0:
        parser.error("learning-rate must be positive; momentum and weight-decay must be nonnegative")
    if not args.check_data and args.output_dir is None:
        parser.error("--output-dir is required for training/evaluation")
    if args.check_data and args.evaluate:
        parser.error("--check-data and --evaluate are separate modes")
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    train_loader, test_loader = build_mid_loaders(args.data_root, args.variant, batch_size=args.batch_size,
                                                seed=args.seed, num_workers=args.num_workers,
                                                augment=not args.no_augment, pin_memory=device.type == "cuda")
    mapping = train_loader.dataset.class_to_idx
    if args.check_data:
        train_images, _ = next(iter(train_loader))
        test_images, _ = next(iter(test_loader))
        result = {"variant": args.variant, "train_samples": len(train_loader.dataset),
                  "test_samples": len(test_loader.dataset), "class_to_idx": mapping,
                  "train_shape": list(train_images.shape), "test_shape": list(test_images.shape), "seed": args.seed}
        print(json.dumps(result, indent=2))
        return result
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Output directory exists; choose a fresh run directory: {output}")
    manifest = args.data_root / "split_manifest.csv"
    if not manifest.is_file():
        raise ValueError("Prepared split_manifest.csv is required for training/evaluation")
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    checkpoint = None
    if args.evaluate:
        checkpoint = torch.load(args.evaluate, map_location="cpu", weights_only=True)
        if checkpoint["class_to_idx"] != mapping or checkpoint["config"]["variant"] != args.variant:
            raise ValueError("Checkpoint class mapping or variant does not match the selected dataset")
        recorded_hash = checkpoint["config"].get("split_manifest_sha256")
        if not recorded_hash or recorded_hash != manifest_hash:
            raise ValueError("Checkpoint split manifest does not match this dataset; use its original prepared split")
        args.model = checkpoint["config"]["model"]
        recorded_revision = checkpoint["config"].get("architecture_revision")
        if args.model == "l" and recorded_revision != MODEL_REVISIONS["l"]:
            raise ValueError("Checkpoint TickNet-L architecture revision does not match this implementation")
    model = build_mid_model(args.model, num_classes=len(mapping), variant=args.variant)
    complexity = profile_model(model, 32 if args.variant == "Mid32" else 224)
    if args.model == "l" and not complexity["within_exam_limits"]:
        raise ValueError("TickNet-L exceeds the exam parameter/FLOP limits")
    model = model.to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    criterion = torch.nn.CrossEntropyLoss()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(device=str(device), num_classes=len(mapping), class_to_idx=mapping,
                  learnable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  python_version=platform.python_version(), torch_version=str(torch.__version__),
                  torchvision_version=str(torchvision.__version__), optimizer="SGD", scheduler="CosineAnnealingLR",
                  train_samples=len(train_loader.dataset), test_samples=len(test_loader.dataset),
                  test_policy="Evaluate final checkpoint only; never select checkpoints using test",
                  normalization="ToTensor [0,1]; model has data_bn", top1_unit="percent")
    config["split_manifest_sha256"] = manifest_hash
    config["architecture_revision"] = MODEL_REVISIONS[args.model]
    config["complexity"] = {key: value for key, value in complexity.items() if key != "layers"}
    if checkpoint is not None:
        config["checkpoint_training_config"] = checkpoint["config"]
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    if checkpoint is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        with (output / "epochs.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("epoch", "learning_rate", "train_loss", "train_top1", "train_samples"))
            writer.writeheader()
            for epoch in range(1, args.epochs + 1):
                learning_rate = optimizer.param_groups[0]["lr"]
                metrics = run_epoch(model, train_loader, criterion, device, optimizer)
                row = {"epoch": epoch, "learning_rate": learning_rate, "train_loss": metrics["loss"],
                       "train_top1": metrics["top1"], "train_samples": metrics["samples"]}
                writer.writerow(row)
                handle.flush()
                scheduler.step()
                temp_checkpoint = output / "last.pt.tmp"
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                            "class_to_idx": mapping, "config": config}, temp_checkpoint)
                temp_checkpoint.replace(output / "last.pt")
                print(json.dumps(row), flush=True)
    result = run_epoch(model, test_loader, criterion, device)
    (output / "test_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"final_test": result, "output_dir": str(output)}), flush=True)
    return result


if __name__ == "__main__":
    main()
