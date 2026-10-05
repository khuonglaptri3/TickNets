#!/usr/bin/env python3
"""Aggregates Grid Search results across CIFAR-10 and CIFAR-100 experiments."""
from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


EXPECTED_EXPERIMENTS = [
    "cifar10_sgd_lr010",
    "cifar10_sgd_lr015",
    "cifar10_adam_lr0001",
    "cifar10_adam_lr00003",
    "cifar100_sgd_lr010",
    "cifar100_sgd_lr015",
    "cifar100_adam_lr0001",
    "cifar100_adam_lr00003",
]


def aggregate_results(runs_dir: Path, output_csv: Optional[Path] = None,
                      output_md: Optional[Path] = None, *, expected_experiments=None,
                      official: bool = False, allow_partial: bool = False) -> pd.DataFrame:
    from scripts.verify_cifar_run import verify_run
    records = []
    expected = EXPECTED_EXPERIMENTS if expected_experiments is None else expected_experiments
    for name in expected:
        path = runs_dir / name
        if not (path / "completion.json").is_file():
            if allow_partial:
                continue
            raise ValueError(f"Missing or incomplete experiment: {name}")
        cfg, metrics, best = verify_run(path, official=official)
        records.append({
            "Experiment": name, "Dataset": cfg["dataset"].upper(),
            "Optimizer": cfg["optimizer"].upper(), "LR": cfg["learning_rate"],
            "Best Val Acc (%)": float(best["val_top1"]), "Best Val Ep": int(best["epoch"]),
            "Test Top-1 (%)": metrics["top1"], "Test Loss": metrics["loss"],
            "Macro F1": metrics["macro_f1"], "Params": cfg["learnable_parameters"],
            "FLOPs (G)": cfg["gflops_forward"],
        })
    df = pd.DataFrame(records)
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
    if output_md:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        header = " | ".join(df.columns)
        lines = ["# Validation-selected CIFAR results", "",
                 f"Completed {len(records)}/{len(expected)} experiments. Select hyperparameters by validation only.",
                 "", "| " + header + " |", "| " + " | ".join("---" for _ in df.columns) + " |"]
        lines += ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False, name=None)]
        output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate Grid Search results.")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"), help="Path to runs directory (default: runs)")
    parser.add_argument("--output-csv", type=Path, default=Path("docs/results/grid_search_summary.csv"), help="Output CSV path")
    parser.add_argument("--output-md", type=Path, default=Path("docs/results/grid_search_summary.md"), help="Output Markdown path")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    aggregate_results(args.runs_dir, args.output_csv, args.output_md, official=True, allow_partial=args.allow_partial)
    return 0


if __name__ == "__main__":
    main()
