#!/usr/bin/env python3
"""Aggregates Grid Search results across CIFAR-10 and CIFAR-100 experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


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


def aggregate_results(runs_dir: Path, output_csv: Optional[Path] = None, output_md: Optional[Path] = None) -> pd.DataFrame:
    records: List[Dict[str, object]] = []

    for exp_name in EXPECTED_EXPERIMENTS:
        p = runs_dir / exp_name
        metric_file = p / "test_metrics.json"
        cfg_file = p / "config.json"
        epochs_file = p / "epochs.csv"

        if not metric_file.is_file() or not cfg_file.is_file():
            print(f"[!] Warning: Experiment {exp_name} is missing or incomplete in {runs_dir}")
            continue

        metrics = json.loads(metric_file.read_text())
        cfg = json.loads(cfg_file.read_text())

        best_val_acc: Optional[float] = None
        best_val_epoch: Optional[int] = None
        if epochs_file.is_file():
            try:
                df_ep = pd.read_csv(epochs_file)
                if "val_acc" in df_ep.columns:
                    idx_max = df_ep["val_acc"].idxmax()
                    best_val_acc = float(df_ep.loc[idx_max, "val_acc"])
                    best_val_epoch = int(df_ep.loc[idx_max, "epoch"]) if "epoch" in df_ep.columns else None
            except Exception:
                pass

        records.append({
            "Experiment": exp_name,
            "Dataset": cfg.get("dataset", "").upper(),
            "Optimizer": cfg.get("optimizer", "").upper(),
            "LR": cfg.get("learning_rate"),
            "Best Val Acc (%)": f"{best_val_acc:.2f}%" if best_val_acc is not None else "N/A",
            "Best Val Ep": best_val_epoch if best_val_epoch is not None else "N/A",
            "Test Top-1 (%)": f"{metrics['top1']:.2f}%",
            "Test Loss": f"{metrics['loss']:.4f}",
            "Macro F1": f"{metrics['macro_f1']:.4f}",
            "Params": f"{cfg.get('learnable_parameters', 0):,}",
            "FLOPs (G)": f"{cfg.get('gflops_forward', 0.0):.4f}G",
        })

    if not records:
        print(f"No completed experiment artifacts found in: {runs_dir}")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    print("\n" + "=" * 80)
    print("                FINAL EXAM: TICKNET-L GRID SEARCH MASTER SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80 + "\n")

    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"[✓] Saved CSV report to: {output_csv}")

    if output_md:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        md_content = f"# Master Grid Search Summary Report\n\n{df.to_markdown(index=False)}\n"
        output_md.write_text(md_content)
        print(f"[✓] Saved Markdown report to: {output_md}")

    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate Grid Search results.")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"), help="Path to runs directory (default: runs)")
    parser.add_argument("--output-csv", type=Path, default=Path("docs/results/grid_search_summary.csv"), help="Output CSV path")
    parser.add_argument("--output-md", type=Path, default=Path("docs/results/grid_search_summary.md"), help="Output Markdown path")
    args = parser.parse_args()

    aggregate_results(args.runs_dir, args.output_csv, args.output_md)
    return 0


if __name__ == "__main__":
    main()
