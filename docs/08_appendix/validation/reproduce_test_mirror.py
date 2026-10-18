"""Replay inference using the UofT CIFAR test mirror, requiring identical raw-data hashes.

Optional dependency: python -m pip install pyarrow
The submission entry points themselves use torchvision's original CIFAR archives.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import urllib.request

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from models.cifar_checkpoint import run_checkpoint
from models.checkpoint_utils import atomic_json, file_sha256, get_test_transform


class MirrorTest(Dataset):
    def __init__(self, path, dataset):
        table = pq.read_table(path)
        label = "label" if dataset == "cifar10" else "fine_label"
        features = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]
        self.data = np.stack([np.asarray(Image.open(io.BytesIO(item["bytes"])).convert("RGB"))
                              for item in table["img"].to_pylist()])
        self.targets = table[label].to_pylist()
        self.classes = features[label]["names"]
        self.transform = get_test_transform(dataset)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.transform(Image.fromarray(self.data[index])), self.targets[index]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", "cifar10", "cifar100"), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    cache = ROOT / "runs/mirror_test"
    cache.mkdir(parents=True, exist_ok=True)
    names = ("cifar10", "cifar100") if args.dataset == "all" else (args.dataset,)
    for dataset in names:
        folder = "plain_text" if dataset == "cifar10" else "cifar100"
        url = f"https://huggingface.co/datasets/uoft-cs/{dataset}/resolve/main/{folder}/test-00000-of-00001.parquet"
        path = cache / f"{dataset}_test.parquet"
        if not path.exists():
            urllib.request.urlretrieve(url, path)
        output = args.output_root / ("checkpoint10" if dataset == "cifar10" else "checkpoint100")
        options = argparse.Namespace(checkpoint=None, output_dir=output, threads=2,
                                     device=args.device, data_root=ROOT / "data", no_download=True,
                                     batch_size=args.batch_size, num_workers=0)
        result = run_checkpoint(dataset, options, test_dataset=MirrorTest(path, dataset))
        result["verification_data_source"] = url
        result["mirror_parquet_sha256"] = file_sha256(path)
        result["verification_note"] = "Raw pixel, target and order fingerprint must match the original Kaggle test set."
        atomic_json(output / "test_metrics.json", result)
        if not result["matches_report"]:
            raise RuntimeError(f"Fresh {dataset} inference differs from the report")


if __name__ == "__main__":
    main()
