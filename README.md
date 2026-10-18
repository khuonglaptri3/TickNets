# TickNet-L — Final Project: CIFAR-10 and CIFAR-100

The submission contains the two evaluation entry points, the final 14-block model, the author's original source, Kaggle notebooks, and the complete report and results.

## Evaluate the submitted checkpoints

Use Python 3.10 or later and a PyTorch installation with CUDA support. From the repository root:

```bash
python -m pip install -r requirements.txt
python checkpoint10.py --batch-size 16
python checkpoint100.py --batch-size 16
```

Both scripts run on **GPU (`cuda:0`)**, verify the checkpoint and dataset fingerprints, and evaluate all 10,000 test images. Missing CIFAR archives are downloaded automatically. Results are written to `runs/checkpoint10/` and `runs/checkpoint100/`. CUDA must be available; the scripts do not silently switch to CPU.

| Entry point | Selected epoch | Test Top-1 | Macro-F1 |
|---|---:|---:|---:|
| [checkpoint10.py](checkpoint10.py) | 187 | **96.04%** | 0.960355 |
| [checkpoint100.py](checkpoint100.py) | 198 | **81.23%** | 0.812142 |

The scripts load the original weights from [the four-run archive](docs/05_results/checkpoints_4_phase/) using [the selection manifest](docs/05_results/selected_checkpoints.json). Weights are stored once. Parameters/FLOPs are 4,453,650 / 0.788601088 GFLOPs for CIFAR-10 and 4,545,900 / 0.788785408 GFLOPs for CIFAR-100; 1 MAC = 2 FLOPs, Conv2d and Linear only.

## Reproduce training on Kaggle

Use the four [Kaggle notebooks](docs/04_training/README.md). Each runnable notebook embeds the original training source, configuration and launch scripts, protected by SHA-256 checks. Enable a GPU accelerator and Internet, then run all cells. No root-level training scripts or external source ZIP are required.

Original executed notebooks and their outputs are retained separately as evidence. [Training details and limitations](docs/04_training/TRAINING_AND_MODEL_REPORT.md).

## Repository contents

- `checkpoint10.py`, `checkpoint100.py`: submitted evaluation entry points.
- `models/`: the final model and inference utilities.
- `author/`: the author's original source, preserved unchanged from the original import.
- `docs/`: the report, eight report sections, Kaggle notebooks, weights, logs, figures and metrics.
- `data/`: dataset instructions; downloaded images and archives are local files.
- `.doc/`: the exam specification.

**[Full report](docs/FINAL_REPORT.md)** · [Documentation index](docs/README.md) · [GPU verification](docs/08_appendix/validation/README.md)

## Reference

Thanh Tuan Nguyen and Thanh Phuong Nguyen. *Efficient tick-shape networks of full-residual point-depth-point blocks for image classification.* Neurocomputing 596 (2024), 127942. [DOI](https://doi.org/10.1016/j.neucom.2024.127942).

```bibtex
@article{neucoTickNetNguyen23,
  author       = {Thanh Tuan Nguyen and Thanh Phuong Nguyen},
  title        = {Efficient tick-shape networks of full-residual point-depth-point blocks for image classification},
  journal      = {Neurocomputing},
  volume       = {596},
  pages        = {127942},
  year         = {2024},
  url          = {https://doi.org/10.1016/j.neucom.2024.127942}
}
```
