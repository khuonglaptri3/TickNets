# TickNet-L: Midterm Image Classification Project

This repository presents **TickNet-L v1**, a lightweight image classification model developed from TickNet-Basic for the Deep Learning midterm project. The model classifies five categories—**bird, cat, dog, frog, and horse**—at two input resolutions: **224 × 224 (Mid224)** and **32 × 32 (Mid32)**.

The submission includes the model implementation, compressed datasets, two selected checkpoints, training logs, evaluation results, and supporting documentation.

- **Full report:** [Midterm Report](docs/MIDTERM_REPORT.md), including confusion matrices, learning curves, per-class metrics, error examples, and complete 200-epoch logs.
- **Assignment:** [Midterm Exam Specification](<.doc/Midterm exam.docx>).
- **Documentation index:** [Report Structure and Evidence](docs/README.md).

## 1. Run the Submitted Checkpoints

Use Python 3.10 or later, PyTorch 2.5 or later, and a compatible torchvision installation. Run the following commands from the repository root:

```bash
python -m pip install torch torchvision numpy pillow
python checkpoint224.py --batch-size 16
python checkpoint32.py --batch-size 16
```

Each script loads its supplied checkpoint, automatically extracts the corresponding dataset archive when the image folders are missing, and evaluates the 250 test images. CUDA is used when available; otherwise, evaluation runs on the CPU.

The output includes model parameters, FLOPs, the recorded training configuration, test Top-1 accuracy, Macro-F1, cross-entropy loss, per-class accuracy, and the confusion matrix. Both scripts also accept `--checkpoint` and `--data-root`; running without arguments uses a batch size of 64.

| Evaluation script | Default checkpoint | Reported test accuracy |
|---|---|---:|
| [checkpoint224.py](checkpoint224.py) | [Mid224 weights](checkpoints/l_mid224_smoothing010/best_val.pt) | **99.20% — 248/250** |
| [checkpoint32.py](checkpoint32.py) | [Mid32 weights](checkpoints/l_mid32_lr015/best_val.pt) | **97.20% — 243/250** |

Both results were reproduced with batch size 16 starting from a directory containing only the two compressed dataset archives. See the [archive-only evaluation record](docs/08_appendix/validation/archive_only_checkpoint_results.json).

## 2. Model and Selected Results

TickNet-L v1 uses **seven PDP blocks** and **1,096,260 learnable parameters**. It adjusts channel widths and block depths, compresses intermediate pointwise channels, and uses mixed 3 × 3 / 5 × 5 depthwise kernels in the later stages. Residual connections and squeeze-and-excitation attention are inherited from the baseline.

| Dataset | Input resolution | Selected training recipe | Selected epoch | Validation Top-1 | Test Top-1 | GFLOPs |
|---|---|---|---:|---:|---:|---:|
| Mid224 | 224 × 224 | Label smoothing 0.10, initial LR 0.10 | 176 | 93.64% | **99.20%** | 0.796760 |
| Mid32 | 32 × 32 | Initial LR 0.15, label smoothing 0 | 186 | 91.04% | **97.20%** | 0.157821 |

The model satisfies the assignment limits of **at most 6 million learnable parameters** and **less than 1 GFLOP** at both resolutions under the stated counting convention: one image, evaluation mode, Conv2d and Linear operations, and **1 MAC = 2 FLOPs**.

Within each run, the checkpoint epoch is selected by validation Top-1, with validation loss as the tie-breaker. The submitted Mid224 recipe was pinned after observing its historical test result; its validation accuracy is below the smoothing/LR baseline's 93.88%. The reported 99.20% therefore represents the best observed historical test result, rather than a new independent estimate of a validation-only recipe selection procedure. The [selection discussion](docs/MIDTERM_REPORT.md#so-sanh) explains this distinction.

## 3. Dataset and Packaging

Both resolutions contain the same class, filename, and train/test membership. Each class has **5,000 training images and 50 test images**, giving **25,000 training images and 250 test images per resolution**. Controlled experiments reserve 2,500 images from the training set for validation, resulting in **22,500 train / 2,500 validation / 250 test** images.

| Dataset | Archive | Compressed size |
|---|---|---:|
| Mid224 | [Mid224.tar.xz](data/Mid224.tar.xz) | 24.61 MB |
| Mid32 | [Mid32.tar.xz](data/Mid32.tar.xz) | 12.31 MB |

Archive sizes use decimal megabytes. Dataset membership and checksums are recorded in [split_manifest.csv](data/split_manifest.csv); split settings and validation results are available in [split_config.json](data/split_config.json) and [validation_report.json](data/validation_report.json).

Git stores the compressed archives and metadata. The extracted `data/Mid224/` and `data/Mid32/` directories are created locally and excluded by `.gitignore`. Manual extraction is unnecessary for checkpoint evaluation.

## 4. Experiments and Training

The report compares Basic with L and documents two independent experiment directions:

1. **Label smoothing and learning rate:** change one setting at a time while keeping the L architecture fixed.
2. **Stage 4 depth:** increase stage 4 from two to three blocks and evaluate the additional computational cost.

The controlled runs use 200 epochs, batch size 64, SGD with momentum 0.9 and weight decay 1e-4, and a cosine learning-rate schedule. The training seed is 42 and the validation split seed is 123. Configurations, logs, and metrics are retained in [the experiment results](docs/05_results/README.md).

To train from a fresh checkout, first extract the archives:

```bash
tar -xf data/Mid32.tar.xz -C data
tar -xf data/Mid224.tar.xz -C data
```

The following commands use the settings of the two submitted recipes, including two data-loading workers:

```bash
python train_mid_experiment.py --config configs/midterm/smoothing_010.json --variant Mid224 --data-root data --num-workers 2 --output-dir runs/l_mid224_smoothing010
python train_mid_experiment.py --config configs/midterm/lr_015.json --variant Mid32 --data-root data --num-workers 2 --output-dir runs/l_mid32_lr015
```

Training writes validation logs and checkpoints. Test evaluation is a separate step using the trainer's `--evaluate` option, as described in the [training protocol](docs/04_training/MIDTERM_PROTOCOL.md). Exact training outcomes can depend on the recorded software environment and hardware.

## 5. Documentation and Verification

The documentation follows the report structure:

| Report section | Location |
|---|---|
| Problem statement and requirements | [01_introduction](docs/01_introduction/README.md) |
| Dataset and preprocessing | [02_dataset](docs/02_dataset/README.md) |
| Model L design and complexity | [03_model](docs/03_model/README.md) |
| Experimental setup | [04_training](docs/04_training/README.md) |
| Results and supporting evidence | [05_results](docs/05_results/README.md) |
| Comparisons, limitations, and conclusions | [06_discussion](docs/06_discussion/README.md) |
| References | [07_references](docs/07_references/README.md) |
| Appendices and verification records | [08_appendix](docs/08_appendix/README.md) |

For figure generation and model complexity checks, install matplotlib:

```bash
python -m pip install matplotlib
python checkmodel.py --model l
python scripts/build_midterm_report.py
```

The report builder verifies the saved predictions and metrics, then regenerates the figures and tables. Adding `--evaluate` runs fresh inference on both submitted checkpoints before rebuilding the report.

## 6. Acknowledgment and Reference

The baseline TickNet architecture and full-residual point-depth-point blocks are inherited from:

Thanh Tuan Nguyen and Thanh Phuong Nguyen. *Efficient tick-shape networks of full-residual point-depth-point blocks for image classification.* Neurocomputing, 596, 127942, 2024. [DOI: 10.1016/j.neucom.2024.127942](https://doi.org/10.1016/j.neucom.2024.127942).

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

The report distinguishes these inherited components from the channel allocation, block configuration, and mixed-kernel choices evaluated in TickNet-L. Additional methodological references are listed in [References](docs/07_references/README.md).
