# TickNet-Basic — Midterm results (seed 42)

This directory reports the upstream TickNet **Basic** model used as the
baseline for the architecture study. The final epoch-200 checkpoints were
evaluated once on the held-out test split; the test set was not used for
checkpoint selection.

| Variant | Train / test | Epochs | Test Top-1 | Macro F1 | Parameters | FLOPs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Mid32 | 25,000 / 250 | 200 | 89.6% | 0.8956 | 1,062,223 | 0.158428 G |
| Mid224 | 25,000 / 250 | 200 | 92.4% | 0.9240 | 1,062,223 | 0.988343 G |

The FLOP convention is `1 MAC = 2 FLOPs`, counting Conv2d and Linear
operations for one batch-size-1 evaluation forward pass. Both variants meet
the exam limits of at most 6 million learnable parameters and less than 1
GFLOP.

## Dataset

Both datasets contain the same five classes in this order: `bird`, `cat`,
`dog`, `frog`, `horse`. Each variant has 5,000 training and 50 test images per
class. Basic Mid32 and Mid224 record the same prepared split-manifest SHA-256:
`9939a6ee404c6fbbdbe1b07a50763dc6d606497709385708512e1e98d71f2780`.

- [Mid32 dataset](https://www.kaggle.com/datasets/hunhxunthanh/mid32-dl-gk-gr06)
- [Mid224 dataset](https://www.kaggle.com/datasets/hunhxunthanh/mid224-dl-gk-gr06)

## Training settings

- Seed: 42
- Batch size: 64
- Optimizer: SGD, learning rate 0.1, momentum 0.9, weight decay 0.0001
- Scheduler: CosineAnnealingLR with the target fixed at 200 epochs
- Training augmentation: reflected random crop and random horizontal flip
- Input normalization: `ToTensor` to `[0, 1]`; the model contains `data_bn`
- Test policy: deterministic loader without augmentation; evaluate only the
  final checkpoint

## Kaggle training notebooks

- [Basic Mid32, epochs 1–200](https://www.kaggle.com/code/khngtrnnh/dl-btgk-modelbaseline-mid32)
- [Basic Mid224, epochs 1–100](https://www.kaggle.com/code/khngtrnnh/dl-btgk-modelbasic-mid224-epoch001-100-notebook)
- [Basic Mid224, epochs 101–200](https://www.kaggle.com/code/khngtrnnh/dl-btgk-modelbasic-mid224-epoch101-200-notebook)

### Why Mid224 uses two Kaggle sessions

Kaggle's 12-hour session limit interrupted the original long Mid224 job, so
the completed run was deliberately executed as epochs 1–100 and 101–200.
At epoch 101, training restored the model, optimizer, scheduler, DataLoader generator,
and Python/NumPy/PyTorch RNG states saved at epoch 100. The cosine
scheduler retained its original 200-epoch target. Consequently, these are two
execution sessions of one logical 200-epoch run—not two independent
experiments and not an average of separate results. The exported history is a
single continuous sequence containing every epoch from 1 through 200.

## Report assets

`midterm_report_assets/` contains:

- dataset counts by variant, split, and class;
- raw 5×5 confusion matrices and labeled PNG heatmaps;
- per-class Precision, Recall, F1, and support, plus accuracy, macro, and
  weighted averages;
- 200-epoch training-loss and training-Top-1 curves;
- the required summary table and layer/stage model profiles.

The corresponding reproducibility artifacts are archived under
`docs/results/training_logs/basic_mid32_seed42/` and
`docs/results/training_logs/basic_mid224_seed42/`. Checkpoint hashes are
indexed by `docs/results/checkpoints/checkpoint_manifest.csv`.

## Scope

These results establish the baseline for the Basic/C/L architecture
comparison. Controlled hyperparameter tuning and a strict one-variable
ablation beyond the completed architecture runs remain **pending** and are not
inferred from the final test scores.
