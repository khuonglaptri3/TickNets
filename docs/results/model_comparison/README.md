# TickNet Basic/L/C — descriptive midterm comparison

This folder compares the three completed seed-42 architecture runs on Mid32
and Mid224. It is a **descriptive architecture comparison**, not a strict
three-way one-variable ablation: Basic and C have directly matching recorded
training settings and split provenance, while the imported L run has protocol
differences described below.

## Reported results

| Model | Variant | Test Top-1 | Macro F1 | Parameters | FLOPs | Batch size |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Basic | Mid32 | 89.6% | 0.8956 | 1,062,223 | 0.158428 G | 64 |
| L | Mid32 | 91.6% | 0.9157 | 1,096,260 | 0.157821 G | 128 |
| C | Mid32 | 91.2% | 0.9117 | 5,155,467 | 0.256830 G | 64 |
| Basic | Mid224 | 92.4% | 0.9240 | 1,062,223 | 0.988343 G | 64 |
| L | Mid224 | 95.6% | 0.9559 | 1,096,260 | 0.796760 G | 64 |
| C | Mid224 | 94.4% | 0.9439 | 5,155,467 | 0.821054 G | 64 |

The exact, unrounded values and source-report paths are in
[`reported_metrics.csv`](reported_metrics.csv). The FLOP convention for all
three models is one evaluation forward pass at batch size 1, with
`1 MAC = 2 FLOPs`, counting Conv2d and Linear operations.

## Observed differences

- C improves over Basic by **+1.6 percentage points** on Mid32 and **+2.0
  points** on Mid224.
- L improves over Basic by **+2.0 points** on Mid32 and **+3.2 points** on
  Mid224.
- L is **+0.4 points** above C on Mid32 and **+1.2 points** above C on
  Mid224.
- Basic Mid224 is close to the exam's `<1G` FLOP limit at 0.988343 GFLOPs.
  C has many more parameters but uses 0.821054 GFLOPs at Mid224; L records
  both the highest accuracy and the lowest Mid224 FLOPs of the three.

Those statements describe these completed runs; they do not prove that one
architectural change alone caused every difference.

## Comparability and split provenance

Basic and C are the most directly controlled pair:

- seed 42, 200 epochs, batch size 64, SGD and cosine settings match;
- both record the same canonical split-manifest SHA-256;
- the canonical manifest pairs Mid32 and Mid224 membership explicitly.

TickNet-L differs in at least these recorded ways:

- its Mid32 batch size is 128 instead of 64;
- it uses two DataLoader workers instead of zero;
- its Kaggle notebook generated a manifest schema
  `variant,split,class_name,filename`, whereas the canonical Basic/C manifest
  stores paired paths and checksums for both resolutions.

The L raw manifest hash therefore cannot be compared directly with the
Basic/C hash. The exact L test-filename manifest is not present in the local
export, so filename equivalence is marked **unverified**, rather than treating
the unequal hashes as proof of different samples. See
[`split_provenance.csv`](split_provenance.csv) for the six per-run records.

## Figures and source reports

- [`accuracy_by_dataset.png`](accuracy_by_dataset.png): grouped test Top-1.
- [`macro_f1_by_dataset.png`](macro_f1_by_dataset.png): grouped Macro F1.
- [`accuracy_vs_compute_mid224.png`](accuracy_vs_compute_mid224.png): Mid224
  accuracy/FLOP tradeoff.
- [TickNet-Basic report](../model_basic/README.md)
- [TickNet-L report](../ticknet_l_midterm_seed42_20260928/README.md)
- [TickNet-C report](../model_c/README.md)

## Remaining experiment scope

Controlled hyperparameter tuning and a narrower ablation that changes only
one factor at a time remain **pending**. No tuning conclusion is inferred from
the final test scores in this report.
