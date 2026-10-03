# l_mid32_lr015

- Variant: `Mid32`
- Recipe: `lr015`
- Seed: `42`; split seed: `123`
- Validation role: Selected winner
- Best validation epoch: **186**
- Validation Top-1: **91.04%**
- Hard-label validation loss: **0.441804**
- Historical test Top-1: **92.40%**
- Historical test Macro F1: **0.923933**
- Historical test loss: **0.305484**

## Class-level observations

The lowest per-class F1 is `dog` at 0.8750. The largest off-diagonal cell is `dog` → `cat` (7 images).

These test results are post-selection descriptive evidence. They did not choose this recipe or checkpoint.

## Evidence

- [Test metrics](test_metrics.json)
- [Predictions](predictions.csv)
- [Classification report](classification_report.csv)
- [Confusion matrix CSV](confusion_matrix.csv)
- [Confusion matrix figure](confusion_matrix.png)
- [Learning curves](learning_curves.png)
- [Copied best validation checkpoint](../../checkpoints/l_mid32_lr015/best_val.pt) — SHA-256 `90b7a870ac7a6b709e23fb6cbd88614d69955e6027f81e21e853ebcd803ec33c`
- [Copied final checkpoint](../../checkpoints/l_mid32_lr015/last.pt) — SHA-256 `e6185294e09949853b6b79fe430ae8e550ce83195bdc537a28a66de93527747b`

## Limitations

This is one seed on a historical 250-image test split. It provides no multi-seed uncertainty estimate. Training objectives differ when label smoothing changes, so training-loss values must not be compared across smoothing recipes.
