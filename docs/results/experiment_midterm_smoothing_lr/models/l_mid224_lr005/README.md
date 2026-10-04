# l_mid224_lr005

- Variant: `Mid224`
- Recipe: `lr005`
- Seed: `42`; split seed: `123`
- Validation role: Not selected; reported as a post-selection diagnostic
- Best validation epoch: **188**
- Validation Top-1: **93.32%**
- Hard-label validation loss: **0.296521**
- Historical test Top-1: **94.00%**
- Historical test Macro F1: **0.940221**
- Historical test loss: **0.217440**

## Class-level observations

The lowest per-class F1 is `dog` at 0.8846. The largest off-diagonal cell is `cat` → `dog` (7 images).

These test results are post-selection descriptive evidence. They did not choose this recipe or checkpoint.

## Evidence

- [Test metrics](test_metrics.json)
- [Predictions](predictions.csv)
- [Classification report](classification_report.csv)
- [Confusion matrix CSV](confusion_matrix.csv)
- [Confusion matrix figure](confusion_matrix.png)
- [Learning curves](learning_curves.png)
- [Copied best validation checkpoint](../../checkpoints/l_mid224_lr005/best_val.pt) — SHA-256 `27ed858125e29687a9f57722f3d6f5b9a1b8ac4a73f2a05551b614909a1e1a01`
- [Copied final checkpoint](../../checkpoints/l_mid224_lr005/last.pt) — SHA-256 `10d05e70bb37d9b1dc0f02dec0a73ba7bef4d0ab9bda191b84c8fbe367a94cbc`

## Limitations

This is one seed on a historical 250-image test split. It provides no multi-seed uncertainty estimate. Training objectives differ when label smoothing changes, so training-loss values must not be compared across smoothing recipes.
