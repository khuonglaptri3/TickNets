# l_mid32_smoothing010

- Variant: `Mid32`
- Recipe: `smoothing010`
- Seed: `42`; split seed: `123`
- Validation role: Not selected; reported as a post-selection diagnostic
- Best validation epoch: **191**
- Validation Top-1: **90.72%**
- Hard-label validation loss: **0.358847**
- Historical test Top-1: **91.20%**
- Historical test Macro F1: **0.911626**
- Historical test loss: **0.347748**

## Class-level observations

The lowest per-class F1 is `dog` at 0.8511. The largest off-diagonal cell is `dog` → `cat` (9 images).

These test results are post-selection descriptive evidence. They did not choose this recipe or checkpoint.

## Evidence

- [Test metrics](test_metrics.json)
- [Predictions](predictions.csv)
- [Classification report](classification_report.csv)
- [Confusion matrix CSV](confusion_matrix.csv)
- [Confusion matrix figure](confusion_matrix.png)
- [Learning curves](learning_curves.png)
- [Copied best validation checkpoint](../../checkpoints/l_mid32_smoothing010/best_val.pt) — SHA-256 `0d09669f7977ed7eeb0b3c6bf39cfe484ce341d2836bf9bfef1bcb67ffdfdc6d`
- [Copied final checkpoint](../../checkpoints/l_mid32_smoothing010/last.pt) — SHA-256 `203f145e70d4cb5b02a22e9a09b7e99292c6176f43554f942d0f44800a335dc9`

## Limitations

This is one seed on a historical 250-image test split. It provides no multi-seed uncertainty estimate. Training objectives differ when label smoothing changes, so training-loss values must not be compared across smoothing recipes.
