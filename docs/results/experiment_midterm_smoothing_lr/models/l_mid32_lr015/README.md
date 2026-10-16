# l_mid32_lr015

- Variant: `Mid32`
- Recipe: `lr015`
- Seed: `42`; split seed: `123`
- Validation role: Selected winner
- Best validation epoch: **186**
- Validation Top-1: **91.04%**
- Hard-label validation loss: **0.441804**
- Historical test Top-1: **97.20%**
- Historical test Macro F1: **0.971982**
- Historical test loss: **0.051278**

## Class-level observations

The lowest per-class F1 is `dog` at 0.9485. The largest off-diagonal cell is `dog` → `cat` (4 images).

These test results are post-selection descriptive evidence. They did not choose this recipe or checkpoint.

## Evidence

- [Test metrics](test_metrics.json)
- [Predictions](predictions.csv)
- [Classification report](classification_report.csv)
- [Confusion matrix CSV](confusion_matrix.csv)
- [Confusion matrix figure](confusion_matrix.png)
- [Learning curves](learning_curves.png)
- [Copied best validation checkpoint](../../checkpoints/l_mid32_lr015/best_val.pt) — SHA-256 `95af8cbae09b0a0bee4263558bf4ead81b6df695f5f7884fa85a09825a862b0c`
- [Copied final checkpoint](../../checkpoints/l_mid32_lr015/last.pt) — SHA-256 `1335aff58a8ce4d501b9f6aaf66ab935b00096ac42732a4966e48ec1f804863a`

## Limitations

This is one seed on a historical 250-image test split. It provides no multi-seed uncertainty estimate. Training objectives differ when label smoothing changes, so training-loss values must not be compared across smoothing recipes.
