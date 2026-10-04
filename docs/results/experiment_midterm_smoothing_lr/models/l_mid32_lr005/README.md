# l_mid32_lr005

- Variant: `Mid32`
- Recipe: `lr005`
- Seed: `42`; split seed: `123`
- Validation role: Not selected; reported as a post-selection diagnostic
- Best validation epoch: **197**
- Validation Top-1: **90.56%**
- Hard-label validation loss: **0.491629**
- Historical test Top-1: **92.00%**
- Historical test Macro F1: **0.919466**
- Historical test loss: **0.340118**

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
- [Copied best validation checkpoint](../../checkpoints/l_mid32_lr005/best_val.pt) — SHA-256 `7132ef52f76719bca8fac30530afe5887bf61b7d1b867699412ad997644ea797`
- [Copied final checkpoint](../../checkpoints/l_mid32_lr005/last.pt) — SHA-256 `25e0aa0e578805c4728c1e0510cc7949be6d75dc54ce37ec979c3ba89d65b5a0`

## Limitations

This is one seed on a historical 250-image test split. It provides no multi-seed uncertainty estimate. Training objectives differ when label smoothing changes, so training-loss values must not be compared across smoothing recipes.
