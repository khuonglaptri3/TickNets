# l_mid224_lr015

- Variant: `Mid224`
- Recipe: `lr015`
- Seed: `42`; split seed: `123`
- Validation role: Not selected; reported as a post-selection diagnostic
- Best validation epoch: **198**
- Validation Top-1: **93.84%**
- Hard-label validation loss: **0.254892**
- Historical test Top-1: **94.80%**
- Historical test Macro F1: **0.947906**
- Historical test loss: **0.185778**

## Class-level observations

The lowest per-class F1 is `cat` at 0.9184. The largest off-diagonal cell is `cat` → `dog` (5 images).

These test results are post-selection descriptive evidence. They did not choose this recipe or checkpoint.

## Evidence

- [Test metrics](test_metrics.json)
- [Predictions](predictions.csv)
- [Classification report](classification_report.csv)
- [Confusion matrix CSV](confusion_matrix.csv)
- [Confusion matrix figure](confusion_matrix.png)
- [Learning curves](learning_curves.png)
- [Copied best validation checkpoint](../../checkpoints/l_mid224_lr015/best_val.pt) — SHA-256 `efaec5e93b1fc49e79b16eb7e439bc6cc543106b98e92f7dc217c9809cbe4cc0`
- [Copied final checkpoint](../../checkpoints/l_mid224_lr015/last.pt) — SHA-256 `a3a202658cc52a50409ff39dda3d70faf825a58f1d34879822cdefc6bc288200`

## Limitations

This is one seed on a historical 250-image test split. It provides no multi-seed uncertainty estimate. Training objectives differ when label smoothing changes, so training-loss values must not be compared across smoothing recipes.
