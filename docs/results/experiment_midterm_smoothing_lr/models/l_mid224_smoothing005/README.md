# l_mid224_smoothing005

- Variant: `Mid224`
- Recipe: `smoothing005`
- Seed: `42`; split seed: `123`
- Validation role: Not selected; reported as a post-selection diagnostic
- Best validation epoch: **160**
- Validation Top-1: **93.44%**
- Hard-label validation loss: **0.236759**
- Historical test Top-1: **96.00%**
- Historical test Macro F1: **0.960132**
- Historical test loss: **0.169884**

## Class-level observations

The lowest per-class F1 is `bird` at 0.9423. The largest off-diagonal cell is `frog` → `bird` (4 images).

These test results are post-selection descriptive evidence. They did not choose this recipe or checkpoint.

## Evidence

- [Test metrics](test_metrics.json)
- [Predictions](predictions.csv)
- [Classification report](classification_report.csv)
- [Confusion matrix CSV](confusion_matrix.csv)
- [Confusion matrix figure](confusion_matrix.png)
- [Learning curves](learning_curves.png)
- [Copied best validation checkpoint](../../checkpoints/l_mid224_smoothing005/best_val.pt) — SHA-256 `8be7400d25a1db5b171ca7b274501b705008c3dd2f9cf71a295efa6fa8685134`
- [Copied final checkpoint](../../checkpoints/l_mid224_smoothing005/last.pt) — SHA-256 `0499333e9b86834a089e00fedacbc769d820fb9ba360c3297aa62717aae33d36`

## Limitations

This is one seed on a historical 250-image test split. It provides no multi-seed uncertainty estimate. Training objectives differ when label smoothing changes, so training-loss values must not be compared across smoothing recipes.
