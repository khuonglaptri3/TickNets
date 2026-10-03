# l_mid32_baseline

- Variant: `Mid32`
- Recipe: `baseline`
- Seed: `42`; split seed: `123`
- Validation role: Not selected; reported as a post-selection diagnostic
- Best validation epoch: **173**
- Validation Top-1: **90.40%**
- Hard-label validation loss: **0.500445**
- Historical test Top-1: **90.40%**
- Historical test Macro F1: **0.903817**
- Historical test loss: **0.405454**

## Class-level observations

The lowest per-class F1 is `dog` at 0.8163. The largest off-diagonal cell is `dog` → `cat` (9 images).

These test results are post-selection descriptive evidence. They did not choose this recipe or checkpoint.

## Evidence

- [Test metrics](test_metrics.json)
- [Predictions](predictions.csv)
- [Classification report](classification_report.csv)
- [Confusion matrix CSV](confusion_matrix.csv)
- [Confusion matrix figure](confusion_matrix.png)
- [Learning curves](learning_curves.png)
- [Copied best validation checkpoint](../../checkpoints/l_mid32_baseline/best_val.pt) — SHA-256 `61b4fb3d4edced38a27ba76e03c0bb7c68ad03f8ecd32dbe837b60d4fb9cf5cd`
- [Copied final checkpoint](../../checkpoints/l_mid32_baseline/last.pt) — SHA-256 `439c4d21ae966bfa5a7f3f04456324a8b4aa2e79800a771117caa39c4f7a30e4`

## Limitations

This is one seed on a historical 250-image test split. It provides no multi-seed uncertainty estimate. Training objectives differ when label smoothing changes, so training-loss values must not be compared across smoothing recipes.
