# l_mid32_smoothing010

- Variant: `Mid32`
- Recipe: `smoothing010`
- Seed: `42`; split seed: `123`
- Validation role: Not selected; reported as a post-selection diagnostic
- Best validation epoch: **191**
- Validation Top-1: **90.72%**
- Hard-label validation loss: **0.358847**
- Historical test Top-1: **94.40%**
- Historical test Macro F1: **0.943555**
- Historical test loss: **0.238940**

## Class-level observations

The lowest per-class F1 is `dog` at 0.8936. The largest off-diagonal cell is `dog` → `cat` (7 images).

These test results are post-selection descriptive evidence. They did not choose this recipe or checkpoint.

## Evidence

- [Test metrics](test_metrics.json)
- [Predictions](predictions.csv)
- [Classification report](classification_report.csv)
- [Confusion matrix CSV](confusion_matrix.csv)
- [Confusion matrix figure](confusion_matrix.png)
- [Learning curves](learning_curves.png)
- [Copied best validation checkpoint](../../checkpoints/l_mid32_smoothing010/best_val.pt) — SHA-256 `4e1d6a89d1bbe51fe1879fc8597fb6435a5f3c5c6270b184781827155f0def87`
- [Copied final checkpoint](../../checkpoints/l_mid32_smoothing010/last.pt) — SHA-256 `e20202b84ccc1458cdc51cd2454e5592249c6aaac67d5c1e0a86e74402e36d76`

## Limitations

This is one seed on a historical 250-image test split. It provides no multi-seed uncertainty estimate. Training objectives differ when label smoothing changes, so training-loss values must not be compared across smoothing recipes.
