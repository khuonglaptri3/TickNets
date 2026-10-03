# l_mid32_smoothing005

- Variant: `Mid32`
- Recipe: `smoothing005`
- Seed: `42`; split seed: `123`
- Validation role: Not selected; reported as a post-selection diagnostic
- Best validation epoch: **167**
- Validation Top-1: **90.64%**
- Hard-label validation loss: **0.335364**
- Historical test Top-1: **89.60%**
- Historical test Macro F1: **0.895510**
- Historical test loss: **0.375396**

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
- [Copied best validation checkpoint](../../checkpoints/l_mid32_smoothing005/best_val.pt) — SHA-256 `02872a71a4f1ab3955263420277cfb730d7a6414e26ca189c9c7db3ad4cbfe70`
- [Copied final checkpoint](../../checkpoints/l_mid32_smoothing005/last.pt) — SHA-256 `2ce3327e0ce737cd4d2a800dd900ed2ae2d10583ee3474f2921b329ab70a5d62`

## Limitations

This is one seed on a historical 250-image test split. It provides no multi-seed uncertainty estimate. Training objectives differ when label smoothing changes, so training-loss values must not be compared across smoothing recipes.
