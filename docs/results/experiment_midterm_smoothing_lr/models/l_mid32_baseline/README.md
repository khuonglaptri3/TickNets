# l_mid32_baseline

- Variant: `Mid32`
- Recipe: `baseline`
- Seed: `42`; split seed: `123`
- Validation role: Not selected; reported as a post-selection diagnostic
- Best validation epoch: **173**
- Validation Top-1: **90.40%**
- Hard-label validation loss: **0.500445**
- Historical test Top-1: **92.40%**
- Historical test Macro F1: **0.923003**
- Historical test loss: **0.261480**

## Class-level observations

The lowest per-class F1 is `dog` at 0.8454. The largest off-diagonal cell is `dog` → `cat` (7 images).

These test results are post-selection descriptive evidence. They did not choose this recipe or checkpoint.

## Evidence

- [Test metrics](test_metrics.json)
- [Predictions](predictions.csv)
- [Classification report](classification_report.csv)
- [Confusion matrix CSV](confusion_matrix.csv)
- [Confusion matrix figure](confusion_matrix.png)
- [Learning curves](learning_curves.png)
- [Copied best validation checkpoint](../../checkpoints/l_mid32_baseline/best_val.pt) — SHA-256 `4ef0bed779f0233e110f8a93e4b9188dfab5c7d129156b0e5764eba561922f2b`
- [Copied final checkpoint](../../checkpoints/l_mid32_baseline/last.pt) — SHA-256 `29bdcab06ae45059100c6cd0aae80f9d65079200d92b3b3c9dda8a4380e35955`

## Limitations

This is one seed on a historical 250-image test split. It provides no multi-seed uncertainty estimate. Training objectives differ when label smoothing changes, so training-loss values must not be compared across smoothing recipes.
