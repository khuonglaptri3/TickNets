# l_mid32_smoothing005

- Variant: `Mid32`
- Recipe: `smoothing005`
- Seed: `42`; split seed: `123`
- Validation role: Not selected; reported as a post-selection diagnostic
- Best validation epoch: **167**
- Validation Top-1: **90.64%**
- Hard-label validation loss: **0.335364**
- Historical test Top-1: **92.80%**
- Historical test Macro F1: **0.927688**
- Historical test loss: **0.273353**

## Class-level observations

The lowest per-class F1 is `dog` at 0.8958. The largest off-diagonal cell is `dog` → `cat` (7 images).

These test results are post-selection descriptive evidence. They did not choose this recipe or checkpoint.

## Evidence

- [Test metrics](test_metrics.json)
- [Predictions](predictions.csv)
- [Classification report](classification_report.csv)
- [Confusion matrix CSV](confusion_matrix.csv)
- [Confusion matrix figure](confusion_matrix.png)
- [Learning curves](learning_curves.png)
- [Copied best validation checkpoint](../../checkpoints/l_mid32_smoothing005/best_val.pt) — SHA-256 `5ef0526c663d43b93340f2d5f727b8129255c268414642acead80cd782eb4b88`
- [Copied final checkpoint](../../checkpoints/l_mid32_smoothing005/last.pt) — SHA-256 `6d73617ef3ef09dc0ca906b8849385ca52c8f8d46f3a463017e198e9c2e8c1f4`

## Limitations

This is one seed on a historical 250-image test split. It provides no multi-seed uncertainty estimate. Training objectives differ when label smoothing changes, so training-loss values must not be compared across smoothing recipes.
