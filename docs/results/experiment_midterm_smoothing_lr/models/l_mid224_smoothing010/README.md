# l_mid224_smoothing010

- Variant: `Mid224`
- Recipe: `smoothing010`
- Seed: `42`; split seed: `123`
- Validation role: Not selected; reported as a post-selection diagnostic
- Best validation epoch: **176**
- Validation Top-1: **93.64%**
- Hard-label validation loss: **0.263301**
- Historical test Top-1: **97.20%**
- Historical test Macro F1: **0.971999**
- Historical test loss: **0.177574**

## Class-level observations

The lowest per-class F1 is `cat` at 0.9600. The largest off-diagonal cell is `bird` → `frog` (2 images).

These test results are post-selection descriptive evidence. They did not choose this recipe or checkpoint.

## Evidence

- [Test metrics](test_metrics.json)
- [Predictions](predictions.csv)
- [Classification report](classification_report.csv)
- [Confusion matrix CSV](confusion_matrix.csv)
- [Confusion matrix figure](confusion_matrix.png)
- [Learning curves](learning_curves.png)
- [Copied best validation checkpoint](../../checkpoints/l_mid224_smoothing010/best_val.pt) — SHA-256 `b04a8ce0aecafc6626d54dbe132baddd7b2c2c193bfe6fd25710435d72cda5d2`
- [Copied final checkpoint](../../checkpoints/l_mid224_smoothing010/last.pt) — SHA-256 `7adda8df75f8bcaa92a41f47a0f7a612331cce801dc0888cc12b283df2558864`

## Limitations

This is one seed on a historical 250-image test split. It provides no multi-seed uncertainty estimate. Training objectives differ when label smoothing changes, so training-loss values must not be compared across smoothing recipes.
