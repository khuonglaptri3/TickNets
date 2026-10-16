# l_mid224_smoothing010

- Variant: `Mid224`
- Recipe: `smoothing010`
- Seed: `42`; split seed: `123`
- Validation role: Not selected; reported as a post-selection diagnostic
- Best validation epoch: **176**
- Validation Top-1: **93.64%**
- Hard-label validation loss: **0.263301**
- Historical test Top-1: **99.20%**
- Historical test Macro F1: **0.992000**
- Historical test loss: **0.132781**

## Class-level observations

The lowest per-class F1 is `cat` at 0.9800. The largest off-diagonal cell is `cat` → `dog` (1 images).

These test results are post-selection descriptive evidence. They did not choose this recipe or checkpoint.

## Evidence

- [Test metrics](test_metrics.json)
- [Predictions](predictions.csv)
- [Classification report](classification_report.csv)
- [Confusion matrix CSV](confusion_matrix.csv)
- [Confusion matrix figure](confusion_matrix.png)
- [Learning curves](learning_curves.png)
- [Copied best validation checkpoint](../../checkpoints/l_mid224_smoothing010/best_val.pt) — SHA-256 `40bd7e1329f0f5dc77a17b441c1bc39bc7fa4bf0349891d772a698b86bb5d7b3`
- [Copied final checkpoint](../../checkpoints/l_mid224_smoothing010/last.pt) — SHA-256 `0d3958ee750474e746f1adb7fe640144aae406e0e4f7e4ef49f15f261c77c067`

## Limitations

This is one seed on a historical 250-image test split. It provides no multi-seed uncertainty estimate. Training objectives differ when label smoothing changes, so training-loss values must not be compared across smoothing recipes.
