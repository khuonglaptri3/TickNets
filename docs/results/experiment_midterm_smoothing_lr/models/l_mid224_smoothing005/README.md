# l_mid224_smoothing005

- Variant: `Mid224`
- Recipe: `smoothing005`
- Seed: `42`; split seed: `123`
- Validation role: Not selected; reported as a post-selection diagnostic
- Best validation epoch: **160**
- Validation Top-1: **93.44%**
- Hard-label validation loss: **0.236759**
- Historical test Top-1: **98.00%**
- Historical test Macro F1: **0.979996**
- Historical test loss: **0.105370**

## Class-level observations

The lowest per-class F1 is `dog` at 0.9697. The largest off-diagonal cell is `dog` → `cat` (2 images).

These test results are post-selection descriptive evidence. They did not choose this recipe or checkpoint.

## Evidence

- [Test metrics](test_metrics.json)
- [Predictions](predictions.csv)
- [Classification report](classification_report.csv)
- [Confusion matrix CSV](confusion_matrix.csv)
- [Confusion matrix figure](confusion_matrix.png)
- [Learning curves](learning_curves.png)
- [Copied best validation checkpoint](../../checkpoints/l_mid224_smoothing005/best_val.pt) — SHA-256 `6630ed16054a1483e7525a064a17baf5e5fe4c0e2542f93e0a48ae41aeeb8b8c`
- [Copied final checkpoint](../../checkpoints/l_mid224_smoothing005/last.pt) — SHA-256 `1dadd688480fe49ddcc3e6e996398e6fd141ea698d7fe49768a2938dae77a45d`

## Limitations

This is one seed on a historical 250-image test split. It provides no multi-seed uncertainty estimate. Training objectives differ when label smoothing changes, so training-loss values must not be compared across smoothing recipes.
