# l_mid224_lr005

- Variant: `Mid224`
- Recipe: `lr005`
- Seed: `42`; split seed: `123`
- Validation role: Not selected; reported as a post-selection diagnostic
- Best validation epoch: **188**
- Validation Top-1: **93.32%**
- Hard-label validation loss: **0.296521**
- Historical test Top-1: **96.00%**
- Historical test Macro F1: **0.960158**
- Historical test loss: **0.112853**

## Class-level observations

The lowest per-class F1 is `dog` at 0.9109. The largest off-diagonal cell is `cat` → `dog` (4 images).

These test results are post-selection descriptive evidence. They did not choose this recipe or checkpoint.

## Evidence

- [Test metrics](test_metrics.json)
- [Predictions](predictions.csv)
- [Classification report](classification_report.csv)
- [Confusion matrix CSV](confusion_matrix.csv)
- [Confusion matrix figure](confusion_matrix.png)
- [Learning curves](learning_curves.png)
- [Copied best validation checkpoint](../../checkpoints/l_mid224_lr005/best_val.pt) — SHA-256 `87de69a06717608eaabe3d78bd02cea14d43d113cb077b685ed614ce57fc0b47`
- [Copied final checkpoint](../../checkpoints/l_mid224_lr005/last.pt) — SHA-256 `4c8cd5b72d5df8a5498ccabdea4617a76516a4a3cba42b067edaeb138d0ca524`

## Limitations

This is one seed on a historical 250-image test split. It provides no multi-seed uncertainty estimate. Training objectives differ when label smoothing changes, so training-loss values must not be compared across smoothing recipes.
