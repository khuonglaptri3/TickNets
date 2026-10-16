# l_mid224_lr015

- Variant: `Mid224`
- Recipe: `lr015`
- Seed: `42`; split seed: `123`
- Validation role: Not selected; reported as a post-selection diagnostic
- Best validation epoch: **198**
- Validation Top-1: **93.84%**
- Hard-label validation loss: **0.254892**
- Historical test Top-1: **96.80%**
- Historical test Macro F1: **0.967957**
- Historical test loss: **0.105013**

## Class-level observations

The lowest per-class F1 is `dog` at 0.9495. The largest off-diagonal cell is `dog` → `cat` (3 images).

These test results are post-selection descriptive evidence. They did not choose this recipe or checkpoint.

## Evidence

- [Test metrics](test_metrics.json)
- [Predictions](predictions.csv)
- [Classification report](classification_report.csv)
- [Confusion matrix CSV](confusion_matrix.csv)
- [Confusion matrix figure](confusion_matrix.png)
- [Learning curves](learning_curves.png)
- [Copied best validation checkpoint](../../checkpoints/l_mid224_lr015/best_val.pt) — SHA-256 `e76dae7e028ecb450175db9ad532f491213946be2b3d42c4c42cff9523ca5fea`
- [Copied final checkpoint](../../checkpoints/l_mid224_lr015/last.pt) — SHA-256 `821fa750f915b358464862149cfbb95e6ba2056f9aa695c60b09515389f27356`

## Limitations

This is one seed on a historical 250-image test split. It provides no multi-seed uncertainty estimate. Training objectives differ when label smoothing changes, so training-loss values must not be compared across smoothing recipes.
