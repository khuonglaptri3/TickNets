# l_mid224_baseline

- Variant: `Mid224`
- Recipe: `baseline`
- Seed: `42`; split seed: `123`
- Validation role: Selected winner
- Best validation epoch: **177**
- Validation Top-1: **93.88%**
- Hard-label validation loss: **0.276317**
- Historical test Top-1: **95.20%**
- Historical test Macro F1: **0.951956**
- Historical test loss: **0.224384**

## Class-level observations

The lowest per-class F1 is `dog` at 0.9293. The largest off-diagonal cell is `dog` → `cat` (4 images).

These test results are post-selection descriptive evidence. They did not choose this recipe or checkpoint.

## Evidence

- [Test metrics](test_metrics.json)
- [Predictions](predictions.csv)
- [Classification report](classification_report.csv)
- [Confusion matrix CSV](confusion_matrix.csv)
- [Confusion matrix figure](confusion_matrix.png)
- [Learning curves](learning_curves.png)
- [Copied best validation checkpoint](../../checkpoints/l_mid224_baseline/best_val.pt) — SHA-256 `3e265df13d32d2f610920ce3088b91703d4f88a95232a9ab1786a8bcaa21920b`
- [Copied final checkpoint](../../checkpoints/l_mid224_baseline/last.pt) — SHA-256 `b38aebae2c0db8e37554c51216035da14233b82ac2009a709a6aa99f13e4a4db`

## Limitations

This is one seed on a historical 250-image test split. It provides no multi-seed uncertainty estimate. Training objectives differ when label smoothing changes, so training-loss values must not be compared across smoothing recipes.
