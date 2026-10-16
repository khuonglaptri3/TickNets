# l_mid224_baseline

- Variant: `Mid224`
- Recipe: `baseline`
- Seed: `42`; split seed: `123`
- Validation role: Selected winner
- Best validation epoch: **177**
- Validation Top-1: **93.88%**
- Hard-label validation loss: **0.276317**
- Historical test Top-1: **96.80%**
- Historical test Macro F1: **0.967948**
- Historical test loss: **0.083773**

## Class-level observations

The lowest per-class F1 is `dog` at 0.9388. The largest off-diagonal cell is `dog` → `cat` (4 images).

These test results are post-selection descriptive evidence. They did not choose this recipe or checkpoint.

## Evidence

- [Test metrics](test_metrics.json)
- [Predictions](predictions.csv)
- [Classification report](classification_report.csv)
- [Confusion matrix CSV](confusion_matrix.csv)
- [Confusion matrix figure](confusion_matrix.png)
- [Learning curves](learning_curves.png)
- [Copied best validation checkpoint](../../checkpoints/l_mid224_baseline/best_val.pt) — SHA-256 `de0f53c603128ed82b7b5a883e7d71dfa61a5d11f4b94bf79b83b925f6a5934e`
- [Copied final checkpoint](../../checkpoints/l_mid224_baseline/last.pt) — SHA-256 `43032500b01be255556a91daec3eb353f2b310e21fcdceb92048916469426ec9`

## Limitations

This is one seed on a historical 250-image test split. It provides no multi-seed uncertainty estimate. Training objectives differ when label smoothing changes, so training-loss values must not be compared across smoothing recipes.
