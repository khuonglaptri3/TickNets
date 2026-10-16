# l_mid32_lr005

- Variant: `Mid32`
- Recipe: `lr005`
- Seed: `42`; split seed: `123`
- Validation role: Not selected; reported as a post-selection diagnostic
- Best validation epoch: **197**
- Validation Top-1: **90.56%**
- Hard-label validation loss: **0.491629**
- Historical test Top-1: **95.60%**
- Historical test Macro F1: **0.955617**
- Historical test loss: **0.137503**

## Class-level observations

The lowest per-class F1 is `dog` at 0.9149. The largest off-diagonal cell is `dog` → `cat` (6 images).

These test results are post-selection descriptive evidence. They did not choose this recipe or checkpoint.

## Evidence

- [Test metrics](test_metrics.json)
- [Predictions](predictions.csv)
- [Classification report](classification_report.csv)
- [Confusion matrix CSV](confusion_matrix.csv)
- [Confusion matrix figure](confusion_matrix.png)
- [Learning curves](learning_curves.png)
- [Copied best validation checkpoint](../../checkpoints/l_mid32_lr005/best_val.pt) — SHA-256 `9f13af6e3b49ce6912819b027a57ee9e47183d61d9e7adea01e6594ec2f53c5e`
- [Copied final checkpoint](../../checkpoints/l_mid32_lr005/last.pt) — SHA-256 `4d6175d03426270069b4ccdd01d8ab0038acb07ef8ff0b756fa5785c511cc124`

## Limitations

This is one seed on a historical 250-image test split. It provides no multi-seed uncertainty estimate. Training objectives differ when label smoothing changes, so training-loss values must not be compared across smoothing recipes.
