# 9. Validation & Metrics: Model Performance Evaluation on CIFAR-10 & CIFAR-100

This document records the mathematical foundations and source code implementation of the evaluation metrics suite: **Top-1 Accuracy, Loss, Macro F1-Score, and Confusion Matrix** across benchmark datasets **CIFAR-10** (10 classes) and **CIFAR-100** (100 classes).

---

## 1. Mathematical Foundations of the Metric Suite

For multi-class image classification:
- **CIFAR-10:** $|\mathcal{C}| = 10$ target classes.
- **CIFAR-100:** $|\mathcal{C}| = 100$ target classes.

For each target class $c \in \mathcal{C}$:
- **$TP_c$ (True Positive)**: Ground-truth is class $c$ and the model correctly predicts class $c$.
- **$FP_c$ (False Positive)**: Ground-truth is NOT $c$, but the model incorrectly predicts class $c$ (*False Alarm / Type I Error*).
- **$FN_c$ (False Negative)**: Ground-truth is class $c$, but the model misses it and predicts another class (*Miss / Type II Error*).
- **$TN_c$ (True Negative)**: Ground-truth is not $c$ and model correctly predicts non-$c$.

### 1.1. Top-1 Accuracy
$$\text{Top-1 Accuracy} = \frac{\sum_{c \in \mathcal{C}} TP_c}{N} \times 100\%$$
- Represents the percentage of samples where the highest predicted class probability matches the true ground-truth label.

### 1.2. Average Cross-Entropy Loss
$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \log \left(\frac{e^{z_{i, y_i}}}{\sum_{j=1}^{C} e^{z_{i, j}}}\right)$$
- Quantifies model calibration and penalizes deviation between predicted probability distributions and ground-truth one-hot encodings.

### 1.3. Macro-Averaged F1 Score (Macro-F1)
$$\text{F1}_c = \frac{2 \times TP_c}{2 \times TP_c + FP_c + FN_c}$$
$$\text{Macro-F1} = \frac{1}{|\mathcal{C}|} \sum_{c \in \mathcal{C}} \text{F1}_c$$
- **Scientific Significance:** F1-score is the harmonic mean of Precision and Recall. Macro-F1 computes the unweighted arithmetic mean of F1 scores across all classes, treating every class equally.
- **Critical Role on CIFAR-100:** With 100 classes (each having exactly 100 test images), Macro-F1 reveals whether the network disproportionately relies on easily recognizable classes while collapsing on fine-grained or difficult classes.

---

## 2. Confusion Matrix

Implemented in the `evaluate` routine within [`train_cifar.py`](file:///home/intern-tdkhuong/Desktop/TickNets/train_cifar.py):
- **CIFAR-10:** A square matrix of size $10 \times 10$ ($10,000$ test samples, $1,000$ per class).
- **CIFAR-100:** A square matrix of size $100 \times 100$ ($10,000$ test samples, $100$ per class).
- **Rows**: Ground-truth class label ($y$).
- **Columns**: Predicted class label ($\hat{y}$).
- The primary diagonal represents $TP_c$. Off-diagonal entries expose pairwise confusion between semantically adjacent classes (e.g., *cat* misclassified as *dog*, or *automobile* misclassified as *truck*).

Automatically saved to `confusion_matrix.csv` and `test_predictions.csv` for downstream heatmap generation in the technical report.

---

## 3. Source Code Implementation Evidence ([`train_cifar.py`](file:///home/intern-tdkhuong/Desktop/TickNets/train_cifar.py))

```python
def evaluate(model, test_loader, device, output_dir, num_classes):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    matrix = [[0] * num_classes for _ in range(num_classes)]
    loss_sum, correct, total = 0.0, 0, 0
    predictions_rows = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

            loss_sum += loss.item() * labels.numel()
            correct += (preds == labels).sum().item()
            total += labels.numel()

            for target, pred in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                matrix[target][pred] += 1
                predictions_rows.append({"sample_index": len(predictions_rows), "target": target, "prediction": pred})

    # Macro F1 computation
    f1_sum = 0.0
    for i in range(num_classes):
        tp = matrix[i][i]
        fp_plus_fn = sum(matrix[i]) + sum(matrix[r][i] for r in range(num_classes)) - 2 * tp
        f1_sum += (2.0 * tp) / max(1, 2 * tp + fp_plus_fn)
    macro_f1 = f1_sum / num_classes

    result = {
        "top1": 100.0 * correct / total,
        "loss": loss_sum / total,
        "macro_f1": macro_f1,
        "samples": total,
        "correct": correct,
    }
    ...
```

---

## 4. Final Examination Evaluation Rubric Alignment

According to the examination grading criteria:
- **Oral Defense (30%):** Instructors probe the specific confusion pairs in the Confusion Matrix and examine the statistical implications of Macro-F1 vs. Top-1 Accuracy.
- **CIFAR-10 Performance (30%) & CIFAR-100 Performance (30%):** Evaluated comparatively based on independent test set Top-1 Accuracy.
- **Technical Report Quality (10%):** Metric tables documenting Top-1, Loss, Macro-F1, and Confusion Matrix heatmaps are indispensable technical artifacts.
