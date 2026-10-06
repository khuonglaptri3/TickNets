# 8. Training Pipeline & Architectural Comparison: TickNet-L vs. Author's TickNet Baseline

This document details the architectural design, technical mechanics, and source code evidence for:
1. **Final Examination Training Pipeline ([`train_cifar.py`](file:///home/intern-tdkhuong/Desktop/TickNets/train_cifar.py))**: The `forward` $\rightarrow$ `loss` $\rightarrow$ `backward` $\rightarrow$ `optimizer.step()` execution loop, numerical stability protections, Cosine Annealing learning rate scheduling, and artifact emission.
2. **Head-to-Head Architectural Comparison**: In-depth analysis comparing the author's original **TickNet-Basic** ([`models/TickNet.py`](file:///home/intern-tdkhuong/Desktop/TickNets/models/TickNet.py)) against the proposed **TickNet-L v1** ([`models/ticknet_l.py`](file:///home/intern-tdkhuong/Desktop/TickNets/models/ticknet_l.py)) on CIFAR-10 & CIFAR-100.

---

## 1. Training Loop Mechanics

Standardized within the `run_epoch` function in [`train_cifar.py`](file:///home/intern-tdkhuong/Desktop/TickNets/train_cifar.py):

```python
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)

    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss encountered during training")

            if training:
                loss.backward()
                optimizer.step()

            batch_count = labels.numel()
            loss_sum += loss.item() * batch_count
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_count

    return {
        "loss": loss_sum / total,
        "top1": 100.0 * correct / total,
        "samples": total,
    }
```

### 1.1. Step-by-Step Technical Design
- **`zero_grad(set_to_none=True)`:** Clears gradients by setting tensors to `None` rather than allocating zero-filled tensors, saving GPU memory allocation overhead and speeding up iteration cycles.
- **`non_blocking=True`:** Transfers tensors asynchronously between host memory and GPU VRAM via direct memory access (DMA) when `pin_memory=True`.
- **Numerical Stability Guard (`torch.isfinite`):** Immediately raises an exception if `NaN` or `Inf` loss is encountered, preventing corrupted weights from propagating into checkpoints.
- **Sample-Weighted Metric Aggregation:** Ensures remainder trailing batches do not distort epoch-level loss or accuracy calculations.
- **Cosine Annealing Learning Rate Schedule:** `scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)` smoothly decays learning rates toward zero across training epochs.

---

## 2. Head-to-Head Architectural Comparison: TickNet-L v1 vs. TickNet-Basic

The codebase supports both architectures via the `--model {l, basic}` CLI flag:

| Technical Feature | TickNet-Basic (Author Baseline) | TickNet-L v1 (Proposed Final Exam Model) |
| :--- | :--- | :--- |
| **Source File** | [`models/TickNet.py`](file:///home/intern-tdkhuong/Desktop/TickNets/models/TickNet.py) | [`models/ticknet_l.py`](file:///home/intern-tdkhuong/Desktop/TickNets/models/ticknet_l.py) |
| **Stem Conv Channels** | 32 channels | **24 channels** (conserves FLOPs at native resolution) |
| **5-Stage Elasticity Config** | [128, 64, 128, 256, 512] | **[112, 64, 144, 288, 512]** (shifts representational focus to stages 3 & 4) |
| **Blocks per Stage** | [1, 1, 1, 1, 1] (5 blocks) | **[1, 1, 2, 2, 1] (7 deeper blocks)** |
| **Pointwise Conv Scheme** | Uncompressed ($C_{in} \rightarrow C_{in}$) | **Bottleneck compression ($0.75 \times C_{in}$)** |
| **Depthwise Conv Scheme** | Uniform $3 \times 3$ across all channels | **Mixed DW (parallel $3 \times 3$ and $5 \times 5$ branches)** |
| **Pre-Pooling Conv Layer** | 1024 channels | **768 channels** |
| **Parameters on CIFAR-10** | **1,067,348** ($\le 6\text{M}$) | **1,100,105** ($\le 6\text{M}$) |
| **FLOPs on CIFAR-10** | **0.1584 GFLOPs** ($< 1\text{G}$) | **0.1578 GFLOPs** ($< 1\text{G}$) |
| **Parameters on CIFAR-100** | **1,159,598** ($\le 6\text{M}$) | **1,169,315** ($\le 6\text{M}$) |
| **FLOPs on CIFAR-100** | **0.1586 GFLOPs** ($< 1\text{G}$) | **0.1580 GFLOPs** ($< 1\text{G}$) |

---

## 3. Comprehensive Artifact Emission System

Upon completion of each training run, [`train_cifar.py`](file:///home/intern-tdkhuong/Desktop/TickNets/train_cifar.py) automatically generates 6 standardized artifacts for auditability and technical reporting:
1. `config.json`: Complete snapshot of hyperparameters, git version, timestamp, parameter count, and FLOP budget.
2. `epochs.csv`: Per-epoch logging of Learning Rate, Train Loss, Train Top-1 Accuracy, Val Loss, and Val Top-1 Accuracy.
3. `best_val.pt`: Checkpoint weights achieving highest validation performance (for submission and final evaluation).
4. `last.pt`: Final epoch checkpoint (includes optimizer and scheduler state for `--resume` continuity).
5. `test_metrics.json`: Final evaluated Top-1 Accuracy (%), Mean Loss, and Macro F1 score on the independent test set.
6. `confusion_matrix.csv`: Full confusion matrix ($10 \times 10$ or $100 \times 100$).
