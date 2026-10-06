# 7. Model Initialization: Weight Initialization and Train from Scratch Strategy

This document details the architectural design, technical mechanics, and source code evidence for **Model Initialization**, affirming the commitment to **Train from Scratch 100% on CIFAR-10 & CIFAR-100** and analyzing the Kaiming / Xavier initialization algorithms.

---

## 1. Final Examination Model Initialization Summary

| Criterion | Implementation in Project | Design Rationale & Code Reference |
| :--- | :--- | :--- |
| **Primary Strategy** | **100% Train from Scratch** | Strictly **no midterm checkpoints or ImageNet pretraining**; initializes completely fresh weights |
| **Backbone Conv Init** | **Kaiming Uniform (He init)** + Zero bias | Optimal for ReLU activation functions across FR-PDP blocks and Stem Conv |
| **Classifier Head Init** | **Xavier Normal for weights; Conv2d default bias** | Initializes 10-class (CIFAR-10) and 100-class (CIFAR-100) classification heads |
| **Batch Normalization** | Weight $\gamma=1.0$, Bias $\beta=0.0$ | Preserves initial variance scaling before running mean/variance updates |
| **Squeeze-and-Excitation** | Default nn.Linear init (Kaiming Uniform with $a=\sqrt{5}$, Uniform bias) | No custom reinitialization required for SE Linear layers |
| **Random Seed** | `seed_everything(args.seed=42)` | Fixes all random seeds across PyTorch (CPU/CUDA), NumPy, and Python standard random |

---

## 2. Why We Train from Scratch on CIFAR

This is the deliberate experimental choice of the team to benchmark configurations from an identical initial state.
The examination DOCX does not mandate pretraining and explicitly permits reusing the team's midterm architectural foundations.

1. **Experimental Consistency:**
   - The final examination requires evaluating the convergence capacity of `TickNet-L` across diverse hyperparameters (SGD 0.10, 0.15; Adam 0.001, 0.0003). Using pretrained weights from another task would distort learning rate comparisons and invalidate convergence trajectory analyses.
2. **Structural Differences in Classification Heads:**
   - Midterm task classified 5 arbitrary classes.
   - Final examination classifies **10 classes (CIFAR-10)** and **100 classes (CIFAR-100)**.
   - Training from scratch allows the entire network—from low-level edge/texture feature extraction to high-level semantic representations—to adapt natively to the CIFAR distribution.

---

## 3. Detailed Initialization Algorithms & Source Code Evidence

Implemented in [`models/ticknet_l.py`](file:///home/intern-tdkhuong/Desktop/TickNets/models/ticknet_l.py) and [`models/TickNet.py`](file:///home/intern-tdkhuong/Desktop/TickNets/models/TickNet.py):

### 3.1. Convolutional Layers (Conv2d): Kaiming Uniform (He Initialization)
```python
# models/ticknet_l.py: only backbone Conv2d layers are re-initialized in this loop
for module in self.backbone.modules():
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
self.classifier.init_params()  # common.py: xavier_normal_(classifier.conv.weight)
# BN and SE Linear layers retain default PyTorch constructor initializations.
```

- **Theoretical Foundation:** With non-linear ReLU activations ($f(x) = \max(0, x)$), half of the neurons output zero for negative inputs. If using standard normal initialization with variance $\frac{1}{\text{fan\_in}}$, the output variance across layers halves at each stage, causing vanishing gradients in deep networks.
- `kaiming_uniform_` samples from $[-\text{bound}, \text{bound}]$ with $\text{bound} = \sqrt{\frac{6}{\text{fan\_in}}}$, doubling variance to accurately compensate for the zeroed activation energy.

### 3.2. Verification of Zero Disconnected Parameters
The test suite in [`tests/test_ticknet_l.py`](file:///home/intern-tdkhuong/Desktop/TickNets/tests/test_ticknet_l.py) under `test_l_trains_at_both_native_resolutions_with_no_disconnected_parameters` executes a single forward + backward step to assert:
```python
for name, param in model.named_parameters():
    assert param.grad is not None, f"Parameter {name} has no gradient!"
    assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient!"
```
Guarantees 100% of network parameters participate in the computational graph and receive valid gradient updates from the very first training epoch.
