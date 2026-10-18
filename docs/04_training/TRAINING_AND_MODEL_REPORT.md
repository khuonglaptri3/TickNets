# TickNet-L Large v2: Training Settings and Model Development

This report describes the four completed final experiments on CIFAR-10 and CIFAR-100. The final model is **TickNet-L Large v2**, a custom 14-block model developed from TickNet-Basic through the earlier TickNet-L design. Both SGD and Adam were evaluated.

The settings and results come from the [saved run artifacts](../05_results/checkpoints_4_phase/), [final experiment summary](../05_results/report_assets/final_required_summary.csv), and source snapshots embedded in the [executed notebooks](notebooks/executed/). Architecture descriptions refer to the model used for these runs.

**1. Completed training experiments**

| Dataset | Optimizer | Initial learning rate | Weight decay | Best validation epoch | Validation Top-1 | Test Top-1 |
|---|---|---:|---:|---:|---:|---:|
| CIFAR-10 | SGD + Nesterov | 0.175 | 0.0001 | 187 | **96.90%** | **96.04%** |
| CIFAR-10 | Adam | 0.001 | 0.0001 | 199 | 95.66% | 95.01% |
| CIFAR-100 | SGD + Nesterov | 0.175 | 0.0005 | 198 | **80.58%** | **81.23%** |
| CIFAR-100 | Adam | 0.001 | 0.0005 | 200 | 77.20% | 78.04% |

Every run completed 200 epochs. The checkpoint with the highest validation Top-1 was selected; lower validation loss resolved ties. SGD also had the higher best validation accuracy for each dataset, so its checkpoint was selected as the final result for that dataset.

These experiments compare two training configurations: optimizer and initial learning rate both change. They do not isolate the effect of the optimizer alone. The final four-run evidence contains one initial learning rate per optimizer and one seed per configuration.

**2. Training arguments and shared settings**

| Argument or setting | Value used | Purpose |
|---|---|---|
| Model identifier | `l_large_v2` in saved configurations | Identifies the final 14-block architecture |
| `--dataset` | `cifar10` or `cifar100` | Selects the dataset and output class count |
| `--optimizer` | `sgd` or `adam` | Selects the weight-update method |
| `--learning-rate` | SGD: `0.175`; Adam: `0.001` | Sets the initial update scale |
| `--epochs` | `200` | Sets the number of passes through the training set |
| `--batch-size` | `128` | Sets the maximum images per training step |
| `--weight-decay` | CIFAR-10: `0.0001`; CIFAR-100: `0.0005` | Regularizes weights to discourage overfitting |
| `--val-fraction` | `0.1` | Reserves 10% of the original training set for validation |
| `--seed` | `42` | Controls initialization, the data split, and random operations |
| Learning-rate scheduler | `CosineAnnealingLR` | Gradually decreases the learning rate |
| Scheduler `T_max` | `200` | Sets the cosine schedule duration in epochs |
| `--eta-min` | `0.0` | Sets the minimum scheduled learning rate |
| Loss | Cross-entropy | Measures classification error |
| `--device` | `cuda:0` | Selects the GPU device |
| Recorded GPU | Tesla T4 | Hardware used for the saved runs |
| `--num-workers` | `2` | Sets data-loading worker processes |
| `--threads` | `2` | Sets CPU computation threads |

The archived training entry points accepted `--model l` and recorded the actual builder as `l_large_v2`. The model identifier in the saved configuration therefore differs from the original command-line selector.

Each dataset uses a stratified split of the original 50,000 training images into **45,000 training images and 5,000 validation images**. The official **10,000 test images** are separate. SGD and Adam use the same split within each dataset.

Training uses randomly initialized weights. The recorded runs have no resume or evaluation checkpoint as their starting point. Convolutional backbone weights use Kaiming uniform initialization, and the classifier convolution uses Xavier normal initialization.

The learning rate follows a cosine decay from its initial value toward zero across the 200-epoch schedule. The scheduler advances after each training and validation epoch.

**3. Optimizer details**

| Setting | SGD | Adam |
|---|---|---|
| Implementation | `torch.optim.SGD` | `torch.optim.Adam` |
| Initial learning rate | `0.175` | `0.001` |
| `--momentum` | `0.9` | Not used by Adam |
| Nesterov | Enabled | Not used by Adam |
| `--adam-beta1` | Not used by SGD | `0.9` |
| `--adam-beta2` | Not used by SGD | `0.999` |
| `--adam-eps` | Not used by SGD | `1e-8` |
| Weight decay | Dataset-specific value above | Dataset-specific value above |

SGD momentum carries information from previous gradients, helping updates move consistently. Nesterov modifies the update using the momentum direction.

Adam maintains moving averages of gradients and squared gradients to adapt the update scale for each parameter. The implementation used is Adam. The `momentum=0.9` field also appears in saved Adam configurations because the arguments are shared, but that field is not passed to Adam.

The selected SGD configuration exceeds Adam by **1.03 percentage points** on CIFAR-10 test accuracy and **3.19 percentage points** on CIFAR-100. These are observed results for the recorded configurations and seed.

The optimizer and scheduler construction is in the original `train_cifar.py` embedded in the [Kaggle notebooks](notebooks/kaggle/).

**4. Preprocessing and augmentation**

| Setting | CIFAR-10 | CIFAR-100 |
|---|---|---|
| Random crop | 32×32 with reflection padding 4 | Same |
| Random horizontal flip | Enabled, probability 0.5 | Same |
| Tensor conversion and normalization | Enabled | Enabled |
| Cutout | Enabled, one hole of length 16 | Disabled |
| RandAugment | Not used | 2 operations, magnitude 7 |
| CutMix | Not used | Alpha `1.0`, probability `0.5` per batch |

| Dataset | Normalization mean, RGB | Normalization standard deviation, RGB |
|---|---|---|
| CIFAR-10 | `(0.4914, 0.4822, 0.4465)` | `(0.2470, 0.2435, 0.2616)` |
| CIFAR-100 | `(0.5071, 0.4867, 0.4408)` | `(0.2675, 0.2565, 0.2761)` |

Cutout hides an image patch to encourage learning from multiple regions. Its nominal length is 16 pixels; a patch near an image boundary can be clipped.

RandAugment applies varied random image transformations. CutMix replaces a patch with one from another image in the batch and mixes their labels according to the actual patch area. Cross-entropy uses the resulting soft targets for CutMix training. Its logged training accuracy is target-weighted, whereas validation and test accuracy use the original class labels.

Cutout, RandAugment, and CutMix are data augmentation techniques. They do not update model weights themselves. Validation and test images receive tensor conversion and normalization without the training augmentations.

The original preprocessing, RandAugment and CutMix implementations are embedded in the [Kaggle notebooks](notebooks/kaggle/), together with their training configurations and launch scripts.

**5. Backbone origin: Basic, Small, or Large?**

The original [TickNet implementation](../../author/models/TickNet.py) defines three variants:

| Architecture | Blocks per stage | Total PDP blocks |
|---|---|---:|
| Author's TickNet-Basic | `[1, 1, 1, 1, 1]` | 5 |
| Author's TickNet-Small | `[1, 2, 3, 3, 1]` | 10 |
| Author's TickNet-Large | `[1, 2, 6, 5, 1]` | 15 |
| Earlier custom TickNet-L v1 | `[1, 1, 2, 2, 1]` | 7 |
| **Final custom TickNet-L Large v2** | **`[1, 2, 5, 5, 1]`** | **14** |

The archived TickNet-L v1 source explicitly identifies **TickNet-Basic** as its origin. The final model retains that custom model's compressed PDP blocks and mixed depthwise kernels, then increases width and depth. Its name, “Large v2,” identifies the custom variant; the author's original Large variant has a different block layout and channel sequence.

The backbone is the feature extractor before the classifier. CIFAR-10 and CIFAR-100 use the same backbone architecture with separately trained weights. The classifier has 10 or 100 output channels to match the dataset.

**6. Final backbone structure**

| Component | Output channels | PDP blocks | Output spatial size |
|---|---:|---:|---|
| Input batch normalization | 3 | — | 32×32 |
| Stem: 3×3 convolution | 32 | — | 32×32 |
| Stage 1 | 160 | 1 | 32×32 |
| Stage 2 | 128 | 2 | 32×32 |
| Stage 3 | 256 | 5 | 16×16 |
| Stage 4 | 512 | 5 | 8×8 |
| Stage 5 | 768 | 1 | 4×4 |
| Final 1×1 convolution | 1024 | — | 4×4 |
| Global average pooling | 1024 | — | 1×1 |
| Classifier: 1×1 convolution and flatten | 10 or 100 | — | Class logits |

The stage strides are `[1, 1, 2, 2, 2]`. Only the first block of a stage applies its stage stride; the remaining blocks use stride 1. The first two stages preserve CIFAR's small spatial resolution before later downsampling.

A final PDP block has this structure:

```text
Input ──→ linear 1×1 compression ──→ depthwise convolution(s)
  │                                  │
  │                               BN + ReLU
  │                                  │
  │                          1×1 projection + BN + ReLU
  │                                  │
  │                              SE attention
  │                                  │
  └──→ identity or projected shortcut ──→ add ──→ Output
```

Stages 1 and 2 use a 3×3 depthwise kernel. Stages 3–5 split the hidden channels equally between 3×3 and 5×5 depthwise convolutions, then concatenate the results before batch normalization and ReLU.

**7. Model changes and their reasons**

| Development | Original TickNet-Basic | Final TickNet-L Large v2 | Design reason |
|---|---|---|---|
| Greater depth | 5 blocks | 14 blocks | Adds more feature-processing steps |
| Wider stages | `[128, 64, 128, 256, 512]` | `[160, 128, 256, 512, 768]` | Increases feature capacity |
| Pointwise compression | First 1×1 maps `Cin → Cin` | Approximately `Cin → 0.75 × Cin` in stages 2–5 | Reduces computation within each block |
| Mixed depthwise kernels | 3×3 throughout | 3×3 and 5×5 branches in stages 3–5 | Captures local detail and broader spatial context |
| Distribution of depth | One block in every stage | Five blocks each in stages 3 and 4 | Concentrates additional processing at smaller spatial sizes |

For the final model, the hidden width follows:

```text
ratio = 1.0 for stage 1; 0.75 for stages 2–5
hidden_channels = max(16, int(in_channels * ratio + 4) // 8 * 8)
```

This rounds the hidden width to a multiple of eight. Stage 1 retains its 32 input channels. For a repeated stage-4 block, the first pointwise convolution compresses **512 → 384 channels**, the mixed depthwise layer processes two groups of 192 channels, and the second pointwise convolution projects **384 → 512 channels**.

Compression reduces the two main-path pointwise convolution costs relative to an uncompressed block with the same input and output widths. It does not imply a 25% reduction in the cost of the complete network, because shortcuts, attention, depth, and stage widths also contribute.

Depthwise convolution applies a spatial filter independently to each channel. Pointwise convolution handles channel mixing. This separation helps keep the model affordable as depth and width increase. The 5×5 branch adds broader spatial coverage at a modest cost compared with using a dense 5×5 convolution across all channels.

The final model also retains components already present in TickNet-Basic:

- **Residual shortcuts:** carry features and gradients through the network. Matching shapes use identity; shape changes use a 1×1 projection with batch normalization and ReLU.
- **SE attention with reduction ratio 16:** learns channel importance through global pooling and a small gating network.
- **A linear first pointwise layer:** uses no batch normalization or activation at that projection.
- **A 32-channel stem and 1024-channel final projection:** retain the Basic model's endpoint widths while the intermediate stages change.
- **Global average pooling:** aggregates spatial features before classification.

SE attention and residual connections are inherited components. The contribution is their combination with the custom width/depth allocation, channel compression, and mixed depthwise kernels. The archived source attributes the mixed-kernel idea to MixConv.

These explanations describe design intent. Establishing how much each modification improves accuracy requires ablation experiments that change one component at a time.

**8. Why the final model still satisfies the lightweight limits**

| Dataset | Learnable parameters | Forward FLOPs per 32×32 image | Parameter limit ≤6M | FLOP limit <1G |
|---|---:|---:|---|---|
| CIFAR-10 | 4,453,650 | 788,601,088 (0.788601088G) | Pass | Pass under the stated convention |
| CIFAR-100 | 4,545,900 | 788,785,408 (0.788785408G) | Pass | Pass under the stated convention |

The [saved profiles](../05_results/report_assets/model_profiles_final.json) count Conv2d and Linear operations for one evaluation forward pass at batch size 1, using **1 MAC = 2 FLOPs**. Bias additions, batch normalization, activations, pooling, residual additions, SE elementwise scaling, and data movement are excluded. This convention should accompany the numbers in the final submission.

The final model uses more capacity than Basic while remaining within the exam budget. Compression, depthwise spatial filtering, and the placement of additional blocks at smaller feature maps make that allocation possible.

**9. Suggested oral-exam explanation**

> We developed our model from TickNet-Basic's full-residual point-depth-point design. We introduced channel compression and mixed depthwise kernels, then increased width and depth to 14 blocks, concentrating additional blocks at smaller spatial resolutions. We retained residual connections and SE attention from the original design. The final model stays below 6 million learnable parameters and 1 billion FLOPs under our documented counting convention. We trained both SGD and Adam configurations for 200 epochs on CIFAR-10 and CIFAR-100. Validation selected SGD for both datasets, with recorded test accuracies of 96.04% and 81.23%.

**10. Evidence and scope**

- [Final summary CSV](../05_results/report_assets/final_required_summary.csv): metrics for the four completed runs.
- [Saved run configurations, logs, checkpoints, and predictions](../05_results/checkpoints_4_phase/): detailed experiment evidence.
- [Model profiles](../05_results/report_assets/model_profiles_final.json): parameter counts and FLOP conventions.
- [Executed notebooks](notebooks/executed/): source snapshots used during the archived training runs.
- [Original TickNet variants](../../author/models/TickNet.py): Basic, Small, and Large definitions.
- [Current consolidated custom model](../../models/ticknet_l.py): readable implementation of the custom 14-block backbone.
- [Experiment analysis and figures](../05_results/README.md): learning curves, confusion matrices, and class-level comparisons.

The reported accuracies originate from the saved Kaggle results. The restructured submission has also reproduced both selected checkpoints on all 10,000 test images per dataset, using the UofT mirror after verifying identical raw pixel, target and ordering fingerprints. See the [fresh inference records](../08_appendix/validation/README.md). Architectural rationales are not claims of individually measured accuracy gains, and the four final runs do not provide a multi-learning-rate sweep within each optimizer or uncertainty across multiple seeds.
