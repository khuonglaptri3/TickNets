# CIFAR Trainer v2: Testing and Reliability Specification for Phases 1–4

This is the active technical specification governing checkpoint recovery, result verification, and Kaggle deployment. Historical v1 specifications and plans represent project records; completion checkboxes denote implementation milestones rather than completed 200-epoch training runs or final examination submission.

## Acceptance Criteria

- SGD/Adam, CIFAR-10/100, and TickNet-L/Basic include unit tests for training, validation checkpoint selection, and test evaluation.
- Multi-session resumption must be bit-for-bit identical to continuous execution regarding model weights, optimizer states, scheduler states, epoch histories, and best validation records. Covered across multi-worker augmentations and CUDA environments (CUDA tests skipped on CPU-only machines).
- Serializes Python, NumPy, PyTorch, and CUDA RNG states alongside train/validation/test DataLoader generators at epoch boundaries.
- `last.pt` commits via atomic replace prior to secondary artifacts, retaining a clone of best validation weights to guarantee recovery after logging crashes or output path changes.
- Resumption strictly rejects recipe tampering, runtime/GPU architectural drift, modified source code, pixel/label corruption, or dataset membership changes.
- Intermediate pauses omit test set evaluation; completed runs generate verifiable completion tokens for safe reuse.
- Local CIFAR caches are verified only when all extracted train, test, and metadata binaries match official checksums.
- Metric calculation routines are tested against closed-form mathematical fixtures and verified against per-sample prediction logs.
- Four Kaggle notebooks package self-contained source and test bundles with verified SHA-256 digests, without git cloning branch heads.
- Subprocesses execute with `check=True`; partial runs emit `recovery.zip` archives.
- Final `results.zip` packaging verifies 200 epochs, 45,000/5,000 train/val splits, 10,000 test samples, validation-based checkpoint selection, and uncorrupted artifact trees.

## Standardized Parameters

- CIFAR-100 normalization: mean `(0.5071, 0.4867, 0.4408)`, std `(0.2675, 0.2565, 0.2761)`.
- CosineAnnealingLR: `T_max=200`, `eta_min=0` across both SGD and Adam.
- TickNet-L CIFAR-10: 1,100,105 learnable parameters, 157,828,544 FLOPs.
- TickNet-L CIFAR-100: 1,169,315 learnable parameters, 157,966,784 FLOPs.
- FLOP convention covers Conv2d/Linear, batch size 1, eval mode forward pass ($1\text{ MAC} = 2\text{ FLOPs}$), excluding BN, activations, pooling, and element-wise ops.
- 100% train-from-scratch is the team's experimental choice (the DOCX exam prompt does not prohibit pretraining). Backbone Conv layers use Kaiming Uniform; classifier weights use Xavier Normal; SE Linear layers and classifier biases retain standard PyTorch constructor defaults.

## Evidence and Scope

Final test outcomes, skipped cases, and deployment instructions are documented in [`docs/kaggle/README.md`](../../kaggle/README.md) and [`docs/kaggle/VALIDATION.md`](../../kaggle/VALIDATION.md). Official dataset integration tests execute only when verified raw files are present locally, avoiding ad-hoc network downloads during pytest. CUDA tests run only when GPUs are available.

Audited source code does not equate to examination completion: the team must still run all eight CIFAR experiments to completion, author the technical PDF report, defend the architectural evolution from the midterm, ensure uniqueness against other teams, prepare for oral defense, and submit on time.
