# Running Phases 1–4 with Audited Source Snapshots

The four notebooks `Phase1_CIFAR10_SGD.ipynb`, `Phase2_CIFAR10_Adam.ipynb`, `Phase3_CIFAR100_SGD.ipynb`, and `Phase4_CIFAR100_Adam.ipynb` bundle an identical snapshot of audited source code and unit tests with verifiable SHA-256 digests. Upload the notebooks directly to Kaggle; there is no need to push intermediate code changes to GitHub for the notebooks to receive updated implementations. Historical baseline and earlier grid-search notebooks still use git cloning; this self-contained snapshot guide applies strictly to Phases 1–4.

## Initial Run

1. Import the corresponding notebook on Kaggle, select a GPU accelerator (Tesla T4 or P100), and enable Internet access.
2. Run the initialization cell. This cell validates the embedded source bundle, checks GPU availability, and verifies profiler imports; it only installs missing testing/reporting dependencies without modifying the preinstalled PyTorch/CUDA environment.
3. Run the preflight test suite cell. Any failed test immediately halts execution before training begins. (GPU-dependent CUDA kernels are tested in local/CI environments with GPU; preflight notebook tests run on CPU).
4. In the configuration cell, set `EPOCHS_PER_SESSION = None` to run all remaining epochs to completion. If splitting across sessions, specify an integer budget such as `50`; this session limit applies per individual run. Two full 200-epoch runs are not guaranteed to fit within a single Kaggle session limit.
5. Run the final training cell. Scripts execute via `subprocess.run(check=True)` so command failures are never silently ignored. Dataset downloads verify exact MD5 hashes for both the downloaded archive and unpacked CIFAR binaries.

The pipeline enforces FP32 arithmetic, disables TF32, and turns on deterministic PyTorch algorithms. Every run strictly fixes seed 42, batch size 128, 200 epochs, and a stratified 45,000 train / 5,000 validation split. Validation and test sets receive no crops, flips, or Cutout. The optimal checkpoint is selected via validation Top-1 accuracy, breaking ties using lower validation loss, and preserving earlier epochs upon complete ties. The learning rate decays each epoch via CosineAnnealingLR (`eta_min=0`). Learning rates must be selected via validation metrics; never select configurations based on test set outcomes.

## Resuming in a New Session

- Download `phaseN_<dataset>_<optimizer>_recovery.zip` before the session expires. If a session times out prior to archiving, save the run directories containing `last.pt` directly from working output.
- Unpack the recovery archive, create a Kaggle Dataset containing the run directories, and attach it to the notebook.
- Set `RESUME_ROOT` to the directory path containing the run folders, for example `/kaggle/input/my-phase1-recovery` (adjust according to your Kaggle dataset structure).
- Reuse the identical notebook, matching PyTorch/TorchVision/NumPy/Pillow versions, and the same GPU architecture. Do not alter learning rate, optimizer, seed, worker counts, or total epoch ceiling upon resuming.
- The checkpoint `last.pt` preserves the random RNG state, optimizer state, scheduler state, epoch history, and a replica of the best validation checkpoint. Even if only `last.pt` survives, the trainer reconstructs `epochs.csv` and `best_val.pt`. Trainer v1 checkpoints lack sufficient state and are rejected by resume validation.

Checkpoints are committed via atomic file replacement after every completed epoch. Logs and optimal checkpoints are reconstructed directly from committed checkpoints, preventing crashed logging steps from duplicating epoch records. Official test evaluation is never executed if stopped before epoch 200. Completed runs are verified and reused when re-running the notebook without re-evaluating test data unnecessarily.

## Inspecting Results

- `*_recovery.zip`: Contains incomplete runs; used strictly for continuation, never as final deliverables.
- `*_results.zip`: Generated only when both runs reach 200 epochs and pass full verification. Includes summary CSV/Markdown tables and `phase_manifest.json` recording artifact SHA-256 digests.
- Each run contains configuration metadata, library/GPU versions, source and data SHA-256 hashes, per-epoch logs, model checkpoints, `test_metrics.json`, confusion matrix, per-sample predictions with negative log-likelihoods, and `completion.json`.
- The validator enforces 45,000/5,000 sample counts every epoch, 10,000 test samples, validation-based checkpoint selection, resource ceilings, SHA-256 integrity, and recomputes Top-1, Macro-F1, and loss from raw prediction artifacts. Any run with missing or corrupted artifacts is rejected.

After downloading all four phases, extract the run folders into `runs/` and run:

```bash
python scripts/aggregate_grid_search.py --runs-dir runs
```

By default, all 8 verified runs are required. The `--allow-partial` flag is intended only for checking intermediate progress; summary tables clearly report the count of completed runs. CSV tables preserve numeric precision without premature rounding.

## Regenerating Notebooks After Code Modifications

```bash
python scripts/generate_kaggle_phase_notebooks.py
python -m pytest tests -q
```

Snapshots are generated deterministically with normalized line endings. CI validates that committed notebooks match snapshots generated directly from checkout. After modifying source code, tests, or configurations, regenerate notebooks prior to uploading.

These automated verifications safeguard empirical rigor and reproducibility. Final classification accuracy must be measured from genuine training runs; static tests cannot prove 200-epoch accuracy or project ranking in advance.
