# Four TickNet-L midterm experiment branches implementation plan

> For agentic workers: use superpowers:executing-plans or independent parallel workers with disjoint file ownership. User authorized implementing four branches in the current conversation.

**Goal:** Deliver four runnable, documented branches derived from `feature/model-l-midterm-final` at `2bcbdb4105c5c8576829af568d6c31bb89a52c8f` for Mixup, CutMix, label smoothing/LR ablation, and one extra Stage 4 block.

**Architecture:** Add a separate experimental trainer so historical L v1 and `train_mid.py` remain reproducible. All branches share stratified validation, explicit held-out evaluation, provenance and checkpoint selection. Each branch adds only its technique and presets after the common infrastructure commit.

**Tech stack:** Python 3.12, PyTorch, torchvision, pytest; existing ImageFolder Mid32/Mid224 data and FLOP profiler.

**Design constraints:** Same class/filename validation membership at both resolutions; split seed 123 independent of training seed 42; validation fraction 0.1; batch 64, 200 epochs, SGD LR 0.1/momentum 0.9/WD 1e-4/cosine for controlled baseline; no automatic test evaluation during development. Final full-train runs may use 25,000 images only after settings/epochs are fixed. Original model revisions and checkpoints stay unchanged. Explicit branch defaults are JSON configs, never silently applied to baseline. Four branches share the source ancestor and use isolated worktrees under ignored `.superpowers/worktrees`.

**Review focus:** Partial-batch CutMix area correction; validation transforms and sample leakage; resume RNG and cosine continuity; wrong checkpoint dataset/architecture rejection; mixed-label train metric interpretation and clean evaluation loss.

## Task 1 — shared validation and trainer

- [x] Write failing tests for reproducible stratified paired splits, no augmentation in validation, trainer writing validation/best/last checkpoints without test metrics, explicit checkpoint evaluation, and resume continuity.
- [x] Implement `models/mid_experiment_data.py` and `train_mid_experiment.py`. Keep model/data/checkpoint fingerprint checks; best checkpoint uses validation Top-1 then loss tie-break. `--full-train` writes no validation-selected checkpoint. Save/restore Python/NumPy/torch RNG, DataLoader generator and optimizer/scheduler state at epoch boundaries.
- [x] Run the targeted tests and existing pipeline/model tests; commit shared infrastructure.

## Task 2 — independent mixing methods

- [x] Write failing tests for deterministic paired labels, convex Mixup images, exact clipped-area CutMix lambda, partial batches, soft-label loss, and gradients.
- [x] Implement dedicated Mixup and CutMix modules with `mix_batch(images, labels, alpha)` returning images, labels_a, labels_b, lambda. Integrate only the relevant module on its branch.
- [x] Run targeted tests and end-to-end one-epoch mixed-label runs; add branch configs and guide; commit.

## Task 3 — smoothing/LR branch

- [x] Write a failing test for train-only smoothing with clean hard-label validation loss.
- [x] Add configs changing smoothing 0→0.05 or 0.1 at LR 0.1, and separate LR 0.05/0.15 configs with smoothing 0; no confounded preset.
- [x] Run targeted and end-to-end tests; add guide and commit.

## Task 4 — Stage 4 depth branch

- [x] Write failing tests for original L unchanged, seven→eight blocks, forward/backward at both resolutions, exact params/FLOPs, and distinct checkpoint revision.
- [x] Add a separate `l_stage4` builder/revision using the existing blocks and configurable Stage 4 depth. Expose the model only in the experiment trainer, keeping the old CLI unchanged.
- [x] Run model/profile/checkpoint tests; document constraints and commit.

## Task 5 — delivery

- [x] Add reproducible branch configs, local/Kaggle commands and experiment comparison output. Run the complete applicable suite on all four worktrees and smoke each default recipe.
- [ ] Review changes, verify common ancestor/clean worktrees/branch-specific diffs, and publish the four branches to origin for teammates if repository credentials permit. Never merge into the source branch or force-push.

## Execution record

- Source checkout has two untracked analysis artifacts from the previous task; they stay in the source checkout.
- Test environment is isolated under `.superpowers/test-venv`; full GPU training is outside this implementation task.

## Review and validation record

Independent read-only review identified four issues: manifest enforcement, interrupted artifact writes, strict JSON recipe types, and source checkout provenance. Each was reproduced in tests before fixing. Added regressions for completed-run recovery, checkpoint evaluation provenance and augmented resume with two workers. Trainer revision is now mid-experiment-v2; original L v1 and historical trainer source remain byte-identical. Final suite results are recorded in docs/experiments/VERIFICATION.json. CPU checks do not establish accuracy improvements or CUDA resume behavior.

Full applicable suite passed on this branch; see VERIFICATION.json for counts and limits. Notebook packaging and real-data preflight are complete. Remote delivery remains pending.
