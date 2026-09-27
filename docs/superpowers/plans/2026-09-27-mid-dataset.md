# Mid dataset preparation and PyTorch pipeline

The user authorized steps 1 and 2: prepare paired 32/224 datasets with a
documented seed and train/test policy, and provide their training/data pipeline.
This plan records that scope; it does not authorize a full training experiment.

## Constraints and decisions

- Five classes: bird, cat, dog, frog, horse; class indices in that order.
- Per resolution: 5,000 train and 50 test images per class.
- Pair images by class and original filename, never split resolutions separately.
- User confirmed: reshuffle all five classes using seed 42, ignoring the old
  train/test membership. Choose exactly 50 test samples within each class.
- Copy exact JPEG bytes into this project's data directory.
- Standard ImageFolder layout: Mid32 or Mid224 / train or test / class / image.
- Manifest, seed/configuration, checksums, Vietnamese Markdown description,
  and individual TAR.XZ archives make the split reviewable and reproducible.
- Reject existing destinations, incomplete pairs, bad dimensions, wrong class
  counts, and identical decoded images crossing train/test.
- Loader uses seeded shuffle/workers, a stable class map and native resolution.
- Training entry point adapts existing TickNet; records per-epoch train metrics,
  configuration and checkpoints, and evaluates held-out test only at the end.
- No new network architecture, dependency installation, or full experiment.

## Tasks

- [x] Test paired splitting, seeds, preserved bytes and invalid inputs; implement
  prepare_mid_dataset.py and its generated manifests/report/archives.
- [x] Test class mapping, native sizes, seeded loading and weighted metrics;
  implement models/mid_data.py and train_mid.py.
- [x] Run preparation on the real source; verify all outputs and archives;
  document the actual split and usage in docs/DATASET_SPLIT.md.
- [x] Run the full test suite and short pipeline checks, obtain an independent
  review, resolve material findings, and report the delivered paths.

## Verification

Use the existing fresher Python environment. Run python -m pytest -q.
Fixtures exercise smaller per-class counts with the same production algorithm.
Actual data verification must show 25,000 train plus 250 test per resolution,
identical sample membership between resolutions, and unchanged image bytes.

Packaging finding: ZIP adds about 4.2 MB of entry metadata per variant and does
not compress across files. TAR.XZ preserves all JPEG bytes and meets the exam
limits (Mid32 12,306,156 bytes; Mid224 24,609,568 bytes). It is now the default.

Completed validation:
- Full suite: 16 passed; includes a one-epoch fixture training run and checkpoint reload.
- Real dataset: 25,250 unique pairs, 25,000 train / 250 test in each resolution.
- Independent seed-42 replay matches the saved manifest for all five classes.
- All 50,500 source/copied JPEGs checked; all archive members checked against SHA-256.
- Real ImageFolder batches: (8, 3, 32, 32) and (8, 3, 224, 224), correct class mapping;
  the 32-pixel check also exercised two Windows DataLoader workers.
- Independent review: no remaining material findings. Checkpoint evaluation now
  rejects mismatched/missing split provenance; direct loader seeding is documented.
- No full-dataset training experiment was run.
