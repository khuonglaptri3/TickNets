# Checkpoint index and integrity evidence

`checkpoint_manifest.csv` indexes the six final epoch-200 checkpoints used in
the Basic/C/L result reports. The checkpoint binaries are stored once under
`docs/results/training_logs/`; this directory only provides the index and does
not duplicate any `.pt` file.

Each training-log directory contains the complete four-file evidence bundle:

- `config.json`: model, variant, class map, hyperparameters, complexity and
  split provenance;
- `epochs.csv`: all 200 per-epoch training records;
- `last.pt`: final model plus training state;
- `test_metrics.json`: one final held-out test evaluation.

## Manifest fields

| Field | Meaning |
| --- | --- |
| `model`, `variant`, `seed` | Experiment identity |
| `epoch` | Final checkpoint epoch; all indexed entries are epoch 200 |
| `file_size_bytes`, `sha256` | Binary size and content-integrity evidence |
| `architecture_revision` | Code-level architecture identity |
| `split_manifest_sha256` | Dataset split provenance recorded during training |
| `test_top1_percent` | Top-1 from the sibling `test_metrics.json` |
| `relative_path` | Single repository-relative location of the binary |

Recompute a checkpoint digest in PowerShell with:

```powershell
(Get-FileHash -Algorithm SHA256 -LiteralPath `
  'docs/results/training_logs/c_mid224_seed42/last.pt').Hash.ToLowerInvariant()
```

The result must equal the corresponding manifest value before the checkpoint
is loaded. Checkpoints are trusted local project artifacts and should be
loaded with `weights_only=True`.

## Mid224 100+100 continuation

The final Basic and C Mid224 checkpoints are the epoch-200 outputs of one
logical training run executed in two Kaggle sessions. Their top-level config
records the epoch-101 resume source, while nested
`checkpoint_training_config` records the phase that intentionally stopped at
epoch 100. The checkpoint restored model, optimizer, scheduler, DataLoader
generator, and RNG states; the combined `epochs.csv` contains every epoch
from 1 through 200 with the original 200-epoch cosine schedule.

## Provenance note

Basic and C share the canonical manifest hash
`9939a6ee404c6fbbdbe1b07a50763dc6d606497709385708512e1e98d71f2780`.
TickNet-L records a different raw hash because its Kaggle notebook constructed
a manifest with a different CSV schema. That raw-hash difference is retained
in the index and is not presented as proof that the test filenames differ.
