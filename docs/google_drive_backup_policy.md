# Google Drive Backup Policy

Last updated: 2026-06-18

Large Colab artifacts should be stored in Google Drive only when they avoid
repeating an expensive simulation or are needed to resume a run. GitHub remains
the source for code, configs, compact metrics, manifests, reports, and plots.

## Current Required Drive Folders

Keep these under:

`/content/drive/MyDrive/Aberration-Simulation-Colab-Backups/`

for the next v15 active-hole run:

1. `v13_1m_*` or any folder containing:
   - `training_results/feature_regression_enhanced/enhanced_v13_1m_spacefill_*/training_features_enhanced.csv`
   - `feature_columns_enhanced.json`
   - `dataset_manifest.json`
   - `dataset_recovery_manifest.json`, if present
   - `label_summary.csv`

2. `v15_active_hole_250k_latest`, once generated:
   - `training_results/feature_regression_enhanced/enhanced_v15_active_hole_250k_*/training_features_enhanced.csv`
   - `feature_columns_enhanced.json`
   - `dataset_manifest.json`
   - `dataset_recovery_manifest.json`
   - compact v15 model-selection outputs, if present

3. Optional but useful if active inference will be repeated without retraining:
   - `v13_seed23_checkpoint_rebuild_*`
   - only if it contains the saved v13 checkpoint used for active-probe
     inference.

## Folders Usually Safe To Move Off Drive

If Drive is nearly full, move these to a local disk or external storage rather
than keeping them in active Drive:

- old `v11_*`, `v12_*`, `v14_*` backup folders after their compact reports and
  metrics have been pushed to GitHub;
- duplicate timestamped backup folders that contain another copy of
  `enhanced_v13_1m_spacefill_*`;
- old full-tree backups created by the previous default `training_results`
  backup policy;
- old model-selection loop folders that do not contain a uniquely needed model
  checkpoint.

Do not delete a large CSV until either:

- a newer dataset CSV supersedes it for the next run, or
- it has been moved to reliable local/external storage and its compact manifest
  remains in GitHub.

## Current Backup Behavior

New v15 backups use a stable Drive folder:

`v15_active_hole_250k_latest`

The workflow updates this folder incrementally instead of creating a new
timestamped backup tree every time. It backs up only:

- the v15 generated dataset folder;
- configs and experiments;
- Colab worker logs;
- the active failed-region report;
- v15 sampling-quality and compact model-selection outputs.

The backup utility excludes checkpoint/archive/raw-array file types by default:

- `.pt`, `.pth`, `.ckpt`, `.zip`, `.npy`, `.npz`, `.h5`

Model checkpoints should be backed up only intentionally, with a named workflow
that explicitly documents why the checkpoint is needed.
