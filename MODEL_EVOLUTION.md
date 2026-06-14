# Model Evolution

This file tracks the feature-to-coefficient regression architectures, the
reason for each change, and the minimum information needed to reproduce a run.
Per-run numeric details are saved by the notebooks in `run_manifest*.json` and
summarized in `model_registry*.csv`.

## 2026-06-14 Current Model-Loop Status

Current baseline:

- Dataset: `enhanced_v9_gap250k`
- Parent dataset: `enhanced_v6_benchmark_gap100k`
- Dataset run:
  - `training_results/feature_regression_enhanced/enhanced_v9_gap250k_20260614_055608_utc`
- Total rows: `250,000`
- Training rows / validation / blind / stress:
  - `242,556 / 1,977 / 2,370 / 3,097`
- Feature family: 66 enhanced harmonic-summary features
- Architecture: grouped-head residual MLP
- Width: 320
- Learning rate: `6e-4`
- Dropout: `0.075`
- Optimizer path: AdamW, SmoothL1 component loss, gradient clipping, plateau LR scheduler
- Split seed: `7`
- Current baseline runs:
  - `D66_grouped_width320_lr6e-4_dropout0.075_v9gap250k_d66_seed23_20260614_062447_utc`
  - `D66_grouped_width320_lr6e-4_dropout0.075_v9gap250k_d66_seed7_20260614_073553_utc`

Comparison against the frozen v8b 66-feature baseline:

| Metric | v8b frozen d66 seed23 | v9 seed23 | v9 seed7 | Direction |
|---|---:|---:|---:|---|
| weighted score | `0.03664` | `0.03051` | `0.03069` | better |
| true hard-target normalized MAE | `0.01853` | `0.01537` | `0.01557` | better |
| validation normalized MAE | `0.01421` | `0.01186` | `0.01199` | better |
| blind normalized MAE | `0.01096` | `0.00969` | `0.00968` | better |
| stress normalized MAE | `0.01021` | `0.00872` | `0.00878` | better |
| high-S3 magnitude MAE | `6.11` | `4.73` | `4.66` | better |
| high-S3 magnitude bias | `-4.44` | `-3.12` | `-3.00` | better |
| high-S3 magnitude slope | `0.812` | `0.865` | `0.851` | better |
| B2 magnitude MAE | `0.0635` | `0.0553` | `0.0552` | better |
| A3 magnitude MAE | `1.381` | `1.208` | `1.237` | better |
| validation C1 MAE | `1.77` | `1.50` | `1.59` | better |
| blind C1 MAE | `1.21` | `1.09` | `1.09` | better |
| stress C1 MAE | `1.25` | `1.02` | `1.05` | better |

Interpretation:

- The v9 250K hard-regime expansion is a clean improvement and replaces v6/v8b
  as the current baseline family.
- The 66-feature representation remains the promoted feature set. The v8b
  full defocus-difference features improved selected high-S3/B2 diagnostics but
  were not seed-stable enough on blind, stress, and A3 metrics to promote.
- C1 improved with v9, but it remains a secondary target because the current
  under/over defocus geometry likely limits direct C1 sensitivity. The method
  measures C1 through differences of already strongly defocused probe features,
  so C1 should not dominate model selection.
- The narrow v10 architecture test did not improve generalization, so the next
  useful experiment is another controlled data-scale step while keeping the v9
  architecture and feature set fixed.

Rejected v10 architecture test:

- Batch config:
  - `configs/model_selection_batch_v10_structured_head_v9_250k.json`
- Architecture:
  - `grouped_heads_structured`
- Candidate runs:
  - `D66_grouped_structured_width320_lr6e-4_dropout0.075_v10_v9gap250k_seed23_20260614_090413_utc`
  - `D66_grouped_structured_width320_lr6e-4_dropout0.075_v10_v9gap250k_seed7_20260614_102033_utc`

| Metric | v9 seed23 | v9 seed7 | v10 seed23 | v10 seed7 | Decision |
|---|---:|---:|---:|---:|---|
| weighted score | `0.03051` | `0.03069` | `0.03118` | `0.03155` | worse |
| true hard-target normalized MAE | `0.01537` | `0.01557` | `0.01584` | `0.01598` | worse |
| validation normalized MAE | `0.01186` | `0.01199` | `0.01199` | `0.01214` | worse/mixed |
| blind normalized MAE | `0.00969` | `0.00968` | `0.00974` | `0.00983` | worse |
| stress normalized MAE | `0.00872` | `0.00878` | `0.00867` | `0.00882` | mixed |
| high-S3 magnitude MAE | `4.73` | `4.66` | `4.71` | `5.06` | mixed/worse |
| high-S3 magnitude slope | `0.865` | `0.851` | `0.851` | `0.844` | worse |
| B2 magnitude MAE | `0.0553` | `0.0552` | `0.0530` | `0.0542` | slightly better |
| A3 magnitude MAE | `1.208` | `1.237` | `1.329` | `1.383` | worse |

Decision:

- Do not promote v10.
- The structured/deeper high-order head slightly helped B2, but worsened the
  main weighted score, true hard-target score, blind score, A3, and seed-stable
  high-S3 behavior.
- Keep `D66_grouped_width320_lr6e-4_dropout0.075_v9gap250k_d66_seed23` as the
  current baseline.

Queued v11 500K data-scale implementation:

- Active worker command:
  - `scripts/run_colab_v11_gap500k_d66_workflow.sh`
- Expansion config:
  - `configs/targeted_expansion_v11_500k.json`
- Batch config:
  - `configs/model_selection_batch_v11_gap500k_d66.json`
- Dataset:
  - parent: `enhanced_v9_gap250k`
  - new dataset: `enhanced_v11_gap500k`
  - expected parent rows: `250,000`
  - expected appended training-only rows: `250,000`
  - expected total rows: `500,000`
- Fixed items:
  - architecture: `grouped_heads`
  - feature count: 66
  - width: 320
  - learning rate: `6e-4`
  - dropout: `0.075`
  - optimizer/loss path: AdamW, SmoothL1, gradient clipping, plateau scheduler
  - split seed: `7`
  - validation/blind/stress: frozen v6 benchmark split manifest
- Resource settings:
  - `batch_size=65536`
  - `eval_batch_size=65536`
  - `predict_batch_size=65536`
  - shuffled mini-batches enabled
  - `max_epochs=2000`
  - `eval_every=10`
  - `patience_epochs=300`
- Jobs:
  - seed23
  - seed7
- Promotion rule:
  - promote only if both seeds improve or preserve weighted score, blind/stress
    metrics, high-S3 magnitude diagnostics, and B2/A3 vector diagnostics
    relative to v9 seed23 without easy-target regression.

v11 500K appended-row distribution:

| Regime | Rows | Reason |
|---|---:|---|
| `coupled_full_random` | `40,000` | broad nonlinear 12D coverage |
| `coupled_sparse_random` | `35,000` | broad sparse hard-regime coverage |
| `S3_high_random` | `24,000` | direct high-S3 tail coverage |
| `coupled_A3_S3_random` | `24,000` | high-order vector coupling |
| `coupled_B2_S3_random` | `20,000` | B2/S3 vector coupling |
| `coupled_A1_B2_S3_random` | `24,000` | A1/B2/S3 coupled hard regime |
| `coupled_C3_A3_S3_random` | `22,000` | C3/A3/S3 high-order coupling |
| `coupled_A1_S3_random` | `16,000` | symmetry-important A1/S3 coupling |
| `coupled_C1_C3_random` | `8,000` | scalar coupling retained |
| `coupled_C1_S3_random` | `8,000` | C1/S3 interaction retained |
| `coupled_C1_A1_S3_random` | `8,000` | C1/A1/S3 interaction retained |
| `coupled_C1_C3_S3_random` | `7,000` | C1/C3/S3 interaction retained |
| `coupled_C1_A3_S3_random` | `6,000` | C1/A3/S3 interaction retained |
| `coupled_A1_B2_random` | `3,000` | secondary vector coupling |
| `coupled_A2_B2_random` | `3,000` | secondary vector coupling |
| `coupled_C3_B2_random` | `2,000` | secondary scalar/vector coupling |

Resource notes for 500K and future 1M:

- The latest Colab session showed about `8.87/15 GB` GPU RAM used at 250K.
  A 500K full-batch run could exceed T4 memory, so v11 uses mini-batch training
  plus chunked evaluation and prediction.
- The runner still stores normalized train/validation/blind/stress tensors on
  GPU. This is expected to be acceptable for 500K because feature/target tensors
  are much smaller than full-batch activations. For 1M, implement CPU-resident
  batch loading if GPU memory becomes tight.
- Disk usage around `49.56/235.68 GB` is not an immediate blocker for 500K. For
  1M, disk can become limiting if several large feature CSV generations remain
  cached at once. Keep only necessary cached datasets in Colab and continue the
  GitHub policy of pushing only compact metrics, manifests, registry CSVs, and
  plots.

## 2026-06-12 Current Model-Loop Status

Current baseline:

- Dataset: `enhanced_v6_benchmark_gap100k`
- Parent dataset: `enhanced_v5_s3_tail60k`
- Dataset run:
  - `training_results/feature_regression_enhanced/enhanced_v6_benchmark_gap100k_20260612_051040_utc`
- Total rows: `100,446`
- Training rows / validation / blind / stress:
  - `93,011 / 1,979 / 2,371 / 3,085`
- Feature family: 66 enhanced harmonic-summary features
- Architecture: grouped-head residual MLP
- Width: 320
- Learning rate: `6e-4`
- Dropout: `0.075`
- Optimizer path: AdamW, SmoothL1 component loss, gradient clipping, plateau LR scheduler
- Split seed: `7`
- Current baseline runs:
  - `D66_grouped_width320_lr6e-4_dropout0.075_v6gap100k_seed23_20260612_051859_utc`
  - `D66_grouped_width320_lr6e-4_dropout0.075_v6gap100k_seed7_20260612_054505_utc`

Comparison against the v5 S3-tail reference:

| Metric | v5 seed23 | v6 seed23 | v6 seed7 | Direction |
|---|---:|---:|---:|---|
| weighted score | `0.05102` | `0.03690` | `0.03714` | better |
| true hard-target normalized MAE | `0.02581` | `0.01850` | `0.01865` | better |
| validation normalized MAE | `0.01900` | `0.01394` | `0.01404` | better |
| blind normalized MAE | `0.01408` | `0.01085` | `0.01092` | better |
| stress normalized MAE | `0.01341` | `0.01018` | `0.01038` | better |
| high-S3 magnitude MAE | `8.20` | `6.27` | `5.96` | better |
| high-S3 magnitude bias | `-6.39` | `-4.65` | `-4.40` | better |
| high-S3 magnitude slope | `0.709` | `0.799` | `0.812` | better |
| B2 magnitude MAE | `0.0798` | `0.0601` | `0.0615` | better |
| A3 magnitude MAE | `1.967` | `1.401` | `1.374` | better |

Interpretation:

- The benchmark-gap-aware v6 expansion is a clear improvement and replaces v5
  as the current baseline family.
- Reject the standalone S3 magnitude-loss direction as the main path. It moved
  selected high-S3 metrics but traded off against overall and B2/A3 behavior.
- The useful data expansion was not generic scale-up; it added rows resembling
  hard parent validation/stress regimes while preserving the same architecture
  and loss.

Rejected follow-up:

- Dataset: `enhanced_v7_c1_gap125k`
- Dataset run:
  - `training_results/feature_regression_enhanced/enhanced_v7_c1_gap125k_20260612_063338_utc`
- Total rows: `125,946`
- Added rows: `25,500` C1-focused training-only rows
- Candidate runs:
  - `D66_grouped_width320_lr6e-4_dropout0.075_v7c1gap125k_seed23_20260612_063900_utc`
  - `D66_grouped_width320_lr6e-4_dropout0.075_v7c1gap125k_seed7_20260612_071130_utc`

v7 result:

| Metric | v6 seed23 | v7 seed23 | v7 seed7 | Direction |
|---|---:|---:|---:|---|
| weighted score | `0.03690` | `0.03752` | `0.03791` | worse |
| validation C1 MAE | `1.77` | `1.82` | `1.89` | worse |
| blind C1 MAE | `1.22` | `1.28` | `1.28` | worse |
| stress C1 MAE | `1.20` | `1.28` | `1.32` | worse |
| high-S3 magnitude MAE | `6.27` | `6.10` | `6.20` | mixed/slightly better |
| high-S3 magnitude slope | `0.799` | `0.847` | `0.893` | better |
| B2 magnitude MAE | `0.0601` | `0.0620` | `0.0624` | worse |
| A3 magnitude MAE | `1.401` | `1.378` | `1.406` | mixed |

Decision:

- Do not promote v7.
- More C1-coupled rows did not improve C1, so C1 is likely feature-limited or
  identifiability-limited rather than simply data-limited.
- This is scientifically plausible because C1 is inferred from the difference
  of features measured under large imposed under/over defocus offsets. The
  residual C1 signal can be a weak incremental perturbation on top of the large
  defocused probe geometry.

Benchmark caveat:

- v6/v7 appended rows were marked `dataset_split_hint=training_only`.
- The stress split still changed by one row (`3,085 -> 3,084`) when v7 was
  appended. This indicates the stress threshold is still recomputed from all
  rows, including training-only rows.
- Before the next scientific comparison, freeze validation/blind/stress row
  membership explicitly or compute split thresholds only from unhinted parent
  benchmark rows.

Next controlled step:

- Do not run another data expansion first.
- Implement fixed benchmark split membership.
- Add a C1 sensitivity audit by C1 magnitude bin and coupling label.
- Add explicit under/over defocus-difference features, especially:
  - `over_Xigma_mean - under_Xigma_mean`
  - normalized defocus difference for `Xigma`, `Mu`, and `Rho`
  - selected under/over harmonic differences and ratios
- Run a v8 no-new-simulation feature batch on the existing v6 dataset before
  deciding whether any further data expansion is justified.

Queued v8 implementation:

- Active worker command:
  - `scripts/run_colab_v8_defocus_difference_workflow.sh`
- Batch config:
  - `configs/model_selection_batch_v8_defocus_difference_features.json`
- No new simulations are queued for v8.
- v8 materializes two derived feature CSV variants from the existing v6 CSV:
  - `enhanced_v8_c1diff_basic`: mean-level under/over defocus differences for
    `Xigma`, `Mu`, and `Rho`
  - `enhanced_v8_c1diff_full`: mean-level plus harmonic under/over differences,
    sums, magnitudes, and normalized differences
- v8 jobs:
  - fixed-split 66-feature baseline, seed23
  - basic defocus-difference features, seed23/seed7
  - full defocus-difference features, seed23/seed7

Prepared next data expansion:

- Config:
  - `configs/targeted_expansion_v9_250k.json`
- Expected size:
  - `100,446` v6 parent rows + `149,554` new training-only rows = `250,000`
    total rows
- This config is intentionally inactive until v8 fixed-split and feature results
  are reviewed.

## 2026-06-11 Current Model-Loop Status

Current champion:

- Dataset: `enhanced_v5_s3_tail60k`
- Parent dataset: `enhanced_v3_targeted25k`
- Dataset run:
  - `training_results/feature_regression_enhanced/enhanced_v5_s3_tail60k_20260611_084005_utc`
- Total rows: `57,446`
- Training-only appended rows: `41,000`
  - `25,000` v3 targeted rows
  - `16,000` v5 high-S3-tail rows
- Feature family: 66 enhanced harmonic-summary features
- Architecture: grouped-head residual MLP
- Width: 320
- Learning rate: `6e-4`
- Dropout: `0.075`
- Optimizer path: AdamW, SmoothL1 component loss, gradient clipping, plateau LR scheduler
- Split seed: `7`
- Current best run:
  - `D66_grouped_width320_lr6e-4_dropout0.075_s3tail60k_plateau_clip_smoothl1_seed23_20260611_090007_utc`

Comparison against the previous v3 SmoothL1 champion:

| Metric | v3 SmoothL1 champion | v5 S3-tail seed23 | Direction |
|---|---:|---:|---|
| weighted score | `0.05568` | `0.05102` | better |
| true hard-target normalized MAE | `0.02770` | `0.02581` | better |
| overall normalized MAE | `0.02057` | `0.01900` | better |
| overall p95 | `0.07983` | `0.07336` | better |
| blind normalized MAE | `0.01509` | `0.01408` | better |
| stress normalized MAE | `0.01536` | `0.01341` | better |
| B2 magnitude MAE | `0.0887` | `0.0798` | better |
| A3 magnitude MAE | `2.154` | `1.967` | better |
| high-S3 magnitude MAE | `9.54` | `8.20` | better |
| high-S3 magnitude bias | `-7.68` | `-6.39` | better |
| high-S3 magnitude slope | `0.721` | `0.709` | not improved |
| high-S3 mean angle error | `5.48 deg` | `4.39 deg` | better |
| high-S3 p95 angle error | `23.40 deg` | `16.95 deg` | better |

Interpretation:

- The v5 high-S3-tail expansion is a real model improvement overall.
- It improved high-S3 magnitude MAE, bias, RMSE, and angular diagnostics.
- It did not fix the high-S3 magnitude compression slope. The slope remained
  near `0.7`, below the desired `0.8+` direction.
- The S3 feature-saturation audit indicates that current 66-feature high-S3
  descriptors have weak linear response in the high-S3 tail:
  - `Eq43_S3_value_magnitude` high-S3 feature-vs-true slope: `0.0388`
  - high-S3 R2: `0.066`
- The requested radial-band, radially weighted, and contour/radius m=2 features
  are not present in the current CSV. This supports feature engineering before
  increasing the dataset directly to `100k`.

Benchmark caveat:

- The appended v5 rows are correctly marked `dataset_split_hint=training_only`.
- The blind split stayed identical to the previous champion.
- Validation/stress were not perfectly identical: one parent row moved from
  stress to validation (`validation 1976 -> 1977`, `stress 3098 -> 3097`).
- Before the next major comparison, lock validation/blind/stress row IDs
  explicitly rather than relying only on stable row hashing.

Next decision:

- Do not jump to `100k` rows yet.
- Keep the v5 seed23 run as the current champion.
- Next controlled work should add high-S3-sensitive features, then rerun the
  same grouped-head SmoothL1 architecture on the v5 dataset with exactly locked
  validation/blind/stress benchmarks.

## 2026-06-10 Automated Model-Loop Status

Current baseline:

- Dataset: `enhanced_v3_targeted25k`
- Feature family: 66 enhanced summary features
- Architecture: grouped-head residual MLP
- Width: 320
- Learning rate: `6e-4`
- Dropout: `0.075`
- Split seed: `7`
- Reference run:
  - `D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_bin_diag_20260609_073514_utc`

Key baseline metrics:

- Weighted selection score: `0.05945`
- True hard-target normalized MAE: `0.02947`
- Validation overall normalized MAE: `0.02188`
- Blind/stress normalized MAE: `0.01575` / `0.01648`
- High-S3 magnitude MAE/bias/slope: `10.84` / `-8.77` / `0.671`
- C1 validation MAE/RMSE/p95: `3.17` / `5.45` / `11.28`

Findings from S3 magnitude-loss tests:

- Strong S3 magnitude loss improved high-S3 magnitude MAE/bias but damaged
  overall, hard-target, B2, A3, and C1 behavior.
- Weaker S3 magnitude-loss variants gave partial high-S3 improvement, but none
  clearly replaced the v3 baseline:
  - `w0.10`: weighted score `0.05936`, high-S3 slope `0.703`, but worse hard,
    overall, blind/stress, and C1 metrics.
  - `w0.25_high2`: high-S3 slope `0.718` and slightly better blind/stress, but
    worse overall and C1 metrics.
- Conclusion: S3 compression is real, but a standalone S3 magnitude penalty
  trades off against C1 and other vector targets. Do not promote these runs as
  the new baseline.

Current training-path interpretation:

- The candidate runner currently trains the selected v3 model with full-batch
  AdamW. Row shuffling has no effect unless mini-batch training is enabled.
- Training component loss is already much lower than validation loss, so the
  bottleneck is generalization, not fitting the training rows.
- The next controlled change should test optimizer stability on the fixed v3
  dataset before changing feature count or generating 50k/100k data.

Next queued direction:

- Keep dataset and feature set fixed.
- Add optimizer/stability candidates:
  - reproducible torch/numpy seed
  - optional mini-batch shuffling
  - gradient clipping
  - ReduceLROnPlateau learning-rate decay
  - SmoothL1 component-loss candidate for heavy-tail C1/S3 errors
- Promote a candidate only if validation, blind, stress, C1, S3, B2, and A3
  diagnostics improve or remain within the accepted regression gates.

## Current Goal

Learn a mapping from probe-derived feature values to aberration coefficients:

```text
Uno / probe-profile feature values -> C1, C3, A1, B2/C21, A2, S3/C32, A3
```

Complex aberration coefficients are represented by real/imaginary target
components.

## Architecture A: Basic Hybrid Regressor

Location:

- `scripts/feature_regression_model.py`
- `notebooks/uno_feature_regression.ipynb`

Structure:

- Input: 12 Uno feature components.
- Output: 12 coefficient-vector components.
- Model: linear layer plus residual MLP.
- Original residual width: 96.
- Original training: 2500 epochs, AdamW.

Observed behavior:

- Worked well for many one-coefficient sweeps.
- Overfit the initial 446-row dataset.
- Train loss kept falling while test loss plateaued.
- Sparse A2 and C1/A1/C3 cases produced large outliers.

Interpretation:

- The original dataset was too small for the residual model.
- Feature cross-coupling became visible once A2 and C1/A1/C3 mixtures were tested.

## Architecture B: Coupled-Data Hybrid Regressor

Location:

- `scripts/feature_regression_model.py`
- `notebooks/uno_feature_regression.ipynb`

Changes:

- Added 10,000 random coupled coefficient cases.
- Kept deterministic anchor sweeps.
- Added batched GPU feature extraction to avoid Colab OOM.
- Reduced and regularized residual training.
- Added best-test checkpoint restore.
- Added run tracking:
  - `run_manifest.json`
  - `model_registry.csv`

Current configured training:

- Input: 12 Uno feature components.
- Output: 12 coefficient-vector components.
- Residual hidden width: 64 in the notebook.
- Epochs: 6000.
- Learning rate: `8e-4`.
- Residual penalty: `3e-3`.
- Weight decay: `1e-4`.

Expected purpose:

- Establish whether the original 12 feature values contain enough information
  for fully coupled coefficient recovery.

## Architecture C: Enhanced Feature Regressor

Location:

- `notebooks/uno_feature_regression_enhanced.ipynb`

Changes:

- Expanded input from 12 Uno components to 66 feature components.
- Added raw under/over-focus `Xigma`, `Mu`, and `Rho` harmonic features.
- Added a larger residual model with LayerNorm.
- Added weighted loss to emphasize previously weak targets.
- Added early stopping.
- Added run tracking:
  - `run_manifest_enhanced.json`
  - `model_registry_enhanced.csv`

Latest observed configuration:

- Input: 66 enhanced feature components.
- Output: 12 coefficient-vector components.
- Residual hidden width: 128.
- Maximum epochs: 8000.
- Best epoch from latest downloaded run: 3075.
- Best weighted test MSE from latest downloaded run: about `0.0614`.

Latest observed behavior:

- Run inspected from local Colab download:
  - `Downloads from Colab/feature_regression_enhanced/`
  - run family: `enhanced_coupled10k`
  - input features: 66
  - hidden width: 128
  - best epoch: 3075
  - best weighted test MSE: about `0.0614`
- Test MAE improved versus Architecture B on every target:
  - `C1`: `12.29 -> 4.93`
  - `C3`: `0.068 -> 0.033`
  - `A1_x/A1_y`: about `3.5-3.6 -> 1.4`
  - `B2_x/B2_y`: about `0.16-0.17 -> 0.09`
  - `A2_x/A2_y`: about `0.49-0.51 -> 0.19-0.21`
  - `S3_x/S3_y`: about `6.9-7.1 -> 4.5-4.7`
  - `A3_x/A3_y`: about `7.0-7.2 -> 1.7-1.8`
- Remaining weak regions:
  - `coupled_full_random`: train MAE about `1.47`, test MAE about `4.27`
  - `coupled_sparse_random`: train MAE about `1.12`, test MAE about `2.77`
  - `S3` high-amplitude bins: test vector RMSE about `22.9`
  - `A3` high-amplitude bins: test vector RMSE about `8.1`

Interpretation:

- Adding raw line-characteristic harmonics was a clear improvement.
- The present failure is now primarily generalization on fully coupled and
  sparse coupled random cases, especially high-amplitude `S3` and mixed
  `C1/S3/A3` behavior.
- The train-test gap is real in the difficult random regimes. The next change
  should improve data coverage and regularization before increasing model size.

## Architecture D: Enhanced V2 Generalization Run

Location:

- `notebooks/uno_feature_regression_enhanced.ipynb`

Decision:

Use the enhanced 66-feature representation from Architecture C, but change the
next Colab experiment to test generalization rather than capacity.

Changes for the next run:

- Run name prefix:
  - `enhanced_v2_coupled16k_stratified_dropout_`
- Increase random coupled cases from `10,000` to `16,000`.
- Put most of the additional cases into the weak regimes:
  - `coupled_full_random`: `2500 -> 5000`
  - `coupled_sparse_random`: `500 -> 3000`
  - `coupled_C1_A1_C3_A2_random`: `1500 -> 2000`
  - `coupled_A3_S3_random`: `1000 -> 1500`
- Keep deterministic anchor sweeps unchanged.
- Use a train/test split stratified by `sweep_label` so every label contributes
  to the test split.
- Add mild dropout (`0.05`) to the residual MLP.
- Keep hidden width at 128, early stopping, weighted target loss, and the same
  66 enhanced feature columns.

Notebook run modes:

- Full Architecture D run:
  - Rerun sections 5-10 in `notebooks/uno_feature_regression_enhanced.ipynb`.
  - This regenerates a new `16,000`-random-case enhanced feature table and then
    trains the v2 model.
- Existing-CSV model-only run:
  - Use section 12 in the same notebook.
  - Set `RUN_EXISTING_CSV_V2_RERUN = True`.
  - This reuses an existing `training_features_enhanced.csv` and tests the v2
    stratified/dropout trainer without running new probe simulations.
  - Run name prefix:
    - `enhanced_v2_existingcsv_stratified_dropout_`
  - This mode is useful for quickly testing the split/training change on a
    previously generated enhanced feature table, but it does not test the
    `16,000` random-case data change.

Purpose:

- Test whether the Architecture C train-test gap is mainly data coverage and
  regularization, not insufficient model capacity.
- Make the test split more stable and easier to compare across model runs.

Primary success criteria:

- Lower test MAE/RMSE on `coupled_full_random` and `coupled_sparse_random`.
- Lower high-amplitude `S3` and `A3` amplitude-bin vector RMSE.
- Maintain the Architecture C gains on `A1`, `A2`, `B2`, and `C3`.
- Avoid a larger train-test gap than Architecture C.

Latest observed Architecture D results:

- Downloaded result inspected:
  - `enhanced_v2_existingcsv_stratified_dropout_20260604_061258_utc_feature_regression_enhanced_results.zip`
- Source dataset:
  - `enhanced_v2_coupled16k_stratified_dropout_20260604_060559_utc`
  - `16,446` rows, `13,158` train, `3,288` test
  - CSV SHA-256: `8cb91ef4ae645529fcd222b2ab18888cd03adb83c9fa121a77b5db9c2d05b9c4`
- Training:
  - best epoch: `7750`
  - best weighted test MSE: about `0.0396`
- Compared with Architecture C, the hard random regimes improved:
  - `coupled_full_random` test MAE: `4.27 -> 3.01`
  - `coupled_sparse_random` test MAE: `2.77 -> 1.81`
  - `coupled_C1_A1_C3_A2_random` test MAE: `1.74 -> 1.21`
- Target-level tradeoff:
  - `C1` test MAE improved: `4.93 -> 3.46`
  - `S3_x/S3_y` improved: about `4.5-4.7 -> 4.1-4.2`
  - `A2_x/A2_y` regressed slightly: about `0.19-0.21 -> 0.22`
  - `A3_x/A3_y` regressed: about `1.7-1.8 -> 2.1`
- Interpretation:
  - The v2 data/split/dropout change improved the intended hard coupled cases.
  - The cost was worse `A3` and slightly worse `A2`.
  - Next improvement should target high-amplitude `S3` without sacrificing `A3`,
    likely with target-group heads or loss weights adjusted separately for
    `A3` versus `S3`.

## Architecture E: Raw-Angle Feature Regressor

Location:

- `notebooks/uno_feature_regression_raw_angles.ipynb`

Decision:

Test whether direct angular samples of the Uno line-profile characteristics
carry useful information that is lost when the curves are compressed into
low-order harmonic features.

Important implementation detail:

- `extract_line_profiles_from_stack()` treats `num_lines` as including the
  duplicated `180 deg` endpoint and then drops that endpoint.
- With `PROFILE_STEP_DEGREES = 10`, the raw-angle notebook uses 18 unique line
  orientations:
  - `0, 10, 20, ..., 170 deg`
- The nonduplicated raw-angle feature count is therefore:
  - `18 angles * 3 characteristics * 2 focus states = 108`
- The total input dimension is:
  - `12 collapsed Uno features + 108 raw-angle features = 120`

Feature representation:

- Keep the original 12 collapsed Uno features.
- Add raw angular samples for:
  - `Xigma(theta_k)`
  - `Mu(theta_k)`
  - `Rho(theta_k)`
- Keep under-focus and over-focus separate:
  - `under_Xigma_theta_000`, ...
  - `over_Rho_theta_170`, ...

Training setup:

- Same v2 random-case distribution as Architecture D:
  - `16,000` random coupled cases
  - extra full-random and sparse-random cases
- Same stratified split by `sweep_label`.
- Same weighted loss, dropout, early stopping, hidden width, and diagnostics.
- Run name prefix:
  - `rawangle18_v1_coupled16k_stratified_dropout_`
- Output root:
  - `training_results/feature_regression_raw_angles/`

Purpose:

- Compare raw orientation-sample features against the 66-feature harmonic
  enhanced model on the same data-generation regime.
- If raw-angle features improve `S3`, `A3`, and full-random cases, then the
  harmonic compression is losing useful coupled-case information.
- If they do not improve those cases, the remaining issue is more likely model
  structure, target coupling, or non-uniqueness.

Primary success criteria:

- Beat Architecture D on:
  - high-amplitude `S3` vector RMSE
  - high-amplitude `A3` vector RMSE
  - `coupled_full_random` test MAE/RMSE
  - `coupled_sparse_random` test MAE/RMSE
- Avoid regressing easy targets such as `C3`, `B2`, and `A2`.

Latest observed Architecture E results:

- Downloaded result inspected:
  - `Downloads from Colab/feature_regression_enhanced/rawangle18_v1_coupled16k_stratified_dropout_20260604_070648_utc_feature_regression_raw_angles_results.zip`
- Additional model-only rerun inspected:
  - `Downloads from Colab/feature_regression_enhanced/rawangle18_existingcsv_stratified_dropout_20260604_071338_utc_feature_regression_raw_angles_results.zip`
- Run identity:
  - run name: `rawangle18_v1_coupled16k_stratified_dropout_20260604_070648_utc`
  - Git commit: `ba7c485`
  - device: `cuda`
- Dataset and model:
  - `16,446` rows, `13,158` train, `3,288` test
  - input features: `120`
  - hidden width: `128`
  - dropout: `0.05`
  - best epoch: `7875`
  - best weighted test MSE: about `0.0397`
- Compared with Architecture D, raw-angle features were mixed rather than a
  clear improvement:
  - overall test MAE/RMSE: `1.027/2.519 -> 1.090/2.592`
  - `coupled_full_random` test MAE improved slightly: `3.006 -> 2.964`
  - `coupled_sparse_random` test MAE regressed: `1.810 -> 1.867`
  - `C1` test MAE improved slightly: `3.462 -> 3.357`
  - `A1`, `B2`, and `A3` mostly regressed.
  - `A2` was essentially unchanged.
  - `S3` vector RMSE improved slightly: `12.337 -> 11.887`
  - `A3` vector RMSE regressed: `5.505 -> 5.832`
- The additional model-only rerun used the same source CSV:
  - source run: `rawangle18_v1_coupled16k_stratified_dropout_20260604_070648_utc`
  - source CSV SHA-256:
    `963de4c3bec82419e546751215a38d271afc9c8a5365ce5bd35a86afa26b8ce0`
  - best epoch: `7875`
  - best weighted test MSE: about `0.0413`
  - overall test MAE/RMSE: `1.088/2.605`
  - `coupled_full_random` test MAE: `3.047`
  - `coupled_sparse_random` test MAE: `1.887`
  - `S3` vector RMSE: `12.334`
  - `A3` vector RMSE: `5.799`
- Interpretation:
  - The 120-feature raw-angle representation did not beat the 66-feature
    harmonic-summary representation overall.
  - Direct angular samples appear to help `C1`, full-random cases, and `S3`
    vector error only marginally.
  - The model-only rerun is slightly worse than the first raw-angle training
    run, so ordinary reruns of this architecture are not moving the bottleneck.
  - The model-only rerun zip is not self-contained because it does not include
    `training_features_raw_angles.csv`; it is reproducible only together with
    the source full-run archive named above, whose CSV hash is recorded in the
    rerun manifest.
  - The larger input dimension likely adds redundant/noisy orientation samples
    that the same 128-wide residual model does not use efficiently.
  - The next architecture should keep the 66-feature representation as the
    baseline and test model structure or target grouping, not simply add more
    raw angular inputs.

## Current Working Hypotheses

- The feature values are useful and monotonic in one-coefficient sweeps.
- Coupled random cases reveal cross-talk between feature values.
- Fully coupled high-amplitude `A3`/`S3`/`C1` cases are the present bottleneck.
- More samples in the weak regimes plus mild regularization should be tested
  before adding another larger neural network.
- Architecture E shows that raw angular samples alone are not enough to solve
  the remaining coupled-case errors.
- The next architecture should encode more structure, such as staged prediction,
  coefficient-group heads, or calibrated correction heads, before adding more
  redundant profile-shape inputs.

## Reproducibility and Tracking Standard

Every model-producing notebook must write one timestamped run folder and one zip
for that folder. The zip is the artifact to download from Colab.

Required run folder naming:

- Include the architecture/run family, key dataset size, and UTC timestamp.
- Example: `enhanced_v2_coupled16k_stratified_dropout_YYYYMMDD_HHMMSS_utc`.

Required files inside each run folder:

- Training data table:
  - `training_features.csv` or `training_features_enhanced.csv`
- Feature columns:
  - `feature_columns*.json`
- Model output:
  - `*.pt`
- Metrics:
  - `metrics*.json`
  - `history*.csv`
  - `predictions*.csv`
  - diagnostic CSVs such as amplitude-bin summaries
- Plots:
  - training-history plot
  - prediction-scatter plot
- Normalization:
  - `normalization*.json`
- Model summary:
  - `model_summary*.txt`
- Run manifest:
  - `run_manifest*.json`

Required manifest fields:

- Run identity:
  - run name
  - UTC creation time
  - Git commit
  - notebook/model architecture family
- Runtime:
  - Python version
  - platform
  - device
- Dataset:
  - CSV path
  - CSV SHA-256
  - row count
  - train/test count
  - label counts
  - random seed and random-case counts
  - batch size and profile-extraction settings
- Features and targets:
  - feature column names
  - target column names
  - target vector convention
- Training:
  - train/test split strategy
  - train/test split seed
  - model dimensions
  - dropout and target weights when used
  - optimizer settings
  - maximum epochs, early-stopping patience, best epoch
  - best validation/test score used for checkpoint restore

Required registry behavior:

- Append one row per run to `model_registry.csv` or
  `model_registry_enhanced.csv`.
- Include the registry CSV in the downloaded zip.
- Include enough summary metrics to compare runs without opening every manifest.

Random number generation:

- Random case generation and train/test splitting should keep explicit seeds in
  the manifest.
- Exact bitwise reproducibility is not required because Colab/GPU operations and
  PyTorch kernels may still be nondeterministic.

## Candidate Next Steps

- Add a linear/ridge baseline to quantify how much the neural residual improves
  over direct calibrated inversion.
- Train grouped heads on top of the 66-feature Architecture D baseline:
  - scalar head for C1/C3,
  - low-order harmonic head for A1/B2/A2,
  - high-order harmonic head for S3/A3.
- Optionally test a hybrid `66 + selected raw angles` model only after checking
  feature importance from Architecture E.
- Train curriculum-style:
  - one-coefficient sweeps,
  - weak coupled cases,
  - high-amplitude full coupled cases.
- Evaluate uniqueness using nearest-neighbor distances in feature space before
  increasing model capacity again.

## Automated Colab Loop Policy

Initial real-loop mode:

- Use existing cached/downloaded feature CSVs first. This tests model structure,
  preprocessing, and hyperparameters before spending GPU time regenerating probe
  simulations.
- Use a one-hour command timeout per Colab worker cycle for now. Revisit this
  limit before starting longer simulation-heavy loops.
- Do not rerun the same model/config silently. The Colab worker records and
  compares config fingerprints; when `require_new_config_each_cycle` is true it
  waits for a changed config before the next cycle.

Model-selection rule:

- Use a weighted selection score rather than overall average error alone.
- Prioritize:
  - `coupled_full_random`
  - `coupled_sparse_random`
  - `S3_x`, `S3_y`
  - `A3_x`, `A3_y`
- Allow small regressions on easier targets, but reject a candidate if easy
  target MAE regresses by more than about `5-10%` versus the current baseline.
- The helper script is:
  - `scripts/select_regression_model.py`

Artifact policy:

- Keep large artifacts out of GitHub:
  - `.pt` models
  - checkpoints
  - large training CSVs
  - prediction CSVs
  - large result folders
- Push only compact artifacts:
  - worker manifests
  - run manifests
  - small metrics JSON files
  - registry CSV rows
  - compact diagnostic CSV summaries
  - small plots
- Colab disk was observed with enough headroom for caching, but duplicate
  simulations should still be avoided. Cached datasets should be identified by
  coefficient case table, profile/focus settings, feature extractor version,
  target convention, and CSV SHA-256.
