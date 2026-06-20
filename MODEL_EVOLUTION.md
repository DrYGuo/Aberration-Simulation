# Model Evolution

This file tracks the feature-to-coefficient regression architectures, the
reason for each change, and the minimum information needed to reproduce a run.
Per-run numeric details are saved by the notebooks in `run_manifest*.json` and
summarized in `model_registry*.csv`.

## 2026-06-20 v15 Active-Hole Retest And v16 Metric Direction

Current promoted baseline remains:

- `D66_grouped_width320_lr6e-4_dropout0.075_v13_1m_d66_seed23_residual_nn_20260615_065556_utc`

Completed diagnostic workflow:

- Worker config id: `v15-active-hole-retest`
- Worker script: `scripts/run_colab_v15_active_hole_retest_workflow.sh`
- Retest report:
  - `training_results/model_selection_reports/v15_active_hole_retest_20260620_082522_utc/active_hole_retest_report.md`
- Drive backup:
  - `/content/drive/MyDrive/Aberration-Simulation-Colab-Backups/v15_active_hole_retest_latest`
- Rebuilt v15 checkpoint:
  - `training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v15_active_hole_expanded_250k_d66_seed23_checkpoint_rebuild_20260620_060115_utc`

The v15 active-hole-expanded model is not promoted as the general baseline
because the old fixed validation/blind/stress benchmark regressed versus v13.
However, the retest confirms that v15 repaired the searched active-hole
regions.

Matched active-hole retest (`n=39,896`):

| active-hole metric | v13 RMSE | v15 RMSE | interpretation |
|---|---:|---:|---|
| weighted normalized abs error | `0.05791` | `0.04434` | v15 better |
| overall mixed-unit abs error | `3.725` | `2.839` | v15 better |
| S3 vector error | `15.84 um` | `13.49 um` | v15 better |
| A3 vector error | `11.42 um` | `7.27 um` | v15 much better |
| B2 vector error | `0.312 um` | `0.250 um` | v15 better |

Earlier v13 top active-hole failures (`n=3,988`) are a harder selected
subpopulation:

| metric | v13 top-failure RMSE |
|---|---:|
| weighted normalized abs | `0.14255` |
| overall mixed-unit abs | `9.30051` |
| C1 | `10.93 nm` |
| C3 | `0.0366 mm` |
| A1 vector | `17.61 nm` |
| B2 vector | `0.669 um` |
| A2 vector | `0.653 um` |
| S3 vector | `35.68 um` |
| A3 vector | `30.73 um` |

The exact v15 result on that same top-3,988 subset still needs a compact
derived report from the full Drive retest artifacts. GitHub has only compact
top-1000 comparison rows and per-run summaries, by artifact policy.

Scientific conclusion:

- The old fixed stress/hard benchmarks are not representative enough for the
  full 12D coefficient space.
- Active 12D hole search found subspaces not adequately covered by the previous
  benchmark suite.
- Pure active-hole training can repair holes but degrade old benchmark balance.
- The next model-selection step should use a benchmark suite, not a single
  fixed validation/stress split.

Proposed v16 model-selection suite:

| suite component | role |
|---|---|
| broad representative benchmark | estimates generalization over the expected 12D use domain |
| frozen active-hole benchmark | tracks repair of known sparse/dense failure regions |
| held-out new-hole challenge benchmark | prevents overfitting to the old active-hole set |
| anchor/easy benchmark | prevents regression on one-coefficient and Uno-success regimes |

Proposed suite score:

```text
score =
  0.35 * broad_blind_stress_score
  0.25 * active_hole_score
  0.15 * new_hole_challenge_score
  0.15 * hard_vector_score
  0.10 * anchor_regression_penalty
```

Suggested v16 balanced expansion:

| component | fraction |
|---|---:|
| known active-hole repair rows | `35%` |
| newly searched hole candidates | `20%` |
| broad coupled-full/coupled-sparse Sobol/LHS rows | `25%` |
| benchmark-preserving anchors | `15%` |
| one-coefficient/Uno-success sweeps | `5%` |

Promotion rule for v16:

- Improve active-hole metrics over v13.
- Do not regress broad blind/stress by more than about `5%`.
- Do not regress anchor/easy targets by more than `5-10%`.
- Require gains to survive a held-out new-hole challenge set.

See also:

- `docs/next_model_selection_strategy_20260620.md`
- `docs/session_handoff_20260620.md`

## 2026-06-18 Active 12D Hole Search And v15 Preparation

Current promoted baseline remains:

- `D66_grouped_width320_lr6e-4_dropout0.075_v13_1m_d66_seed23_residual_nn_20260615_065556_utc`

Active-search result:

- Completed active 12D hole-search cycles:
  - `v13_active_12d_hole_search_20260618_025610_utc`
  - `v13_active_12d_hole_search_v2_focused_sparse_20260618_030707_utc`
  - `v13_active_12d_hole_search_v3_high_amp_alignment_20260618_031730_utc`
  - `v13_active_12d_hole_search_v4_dense_alignment_bridge_20260618_032829_utc`
- Synthesis:
  - `training_results/model_selection_reports/active_12d_hole_search_synthesis_20260618_033955_utc/active_12d_hole_search_synthesis_report.md`
- Failed-region error report:
  - `training_results/model_selection_reports/active_failed_region_error_report_20260618_043212_utc/active_failed_region_error_report.md`

The active search probed the frozen v13 model in normalized 12D coefficient
space using GA, far-NN, Sobol/LHS, high-amplitude alignment, bridge, and
residual-jitter proposals. It found:

| active-search aggregate | count |
|---|---:|
| coverage-limited sparse failures | `1206` |
| mixed failures | `261` |
| dense/model-feature-loss failures | `152` |

The top failures are overwhelmingly in `coupled_full_random` and concentrate in
high-amplitude S3/A3/B2/A1 vector corners. The failed-region local errors are:

| target | MAE | RMSE | p95 |
|---|---:|---:|---:|
| C1 (nm) | `7.83` | `10.36` | `19.30` |
| C3 (mm) | `0.0263` | `0.0344` | `0.0619` |
| A1 vector (nm) | `17.41` | `22.12` | `44.31` |
| B2 vector (um) | `0.558` | `0.730` | `1.571` |
| A2 vector (um) | `0.554` | `0.662` | `1.228` |
| S3 vector (um) | `36.83` | `41.19` | `68.23` |
| A3 vector (um) | `24.75` | `33.82` | `75.13` |

These active failed-region errors are far larger than the ordinary v13
validation/blind/stress errors, so they represent real local holes rather than
normal benchmark uncertainty.

Prepared next test:

- Dataset config:
  - `configs/targeted_expansion_v15_active_hole_250k.json`
- Batch config:
  - `configs/model_selection_batch_v15_active_hole_250k_d66.json`
- Worker script:
  - `scripts/run_colab_v15_active_hole_250k_workflow.sh`
- Worker config:
  - `experiments/colab_worker_model_loop.json`
  - config id: `v15-active-hole-250k-d66`

v15 is a controlled data-only test. It appends `250,000` training-only rows to
the v13 1M parent dataset, using:

- `30%` jitter around active failed-subspace cluster centers;
- `25%` high-amplitude A1/B2/S3/A3 angle-structured cases;
- `20%` far/bridge coupled-full random coverage;
- `10%` coupled sparse controls;
- `10%` global/bridge controls;
- `5%` orthogonal-style diagnostic controls.

The model architecture remains fixed: 66-feature grouped-head residual MLP,
width `320`, dropout `0.075`, learning rate `6e-4`, seed `23`, SmoothL1,
gradient clipping, plateau LR scheduler, and frozen benchmark-v2
validation/blind/stress splits.

Promotion rule:

- promote v15 only if it reduces active sparse-tail failures and improves or
  preserves weighted score, blind/stress metrics, S3/A3/A1 diagnostics, and
  easy targets relative to v13 seed23;
- do not promote v15 merely because training loss decreases;
- keep dense high-amplitude failures as a separate diagnostic group after v15.

## 2026-06-15 Current Model-Loop Status

Current baseline:

- Dataset: `enhanced_v13_1m_spacefill`
- Parent dataset: `enhanced_v12_benchmark_v2`
- Dataset run:
  - `training_results/feature_regression_enhanced/enhanced_v13_1m_spacefill_20260615_021848_utc`
- Total rows: `1,075,000`
- New v13 training-only rows: `500,000`
- Training / validation / blind / stress rows:
  - `992,556 / 26,977 / 27,370 / 28,097`
- Benchmark split manifest:
  - `configs/benchmark_split_v12_v2_row_keys.json`
- Feature family: 66 enhanced harmonic-summary features
- Architecture: grouped-head residual MLP
- Width: 320
- Learning rate: `6e-4`
- Dropout: `0.075`
- Optimizer path: AdamW, SmoothL1 component loss, gradient clipping, plateau LR scheduler
- Split seed: `7`
- Current baseline run:
  - `D66_grouped_width320_lr6e-4_dropout0.075_v13_1m_d66_seed7_20260615_042743_utc`

Best promoted run:

- `D66_grouped_width320_lr6e-4_dropout0.075_v13_1m_d66_seed7_20260615_042743_utc`

Comparison against the v12 500K benchmark-v2 baseline:

| Metric | v12 500K benchmark-v2 | v13 1M | Direction |
|---|---:|---:|---|
| weighted score | `0.01342` | `0.01258` | better |
| true hard-target normalized MAE | `0.00717` | `0.00674` | better |
| hard-label normalized MAE | `0.00580` | `0.00554` | better |
| validation normalized MAE | `0.00547` | `0.00521` | better |
| blind normalized MAE | `0.00601` | `0.00565` | better |
| stress normalized MAE | `0.00623` | `0.00584` | better |
| high-S3 magnitude MAE | `0.949` | `0.876` | better |
| high-S3 magnitude bias | `-0.580` | `-0.480` | better |
| high-S3 magnitude slope | `0.970` | `0.974` | slightly better |
| high-S3 mean angle error | `0.419 deg` | `0.342 deg` | better |
| high-S3 p95 angle error | `1.683 deg` | `1.349 deg` | better |
| B2 magnitude MAE | `0.0225` | `0.0220` | slightly better |
| A3 magnitude MAE | `0.592` | `0.559` | better |
| C1 validation MAE | `0.748` | `0.755` | slightly worse |
| C1 blind MAE | `0.737` | `0.717` | better |
| C1 stress MAE | `0.715` | `0.695` | better |

Interpretation:

- The v13 1M expansion is the current promoted baseline. It improves weighted
  score, true hard-target MAE, validation/blind/stress MAE, S3 high-magnitude
  compression, and B2/A3 vector diagnostics relative to the v12 500K
  benchmark-v2 run.
- The v13 result used the same v12 benchmark-v2 validation/blind/stress split
  manifest as v12, so the v12 -> v13 comparison is a clean training-data-scale
  comparison.
- The main data-scale trend remains positive: the latest learning-curve report
  shows v9 250K -> v13 1M improvements of `58.8%` weighted score, `56.1%`
  true hard-target MAE, `41.7%` blind MAE, and `33.0%` stress MAE.
- The 66-feature representation remains the promoted feature set. The v8b
  full defocus-difference features improved selected high-S3/B2 diagnostics but
  were not seed-stable enough on blind, stress, and A3 metrics to promote.
- C1 remains a secondary target because the current
  under/over defocus geometry likely limits direct C1 sensitivity. The method
  measures C1 through differences of already strongly defocused probe features,
  so C1 should not dominate model selection.
- The narrow v10 architecture test did not improve generalization. The successful
  route since v9 has been controlled data-scale expansion while keeping the
  grouped-head 66-feature architecture fixed.

Sampling-quality status:

- Current v13 results include the standalone sampling-quality dashboard:
  - `training_results/model_selection_reports/sampling_quality_v13_1m_d66/sampling_quality_report.md`
  - `training_results/model_selection_reports/sampling_quality_v13_1m_d66/sampling_quality_summary.json`
- Sampling-quality recommendation: `PASS`.
- The v13 dataset has `1,075,000` total rows, including `500,000` new
  training-only v13 rows and the `75,000` row v12 benchmark-v2 split.
- Training-only leakage into validation/blind/stress is `0`.
- Relative-angle coverage for A1-S3, B2-S3, and A3-S3 is balanced across
  aligned, orthogonal, anti-aligned, and random categories.
- Coverage diagnostics still show nonuniform marginal and pairwise density, as
  expected for a physics-weighted hard-regime sampler rather than a uniform
  12D grid. The dashboard is used to diagnose coverage, not as the sole
  criterion for model selection.

Coefficient uncertainty table:

- Presentation table:
  - `docs/regression_uncertainty_by_training_size.md`
- The table reports validation p95 absolute error. For vector coefficients it
  uses the worse x/y component as a conservative one-number uncertainty level.
  This is more useful as an error-bar style presentation number than RMSE
  because it describes a 95% absolute-error envelope.

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
- Keep the original grouped-head architecture for the next data-scale step. The
  later v11 500K run superseded v9, and the later v13 1M run now supersedes
  both.

Completed v11 500K data-scale implementation:

- Active worker command:
  - `scripts/run_colab_v11_gap500k_d66_workflow.sh`
- Expansion config:
  - `configs/targeted_expansion_v11_500k.json`
- Batch config:
  - `configs/model_selection_batch_v11_gap500k_d66.json`
- Dataset:
  - parent: `enhanced_v9_gap250k`
  - new dataset: `enhanced_v11_gap500k_20260614_205607_utc`
  - parent rows: `250,000`
  - appended training-only rows: `250,000`
  - total rows: `500,000`
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
- Completed jobs:
  - seed23
  - seed7
- Sampling strategy:
  - per-regime Latin-hypercube sampling for all v11 labels
  - physics-guided regime proportions retained
  - high-S3 tail forcing retained for S3-linked labels
  - C1 magnitude-bin balancing retained for selected C1 labels
  - balanced relative-angle strata for vector-vector S3 couplings:
    aligned, orthogonal, anti-aligned, and random
- Coverage diagnostics:
  - coefficient marginal bin counts
  - selected pairwise occupancy counts
  - A1-S3, B2-S3, and A3-S3 relative-angle coverage
  - sampled nearest-neighbor distances in normalized 12D target space
  - standalone sampling-quality dashboard was planned but did not run in the
    current pushed result; see sampling-quality status above
- Learning-curve diagnostics:
  - `scripts/report_data_scale_learning_curve.py` compared v6 100K, v9 250K,
    and v11 500K batch summaries.
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
