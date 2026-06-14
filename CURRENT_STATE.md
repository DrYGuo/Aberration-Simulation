# Current Project State

Last updated: 2026-06-14

## Stable Commit

- Latest evaluated Colab result commit: `3cee6f8`
- Documentation in this file reflects the evaluated Colab results through that commit.
- Repository: `https://github.com/DrYGuo/Aberration-Simulation`
- Branch: `main`

## Workflow

1. Codex edits the local repository.
2. Codex commits and pushes changes to GitHub.
3. The Colab worker notebook pulls the latest GitHub code.
4. The Colab worker runs the configured simulation/training/evaluation command.
5. Colab pushes compact artifacts back to GitHub.
6. Codex pulls and evaluates the pushed results.

Do not treat local Mac CUDA failures as project failures. The local Codex shell is not the Colab GPU runtime.

Large artifacts stay out of GitHub. Do not push training feature CSVs, model
checkpoints, raw predictions, zip files, or large result folders. The Colab
worker should push only manifests, compact metrics JSON, registry/batch CSVs,
small plots, and concise reports.

## Main Files

- Main coefficient relationship notebook: `notebooks/uno_coefficient_relationships.ipynb`
- Feature-regression notebook: `notebooks/uno_feature_regression.ipynb`
- Enhanced harmonic-summary regression notebook: `notebooks/uno_feature_regression_enhanced.ipynb`
- Raw-angle regression notebook: `notebooks/uno_feature_regression_raw_angles.ipynb`
- Colab model-selection worker notebook: `notebooks/colab_worker_model_loop.ipynb`
- Colab worker config: `experiments/colab_worker_model_loop.json`
- Latest completed model-selection batch config: `configs/model_selection_batch_v9_gap250k_d66.json`
- Current Colab worker config: `experiments/colab_worker_model_loop.json`
- Next queued model-selection batch config:
  - `configs/model_selection_batch_v10_structured_head_v9_250k.json`
- Completed 250k expansion config:
  - `configs/targeted_expansion_v9_250k.json`
- Main Colab smoke-test notebook: `notebooks/colab_gpu_smoke_test.ipynb`
- GPU optics implementation: `src/aberration_simulation/gpu_optics.py`
- CPU optics implementation: `src/aberration_simulation/cpu_optics.py`
- Harmonic phase conventions: `src/aberration_simulation/uno_conventions.py`
- Documentation: `README.md`

## Current Value Definitions

The relationship notebook documents and computes these values:

- `Cdf_value = (1/N) sum_k (Xigma_under,k - Xigma_over,k)`
- `C1_value = -Cdf_value`
- `A1_value = (2/N) sum_k (Xigma_under,k - Xigma_over,k) exp(2 i theta_k)`
- `B2_value = (2/N) sum_k (Mu_under,k + Mu_over,k) exp(i theta_k)`
- `A2_value = (2/N) sum_k (Mu_under,k + Mu_over,k) exp(3 i theta_k)`
- `Cs_value = (1/N) sum_k (Rho_under,k - Rho_over,k)`
- `C3_value = -mean(Rho_over) = -(1/N) sum_k Rho_over,k`
- `S3_value = (2/N) sum_k (Rho_under,k - Rho_over,k) exp(2 i theta_k)`
- `A3_value = (2/N) sum_k (Xigma_under,k - Xigma_over,k) exp(4 i theta_k)`

## Current Convention Notes

- Harmonic phase conventions are centralized in `src/aberration_simulation/uno_conventions.py`.
- The relationship notebook imports `UNO_HARMONIC_ORDERS`, `PRIMARY_PHASE_CONVENTIONS`, and `add_complex_columns()` from that module.
- Probe simulation uses the EM Fourier-transform convention comments in the optics code.
- Full vectorized GPU simulation is currently restored in `notebooks/uno_coefficient_relationships.ipynb`; batching was removed to keep GPU speed on higher-RAM Colab GPUs.

## Current Notebook Features

`notebooks/uno_coefficient_relationships.ipynb` currently includes:

- One-coefficient sweeps for C1, C3, A1, B2/C21, A2, A3, and S3/C32.
- Probe-shape galleries with color bars.
- Coupled C1/C3 scalar response maps:
  - `C1_value` over `(C1, C3)`
  - `C3_value` over `(C1, C3)`
- Coupled A1/S3 response maps:
  - `A1_value_abs` over `(A1_amp, S3_amp)`, faceted by `(A1_phase, S3_phase)`
  - `S3_value_abs` over `(A1_amp, S3_amp)`, faceted by `(A1_phase, S3_phase)`
- Sampled-grid uniqueness diagnostics:
  - `uniqueness_c1_c3.png`
  - `uniqueness_a1_s3_complex.png`
- A pre-download output check that raises an error if expected relationship outputs are missing.

## Regression Notebook Experiments

`notebooks/uno_feature_regression_enhanced.ipynb` is the current 66-input
enhanced harmonic-summary reference. It uses the original 12 collapsed Uno
features plus 54 raw under/over-focus `Xigma`, `Mu`, and `Rho` harmonic-summary
features.

`notebooks/uno_feature_regression_raw_angles.ipynb` is the Architecture E
experiment. It tests direct raw-angle `Xigma`, `Mu`, and `Rho` inputs on the
same v2 16k coupled-case generator and stratified/dropout trainer. The current
line-profile extractor drops the duplicated `180 deg` endpoint, so the raw-angle
input set has `18 * 3 * 2 = 108` raw under/over-focus angle features plus the
12 collapsed Uno features, for `120` total inputs. Outputs go under
`training_results/feature_regression_raw_angles/`.

Model decisions, run history, and the reproducibility/tracking standard are in
`MODEL_EVOLUTION.md`.

## Current Regression State

Current baseline:

- Run:
  - `D66_grouped_width320_lr6e-4_dropout0.075_v9gap250k_d66_seed23_20260614_062447_utc`
  - seed-repeat check: `D66_grouped_width320_lr6e-4_dropout0.075_v9gap250k_d66_seed7_20260614_073553_utc`
- Dataset:
  - `enhanced_v9_gap250k_20260614_055608_utc`
  - `250,000` total rows
  - `149,554` targeted hard-regime training-only rows appended to the v6 parent
  - train/validation/blind/stress: `242,556 / 1,977 / 2,370 / 3,097`
- Feature family:
  - 66 enhanced harmonic-summary features
- Model:
  - grouped-head residual MLP
  - width `320`
  - dropout `0.075`
  - learning rate `6e-4`
  - SmoothL1 component loss
  - gradient clipping
  - plateau LR scheduler
  - split seed `7`

Key metrics for the current baseline:

- Weighted score: `0.03051` for seed23, `0.03069` for seed7
- True hard-target normalized MAE: `0.01537` / `0.01557`
- Overall validation normalized MAE: `0.01186` / `0.01199`
- Blind/stress normalized MAE:
  - seed23: `0.00969` / `0.00872`
  - seed7: `0.00968` / `0.00878`
- B2/A3 magnitude MAE:
  - seed23: `0.0553` / `1.208`
  - seed7: `0.0552` / `1.237`
- High-S3 magnitude MAE/bias/slope:
  - seed23: `4.73` / `-3.12` / `0.865`
  - seed7: `4.66` / `-3.00` / `0.851`
- C1 validation/blind/stress MAE for seed23:
  - `1.50` / `1.09` / `1.02`

Current interpretation:

- The v9 250K hard-regime expansion is the current best result and replaces v6
  as the baseline family.
- Standalone S3 magnitude-loss remains rejected as the main direction.
- The v7 C1-focused expansion is rejected, and v8/v8b defocus-difference
  feature variants are not promoted. The v8b full defocus-difference features
  helped some high-S3/B2 diagnostics but did not consistently improve blind,
  stress, and A3 behavior across seeds.
- C1 is likely limited by the current defocus measurement geometry. C1 is
  inferred from differences between features measured under large imposed
  under/over defocus offsets, so its residual signal can be weaker than the
  defocused probe geometry itself.
- The active next step is an architecture-only test on the v9 250K 66-feature
  dataset:
  - new architecture option: `grouped_heads_structured`
  - batch config: `configs/model_selection_batch_v10_structured_head_v9_250k.json`
  - worker script: `scripts/run_colab_v10_structured_head_v9_250k_workflow.sh`
  - active worker command: `bash scripts/run_colab_v10_structured_head_v9_250k_workflow.sh`
  - no new simulations are expected if the v9 cached CSV is still alive in
    Colab; the workflow can regenerate v9 only if the cache is missing.

Benchmark note:

- Appended rows are marked `dataset_split_hint=training_only`.
- Validation/blind/stress are controlled by the frozen v6 benchmark split
  manifest. New appended rows are training-only and should not leak into
  model-selection validation, blind, or stress splits.

## Recent Interpretation

- On the sampled C1/C3 grid, `(C1_value, C3_value)` uniquely identifies the sampled `(C1, C3)` pair, but this is not proof of global uniqueness over continuous coefficient space.
- On the sampled A1/S3 grid, complex `(A1_value, S3_value)` uniquely identifies the sampled physical input points after collapsing zero-amplitude phase duplicates, but this is also not proof of global uniqueness.
- The recommended diagnostic for uniqueness is pairwise input-space distance versus output-space distance plus nearest-output-neighbor plots.

## Known Local Dirty State

At the time this note was created, these local items existed and should not be assumed to be part of the current committed state unless explicitly committed later:

- Modified notebooks:
  - `notebooks/colab_gpu_smoke_test.ipynb`
  - `notebooks/gpu_smoke_test.ipynb`
  - `notebooks/original_aberration_simulation.ipynb`
  - `notebooks/uno_coefficient_relationships.ipynb`
  - `notebooks/uno_feature_regression.ipynb`
  - `notebooks/uno_feature_regression_enhanced.ipynb`
- Untracked downloaded Colab outputs:
  - `Downloads from Colab/`
- Other untracked local presentation/notebook files:
  - `docs/neural_network_evolution_presentation.md`
  - `docs/neural_network_evolution_presentation.svg`
  - `notebooks/uno_feature_regression_raw_angles.ipynb`
