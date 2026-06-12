# Current Project State

Last updated: 2026-06-12

## Stable Commit

- Latest evaluated Colab result commit: `537c92d`
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
- Latest completed model-selection batch config: `configs/model_selection_batch_v7_c1_gap125k.json`
- Current Colab worker config: `experiments/colab_worker_model_loop.json`
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
  - `D66_grouped_width320_lr6e-4_dropout0.075_v6gap100k_seed23_20260612_051859_utc`
  - seed-repeat check: `D66_grouped_width320_lr6e-4_dropout0.075_v6gap100k_seed7_20260612_054505_utc`
- Dataset:
  - `enhanced_v6_benchmark_gap100k_20260612_051040_utc`
  - `100,446` total rows
  - `43,000` benchmark-gap-aware hard-regime training-only rows appended to v5
  - train/validation/blind/stress: `93,011 / 1,979 / 2,371 / 3,085`
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

- Weighted score: `0.03690` for seed23, `0.03714` for seed7
- True hard-target normalized MAE: `0.01850` / `0.01865`
- Overall validation normalized MAE: `0.01394` / `0.01404`
- Blind/stress normalized MAE: `0.01085` / `0.01018` for seed23
- B2/A3 magnitude MAE: `0.0601` / `1.401` for seed23
- High-S3 magnitude MAE/bias/slope:
  - seed23: `6.27` / `-4.65` / `0.799`
  - seed7: `5.96` / `-4.40` / `0.812`
- C1 validation/blind/stress MAE for seed23:
  - `1.77` / `1.22` / `1.20`

Current interpretation:

- The v6 benchmark-gap-aware expansion is the current best result and replaces
  v5 as the baseline family.
- Standalone S3 magnitude-loss is rejected as the main direction.
- The v7 C1-focused expansion to `125,946` rows is rejected:
  - weighted score worsened: `0.03690 -> 0.03752` / `0.03791`
  - validation C1 MAE worsened: `1.77 -> 1.82` / `1.89`
  - blind/stress C1 MAE also worsened.
- C1 is likely feature-limited or identifiability-limited in the current
  representation. This is plausible because C1 is inferred from differences
  between features measured under large imposed under/over defocus offsets.
- The next step should be no-new-simulation feature/split infrastructure:
  fixed benchmark split membership, C1 sensitivity diagnostics, and explicit
  under/over defocus-difference features.

Benchmark caveat:

- Appended rows are marked `dataset_split_hint=training_only`.
- Validation/blind/stress are intended to come only from parent benchmark rows.
- Stress split still changed by one row between v6 and v7 (`3,085 -> 3,084`),
  so future comparisons must explicitly freeze benchmark row membership or
  compute split thresholds only from unhinted parent rows.

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
- Untracked downloaded Colab outputs:
  - `Downloads from Colab/`
