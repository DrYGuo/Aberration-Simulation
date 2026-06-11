# Current Project State

Last updated: 2026-06-11

## Stable Commit

- Current GitHub `main` commit: `9d41407`
- Current regression-prep commit: `066c808`
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
- Current model-selection batch config: `configs/model_selection_batch_v5_s3_tail60k.json`
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

Current champion:

- Run:
  - `D66_grouped_width320_lr6e-4_dropout0.075_s3tail60k_plateau_clip_smoothl1_seed23_20260611_090007_utc`
- Dataset:
  - `enhanced_v5_s3_tail60k_20260611_084005_utc`
  - `57,446` total rows
  - `16,000` new high-S3-tail training-only rows appended to v3
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

Key metrics for the current champion:

- Weighted score: `0.05102`
- True hard-target normalized MAE: `0.02581`
- Overall validation normalized MAE: `0.01900`
- Blind/stress normalized MAE: `0.01408` / `0.01341`
- B2/A3 magnitude MAE: `0.0798` / `1.967`
- High-S3 magnitude MAE/bias/slope: `8.20` / `-6.39` / `0.709`
- High-S3 mean/p95 angle error: `4.39 deg` / `16.95 deg`

Current interpretation:

- The v5 S3-tail expansion improved overall regression, blind/stress metrics,
  B2/A3 vector metrics, and high-S3 magnitude MAE/bias.
- The high-S3 magnitude slope remains compressed near `0.7`, so more data alone
  has not solved S3 calibration.
- The S3 feature saturation audit shows weak high-S3 linear response for the
  current Eq. 43-like and h2 harmonic features. The current CSV does not contain
  radial-band, radially weighted, or contour/radius m=2 features.
- The next step should be high-S3-sensitive feature engineering with locked
  validation/blind/stress row IDs, not immediate expansion to `100k` rows.

Benchmark caveat:

- v5 appended rows are marked `dataset_split_hint=training_only`.
- Blind split remained identical to the previous champion.
- Validation/stress changed by one parent row, so future comparisons should
  write and reuse explicit benchmark row IDs instead of relying only on stable
  hashing.

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
