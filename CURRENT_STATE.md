# Current Project State

Last updated: 2026-06-03

## Stable Commit

- Current GitHub `main` commit: `c0d9de5` (`Expand coupled regression training data`)
- Current good helper-refactor commit before the regression expansion: `2cb7e0e`
- Repository: `https://github.com/DrYGuo/Aberration-Simulation`
- Branch: `main`
- Local `main` is aligned with `origin/main` at `c0d9de5`.

## Workflow

1. Codex edits the local repository.
2. Codex commits and pushes changes to GitHub.
3. The Colab-connected notebook pulls/clones the latest GitHub code.
4. The user runs notebooks manually on the Colab GPU runtime.
5. The user manually downloads Colab outputs into `Downloads from Colab/`.

Do not treat local Mac CUDA failures as project failures. The local Codex shell is not the Colab GPU runtime.

## Main Files

- Main coefficient relationship notebook: `notebooks/uno_coefficient_relationships.ipynb`
- Short imported-code relationship notebook: `notebooks/uno_coefficient_relationship_short.ipynb`
- Feature-regression notebook: `notebooks/uno_feature_regression.ipynb`
- Enhanced feature-regression notebook: `notebooks/uno_feature_regression_enhanced.ipynb`
- Main Colab smoke-test notebook: `notebooks/colab_gpu_smoke_test.ipynb`
- GPU optics implementation: `src/aberration_simulation/gpu_optics.py`
- CPU optics implementation: `src/aberration_simulation/cpu_optics.py`
- Uno value definitions and harmonic phase conventions: `src/aberration_simulation/uno_conventions.py`
- Relationship plotting helpers: `scripts/plots_uno_convention.py`
- Probe-shape plotting helpers: `scripts/plot_probe_shapes.py`
- Hybrid feature-regression helper: `scripts/feature_regression_model.py`
- Model architecture history: `MODEL_EVOLUTION.md`
- Documentation: `README.md`

## GitHub Dependency Surface

Keep these files together when pushing notebook changes to GitHub:

- Colab smoke test:
  - `notebooks/colab_gpu_smoke_test.ipynb`
  - `scripts/run_smoke_test.py`
  - `scripts/plot_line_profiles.py`
  - `scripts/check_backend.py`
  - `src/aberration_simulation/backend.py`
  - `src/aberration_simulation/optics.py`
  - `src/aberration_simulation/gpu_optics.py`
  - `src/aberration_simulation/cpu_optics.py`
  - `src/aberration_simulation/line_profiles.py`
  - `requirements.txt`
- Short Uno coefficient relationship notebook:
  - `notebooks/uno_coefficient_relationship_short.ipynb`
  - `src/aberration_simulation/uno_conventions.py`
  - `src/aberration_simulation/line_profiles.py`
  - `src/aberration_simulation/optics.py`
  - `scripts/plots_uno_convention.py`
  - `scripts/plot_probe_shapes.py`
  - `requirements.txt`
- Feature-regression notebook:
  - `notebooks/uno_feature_regression.ipynb`
  - `notebooks/uno_feature_regression_enhanced.ipynb`
  - `scripts/feature_regression_model.py`
  - `src/aberration_simulation/uno_conventions.py`
  - `src/aberration_simulation/line_profiles.py`
  - `src/aberration_simulation/optics.py`
  - `training_results/feature_regression/README.md`
  - `requirements.txt`

The committed package dependencies are `numpy`, `scipy`, `matplotlib`, and `h5py`.
The Colab notebooks install or verify `requirements.txt` and then install a
CUDA-matched `cupy-cuda12x` package when CuPy is missing. The regression helper
also imports PyTorch inside `train_hybrid_regressor()`, relying on Colab's
runtime torch install rather than listing torch in `requirements.txt`.

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
- `extract_line_profiles_from_stack()` is vectorized through the active backend; with CuPy active, line-profile extraction runs on the GPU.

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

`notebooks/uno_coefficient_relationship_short.ipynb` preserves the same workflow but imports the reusable formula and plotting code from Python files. Pairwise uniqueness diagnostics remain inline in the short notebook for active adjustment.

The short notebook also includes wide-amplitude A3 and S3/C32 sweeps up to `100`, producing `relationship_A3_value_wide.png`, `relationship_S3_value_wide.png`, `probe_shapes_a3_wide.png`, and `probe_shapes_s3_c32_wide.png`. A3 phases are sampled in `[0, 90]`; S3/C32 phases are sampled in `[0, 180]`. Additional coupled C1/A1/C3 and A1/B2/S3 grids are included for regression training.

`notebooks/uno_feature_regression.ipynb` is separate from the relationship notebooks. It generates its own training coefficient combinations, runs the GPU probe simulation, extracts Uno feature values in GPU batches, saves `training_features.csv` inside a timestamped run folder under `training_results/feature_regression/`, and trains the first feature-to-coefficient hybrid model from that table. The current generator keeps the deterministic anchor sweeps and adds `10,000` random coupled cases across full-random, targeted weak-coupling, and sparse-random mixtures. It maps feature values to aberration coefficient vectors using real/imaginary complex representations. Each run writes `run_manifest.json` and appends `model_registry.csv` with model parameters, dataset fingerprint, and summary metrics. Colab can optionally mirror the run folder to Google Drive.

`notebooks/uno_feature_regression_enhanced.ipynb` is the next regression experiment.
It keeps the same training-case generator, adds raw under/over-focus `Xigma`,
`Mu`, and `Rho` harmonic features to the original 12 Uno features, trains a
larger weighted residual model with early stopping, and writes test-only
colored scatter diagnostics plus amplitude-bin error summaries under
`training_results/feature_regression_enhanced/`.

`notebooks/colab_gpu_smoke_test.ipynb` is the GitHub/Colab entry point. It checks
`nvidia-smi`, clones or pulls `main`, verifies CuPy through `scripts/check_backend.py`,
runs `scripts/run_smoke_test.py`, then runs `scripts/plot_line_profiles.py`.
The current checked-in smoke grid generates `486` paired C1-offset cases:
baseline, C3, A1, A2, B2/C21, A3, and S3/C32 families.

## Recent Interpretation

- On the sampled C1/C3 grid, `(C1_value, C3_value)` uniquely identifies the sampled `(C1, C3)` pair, but this is not proof of global uniqueness over continuous coefficient space.
- On the sampled A1/S3 grid, complex `(A1_value, S3_value)` uniquely identifies the sampled physical input points after collapsing zero-amplitude phase duplicates, but this is also not proof of global uniqueness.
- The recommended diagnostic for uniqueness is pairwise input-space distance versus output-space distance plus nearest-output-neighbor plots.

## Known Local Dirty State

At the time this note was updated, these local items existed and should not be assumed to be part of the current committed state unless explicitly committed later:

- Modified notebooks:
  - `notebooks/colab_gpu_smoke_test.ipynb`
  - `notebooks/gpu_smoke_test.ipynb`
  - `notebooks/original_aberration_simulation.ipynb`
- The modified notebooks have the same cell sources and metadata as `HEAD`; the
  dirty diff is execution output, execution counts, and notebook-generated cell
  IDs. `notebooks/colab_gpu_smoke_test.ipynb` has a very large output diff with
  embedded plot images from a Colab run.
- Untracked downloaded Colab outputs:
  - `Downloads from Colab/`
- Downloaded Colab artifacts currently include:
  - `uno_coefficient_relationships.csv`
  - `uno_coefficient_relationship_summary.png`
  - `uno_relationship_probe_images.npz`
  - `training_features.csv`
  - `history.csv`
  - `predictions.csv`
  - `prediction_scatter.png`
  - ignored zip bundles such as `uno_auto_convention_results.zip`,
    `uno_coefficient_relationships.zip`, and `feature_regression_results.zip`

For a clean GitHub commit, either leave those notebook execution outputs out of
the commit or intentionally strip notebook outputs before committing. Do not
commit `Downloads from Colab/` unless the project policy changes; generated
outputs and zip bundles are currently treated as local artifacts.
