# Aberration Simulation

Simulation of electron probes with aberrations under GPU parallel computing.

Refactored Python project from the notebook:

`Another copy of Batch process of Modularised code for GPU parallel computing of aberrated probes.ipynb`

The original notebook mixed module generation, GPU execution, feature extraction experiments, and plotting. This folder separates those pieces into importable modules and small scripts that are easier to run, test, and publish on GitHub.

## What is included

- `src/aberration_simulation/optics.py` - compatibility import that selects the GPU implementation when CuPy is available, otherwise the CPU implementation.
- `src/aberration_simulation/gpu_optics.py` - CuPy-only implementation with CTF generation vectorized across aberration coefficient combinations in each batch.
- `src/aberration_simulation/cpu_optics.py` - NumPy/SciPy implementation for local development and CPU fallback.
- `src/aberration_simulation/line_profiles.py` - reusable line-profile extraction for image stacks.
- `src/aberration_simulation/uno_conventions.py` - fitted Uno et al. phase-convention constants and helpers shared by notebooks and scripts.
- `scripts/run_smoke_test.py` - small CPU/GPU-friendly simulation over a reduced aberration grid.
- `scripts/plot_line_profiles.py` - generates probe and radial line-profile plots for selected nonzero aberration coefficients.
- `outputs/` - generated smoke-test data and plots.

The original notebook used CuPy arrays but still looped over aberration coefficient combinations when constructing the CTF tensor. The refactored GPU path keeps the CuPy code explicit (`import cupy as cp`) and vectorizes the CTF phase calculation across all coefficient combinations in the selected batch. If CuPy is unavailable, the public `aberration_simulation.optics` module falls back to the CPU implementation.

The vectorized GPU path can use more memory than the old per-combination loop, so use the existing `batch_size` argument to split large parameter grids into manageable chunks.

## Setup

```bash
python -m pip install -r requirements.txt
```

For GPU acceleration, install the CuPy package that matches your CUDA version separately. The smoke test does not require CuPy.

## Run a small simulation

```bash
python scripts/run_smoke_test.py
```

The smoke test uses 486 targeted cases: one no-aberration baseline, ten `C3`-only combinations, sixty-four `A1`-only combinations, twenty-five `A2`-only combinations spanning the original notebook's larger A2 amplitude steps and phases, sixty-three pure `B2/C21` combinations with amplitude `0..3` in `0.5` steps and phase `0..360` degrees in `45` degree steps, forty `A3` combinations, and forty `S3/C32` combinations, each evaluated at `C1_offset=-909 nm` and `C1_offset=+909 nm`. This keeps the test interpretable while covering isolated spherical aberration, 2-fold astigmatism, 3-fold astigmatism, axial coma, four-fold astigmatism, axial star aberration, and the two requested C1-offset conditions.

This writes:

- `outputs/smoke_probe_images.npz`
- `outputs/smoke_parameters.csv`

## Generate line-profile plots

```bash
python scripts/plot_line_profiles.py
```

This writes PNG files under `outputs/plots/`.

Each plot compares the `C1_offset=-909 nm` and `C1_offset=+909 nm` results for the same underlying aberration combination in one figure. Plot filenames include the aberration family and values, for example `line_profiles_001_a2_amp1_phase0.png` or `line_profiles_0xx_b2_amp0p5_phase0.png`; A2, B2/C21, A3, and S3/C32 plots are ordered immediately after the baseline plot. The probe images are overlaid with the sampled line directions, using the same colors as the line-profile curves. Line-profile angles increase counter-clockwise in displayed probe coordinates. The script also writes A2, B2/C21, A3, and S3/C32 summary grids, one for each C1 offset, so those sweeps are visible near the top of the notebook output.

## Conventions

The probe simulation follows a Nion-style aberration convention adapted to the project phase-sign choice. Aberration input phases are stored directly as internal angles, then evaluated as `cos(m * (qphi - angle))`. For example, an input phase `A1_phase` is stored internally as `A1_angle = radians(A1_phase)`. The same convention is implemented in the compatibility `Aberration`/`chi()` path and in the vectorized GPU path, but those paths are alternatives; the phase sign is not applied twice in one calculation.

The reciprocal-space phase is inserted into the CTF as `exp(+1j * chi)`. This removes the previous paired signs `angle = -input_phase` and `CTF = exp(-i chi)`. The probe wave is then computed from the CTF with the project EM inverse FFT wrapper, whose reciprocal-to-real convention uses the negative exponential sign. In short, the current implemented simulation convention is:

```text
angle = input_phase
chi term = amplitude * cos(m * (qphi - angle))
CTF = exp(+i chi)
probe wave = fft2(CTF)  # EM convention: ifft is forward, fft is inverse
probe intensity = normalized(abs(probe wave)^2)
```

Changing both of the previous signs is not generally an algebraic no-op for a fixed input phase. The old angular mapping gave `cos(m * (qphi + input_phase))`, while the current one gives `cos(m * (qphi - input_phase))`; the CTF sign change then complex-conjugates the phase factor. Depending on symmetry, this can appear as a reflected or conjugated probe, but directional aberration phase reporting should be refit after changing the convention.

Fourier-transform sign convention is handled explicitly. Electron microscopy and crystallography often define the real-to-reciprocal transform with a positive exponential and the reciprocal-to-real inverse with a negative exponential. NumPy/CuPy use the opposite signs: `fft2` has the negative exponential and `ifft2` has the positive exponential plus normalization. Therefore, the project keeps exact wrappers in both CPU and GPU code: `fft2_em(f) = np.fft.ifft2(f) * N_axes` and `ifft2_em(F) = np.fft.fft2(F) / N_axes`, with the same expressions implemented through CuPy for the GPU path. Here `N_axes` is the product of the transformed axes only, so stacked probe batches are normalized by image size, not by the number of coefficient combinations. The hot probe-formation path calls `np.fft.fft2` / `cp.fft.fft2` directly with an inline EM-convention comment, then normalizes each smoothed probe image to unit summed intensity. This avoids the wrapper call and scalar division in the performance-critical path.

Displayed line-profile angles increase counter-clockwise: `0 deg` points right and `90 deg` points up. This is implemented by sampling `x = x_center + cos(theta) * offset` and `y = y_center - sin(theta) * offset`, because image row coordinates increase downward.

For the Uno et al. 2005 digitized-aberration workflow, the fitted harmonic phase convention maps Uno's raw complex profile coefficients onto the simulation convention. The constants live in `src/aberration_simulation/uno_conventions.py` so scripts and notebooks use the same values. The primary reported Uno phase is computed as `sign * raw_complex_phase / harmonic_order + offset`, wrapped to the coefficient period. The most recent fitted summary from `notebooks/uno_et_al_2005_optik_auto_convention_search.ipynb` and the Colab-downloaded result `uno_auto_convention_results.zip` was:

| coefficient | sign | offset | period | mean abs error |
| --- | ---: | ---: | ---: | ---: |
| `A1_value` | `-1` | `90 deg` | `180 deg` | `0.023 deg` |
| `B2_value` / `C21` | `-1` | `0 deg` | `360 deg` | `<1e-12 deg` |
| `A2_value` | `-1` | `0 deg` | `120 deg` | `<1e-12 deg` |
| `A3_value` | `-1` | `45 deg` | `90 deg` | `<1e-12 deg` |
| `S3_value` / `C32` | `-1` | `0 deg` | `180 deg` | `0.034 deg` |

The downloaded auto-search JSON reported `360 deg` for `B2_value`, which is equivalent to `0 deg` because B2/C21 has a `360 deg` phase period. The raw Colab result also exposed an A2 tie-breaking issue: sorting candidate conventions by median error first chose `sign=+1`, producing a mean A2 error of `24 deg` and max error of `60 deg`. The notebook now chooses the convention by mean error first, then max and median error, which selects the correct `A2_value` convention `sign=-1, offset=0 deg`.

## Run on Colab GPU

Open `notebooks/colab_gpu_smoke_test.ipynb` in Google Colab, choose a GPU runtime, and run the cells. The notebook clones or pulls the latest `main` branch from GitHub, checks `nvidia-smi`, installs CuPy if needed, verifies the active backend, runs the smoke test, and displays the generated line-profile plots.

Direct Colab URL:

```text
https://colab.research.google.com/github/DrYGuo/Aberration-Simulation/blob/main/notebooks/colab_gpu_smoke_test.ipynb
```

For a Jupyter/VS Code notebook smoke test, run `notebooks/gpu_smoke_test.ipynb`. It uses the same project modules, prints whether CuPy is active, runs the reduced coefficient grid, and displays probe images plus line profiles inline.

For the Uno et al. 2005 digitized-aberration workflow, run `notebooks/uno_et_al_2005_optik.ipynb`. It follows the Colab GPU smoke-test setup, extracts line profiles every `10` counter-clockwise degrees, computes the profile quantities `Xigma`, `Mu`, and `Rho` from formulas `(45)-(47)`, then computes `Cdf_value`, `A1_value`, `B2_value`, `A2_value`, `Cs_value`, `S3_value`, and `A3_value` from formulas `(38)-(44)`.

For one-coefficient-at-a-time calibration plots, run `notebooks/uno_coefficient_relationships.ipynb`. It sweeps C1, C3, A1, B2/C21, A2, A3, and S3/C32 separately, computes the corresponding Uno values from paired under/over-focus line profiles, and saves scalar, amplitude, and phase relationship plots, Fig. 8-style probe-shape galleries, coupled C1/C3 scalar-response color maps, coupled A1/S3 response maps, sampled-grid uniqueness diagnostics, and a CSV under `outputs/uno_relationships/`.

The relationship notebook uses these value definitions, where `theta_k` are the sampled line-profile angles and `N` is the number of sampled angles: `Cdf_value = (1/N) sum_k (Xigma_under,k - Xigma_over,k)`, `C1_value = -Cdf_value`, `A1_value = (2/N) sum_k (Xigma_under,k - Xigma_over,k) exp(2 i theta_k)`, `B2_value = (2/N) sum_k (Mu_under,k + Mu_over,k) exp(i theta_k)`, `A2_value = (2/N) sum_k (Mu_under,k + Mu_over,k) exp(3 i theta_k)`, `Cs_value = (1/N) sum_k (Rho_under,k - Rho_over,k)`, `C3_value = -mean(Rho_over) = -(1/N) sum_k Rho_over,k`, `S3_value = (2/N) sum_k (Rho_under,k - Rho_over,k) exp(2 i theta_k)`, and `A3_value = (2/N) sum_k (Xigma_under,k - Xigma_over,k) exp(4 i theta_k)`.

A shorter imported-code version is available at `notebooks/uno_coefficient_relationship_short.ipynb`. It imports the Uno value formulas from `src/aberration_simulation/uno_conventions.py`, relationship plots from `scripts/plots_uno_convention.py`, and probe galleries from `scripts/plot_probe_shapes.py`. The short notebook also includes wide-amplitude A3 and S3/C32 sweeps up to `100`, using A3 phases in `[0, 90]` and S3/C32 phases in `[0, 180]`, with matching wide-range probe-shape galleries. It also adds coupled C1/A1/C3 and A1/B2/S3 grids for regression training.

For the first feature-to-coefficient regression test, run `notebooks/uno_feature_regression.ipynb`. This notebook is intentionally separate from `notebooks/uno_coefficient_relationship_short.ipynb`: it generates its own training coefficient combinations, runs the GPU probe simulation, extracts Uno feature values, and trains a hybrid linear-baseline plus residual-MLP model from feature values to aberration coefficient vectors. It saves `training_features.csv` and temporary model outputs under `training_results/feature_regression/`, with optional Google Drive mirroring from Colab.

You can also check the active backend from a terminal:

```bash
python scripts/check_backend.py
```

## Notes

The smoke test follows the original notebook sampling: `pix_dim=(256, 256)`, `real_dim=(1280, 1280)`, `eV=0.8e3`, `app=30 mrad`, Gaussian blurring with `Sigma=2`, and line-profile sampling with radius `r=80` over 36 angular directions. `Sigma` is the Gaussian standard deviation in image pixels; with this sampling it corresponds to about `1.0 nm`. Increase the coefficient sequences in `scripts/run_smoke_test.py` only after this run succeeds.
