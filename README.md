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

The smoke test uses 326 targeted cases: one no-aberration baseline, ten `C3`-only combinations, sixty-four `A1`-only combinations, twenty-five `A2`-only combinations spanning the original notebook's larger A2 amplitude steps and phases, and sixty-three pure `B2/C21` combinations with amplitude `0..3` in `0.5` steps and phase `0..360` degrees in `45` degree steps, each evaluated at `C1_offset=-909 nm` and `C1_offset=+909 nm`. This keeps the test interpretable while covering isolated spherical aberration, 2-fold astigmatism, 3-fold astigmatism, axial coma, and the two requested C1-offset conditions.

This writes:

- `outputs/smoke_probe_images.npz`
- `outputs/smoke_parameters.csv`

## Generate line-profile plots

```bash
python scripts/plot_line_profiles.py
```

This writes PNG files under `outputs/plots/`.

Each plot compares the `C1_offset=-909 nm` and `C1_offset=+909 nm` results for the same underlying aberration combination in one figure. Plot filenames include the aberration family and values, for example `line_profiles_001_a2_amp1_phase0.png` or `line_profiles_0xx_b2_amp0p5_phase0.png`; A2 and B2/C21 plots are ordered immediately after the baseline plot. The probe images are overlaid with the sampled line directions, using the same colors as the line-profile curves. The script also writes A2 and B2/C21 summary grids, one for each C1 offset, so those sweeps are visible near the top of the notebook output.

## Run on Colab GPU

Open `notebooks/colab_gpu_smoke_test.ipynb` in Google Colab, choose a GPU runtime, and run the cells. The notebook clones or pulls the latest `main` branch from GitHub, checks `nvidia-smi`, installs CuPy if needed, verifies the active backend, runs the smoke test, and displays the generated line-profile plots.

Direct Colab URL:

```text
https://colab.research.google.com/github/DrYGuo/Aberration-Simulation/blob/main/notebooks/colab_gpu_smoke_test.ipynb
```

For a Jupyter/VS Code notebook smoke test, run `notebooks/gpu_smoke_test.ipynb`. It uses the same project modules, prints whether CuPy is active, runs the reduced coefficient grid, and displays probe images plus line profiles inline.

For the Uno et al. 2005 digitized-aberration workflow, run `notebooks/uno_et_al_2005_optik.ipynb`. It follows the Colab GPU smoke-test setup, extracts line profiles every `10` degrees, computes the profile quantities `Xigma`, `Mu`, and `Rho` from formulas `(45)-(47)`, then computes `Cdf_value`, `A1_value`, `B2_value`, `A2_value`, `Cs_value`, `S3_value`, and `A3_value` from formulas `(38)-(44)`.

You can also check the active backend from a terminal:

```bash
python scripts/check_backend.py
```

## Notes

The smoke test follows the original notebook sampling: `pix_dim=(256, 256)`, `real_dim=(1280, 1280)`, `eV=0.8e3`, `app=30 mrad`, Gaussian blurring with `Sigma=2`, and line-profile sampling with radius `r=80` over 36 angular directions. `Sigma` is the Gaussian standard deviation in image pixels; with this sampling it corresponds to about `1.0 nm`. Increase the coefficient sequences in `scripts/run_smoke_test.py` only after this run succeeds.
