# Aberration Simulation

Simulation of electron probes with aberrations under GPU parallel computing.

Refactored Python project from the notebook:

`Another copy of Batch process of Modularised code for GPU parallel computing of aberrated probes.ipynb`

The original notebook mixed module generation, GPU execution, feature extraction experiments, and plotting. This folder separates those pieces into importable modules and small scripts that are easier to run, test, and publish on GitHub.

## What is included

- `src/aberration_simulation/optics.py` - aberration model, contrast transfer function generation, and probe image computation.
- `src/aberration_simulation/line_profiles.py` - reusable line-profile extraction for image stacks.
- `scripts/run_smoke_test.py` - small CPU/GPU-friendly simulation over a reduced aberration grid.
- `scripts/plot_line_profiles.py` - generates probe and radial line-profile plots for selected nonzero aberration coefficients.
- `outputs/` - generated smoke-test data and plots.

The code uses CuPy automatically when it is installed. If CuPy is unavailable, it falls back to NumPy/SciPy, which is useful for small tests and GitHub/CI environments.

## Setup

```bash
python -m pip install -r requirements.txt
```

For GPU acceleration, install the CuPy package that matches your CUDA version separately. The smoke test does not require CuPy.

## Run a small simulation

```bash
python scripts/run_smoke_test.py
```

This writes:

- `outputs/smoke_probe_images.npz`
- `outputs/smoke_parameters.csv`

## Generate line-profile plots

```bash
python scripts/plot_line_profiles.py
```

This writes PNG files under `outputs/plots/`.

## Run on Colab GPU

Open `notebooks/colab_gpu_smoke_test.ipynb` in Google Colab, choose a GPU runtime, and run the cells. The notebook checks `nvidia-smi`, installs CuPy if needed, verifies the active backend, runs the smoke test, and displays the generated line-profile plots.

For a Jupyter/VS Code notebook smoke test, run `notebooks/gpu_smoke_test.ipynb`. It uses the same project modules, prints whether CuPy is active, runs the reduced coefficient grid, and displays probe images plus line profiles inline.

You can also check the active backend from a terminal:

```bash
python scripts/check_backend.py
```

## Notes

The default smoke-test grid intentionally uses small image dimensions and a small set of aberration coefficient combinations. Increase `pix_dim`, `real_dim`, or the coefficient sequences in `scripts/run_smoke_test.py` only after the small run succeeds.
