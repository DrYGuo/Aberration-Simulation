# Model Evolution

This file tracks the feature-to-coefficient regression architectures and the
reason for each change. Per-run numeric details are saved by the notebooks in
`run_manifest*.json` and summarized in `model_registry*.csv`.

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

- Deterministic single-coefficient sweeps remain good.
- B2 and A2 are acceptable.
- A1 phase behavior is reasonable, but amplitude/vector errors remain nonzero.
- Large coupled full-random cases remain poor.
- S3/C32 and A3 errors grow strongly at high amplitudes.
- Train error is also large in the difficult coupled regions, so this is not
  only overfitting.

Interpretation:

- Adding more features and a larger model did not solve the fully coupled
  high-amplitude regime.
- The current feature set may be insufficient for unique recovery in the most
  coupled region, or the regression target should be decomposed into staged or
  physics-guided subproblems.

## Current Working Hypotheses

- The feature values are useful and monotonic in one-coefficient sweeps.
- Coupled random cases reveal cross-talk between feature values.
- Fully coupled high-amplitude A3/S3/C1 cases are the present bottleneck.
- The next architecture should not simply be larger; it should encode more
  structure, such as staged prediction, coefficient-group heads, or additional
  probe-shape/profile features.

## Candidate Next Steps

- Add a linear/ridge baseline to quantify how much the neural residual improves
  over direct calibrated inversion.
- Train grouped heads:
  - scalar head for C1/C3,
  - low-order harmonic head for A1/B2/A2,
  - high-order harmonic head for S3/A3.
- Add direct line-profile samples or compressed profile coefficients as model
  inputs, not only scalar feature values.
- Train curriculum-style:
  - one-coefficient sweeps,
  - weak coupled cases,
  - high-amplitude full coupled cases.
- Evaluate uniqueness using nearest-neighbor distances in feature space before
  increasing model capacity again.
