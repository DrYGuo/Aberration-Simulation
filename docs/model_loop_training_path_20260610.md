# Model Loop Training Path - 2026-06-10

## Current Baseline

- Dataset: `enhanced_v3_targeted25k`
- Feature family: 66 enhanced summary features
- Architecture: grouped-head residual MLP
- Width: 320
- Learning rate: `6e-4`
- Dropout: `0.075`
- Split seed: `7`
- Reference run:
  - `D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_bin_diag_20260609_073514_utc`

Baseline metrics:

- Weighted selection score: `0.05945`
- True hard-target normalized MAE: `0.02947`
- Validation overall normalized MAE: `0.02188`
- Blind/stress normalized MAE: `0.01575` / `0.01648`
- High-S3 magnitude MAE/bias/slope: `10.84` / `-8.77` / `0.671`
- C1 validation MAE/RMSE/p95: `3.17` / `5.45` / `11.28`

## S3 Magnitude-Loss Finding

The S3 magnitude-loss variants improved selected high-S3 magnitude metrics but
did not clearly replace the v3 baseline.

- Strong S3 magnitude loss improved high-S3 MAE/bias but damaged overall,
  hard-target, B2, A3, and C1 behavior.
- `w0.10` slightly improved weighted score but worsened hard-target, overall,
  blind/stress, and C1 metrics.
- `w0.25_high2` improved high-S3 slope and slightly improved blind/stress, but
  worsened overall and C1 metrics.

Decision: do not promote the S3 magnitude-loss runs as the new baseline.

## Training-Path Interpretation

- The current runner has been using full-batch AdamW.
- Row shuffling does not matter unless mini-batch training is enabled.
- Training component loss is much lower than validation loss, so the current
  bottleneck is generalization rather than fitting the training rows.
- C1 and S3 must be tracked together; optimizing S3 alone can degrade C1.

## Next Controlled Batch

Keep dataset and feature set fixed. Test optimizer/stability changes only:

- reproducible torch/numpy seed
- optional mini-batch shuffling
- gradient clipping
- `ReduceLROnPlateau`
- SmoothL1 component-loss candidate for heavy-tail C1/S3 residuals

Promote a candidate only if validation, blind, stress, C1, S3, B2, and A3
diagnostics improve or remain within accepted regression gates.
