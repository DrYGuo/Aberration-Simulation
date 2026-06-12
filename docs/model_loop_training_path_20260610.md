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

## 2026-06-11 Update

The SmoothL1 plateau/clip path was promoted over the earlier v3 baseline, then
tested with a targeted high-S3-tail data expansion.

Current champion:

- Run:
  - `D66_grouped_width320_lr6e-4_dropout0.075_s3tail60k_plateau_clip_smoothl1_seed23_20260611_090007_utc`
- Dataset:
  - `enhanced_v5_s3_tail60k_20260611_084005_utc`
  - `57,446` total rows
  - `16,000` new high-S3-tail training-only rows

Comparison against the previous v3 SmoothL1 champion:

- Weighted score improved: `0.05568 -> 0.05102`.
- True hard-target normalized MAE improved: `0.02770 -> 0.02581`.
- Overall normalized MAE improved: `0.02057 -> 0.01900`.
- Blind/stress normalized MAE improved: `0.01509/0.01536 -> 0.01408/0.01341`.
- B2/A3 magnitude MAE improved: `0.0887/2.154 -> 0.0798/1.967`.
- High-S3 magnitude MAE/bias improved: `9.54/-7.68 -> 8.20/-6.39`.
- High-S3 angle errors improved: mean `5.48 -> 4.39 deg`, p95 `23.40 -> 16.95 deg`.
- High-S3 magnitude slope did not improve: `0.721 -> 0.709`.

Decision:

- Treat the v5 seed23 run as the current champion.
- Do not jump directly to `100k` rows.
- The remaining high-S3 bottleneck is likely feature sensitivity/calibration,
  not only data coverage.
- Next controlled step: add high-S3-sensitive features such as radial-band,
  radially weighted, or contour/radius m=2 descriptors, then rerun the champion
  architecture on v5 with fixed benchmark row IDs.

Benchmark note:

- v5 appended rows were training-only.
- The blind split stayed identical.
- Validation/stress shifted by one parent row, so future serious comparisons
  should explicitly persist benchmark row IDs.

## 2026-06-12 Update

The next two data-scale experiments were completed through the Colab worker.

### v6 Benchmark-Gap 100k

v6 appended `43,000` benchmark-gap-aware hard-regime training-only rows to the
v5 S3-tail dataset, producing `100,446` total rows.

Current baseline runs:

- `D66_grouped_width320_lr6e-4_dropout0.075_v6gap100k_seed23_20260612_051859_utc`
- `D66_grouped_width320_lr6e-4_dropout0.075_v6gap100k_seed7_20260612_054505_utc`

Compared with v5 seed23:

- Weighted score improved: `0.05102 -> 0.03690` / `0.03714`.
- True hard-target normalized MAE improved: `0.02581 -> 0.01850` / `0.01865`.
- Blind/stress normalized MAE improved to about `0.0109` / `0.0102-0.0104`.
- High-S3 magnitude MAE improved: `8.20 -> 6.27` / `5.96`.
- High-S3 magnitude slope improved: `0.709 -> 0.799` / `0.812`.
- B2/A3 vector magnitude MAE improved to about `0.060-0.062` / `1.37-1.40`.

Decision: promote the v6 seed-repeat family as the current baseline.

### v7 C1-Gap 125k

v7 appended `25,500` C1-focused training-only rows to v6, producing `125,946`
total rows.

Candidate runs:

- `D66_grouped_width320_lr6e-4_dropout0.075_v7c1gap125k_seed23_20260612_063900_utc`
- `D66_grouped_width320_lr6e-4_dropout0.075_v7c1gap125k_seed7_20260612_071130_utc`

Compared with v6 seed23:

- Weighted score worsened: `0.03690 -> 0.03752` / `0.03791`.
- Validation C1 MAE worsened: `1.77 -> 1.82` / `1.89`.
- Blind C1 MAE worsened: `1.22 -> 1.28` / `1.28`.
- Stress C1 MAE worsened: `1.20 -> 1.28` / `1.32`.
- High-S3 slope improved, but this does not compensate for C1 and weighted-score
  regression.

Decision: reject v7 as a model-improvement direction.

### Interpretation

C1 now appears less like a data-volume problem and more like an
observability/feature-identifiability problem. The simulations use large
under/over defocus offsets, and C1 is inferred mainly from differences in
defocused probe features such as `Xigma`. The residual C1 signal can be a weak
incremental perturbation on top of the imposed defocus geometry, so simply
adding more C1-coupled rows is not sufficient.

### Next Controlled Work

Do not run another data expansion first. The next model-loop patch should:

- freeze validation/blind/stress row membership explicitly;
- fix the stress-threshold issue that moved one row between v6 and v7;
- add C1 sensitivity diagnostics by C1 magnitude bin and coupling label;
- add explicit under/over defocus-difference features, including raw and
  normalized differences for `Xigma`, `Mu`, `Rho`, and selected harmonics;
- run a no-new-simulation v8 feature batch on the existing v6 data before
  deciding whether more simulations are justified.
