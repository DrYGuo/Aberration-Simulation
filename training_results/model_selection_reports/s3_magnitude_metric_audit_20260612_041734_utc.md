# S3 Magnitude Metric Audit

Created UTC: 2026-06-12T04:17:38.333582+00:00

## Formula And Code-Path Check

- `true_mag = sqrt(true_S3_x^2 + true_S3_y^2)`.
- `pred_mag = sqrt(pred_S3_x^2 + pred_S3_y^2)`.
- High-S3 bin is selected by true magnitude: `true_mag > 0.7 * vector_scale`.
- `vector_scale` is the 95th percentile of true vector magnitude on the training split.
- The model runner inverse-transforms normalized predictions once with `y_scaler.inverse_transform(pred_scaled)` before diagnostics.
- `S3_x` and `S3_y` are adjacent target columns from the same physical target vector and use the same coefficient units.

## Artifact Limitation

Compact-only historical runs do not include raw predictions or checkpoints, so intercepts, through-origin slopes, component slopes, and residual plots cannot be recomputed for those runs. Runs with `s3_magnitude_metric_audit_validation.csv` are fully recomputed below, including OLS intercepts, through-origin slopes, component slopes, and residual plots.

## Stored Metric Comparison

| run | has raw predictions | has S3 audit CSV | weighted score | high n | stored high OLS slope | stored high MAE | stored high bias | stored high RMSE | high angle mean deg | high angle p95 deg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old_v3_bin_diag | False | False | 0.05945 | 321 | 0.6711 | 10.84 | -8.767 | 18.39 | 5.703 | 20.93 |
| v3_smoothl1_champion | False | False | 0.05568 | 321 | 0.7209 | 9.54 | -7.683 | 16.72 | 5.484 | 23.4 |
| v5_s3tail60k_seed23 | False | False | 0.05102 | 289 | 0.7091 | 8.201 | -6.386 | 14.81 | 4.392 | 16.95 |
| v5_logging_ref_seed23 | False | True | 0.05102 | 289 | 0.7091 | 8.201 | -6.386 | 14.81 | 4.392 | 16.95 |
| v5_s3magloss_w0p05_seed23 | False | True | 0.052 | 289 | 0.7401 | 8.296 | -6.374 | 14.96 | 4.952 | 17.23 |
| v5_s3magloss_w0p10_seed23 | False | True | 0.05353 | 289 | 0.7654 | 8.195 | -6.3 | 14.54 | 4.187 | 16.39 |

## Full Recompute Tables

### Magnitude

| run | subset | OLS slope | intercept | through-origin slope | corr | R2 | MAE | bias | RMSE | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v5_logging_ref_seed23 | all_magnitude | 0.8822 | 3.013 | 0.9278 | 0.9635 | 0.9284 | 4.477 | 0.296 | 8.568 | 1977 |
| v5_logging_ref_seed23 | high_magnitude | 0.7091 | 17.92 | 0.9207 | 0.4695 | 0.2204 | 8.201 | -6.386 | 14.81 | 289 |
| v5_s3magloss_w0p05_seed23 | all_magnitude | 0.881 | 3.124 | 0.9283 | 0.961 | 0.9235 | 4.536 | 0.38 | 8.831 | 1977 |
| v5_s3magloss_w0p05_seed23 | high_magnitude | 0.7401 | 15.35 | 0.9212 | 0.4786 | 0.2291 | 8.296 | -6.374 | 14.96 | 289 |
| v5_s3magloss_w0p10_seed23 | all_magnitude | 0.8844 | 3.129 | 0.9318 | 0.9633 | 0.9279 | 4.488 | 0.4632 | 8.589 | 1977 |
| v5_s3magloss_w0p10_seed23 | high_magnitude | 0.7654 | 13.3 | 0.9225 | 0.5024 | 0.2524 | 8.195 | -6.3 | 14.54 | 289 |

### Components

| run | component subset | OLS slope | intercept | through-origin slope | corr | R2 | MAE | bias | RMSE | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v5_logging_ref_seed23 | S3_x_all | 0.9271 | 0.3777 | 0.9269 | 0.967 | 0.9351 | 3.524 | 0.4089 | 6.95 | 1977 |
| v5_logging_ref_seed23 | S3_y_all | 0.9053 | 0.1008 | 0.9053 | 0.9599 | 0.9215 | 3.61 | 0.1508 | 7.857 | 1977 |
| v5_logging_ref_seed23 | S3_x_high | 0.9284 | 1.394 | 0.9268 | 0.9827 | 0.9656 | 6.536 | 1.68 | 11.14 | 289 |
| v5_logging_ref_seed23 | S3_y_high | 0.9009 | -0.3686 | 0.9013 | 0.974 | 0.9488 | 7.072 | -0.003793 | 14.02 | 289 |
| v5_s3magloss_w0p05_seed23 | S3_x_all | 0.9283 | 0.1378 | 0.9282 | 0.9656 | 0.9324 | 3.513 | 0.1684 | 7.079 | 1977 |
| v5_s3magloss_w0p05_seed23 | S3_y_all | 0.9053 | 0.227 | 0.9051 | 0.9582 | 0.9182 | 3.705 | 0.277 | 8.019 | 1977 |
| v5_s3magloss_w0p05_seed23 | S3_x_high | 0.9279 | 1.175 | 0.9265 | 0.9828 | 0.9659 | 6.304 | 1.463 | 11.07 | 289 |
| v5_s3magloss_w0p05_seed23 | S3_y_high | 0.9019 | -0.3013 | 0.9022 | 0.9726 | 0.946 | 7.296 | 0.05988 | 14.33 | 289 |
| v5_s3magloss_w0p10_seed23 | S3_x_all | 0.9278 | 0.3409 | 0.9276 | 0.9649 | 0.931 | 3.626 | 0.3718 | 7.159 | 1977 |
| v5_s3magloss_w0p10_seed23 | S3_y_all | 0.9081 | 0.152 | 0.908 | 0.9571 | 0.916 | 3.693 | 0.2005 | 8.117 | 1977 |
| v5_s3magloss_w0p10_seed23 | S3_x_high | 0.9288 | 1.134 | 0.9275 | 0.9832 | 0.9666 | 6.473 | 1.419 | 10.96 | 289 |
| v5_s3magloss_w0p10_seed23 | S3_y_high | 0.905 | -0.2666 | 0.9053 | 0.9758 | 0.9522 | 7.147 | 0.08306 | 13.56 | 289 |

## Plots

### old_v3_bin_diag
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260612_041734_utc/plots/old_v3_bin_diag_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260612_041734_utc/plots/old_v3_bin_diag_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260612_041734_utc/plots/old_v3_bin_diag_S3_pred_vs_true_angle.png`

### v3_smoothl1_champion
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260612_041734_utc/plots/v3_smoothl1_champion_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260612_041734_utc/plots/v3_smoothl1_champion_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260612_041734_utc/plots/v3_smoothl1_champion_S3_pred_vs_true_angle.png`

### v5_s3tail60k_seed23
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260612_041734_utc/plots/v5_s3tail60k_seed23_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260612_041734_utc/plots/v5_s3tail60k_seed23_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260612_041734_utc/plots/v5_s3tail60k_seed23_S3_pred_vs_true_angle.png`

### v5_logging_ref_seed23
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260612_041734_utc/plots/v5_logging_ref_seed23_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260612_041734_utc/plots/v5_logging_ref_seed23_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260612_041734_utc/plots/v5_logging_ref_seed23_S3_pred_vs_true_angle.png`
Generated full audit plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260612_041734_utc/plots/v5_logging_ref_seed23_S3_x_pred_vs_true.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260612_041734_utc/plots/v5_logging_ref_seed23_S3_y_pred_vs_true.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260612_041734_utc/plots/v5_logging_ref_seed23_S3_mag_pred_vs_true_all.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260612_041734_utc/plots/v5_logging_ref_seed23_S3_mag_pred_vs_true_high.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260612_041734_utc/plots/v5_logging_ref_seed23_S3_mag_residual_vs_true_mag.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260612_041734_utc/plots/v5_logging_ref_seed23_S3_vector_residual_vs_true_mag.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260612_041734_utc/plots/v5_logging_ref_seed23_S3_angle_error_vs_true_mag.png`

### v5_s3magloss_w0p05_seed23
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260612_041734_utc/plots/v5_s3magloss_w0p05_seed23_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260612_041734_utc/plots/v5_s3magloss_w0p05_seed23_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260612_041734_utc/plots/v5_s3magloss_w0p05_seed23_S3_pred_vs_true_angle.png`
Generated full audit plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260612_041734_utc/plots/v5_s3magloss_w0p05_seed23_S3_x_pred_vs_true.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260612_041734_utc/plots/v5_s3magloss_w0p05_seed23_S3_y_pred_vs_true.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260612_041734_utc/plots/v5_s3magloss_w0p05_seed23_S3_mag_pred_vs_true_all.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260612_041734_utc/plots/v5_s3magloss_w0p05_seed23_S3_mag_pred_vs_true_high.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260612_041734_utc/plots/v5_s3magloss_w0p05_seed23_S3_mag_residual_vs_true_mag.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260612_041734_utc/plots/v5_s3magloss_w0p05_seed23_S3_vector_residual_vs_true_mag.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260612_041734_utc/plots/v5_s3magloss_w0p05_seed23_S3_angle_error_vs_true_mag.png`

### v5_s3magloss_w0p10_seed23
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260612_041734_utc/plots/v5_s3magloss_w0p10_seed23_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260612_041734_utc/plots/v5_s3magloss_w0p10_seed23_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260612_041734_utc/plots/v5_s3magloss_w0p10_seed23_S3_pred_vs_true_angle.png`
Generated full audit plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260612_041734_utc/plots/v5_s3magloss_w0p10_seed23_S3_x_pred_vs_true.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260612_041734_utc/plots/v5_s3magloss_w0p10_seed23_S3_y_pred_vs_true.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260612_041734_utc/plots/v5_s3magloss_w0p10_seed23_S3_mag_pred_vs_true_all.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260612_041734_utc/plots/v5_s3magloss_w0p10_seed23_S3_mag_pred_vs_true_high.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260612_041734_utc/plots/v5_s3magloss_w0p10_seed23_S3_mag_residual_vs_true_mag.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260612_041734_utc/plots/v5_s3magloss_w0p10_seed23_S3_vector_residual_vs_true_mag.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260612_041734_utc/plots/v5_s3magloss_w0p10_seed23_S3_angle_error_vs_true_mag.png`

## Answers To Main Questions

A. The stored high-S3 magnitude slope near `0.71` is present in `vector_diagnostics.json` and the batch summary for the v5 seed23 run.

B. Code inspection shows it is an OLS-with-intercept slope from `np.polyfit(true_magnitude, pred_magnitude, 1)`. The intercept is recovered for runs with `s3_magnitude_metric_audit_validation.csv`; compact-only historical runs cannot provide it.

C. The through-origin slope is computed when `s3_magnitude_metric_audit_validation.csv` or raw validation predictions are available; otherwise it cannot be recovered from historical compact artifacts.

D. Component slopes for `S3_x` and `S3_y` are computed from the S3 audit CSV when present. For older compact-only runs, existing component scatter plots are images only.

E. A low OLS slope can be affected by high-bin range restriction and nonzero intercept. Runs with the S3 audit CSV provide enough data to distinguish OLS-with-intercept behavior from through-origin/component slopes.

F. Whether magnitude bias becomes more negative at larger true `|S3|` requires residual-vs-true-magnitude data. This is available for runs with the S3 audit CSV and unavailable for compact-only historical runs.

G. Feature-bottleneck conclusions should be based on the full recompute rows when available, not on the high-bin OLS slope alone.

## Recommendation

Use the recomputed S3 audit CSV metrics when deciding whether S3 magnitude-loss candidates improve high-S3 magnitude MAE and bias without damaging component slopes or blind/stress performance.
