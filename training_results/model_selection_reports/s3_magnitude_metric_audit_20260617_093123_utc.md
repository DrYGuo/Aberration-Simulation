# S3 Magnitude Metric Audit

Created UTC: 2026-06-17T09:31:26.091336+00:00

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
| v14_1p5m_far_nn_seed23 | False | True | 0.01233 | 13437 | 0.9719 | 0.8386 | -0.4054 | 2.4 | 0.3646 | 1.411 |

## Full Recompute Tables

### Magnitude

| run | subset | OLS slope | intercept | through-origin slope | corr | R2 | MAE | bias | RMSE | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v14_1p5m_far_nn_seed23 | all_magnitude | 0.9835 | 0.8732 | 0.9947 | 0.9982 | 0.9964 | 1.017 | 0.03186 | 2.299 | 26977 |
| v14_1p5m_far_nn_seed23 | high_magnitude | 0.9719 | 1.916 | 0.9948 | 0.97 | 0.941 | 0.8386 | -0.4054 | 2.4 | 13437 |

### Components

| run | component subset | OLS slope | intercept | through-origin slope | corr | R2 | MAE | bias | RMSE | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v14_1p5m_far_nn_seed23 | S3_x_all | 0.9948 | 0.02377 | 0.9948 | 0.999 | 0.998 | 0.8093 | 0.02464 | 1.995 | 26977 |
| v14_1p5m_far_nn_seed23 | S3_y_all | 0.9939 | -0.1034 | 0.9939 | 0.9991 | 0.9982 | 0.8089 | -0.1017 | 1.919 | 26977 |
| v14_1p5m_far_nn_seed23 | S3_x_high | 0.9951 | 0.028 | 0.9951 | 0.9994 | 0.9988 | 0.6801 | 0.0294 | 2.047 | 13437 |
| v14_1p5m_far_nn_seed23 | S3_y_high | 0.9941 | -0.1235 | 0.9941 | 0.9995 | 0.9989 | 0.6888 | -0.1212 | 1.962 | 13437 |

## Plots

### old_v3_bin_diag
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260617_093123_utc/plots/old_v3_bin_diag_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260617_093123_utc/plots/old_v3_bin_diag_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260617_093123_utc/plots/old_v3_bin_diag_S3_pred_vs_true_angle.png`

### v3_smoothl1_champion
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260617_093123_utc/plots/v3_smoothl1_champion_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260617_093123_utc/plots/v3_smoothl1_champion_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260617_093123_utc/plots/v3_smoothl1_champion_S3_pred_vs_true_angle.png`

### v5_s3tail60k_seed23
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260617_093123_utc/plots/v5_s3tail60k_seed23_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260617_093123_utc/plots/v5_s3tail60k_seed23_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260617_093123_utc/plots/v5_s3tail60k_seed23_S3_pred_vs_true_angle.png`

### v14_1p5m_far_nn_seed23
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260617_093123_utc/plots/v14_1p5m_far_nn_seed23_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260617_093123_utc/plots/v14_1p5m_far_nn_seed23_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260617_093123_utc/plots/v14_1p5m_far_nn_seed23_S3_pred_vs_true_angle.png`
Generated full audit plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260617_093123_utc/plots/v14_1p5m_far_nn_seed23_S3_x_pred_vs_true.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260617_093123_utc/plots/v14_1p5m_far_nn_seed23_S3_y_pred_vs_true.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260617_093123_utc/plots/v14_1p5m_far_nn_seed23_S3_mag_pred_vs_true_all.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260617_093123_utc/plots/v14_1p5m_far_nn_seed23_S3_mag_pred_vs_true_high.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260617_093123_utc/plots/v14_1p5m_far_nn_seed23_S3_mag_residual_vs_true_mag.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260617_093123_utc/plots/v14_1p5m_far_nn_seed23_S3_vector_residual_vs_true_mag.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260617_093123_utc/plots/v14_1p5m_far_nn_seed23_S3_angle_error_vs_true_mag.png`

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
