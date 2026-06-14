# S3 Magnitude Metric Audit

Created UTC: 2026-06-13T23:14:27.639682+00:00

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
| v6_benchmark_gap100k_seed23 | False | True | 0.0369 | 289 | 0.7986 | 6.274 | -4.647 | 11.76 | 3.144 | 9.773 |
| v6_benchmark_gap100k_seed7 | False | True | 0.03714 | 289 | 0.8119 | 5.96 | -4.404 | 10.85 | 3.062 | 10.21 |

## Full Recompute Tables

### Magnitude

| run | subset | OLS slope | intercept | through-origin slope | corr | R2 | MAE | bias | RMSE | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v6_benchmark_gap100k_seed23 | all_magnitude | 0.9173 | 1.931 | 0.9465 | 0.9793 | 0.9589 | 3.255 | 0.02571 | 6.522 | 1979 |
| v6_benchmark_gap100k_seed23 | high_magnitude | 0.7986 | 12.18 | 0.9424 | 0.593 | 0.3516 | 6.274 | -4.647 | 11.76 | 289 |
| v6_benchmark_gap100k_seed7 | all_magnitude | 0.9203 | 1.901 | 0.9491 | 0.98 | 0.9603 | 3.266 | 0.06497 | 6.403 | 1979 |
| v6_benchmark_gap100k_seed7 | high_magnitude | 0.8119 | 11.31 | 0.9455 | 0.6323 | 0.3998 | 5.96 | -4.404 | 10.85 | 289 |

### Components

| run | component subset | OLS slope | intercept | through-origin slope | corr | R2 | MAE | bias | RMSE | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v6_benchmark_gap100k_seed23 | S3_x_all | 0.9439 | 0.2346 | 0.9437 | 0.9817 | 0.9638 | 2.537 | 0.2586 | 5.215 | 1979 |
| v6_benchmark_gap100k_seed23 | S3_y_all | 0.9369 | 0.1014 | 0.9368 | 0.9778 | 0.9561 | 2.623 | 0.1347 | 5.888 | 1979 |
| v6_benchmark_gap100k_seed23 | S3_x_high | 0.9416 | 0.9031 | 0.9405 | 0.9893 | 0.9788 | 4.911 | 1.137 | 8.82 | 289 |
| v6_benchmark_gap100k_seed23 | S3_y_high | 0.936 | -0.175 | 0.9362 | 0.9841 | 0.9685 | 5.368 | 0.06061 | 10.93 | 289 |
| v6_benchmark_gap100k_seed7 | S3_x_all | 0.9465 | 0.1388 | 0.9464 | 0.9811 | 0.9626 | 2.586 | 0.1616 | 5.285 | 1979 |
| v6_benchmark_gap100k_seed7 | S3_y_all | 0.9386 | 0.1473 | 0.9385 | 0.9786 | 0.9576 | 2.647 | 0.1797 | 5.789 | 1979 |
| v6_benchmark_gap100k_seed7 | S3_x_high | 0.9463 | 1.008 | 0.9452 | 0.9893 | 0.9787 | 4.876 | 1.222 | 8.792 | 289 |
| v6_benchmark_gap100k_seed7 | S3_y_high | 0.9379 | -0.07834 | 0.938 | 0.9873 | 0.9747 | 5.087 | 0.1504 | 9.888 | 289 |

## Plots

### old_v3_bin_diag
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260613_231425_utc/plots/old_v3_bin_diag_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260613_231425_utc/plots/old_v3_bin_diag_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260613_231425_utc/plots/old_v3_bin_diag_S3_pred_vs_true_angle.png`

### v3_smoothl1_champion
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260613_231425_utc/plots/v3_smoothl1_champion_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260613_231425_utc/plots/v3_smoothl1_champion_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260613_231425_utc/plots/v3_smoothl1_champion_S3_pred_vs_true_angle.png`

### v5_s3tail60k_seed23
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260613_231425_utc/plots/v5_s3tail60k_seed23_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260613_231425_utc/plots/v5_s3tail60k_seed23_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260613_231425_utc/plots/v5_s3tail60k_seed23_S3_pred_vs_true_angle.png`

### v6_benchmark_gap100k_seed23
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260613_231425_utc/plots/v6_benchmark_gap100k_seed23_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260613_231425_utc/plots/v6_benchmark_gap100k_seed23_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260613_231425_utc/plots/v6_benchmark_gap100k_seed23_S3_pred_vs_true_angle.png`
Generated full audit plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260613_231425_utc/plots/v6_benchmark_gap100k_seed23_S3_x_pred_vs_true.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260613_231425_utc/plots/v6_benchmark_gap100k_seed23_S3_y_pred_vs_true.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260613_231425_utc/plots/v6_benchmark_gap100k_seed23_S3_mag_pred_vs_true_all.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260613_231425_utc/plots/v6_benchmark_gap100k_seed23_S3_mag_pred_vs_true_high.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260613_231425_utc/plots/v6_benchmark_gap100k_seed23_S3_mag_residual_vs_true_mag.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260613_231425_utc/plots/v6_benchmark_gap100k_seed23_S3_vector_residual_vs_true_mag.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260613_231425_utc/plots/v6_benchmark_gap100k_seed23_S3_angle_error_vs_true_mag.png`

### v6_benchmark_gap100k_seed7
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260613_231425_utc/plots/v6_benchmark_gap100k_seed7_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260613_231425_utc/plots/v6_benchmark_gap100k_seed7_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260613_231425_utc/plots/v6_benchmark_gap100k_seed7_S3_pred_vs_true_angle.png`
Generated full audit plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260613_231425_utc/plots/v6_benchmark_gap100k_seed7_S3_x_pred_vs_true.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260613_231425_utc/plots/v6_benchmark_gap100k_seed7_S3_y_pred_vs_true.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260613_231425_utc/plots/v6_benchmark_gap100k_seed7_S3_mag_pred_vs_true_all.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260613_231425_utc/plots/v6_benchmark_gap100k_seed7_S3_mag_pred_vs_true_high.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260613_231425_utc/plots/v6_benchmark_gap100k_seed7_S3_mag_residual_vs_true_mag.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260613_231425_utc/plots/v6_benchmark_gap100k_seed7_S3_vector_residual_vs_true_mag.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260613_231425_utc/plots/v6_benchmark_gap100k_seed7_S3_angle_error_vs_true_mag.png`

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
