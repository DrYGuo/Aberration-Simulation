# S3 Magnitude Metric Audit

Created UTC: 2026-06-14T05:47:06.623158+00:00

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
| v8b_frozen_d66_seed23 | False | True | 0.03664 | 289 | 0.8124 | 6.11 | -4.443 | 11.62 | 2.918 | 8.975 |
| v8b_frozen_c1diff_full_seed23 | False | True | 0.03636 | 289 | 0.8288 | 5.779 | -4.318 | 10.25 | 2.879 | 10.4 |
| v8b_frozen_c1diff_full_seed7 | False | True | 0.03693 | 289 | 0.8083 | 5.803 | -4.541 | 10.73 | 3.357 | 10.73 |

## Full Recompute Tables

### Magnitude

| run | subset | OLS slope | intercept | through-origin slope | corr | R2 | MAE | bias | RMSE | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v8b_frozen_d66_seed23 | all_magnitude | 0.9192 | 1.941 | 0.9486 | 0.9785 | 0.9575 | 3.256 | 0.07716 | 6.611 | 1977 |
| v8b_frozen_d66_seed23 | high_magnitude | 0.8124 | 11.23 | 0.945 | 0.6011 | 0.3613 | 6.11 | -4.443 | 11.62 | 289 |
| v8b_frozen_c1diff_full_seed23 | all_magnitude | 0.9221 | 1.904 | 0.951 | 0.983 | 0.9663 | 3.086 | 0.1084 | 5.958 | 1977 |
| v8b_frozen_c1diff_full_seed23 | high_magnitude | 0.8288 | 9.988 | 0.9467 | 0.6635 | 0.4403 | 5.779 | -4.318 | 10.25 | 289 |
| v8b_frozen_c1diff_full_seed7 | all_magnitude | 0.9198 | 1.965 | 0.9495 | 0.9824 | 0.9651 | 3.151 | 0.1148 | 6.066 | 1977 |
| v8b_frozen_c1diff_full_seed7 | high_magnitude | 0.8083 | 11.47 | 0.9438 | 0.6387 | 0.4079 | 5.803 | -4.541 | 10.73 | 289 |

### Components

| run | component subset | OLS slope | intercept | through-origin slope | corr | R2 | MAE | bias | RMSE | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v8b_frozen_d66_seed23 | S3_x_all | 0.9493 | 0.1763 | 0.9492 | 0.9814 | 0.9631 | 2.554 | 0.1979 | 5.25 | 1977 |
| v8b_frozen_d66_seed23 | S3_y_all | 0.9351 | 0.1233 | 0.935 | 0.9767 | 0.954 | 2.605 | 0.1575 | 6.027 | 1977 |
| v8b_frozen_d66_seed23 | S3_x_high | 0.9505 | 0.9575 | 0.9493 | 0.9903 | 0.9807 | 4.614 | 1.155 | 8.349 | 289 |
| v8b_frozen_d66_seed23 | S3_y_high | 0.9324 | -0.2053 | 0.9326 | 0.9831 | 0.9665 | 5.28 | 0.04356 | 11.28 | 289 |
| v8b_frozen_c1diff_full_seed23 | S3_x_all | 0.948 | 0.1793 | 0.9479 | 0.983 | 0.9662 | 2.5 | 0.2015 | 5.033 | 1977 |
| v8b_frozen_c1diff_full_seed23 | S3_y_all | 0.9434 | 0.1669 | 0.9433 | 0.9831 | 0.9665 | 2.487 | 0.1968 | 5.171 | 1977 |
| v8b_frozen_c1diff_full_seed23 | S3_x_high | 0.9487 | 0.976 | 0.9475 | 0.9914 | 0.983 | 4.654 | 1.181 | 7.943 | 289 |
| v8b_frozen_c1diff_full_seed23 | S3_y_high | 0.9401 | 0.2698 | 0.9399 | 0.9888 | 0.9778 | 5.266 | 0.4902 | 9.331 | 289 |
| v8b_frozen_c1diff_full_seed7 | S3_x_all | 0.948 | 0.08236 | 0.9479 | 0.9811 | 0.9626 | 2.542 | 0.1046 | 5.28 | 1977 |
| v8b_frozen_c1diff_full_seed7 | S3_y_all | 0.9389 | 0.1827 | 0.9388 | 0.9823 | 0.9649 | 2.564 | 0.2149 | 5.303 | 1977 |
| v8b_frozen_c1diff_full_seed7 | S3_x_high | 0.9467 | 1.099 | 0.9454 | 0.988 | 0.9762 | 4.823 | 1.312 | 9.231 | 289 |
| v8b_frozen_c1diff_full_seed7 | S3_y_high | 0.9334 | 0.2067 | 0.9332 | 0.9876 | 0.9754 | 5.248 | 0.4519 | 9.849 | 289 |

## Plots

### old_v3_bin_diag
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_054703_utc/plots/old_v3_bin_diag_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_054703_utc/plots/old_v3_bin_diag_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_054703_utc/plots/old_v3_bin_diag_S3_pred_vs_true_angle.png`

### v3_smoothl1_champion
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_054703_utc/plots/v3_smoothl1_champion_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_054703_utc/plots/v3_smoothl1_champion_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_054703_utc/plots/v3_smoothl1_champion_S3_pred_vs_true_angle.png`

### v5_s3tail60k_seed23
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_054703_utc/plots/v5_s3tail60k_seed23_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_054703_utc/plots/v5_s3tail60k_seed23_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_054703_utc/plots/v5_s3tail60k_seed23_S3_pred_vs_true_angle.png`

### v8b_frozen_d66_seed23
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_054703_utc/plots/v8b_frozen_d66_seed23_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_054703_utc/plots/v8b_frozen_d66_seed23_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_054703_utc/plots/v8b_frozen_d66_seed23_S3_pred_vs_true_angle.png`
Generated full audit plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_054703_utc/plots/v8b_frozen_d66_seed23_S3_x_pred_vs_true.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_054703_utc/plots/v8b_frozen_d66_seed23_S3_y_pred_vs_true.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_054703_utc/plots/v8b_frozen_d66_seed23_S3_mag_pred_vs_true_all.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_054703_utc/plots/v8b_frozen_d66_seed23_S3_mag_pred_vs_true_high.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_054703_utc/plots/v8b_frozen_d66_seed23_S3_mag_residual_vs_true_mag.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_054703_utc/plots/v8b_frozen_d66_seed23_S3_vector_residual_vs_true_mag.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_054703_utc/plots/v8b_frozen_d66_seed23_S3_angle_error_vs_true_mag.png`

### v8b_frozen_c1diff_full_seed23
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_054703_utc/plots/v8b_frozen_c1diff_full_seed23_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_054703_utc/plots/v8b_frozen_c1diff_full_seed23_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_054703_utc/plots/v8b_frozen_c1diff_full_seed23_S3_pred_vs_true_angle.png`
Generated full audit plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_054703_utc/plots/v8b_frozen_c1diff_full_seed23_S3_x_pred_vs_true.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_054703_utc/plots/v8b_frozen_c1diff_full_seed23_S3_y_pred_vs_true.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_054703_utc/plots/v8b_frozen_c1diff_full_seed23_S3_mag_pred_vs_true_all.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_054703_utc/plots/v8b_frozen_c1diff_full_seed23_S3_mag_pred_vs_true_high.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_054703_utc/plots/v8b_frozen_c1diff_full_seed23_S3_mag_residual_vs_true_mag.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_054703_utc/plots/v8b_frozen_c1diff_full_seed23_S3_vector_residual_vs_true_mag.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_054703_utc/plots/v8b_frozen_c1diff_full_seed23_S3_angle_error_vs_true_mag.png`

### v8b_frozen_c1diff_full_seed7
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_054703_utc/plots/v8b_frozen_c1diff_full_seed7_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_054703_utc/plots/v8b_frozen_c1diff_full_seed7_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_054703_utc/plots/v8b_frozen_c1diff_full_seed7_S3_pred_vs_true_angle.png`
Generated full audit plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_054703_utc/plots/v8b_frozen_c1diff_full_seed7_S3_x_pred_vs_true.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_054703_utc/plots/v8b_frozen_c1diff_full_seed7_S3_y_pred_vs_true.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_054703_utc/plots/v8b_frozen_c1diff_full_seed7_S3_mag_pred_vs_true_all.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_054703_utc/plots/v8b_frozen_c1diff_full_seed7_S3_mag_pred_vs_true_high.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_054703_utc/plots/v8b_frozen_c1diff_full_seed7_S3_mag_residual_vs_true_mag.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_054703_utc/plots/v8b_frozen_c1diff_full_seed7_S3_vector_residual_vs_true_mag.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_054703_utc/plots/v8b_frozen_c1diff_full_seed7_S3_angle_error_vs_true_mag.png`

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
