# S3 Magnitude Metric Audit

Created UTC: 2026-06-14T23:23:12.682147+00:00

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
| v11_gap500k_d66_seed23 | False | True | 0.0271 | 281 | 0.8576 | 4.318 | -3.045 | 7.656 | 2.234 | 7.943 |
| v11_gap500k_d66_seed7 | False | True | 0.02619 | 281 | 0.8453 | 4.093 | -2.913 | 7.782 | 2.15 | 6.999 |

## Full Recompute Tables

### Magnitude

| run | subset | OLS slope | intercept | through-origin slope | corr | R2 | MAE | bias | RMSE | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v11_gap500k_d66_seed23 | all_magnitude | 0.9434 | 1.639 | 0.9682 | 0.9914 | 0.9828 | 2.406 | 0.3324 | 4.326 | 1977 |
| v11_gap500k_d66_seed23 | high_magnitude | 0.8576 | 8.917 | 0.9624 | 0.7649 | 0.5851 | 4.318 | -3.045 | 7.656 | 281 |
| v11_gap500k_d66_seed7 | all_magnitude | 0.9456 | 1.584 | 0.9695 | 0.9913 | 0.9827 | 2.328 | 0.3286 | 4.326 | 1977 |
| v11_gap500k_d66_seed7 | high_magnitude | 0.8453 | 10.09 | 0.9638 | 0.7523 | 0.5659 | 4.093 | -2.913 | 7.782 | 281 |

### Components

| run | component subset | OLS slope | intercept | through-origin slope | corr | R2 | MAE | bias | RMSE | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v11_gap500k_d66_seed23 | S3_x_all | 0.9679 | 0.06808 | 0.9679 | 0.9914 | 0.9828 | 1.924 | 0.0818 | 3.59 | 1977 |
| v11_gap500k_d66_seed23 | S3_y_all | 0.9633 | -0.02442 | 0.9633 | 0.9913 | 0.9828 | 1.903 | -0.005067 | 3.714 | 1977 |
| v11_gap500k_d66_seed23 | S3_x_high | 0.9644 | 0.4029 | 0.964 | 0.9954 | 0.9908 | 3.506 | 0.5218 | 5.846 | 281 |
| v11_gap500k_d66_seed23 | S3_y_high | 0.9575 | 0.1505 | 0.9573 | 0.9934 | 0.9869 | 3.887 | 0.3285 | 7.198 | 281 |
| v11_gap500k_d66_seed7 | S3_x_all | 0.9689 | 0.1747 | 0.9688 | 0.9925 | 0.985 | 1.798 | 0.188 | 3.364 | 1977 |
| v11_gap500k_d66_seed7 | S3_y_all | 0.9655 | 0.1062 | 0.9655 | 0.9906 | 0.9814 | 1.884 | 0.1243 | 3.849 | 1977 |
| v11_gap500k_d66_seed7 | S3_x_high | 0.966 | 0.6652 | 0.9654 | 0.9962 | 0.9924 | 3.21 | 0.7785 | 5.374 | 281 |
| v11_gap500k_d66_seed7 | S3_y_high | 0.9592 | 0.172 | 0.959 | 0.9927 | 0.9854 | 3.804 | 0.3428 | 7.537 | 281 |

## Plots

### old_v3_bin_diag
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_232309_utc/plots/old_v3_bin_diag_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_232309_utc/plots/old_v3_bin_diag_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_232309_utc/plots/old_v3_bin_diag_S3_pred_vs_true_angle.png`

### v3_smoothl1_champion
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_232309_utc/plots/v3_smoothl1_champion_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_232309_utc/plots/v3_smoothl1_champion_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_232309_utc/plots/v3_smoothl1_champion_S3_pred_vs_true_angle.png`

### v5_s3tail60k_seed23
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_232309_utc/plots/v5_s3tail60k_seed23_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_232309_utc/plots/v5_s3tail60k_seed23_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_232309_utc/plots/v5_s3tail60k_seed23_S3_pred_vs_true_angle.png`

### v11_gap500k_d66_seed23
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_232309_utc/plots/v11_gap500k_d66_seed23_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_232309_utc/plots/v11_gap500k_d66_seed23_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_232309_utc/plots/v11_gap500k_d66_seed23_S3_pred_vs_true_angle.png`
Generated full audit plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_232309_utc/plots/v11_gap500k_d66_seed23_S3_x_pred_vs_true.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_232309_utc/plots/v11_gap500k_d66_seed23_S3_y_pred_vs_true.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_232309_utc/plots/v11_gap500k_d66_seed23_S3_mag_pred_vs_true_all.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_232309_utc/plots/v11_gap500k_d66_seed23_S3_mag_pred_vs_true_high.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_232309_utc/plots/v11_gap500k_d66_seed23_S3_mag_residual_vs_true_mag.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_232309_utc/plots/v11_gap500k_d66_seed23_S3_vector_residual_vs_true_mag.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_232309_utc/plots/v11_gap500k_d66_seed23_S3_angle_error_vs_true_mag.png`

### v11_gap500k_d66_seed7
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_232309_utc/plots/v11_gap500k_d66_seed7_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_232309_utc/plots/v11_gap500k_d66_seed7_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_232309_utc/plots/v11_gap500k_d66_seed7_S3_pred_vs_true_angle.png`
Generated full audit plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_232309_utc/plots/v11_gap500k_d66_seed7_S3_x_pred_vs_true.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_232309_utc/plots/v11_gap500k_d66_seed7_S3_y_pred_vs_true.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_232309_utc/plots/v11_gap500k_d66_seed7_S3_mag_pred_vs_true_all.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_232309_utc/plots/v11_gap500k_d66_seed7_S3_mag_pred_vs_true_high.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_232309_utc/plots/v11_gap500k_d66_seed7_S3_mag_residual_vs_true_mag.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_232309_utc/plots/v11_gap500k_d66_seed7_S3_vector_residual_vs_true_mag.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_232309_utc/plots/v11_gap500k_d66_seed7_S3_angle_error_vs_true_mag.png`

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
