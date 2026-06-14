# S3 Magnitude Metric Audit

Created UTC: 2026-06-14T11:37:22.906579+00:00

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
| v10_structured_head_v9_250k_seed23 | False | True | 0.03118 | 289 | 0.8507 | 4.707 | -3.625 | 8.399 | 2.411 | 8.39 |
| v10_structured_head_v9_250k_seed7 | False | True | 0.03155 | 289 | 0.8442 | 5.061 | -3.565 | 9.757 | 2.284 | 7.968 |

## Full Recompute Tables

### Magnitude

| run | subset | OLS slope | intercept | through-origin slope | corr | R2 | MAE | bias | RMSE | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v10_structured_head_v9_250k_seed23 | all_magnitude | 0.9373 | 1.539 | 0.9606 | 0.9885 | 0.9771 | 2.644 | 0.09296 | 4.935 | 1977 |
| v10_structured_head_v9_250k_seed23 | high_magnitude | 0.8507 | 8.852 | 0.9552 | 0.746 | 0.5566 | 4.707 | -3.625 | 8.399 | 289 |
| v10_structured_head_v9_250k_seed7 | all_magnitude | 0.939 | 1.518 | 0.962 | 0.9866 | 0.9734 | 2.681 | 0.1109 | 5.259 | 1977 |
| v10_structured_head_v9_250k_seed7 | high_magnitude | 0.8442 | 9.454 | 0.9558 | 0.6783 | 0.4601 | 5.061 | -3.565 | 9.757 | 289 |

### Components

| run | component subset | OLS slope | intercept | through-origin slope | corr | R2 | MAE | bias | RMSE | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v10_structured_head_v9_250k_seed23 | S3_x_all | 0.9588 | 0.07919 | 0.9588 | 0.9882 | 0.9765 | 2.101 | 0.09679 | 4.202 | 1977 |
| v10_structured_head_v9_250k_seed23 | S3_y_all | 0.9552 | -0.003058 | 0.9552 | 0.9888 | 0.9777 | 2.158 | 0.02059 | 4.228 | 1977 |
| v10_structured_head_v9_250k_seed23 | S3_x_high | 0.9554 | 0.6785 | 0.9546 | 0.9933 | 0.9867 | 3.956 | 0.8566 | 7.005 | 289 |
| v10_structured_head_v9_250k_seed23 | S3_y_high | 0.9512 | -0.0482 | 0.9513 | 0.9932 | 0.9865 | 4.123 | 0.1313 | 7.348 | 289 |
| v10_structured_head_v9_250k_seed7 | S3_x_all | 0.9648 | 0.1744 | 0.9647 | 0.9877 | 0.9756 | 2.094 | 0.1894 | 4.268 | 1977 |
| v10_structured_head_v9_250k_seed7 | S3_y_all | 0.9527 | 0.1063 | 0.9526 | 0.9875 | 0.9751 | 2.167 | 0.1312 | 4.464 | 1977 |
| v10_structured_head_v9_250k_seed7 | S3_x_high | 0.964 | 1.031 | 0.9628 | 0.9929 | 0.9858 | 3.755 | 1.175 | 7.141 | 289 |
| v10_structured_head_v9_250k_seed7 | S3_y_high | 0.9454 | 0.1875 | 0.9452 | 0.9905 | 0.981 | 4.56 | 0.3885 | 8.628 | 289 |

## Plots

### old_v3_bin_diag
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_113720_utc/plots/old_v3_bin_diag_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_113720_utc/plots/old_v3_bin_diag_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_113720_utc/plots/old_v3_bin_diag_S3_pred_vs_true_angle.png`

### v3_smoothl1_champion
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_113720_utc/plots/v3_smoothl1_champion_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_113720_utc/plots/v3_smoothl1_champion_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_113720_utc/plots/v3_smoothl1_champion_S3_pred_vs_true_angle.png`

### v5_s3tail60k_seed23
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_113720_utc/plots/v5_s3tail60k_seed23_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_113720_utc/plots/v5_s3tail60k_seed23_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_113720_utc/plots/v5_s3tail60k_seed23_S3_pred_vs_true_angle.png`

### v10_structured_head_v9_250k_seed23
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_113720_utc/plots/v10_structured_head_v9_250k_seed23_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_113720_utc/plots/v10_structured_head_v9_250k_seed23_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_113720_utc/plots/v10_structured_head_v9_250k_seed23_S3_pred_vs_true_angle.png`
Generated full audit plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_113720_utc/plots/v10_structured_head_v9_250k_seed23_S3_x_pred_vs_true.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_113720_utc/plots/v10_structured_head_v9_250k_seed23_S3_y_pred_vs_true.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_113720_utc/plots/v10_structured_head_v9_250k_seed23_S3_mag_pred_vs_true_all.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_113720_utc/plots/v10_structured_head_v9_250k_seed23_S3_mag_pred_vs_true_high.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_113720_utc/plots/v10_structured_head_v9_250k_seed23_S3_mag_residual_vs_true_mag.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_113720_utc/plots/v10_structured_head_v9_250k_seed23_S3_vector_residual_vs_true_mag.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_113720_utc/plots/v10_structured_head_v9_250k_seed23_S3_angle_error_vs_true_mag.png`

### v10_structured_head_v9_250k_seed7
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_113720_utc/plots/v10_structured_head_v9_250k_seed7_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_113720_utc/plots/v10_structured_head_v9_250k_seed7_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_113720_utc/plots/v10_structured_head_v9_250k_seed7_S3_pred_vs_true_angle.png`
Generated full audit plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_113720_utc/plots/v10_structured_head_v9_250k_seed7_S3_x_pred_vs_true.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_113720_utc/plots/v10_structured_head_v9_250k_seed7_S3_y_pred_vs_true.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_113720_utc/plots/v10_structured_head_v9_250k_seed7_S3_mag_pred_vs_true_all.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_113720_utc/plots/v10_structured_head_v9_250k_seed7_S3_mag_pred_vs_true_high.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_113720_utc/plots/v10_structured_head_v9_250k_seed7_S3_mag_residual_vs_true_mag.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_113720_utc/plots/v10_structured_head_v9_250k_seed7_S3_vector_residual_vs_true_mag.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_113720_utc/plots/v10_structured_head_v9_250k_seed7_S3_angle_error_vs_true_mag.png`

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
