# S3 Magnitude Metric Audit

Created UTC: 2026-06-14T08:47:02.084318+00:00

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
| v9_gap250k_d66_seed23 | False | True | 0.03051 | 289 | 0.8655 | 4.729 | -3.119 | 8.64 | 2.399 | 8.075 |
| v9_gap250k_d66_seed7 | False | True | 0.03069 | 289 | 0.8514 | 4.656 | -3.003 | 8.306 | 2.315 | 8.064 |

## Full Recompute Tables

### Magnitude

| run | subset | OLS slope | intercept | through-origin slope | corr | R2 | MAE | bias | RMSE | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v9_gap250k_d66_seed23 | all_magnitude | 0.9428 | 1.47 | 0.965 | 0.988 | 0.9762 | 2.654 | 0.1507 | 4.978 | 1977 |
| v9_gap250k_d66_seed23 | high_magnitude | 0.8655 | 8.123 | 0.9614 | 0.7293 | 0.5319 | 4.729 | -3.119 | 8.64 | 289 |
| v9_gap250k_d66_seed7 | all_magnitude | 0.9414 | 1.514 | 0.9643 | 0.9886 | 0.9774 | 2.622 | 0.1623 | 4.873 | 1977 |
| v9_gap250k_d66_seed7 | high_magnitude | 0.8514 | 9.414 | 0.9625 | 0.7387 | 0.5457 | 4.656 | -3.003 | 8.306 | 289 |

### Components

| run | component subset | OLS slope | intercept | through-origin slope | corr | R2 | MAE | bias | RMSE | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v9_gap250k_d66_seed23 | S3_x_all | 0.9649 | 0.1732 | 0.9648 | 0.988 | 0.9762 | 2.116 | 0.1882 | 4.217 | 1977 |
| v9_gap250k_d66_seed23 | S3_y_all | 0.9576 | 0.02825 | 0.9576 | 0.988 | 0.9762 | 2.104 | 0.05061 | 4.353 | 1977 |
| v9_gap250k_d66_seed23 | S3_x_high | 0.9656 | 0.7289 | 0.9647 | 0.9937 | 0.9875 | 3.793 | 0.8664 | 6.676 | 289 |
| v9_gap250k_d66_seed23 | S3_y_high | 0.9532 | 0.01359 | 0.9532 | 0.9912 | 0.9825 | 4.227 | 0.186 | 8.206 | 289 |
| v9_gap250k_d66_seed7 | S3_x_all | 0.962 | 0.1858 | 0.9619 | 0.9883 | 0.9767 | 2.144 | 0.202 | 4.178 | 1977 |
| v9_gap250k_d66_seed7 | S3_y_all | 0.9599 | 0.1103 | 0.9598 | 0.9893 | 0.9788 | 2.043 | 0.1315 | 4.111 | 1977 |
| v9_gap250k_d66_seed7 | S3_x_high | 0.9634 | 0.7578 | 0.9625 | 0.9936 | 0.9873 | 3.882 | 0.9041 | 6.764 | 289 |
| v9_gap250k_d66_seed7 | S3_y_high | 0.9583 | 0.01195 | 0.9583 | 0.9929 | 0.9859 | 3.942 | 0.1654 | 7.392 | 289 |

## Plots

### old_v3_bin_diag
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_084659_utc/plots/old_v3_bin_diag_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_084659_utc/plots/old_v3_bin_diag_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_084659_utc/plots/old_v3_bin_diag_S3_pred_vs_true_angle.png`

### v3_smoothl1_champion
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_084659_utc/plots/v3_smoothl1_champion_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_084659_utc/plots/v3_smoothl1_champion_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_084659_utc/plots/v3_smoothl1_champion_S3_pred_vs_true_angle.png`

### v5_s3tail60k_seed23
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_084659_utc/plots/v5_s3tail60k_seed23_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_084659_utc/plots/v5_s3tail60k_seed23_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_084659_utc/plots/v5_s3tail60k_seed23_S3_pred_vs_true_angle.png`

### v9_gap250k_d66_seed23
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_084659_utc/plots/v9_gap250k_d66_seed23_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_084659_utc/plots/v9_gap250k_d66_seed23_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_084659_utc/plots/v9_gap250k_d66_seed23_S3_pred_vs_true_angle.png`
Generated full audit plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_084659_utc/plots/v9_gap250k_d66_seed23_S3_x_pred_vs_true.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_084659_utc/plots/v9_gap250k_d66_seed23_S3_y_pred_vs_true.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_084659_utc/plots/v9_gap250k_d66_seed23_S3_mag_pred_vs_true_all.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_084659_utc/plots/v9_gap250k_d66_seed23_S3_mag_pred_vs_true_high.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_084659_utc/plots/v9_gap250k_d66_seed23_S3_mag_residual_vs_true_mag.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_084659_utc/plots/v9_gap250k_d66_seed23_S3_vector_residual_vs_true_mag.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_084659_utc/plots/v9_gap250k_d66_seed23_S3_angle_error_vs_true_mag.png`

### v9_gap250k_d66_seed7
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_084659_utc/plots/v9_gap250k_d66_seed7_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_084659_utc/plots/v9_gap250k_d66_seed7_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_084659_utc/plots/v9_gap250k_d66_seed7_S3_pred_vs_true_angle.png`
Generated full audit plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_084659_utc/plots/v9_gap250k_d66_seed7_S3_x_pred_vs_true.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_084659_utc/plots/v9_gap250k_d66_seed7_S3_y_pred_vs_true.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_084659_utc/plots/v9_gap250k_d66_seed7_S3_mag_pred_vs_true_all.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_084659_utc/plots/v9_gap250k_d66_seed7_S3_mag_pred_vs_true_high.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_084659_utc/plots/v9_gap250k_d66_seed7_S3_mag_residual_vs_true_mag.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_084659_utc/plots/v9_gap250k_d66_seed7_S3_vector_residual_vs_true_mag.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260614_084659_utc/plots/v9_gap250k_d66_seed7_S3_angle_error_vs_true_mag.png`

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
