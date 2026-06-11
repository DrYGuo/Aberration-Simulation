# S3 Magnitude Metric Audit

Created UTC: 2026-06-11T19:18:28.947811+00:00

## Formula And Code-Path Check

- `true_mag = sqrt(true_S3_x^2 + true_S3_y^2)`.
- `pred_mag = sqrt(pred_S3_x^2 + pred_S3_y^2)`.
- High-S3 bin is selected by true magnitude: `true_mag > 0.7 * vector_scale`.
- `vector_scale` is the 95th percentile of true vector magnitude on the training split.
- The model runner inverse-transforms normalized predictions once with `y_scaler.inverse_transform(pred_scaled)` before diagnostics.
- `S3_x` and `S3_y` are adjacent target columns from the same physical target vector and use the same coefficient units.

## Artifact Limitation

The current GitHub artifact policy excludes raw predictions and model checkpoints. This report therefore verifies the stored metric and code path, but cannot recompute the 0.71 slope, through-origin slope, component slopes, or intercept from saved raw predictions.

## Stored Metric Comparison

| run | has raw predictions | has S3 audit CSV | weighted score | high n | stored high OLS slope | stored high MAE | stored high bias | stored high RMSE | high angle mean deg | high angle p95 deg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old_v3_bin_diag | False | False | 0.05945 | 321 | 0.6711 | 10.84 | -8.767 | 18.39 | 5.703 | 20.93 |
| v3_smoothl1_champion | False | False | 0.05568 | 321 | 0.7209 | 9.54 | -7.683 | 16.72 | 5.484 | 23.4 |
| v5_s3tail60k_seed23 | False | False | 0.05102 | 289 | 0.7091 | 8.201 | -6.386 | 14.81 | 4.392 | 16.95 |
| v3_smoothl1_audit | False | True | 0.05568 | 321 | 0.7209 | 9.54 | -7.683 | 16.72 | 5.484 | 23.4 |
| v5_s3tail60k_seed23_audit | False | True | 0.05102 | 289 | 0.7091 | 8.201 | -6.386 | 14.81 | 4.392 | 16.95 |

## Full Recompute Tables

### Magnitude

| run | subset | OLS slope | intercept | through-origin slope | corr | R2 | MAE | bias | RMSE | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v3_smoothl1_audit | all_magnitude | 0.8661 | 2.752 | 0.9078 | 0.9569 | 0.9157 | 4.666 | -0.3377 | 9.3 | 1976 |
| v3_smoothl1_audit | high_magnitude | 0.7209 | 15.14 | 0.9029 | 0.471 | 0.2219 | 9.54 | -7.683 | 16.72 | 321 |
| v5_s3tail60k_seed23_audit | all_magnitude | 0.8822 | 3.013 | 0.9278 | 0.9635 | 0.9284 | 4.477 | 0.296 | 8.568 | 1977 |
| v5_s3tail60k_seed23_audit | high_magnitude | 0.7091 | 17.92 | 0.9207 | 0.4695 | 0.2204 | 8.201 | -6.386 | 14.81 | 289 |

### Components

| run | component subset | OLS slope | intercept | through-origin slope | corr | R2 | MAE | bias | RMSE | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v3_smoothl1_audit | S3_x_all | 0.9051 | 0.1815 | 0.905 | 0.9615 | 0.9244 | 3.545 | 0.2221 | 7.508 | 1976 |
| v3_smoothl1_audit | S3_y_all | 0.8826 | 0.3476 | 0.8824 | 0.952 | 0.9063 | 3.779 | 0.4096 | 8.605 | 1976 |
| v3_smoothl1_audit | S3_x_high | 0.9021 | 1.335 | 0.9009 | 0.974 | 0.9487 | 7.158 | 1.623 | 13.26 | 321 |
| v3_smoothl1_audit | S3_y_high | 0.8847 | 0.05411 | 0.8846 | 0.966 | 0.9331 | 7.776 | 0.3968 | 15.71 | 321 |
| v5_s3tail60k_seed23_audit | S3_x_all | 0.9271 | 0.3777 | 0.9269 | 0.967 | 0.9351 | 3.524 | 0.4089 | 6.95 | 1977 |
| v5_s3tail60k_seed23_audit | S3_y_all | 0.9053 | 0.1008 | 0.9053 | 0.9599 | 0.9215 | 3.61 | 0.1508 | 7.857 | 1977 |
| v5_s3tail60k_seed23_audit | S3_x_high | 0.9284 | 1.394 | 0.9268 | 0.9827 | 0.9656 | 6.536 | 1.68 | 11.14 | 289 |
| v5_s3tail60k_seed23_audit | S3_y_high | 0.9009 | -0.3686 | 0.9013 | 0.974 | 0.9488 | 7.072 | -0.003793 | 14.02 | 289 |

## Plots

### old_v3_bin_diag
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260611_191826_utc/plots/old_v3_bin_diag_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260611_191826_utc/plots/old_v3_bin_diag_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260611_191826_utc/plots/old_v3_bin_diag_S3_pred_vs_true_angle.png`

### v3_smoothl1_champion
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260611_191826_utc/plots/v3_smoothl1_champion_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260611_191826_utc/plots/v3_smoothl1_champion_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260611_191826_utc/plots/v3_smoothl1_champion_S3_pred_vs_true_angle.png`

### v5_s3tail60k_seed23
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260611_191826_utc/plots/v5_s3tail60k_seed23_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260611_191826_utc/plots/v5_s3tail60k_seed23_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260611_191826_utc/plots/v5_s3tail60k_seed23_S3_pred_vs_true_angle.png`

### v3_smoothl1_audit
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260611_191826_utc/plots/v3_smoothl1_audit_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260611_191826_utc/plots/v3_smoothl1_audit_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260611_191826_utc/plots/v3_smoothl1_audit_S3_pred_vs_true_angle.png`
Generated full audit plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260611_191826_utc/plots/v3_smoothl1_audit_S3_x_pred_vs_true.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260611_191826_utc/plots/v3_smoothl1_audit_S3_y_pred_vs_true.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260611_191826_utc/plots/v3_smoothl1_audit_S3_mag_pred_vs_true_all.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260611_191826_utc/plots/v3_smoothl1_audit_S3_mag_pred_vs_true_high.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260611_191826_utc/plots/v3_smoothl1_audit_S3_mag_residual_vs_true_mag.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260611_191826_utc/plots/v3_smoothl1_audit_S3_vector_residual_vs_true_mag.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260611_191826_utc/plots/v3_smoothl1_audit_S3_angle_error_vs_true_mag.png`

### v5_s3tail60k_seed23_audit
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260611_191826_utc/plots/v5_s3tail60k_seed23_audit_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260611_191826_utc/plots/v5_s3tail60k_seed23_audit_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260611_191826_utc/plots/v5_s3tail60k_seed23_audit_S3_pred_vs_true_angle.png`
Generated full audit plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260611_191826_utc/plots/v5_s3tail60k_seed23_audit_S3_x_pred_vs_true.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260611_191826_utc/plots/v5_s3tail60k_seed23_audit_S3_y_pred_vs_true.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260611_191826_utc/plots/v5_s3tail60k_seed23_audit_S3_mag_pred_vs_true_all.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260611_191826_utc/plots/v5_s3tail60k_seed23_audit_S3_mag_pred_vs_true_high.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260611_191826_utc/plots/v5_s3tail60k_seed23_audit_S3_mag_residual_vs_true_mag.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260611_191826_utc/plots/v5_s3tail60k_seed23_audit_S3_vector_residual_vs_true_mag.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260611_191826_utc/plots/v5_s3tail60k_seed23_audit_S3_angle_error_vs_true_mag.png`

## Answers To Main Questions

A. The stored high-S3 magnitude slope near `0.71` is present in `vector_diagnostics.json` and the batch summary for the v5 seed23 run.

B. Code inspection shows it is an OLS-with-intercept slope from `np.polyfit(true_magnitude, pred_magnitude, 1)`. The current compact JSON does not store the intercept, so the intercept cannot be recovered without raw predictions.

C. The through-origin slope is computed when `s3_magnitude_metric_audit_validation.csv` or raw validation predictions are available; otherwise it cannot be recovered from historical compact artifacts.

D. Component slopes for `S3_x` and `S3_y` are computed from the S3 audit CSV when present. For older compact-only runs, existing component scatter plots are images only.

E. A low OLS slope could be affected by high-bin range restriction and nonzero intercept, but this cannot be confirmed or rejected without raw predictions and the intercept.

F. Whether magnitude bias becomes more negative at larger true `|S3|` requires residual-vs-true-magnitude data; current compact artifacts do not contain the raw residuals.

G. The feature-bottleneck conclusion should be treated as plausible but not final until the full magnitude-slope audit is run with raw validation predictions or compact S3 audit arrays.

## Recommendation

Before feature engineering, run one non-training diagnostic pass on Colab that saves compact `validation_predictions_s3_audit.npz` or an equivalent small S3-only audit CSV for the v3/v5 runs. Then rerun this script to compute through-origin slopes, component slopes, intercepts, residual plots, and decide whether the `0.71` slope is a real compression effect or an OLS/intercept artifact.
