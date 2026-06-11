# S3 Magnitude Metric Audit

Created UTC: 2026-06-11T09:36:52.855875+00:00

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

| run | has raw predictions | weighted score | high n | stored high OLS slope | stored high MAE | stored high bias | stored high RMSE | high angle mean deg | high angle p95 deg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old_v3_bin_diag | False | 0.05945 | 321 | 0.6711 | 10.84 | -8.767 | 18.39 | 5.703 | 20.93 |
| v3_smoothl1_champion | False | 0.05568 | 321 | 0.7209 | 9.54 | -7.683 | 16.72 | 5.484 | 23.4 |
| v5_s3tail60k_seed23 | False | 0.05102 | 289 | 0.7091 | 8.201 | -6.386 | 14.81 | 4.392 | 16.95 |

## Full Recompute Tables

No run folder contained `validation_predictions_s3_audit.npz`, so through-origin slopes and component slopes could not be recomputed from saved predictions in this local audit.

## Plots

### old_v3_bin_diag
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260611_093652_utc/plots/old_v3_bin_diag_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260611_093652_utc/plots/old_v3_bin_diag_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260611_093652_utc/plots/old_v3_bin_diag_S3_pred_vs_true_angle.png`

### v3_smoothl1_champion
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260611_093652_utc/plots/v3_smoothl1_champion_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260611_093652_utc/plots/v3_smoothl1_champion_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260611_093652_utc/plots/v3_smoothl1_champion_S3_pred_vs_true_angle.png`

### v5_s3tail60k_seed23
Copied existing compact plots:
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260611_093652_utc/plots/v5_s3tail60k_seed23_S3_pred_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260611_093652_utc/plots/v5_s3tail60k_seed23_S3_angle_error_vs_true_magnitude.png`
- `training_results/model_selection_reports/s3_magnitude_metric_audit_20260611_093652_utc/plots/v5_s3tail60k_seed23_S3_pred_vs_true_angle.png`

## Answers To Main Questions

A. The stored high-S3 magnitude slope near `0.71` is present in `vector_diagnostics.json` and the batch summary for the v5 seed23 run.

B. Code inspection shows it is an OLS-with-intercept slope from `np.polyfit(true_magnitude, pred_magnitude, 1)`. The current compact JSON does not store the intercept, so the intercept cannot be recovered without raw predictions.

C. The through-origin slope cannot be computed from the current compact artifacts because raw validation predictions are not pushed.

D. Component slopes for `S3_x` and `S3_y` cannot be computed from current compact artifacts for the same reason. Existing component scatter plots are images only.

E. A low OLS slope could be affected by high-bin range restriction and nonzero intercept, but this cannot be confirmed or rejected without raw predictions and the intercept.

F. Whether magnitude bias becomes more negative at larger true `|S3|` requires residual-vs-true-magnitude data; current compact artifacts do not contain the raw residuals.

G. The feature-bottleneck conclusion should be treated as plausible but not final until the full magnitude-slope audit is run with raw validation predictions or compact S3 audit arrays.

## Recommendation

Before feature engineering, run one non-training diagnostic pass on Colab that saves compact `validation_predictions_s3_audit.npz` or an equivalent small S3-only audit CSV for the v3/v5 runs. Then rerun this script to compute through-origin slopes, component slopes, intercepts, residual plots, and decide whether the `0.71` slope is a real compression effect or an OLS/intercept artifact.
