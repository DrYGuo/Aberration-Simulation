# Training-Validation Loss Gap Audit

Created UTC: 2026-06-15T01:41:35.616971+00:00

## Summary conclusion

The large visual gap is real in the stored history curves, but the plotted curves are not the same objective optimized by SmoothL1 runs. The PNG plots eval-mode weighted scaled MSE for train and validation on a log y-axis. SmoothL1 runs optimize SmoothL1 component loss plus residual penalty, with optional S3 magnitude loss. That makes the plot label technically correct but incomplete for SmoothL1 interpretation.

The gap first becomes large after the targeted25k training-only rows enter the workflow. Later v4b and SmoothL1 variants increase the validation/train MSE ratio further, but the step change is already visible before SmoothL1. The v5 S3-tail run improves validation/blind/stress normalized MAE and weighted score while keeping a large MSE-history gap, so the current model comparison remains valid under the stored selection metrics.

Classification: `C_loss_logging_mismatch_plus_D_split_benchmark_distribution_difference`, with a real validation/generalization gap on parent benchmark rows. The audit cannot prove `B_MSE_tail_effect` or identify top training outlier rows from existing compact artifacts because raw train predictions/checkpoints were intentionally not pushed.

## Source-code verification

- `plot_history()` plots `train_loss` and `validation_loss` with y-axis label `weighted scaled MSE` and `log` scaling.
- `train_loss` is eval-mode `weighted_mse(train_pred_eval, y_train, target_weights)` on standardized targets.
- `validation_loss` is eval-mode `weighted_mse(model(x_validation), y_validation, target_weights)` on standardized targets.
- SmoothL1 runs optimize `weighted_component_loss(..., loss_kind=smooth_l1) + S3 magnitude loss + residual penalty`, not the plotted MSE.
- `x_scaler` and `y_scaler` are fit on `train_index` only; the same transformed target scale and target weights are used for train and validation loss.

## History gap table

| run | dataset | train_rows | validation_rows | training_only | loss_kind | best_epoch | best_val_mse | final_train_mse | final_val_mse | val/train final | weighted_score | overall_norm_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D66_grouped_width320_lr6e-4_dropout0.075_v12benchmarkv2_500k_seed7_20260615_005333_utc | enhanced_v9_gap250k | 492556 | 26977 | 483554 | smooth_l1 | 1930 | 0.002057 | 0.001725 | 0.002098 | 1.216 | 0.01342 | 0.005347 |


## Dataset split/source composition

| dataset | split | n | training_only | unhinted_parent | s3_near_zero | s3_low | s3_medium | s3_high | sources |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| enhanced_v9_gap250k | train | 242566 | 233554 | 9012 | 95131 | 11890 | 33542 | 102003 | {"enhanced_v3_targeted25k": 25000, "enhanced_v5_s3_tail60k": 16000, "enhanced_v6_benchmark_gap100k": 43000, "enhanced_v9_gap250k": 149554, "parent": 9012} |
| enhanced_v9_gap250k | validation | 1979 | 0 | 1979 | 1161 | 162 | 367 | 289 | {"parent": 1979} |
| enhanced_v9_gap250k | blind | 2371 | 0 | 2371 | 1972 | 73 | 184 | 142 | {"parent": 2371} |
| enhanced_v9_gap250k | stress | 3084 | 0 | 3084 | 885 | 468 | 902 | 829 | {"parent": 3084} |


## Per-target train vs validation metrics

These are physical-unit RMSE and physical-scale normalized MAE from `metrics_model_loop.json`, not the standardized weighted MSE used by the history PNG.

_No rows available._


## Hard-regime label metrics

_No rows available._


## Requested decomposition status

- `train parent only` vs `validation parent`: unavailable for historical runs without per-row predictions or checkpoints.
- `training_only`, `coupled`, and `high-S3` fraction of training MSE: unavailable for historical runs without per-row train predictions.
- top 20 training rows by squared error: unavailable for historical runs without per-row train predictions.
- benchmark row drift: split counts are stable in manifests, and training-only rows are explicitly excluded from validation/blind/stress by policy.

## Interpretation

The gap first appears when training-only targeted/coupled rows are appended and the split policy changes to parent benchmark rows plus training-only append rows. Because the curve is train MSE much lower than validation MSE, the stored history does not support the explanation that hard training-only rows are raising train MSE. Instead, the model fits the enlarged training set very closely while validation remains a harder parent benchmark distribution. Coupled-full and coupled-sparse validation/stress labels remain the dominant difficult regimes in aggregate metrics.

The log y-axis visually amplifies the separation after train MSE becomes very small. For SmoothL1 runs, the plot also hides the objective actually optimized during training, so users can easily compare a plotted MSE curve against optimizer behavior incorrectly.

## Recommendation

Do not use this audit alone to justify feature engineering or a jump to 100k rows. The immediate fix should be logging: future training-history output should show `train_all_weighted_mse`, `validation_weighted_mse`, `train_total_objective`, and, when per-row predictions are available, `train_parent`, `train_training_only`, `blind`, and `stress` curves. For source-level loss decomposition, either save compact split/source loss summaries during each run or rerun the audited candidates with the diagnostic CSV enabled.

## Plots

- `training_results/model_selection_reports/training_validation_loss_gap_audit_20260615_014112_utc/plots/audited_train_validation_history.png`
- `training_results/model_selection_reports/training_validation_loss_gap_audit_20260615_014112_utc/plots/validation_over_train_ratio.png`
- `training_results/model_selection_reports/training_validation_loss_gap_audit_20260615_014112_utc/plots/split_normalized_mae.png`
- `training_results/model_selection_reports/training_validation_loss_gap_audit_20260615_014112_utc/plots/enhanced_v9_gap250k_s3_bin_counts.png`
