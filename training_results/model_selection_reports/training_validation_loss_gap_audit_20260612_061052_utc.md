# Training-Validation Loss Gap Audit

Created UTC: 2026-06-12T06:11:11.578536+00:00

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
| D66_width128_lr4e-4_dropout0.05_20260606_102435_utc | NA | 13158 | NA | NA | mse | 6000 | 0.04187 | 0.009626 | 0.04187 | 4.35 | 4.012 | NA |
| D66_grouped_width192_lr6e-4_dropout0.075_targeted25k_20260607_050604_utc | parent_cached_dataset | 35424 | 1976 | 25000 | mse | 5825 | 0.03508 | 0.003997 | 0.03553 | 8.89 | 0.06098 | 0.01138 |
| D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_bin_diag_20260609_073514_utc | enhanced_v3_targeted25k | 34002 | 1976 | 25000 | mse | 5925 | 0.03431 | 0.001477 | 0.03443 | 23.32 | 0.05945 | 0.008191 |
| D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_plateau_clip_smoothl1_20260610_071108_utc | enhanced_v3_targeted25k | 34002 | 1976 | 25000 | smooth_l1 | 5850 | 0.03211 | 0.001179 | 0.03224 | 27.34 | 0.05568 | 0.007392 |
| D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_plateau_clip_smoothl1_seed11_20260611_072635_utc | enhanced_v3_targeted25k | 34002 | 1976 | 25000 | smooth_l1 | 5500 | 0.03133 | 0.001106 | 0.03154 | 28.53 | 0.05608 | 0.007325 |
| D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_plateau_clip_smoothl1_seed23_20260611_073754_utc | enhanced_v3_targeted25k | 34002 | 1976 | 25000 | smooth_l1 | 5575 | 0.03114 | 0.001181 | 0.03131 | 26.51 | 0.05656 | 0.007388 |
| D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_plateau_clip_smoothl1_seed37_20260611_074903_utc | enhanced_v3_targeted25k | 34002 | 1976 | 25000 | smooth_l1 | 5350 | 0.03319 | 0.00125 | 0.03326 | 26.6 | 0.05777 | 0.007533 |
| D66_grouped_width320_lr6e-4_dropout0.075_s3tail60k_plateau_clip_smoothl1_seed7_20260611_084343_utc | enhanced_v5_s3_tail60k | 50002 | 1977 | 41000 | smooth_l1 | 5925 | 0.02124 | 0.001098 | 0.02128 | 19.38 | 0.05249 | 0.006251 |
| D66_grouped_width320_lr6e-4_dropout0.075_s3tail60k_plateau_clip_smoothl1_seed23_20260611_090007_utc | enhanced_v5_s3_tail60k | 50002 | 1977 | 41000 | smooth_l1 | 5950 | 0.0203 | 0.0009905 | 0.02034 | 20.54 | 0.05102 | 0.006029 |
| D66_grouped_width320_lr6e-4_dropout0.075_v6gap100k_seed23_20260612_051859_utc | enhanced_v6_benchmark_gap100k | 93011 | 1979 | 84000 | smooth_l1 | 5875 | 0.009917 | 0.001674 | 0.009972 | 5.956 | 0.0369 | 0.006574 |
| D66_grouped_width320_lr6e-4_dropout0.075_v6gap100k_seed7_20260612_054505_utc | enhanced_v6_benchmark_gap100k | 93011 | 1979 | 84000 | smooth_l1 | 5975 | 0.00989 | 0.001659 | 0.009954 | 5.999 | 0.03714 | 0.0066 |


## Dataset split/source composition

| dataset | split | n | training_only | unhinted_parent | s3_near_zero | s3_low | s3_medium | s3_high | sources |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| enhanced_v3_targeted25k | train | 34002 | 25000 | 9002 | 17755 | 3308 | 6406 | 6533 | {"enhanced_v3_targeted25k": 25000, "parent": 9002} |
| enhanced_v3_targeted25k | validation | 1976 | 0 | 1976 | 1155 | 153 | 347 | 321 | {"parent": 1976} |
| enhanced_v3_targeted25k | blind | 2370 | 0 | 2370 | 1971 | 69 | 161 | 169 | {"parent": 2370} |
| enhanced_v3_targeted25k | stress | 3098 | 0 | 3098 | 888 | 435 | 863 | 912 | {"parent": 3098} |
| enhanced_v5_s3_tail60k | train | 50002 | 41000 | 9002 | 17843 | 3489 | 8363 | 20307 | {"enhanced_v3_targeted25k": 25000, "enhanced_v5_s3_tail60k": 16000, "parent": 9002} |
| enhanced_v5_s3_tail60k | validation | 1977 | 0 | 1977 | 1158 | 163 | 367 | 289 | {"parent": 1977} |
| enhanced_v5_s3_tail60k | blind | 2370 | 0 | 2370 | 1971 | 73 | 183 | 143 | {"parent": 2370} |
| enhanced_v5_s3_tail60k | stress | 3097 | 0 | 3097 | 897 | 466 | 900 | 834 | {"parent": 3097} |
| enhanced_v6_benchmark_gap100k | train | 93011 | 84000 | 9011 | 32660 | 6054 | 15263 | 39034 | {"enhanced_v3_targeted25k": 25000, "enhanced_v5_s3_tail60k": 16000, "enhanced_v6_benchmark_gap100k": 43000, "parent": 9011} |
| enhanced_v6_benchmark_gap100k | validation | 1979 | 0 | 1979 | 1161 | 162 | 367 | 289 | {"parent": 1979} |
| enhanced_v6_benchmark_gap100k | blind | 2371 | 0 | 2371 | 1972 | 73 | 184 | 142 | {"parent": 2371} |
| enhanced_v6_benchmark_gap100k | stress | 3085 | 0 | 3085 | 886 | 467 | 900 | 832 | {"parent": 3085} |


## Per-target train vs validation metrics

These are physical-unit RMSE and physical-scale normalized MAE from `metrics_model_loop.json`, not the standardized weighted MSE used by the history PNG.

| run | target | train_normalized_mae | validation_normalized_mae | train_rmse | validation_rmse |
| --- | --- | --- | --- | --- | --- |
| D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_plateau_clip_smoothl1_20260610_071108_utc | C1 | 0.008076 | 0.03013 | 1.195 | 5.159 |
| D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_plateau_clip_smoothl1_20260610_071108_utc | C3 | 0.003587 | 0.008273 | 0.01075 | 0.02775 |
| D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_plateau_clip_smoothl1_20260610_071108_utc | A1_x | 0.005247 | 0.01651 | 0.5216 | 1.887 |
| D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_plateau_clip_smoothl1_20260610_071108_utc | A1_y | 0.005138 | 0.01733 | 0.5038 | 2.212 |
| D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_plateau_clip_smoothl1_20260610_071108_utc | B2_x | 0.006888 | 0.0239 | 0.03519 | 0.1255 |
| D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_plateau_clip_smoothl1_20260610_071108_utc | B2_y | 0.006774 | 0.02418 | 0.03429 | 0.1319 |
| D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_plateau_clip_smoothl1_20260610_071108_utc | A2_x | 0.003375 | 0.008933 | 0.09581 | 0.2336 |
| D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_plateau_clip_smoothl1_20260610_071108_utc | A2_y | 0.003401 | 0.009262 | 0.09592 | 0.2392 |
| D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_plateau_clip_smoothl1_20260610_071108_utc | S3_x | 0.00655 | 0.03545 | 1.108 | 7.508 |
| D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_plateau_clip_smoothl1_20260610_071108_utc | S3_y | 0.006478 | 0.03779 | 1.084 | 8.605 |
| D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_plateau_clip_smoothl1_20260610_071108_utc | A3_x | 0.004434 | 0.01789 | 0.7489 | 3.718 |
| D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_plateau_clip_smoothl1_20260610_071108_utc | A3_y | 0.004411 | 0.01722 | 0.7469 | 3.06 |
| D66_grouped_width320_lr6e-4_dropout0.075_s3tail60k_plateau_clip_smoothl1_seed23_20260611_090007_utc | C1 | 0.006378 | 0.02596 | 1.02 | 4.572 |
| D66_grouped_width320_lr6e-4_dropout0.075_s3tail60k_plateau_clip_smoothl1_seed23_20260611_090007_utc | C3 | 0.002802 | 0.007159 | 0.009048 | 0.02399 |
| D66_grouped_width320_lr6e-4_dropout0.075_s3tail60k_plateau_clip_smoothl1_seed23_20260611_090007_utc | A1_x | 0.004599 | 0.01623 | 0.4823 | 1.793 |
| D66_grouped_width320_lr6e-4_dropout0.075_s3tail60k_plateau_clip_smoothl1_seed23_20260611_090007_utc | A1_y | 0.004556 | 0.01639 | 0.4713 | 1.967 |
| D66_grouped_width320_lr6e-4_dropout0.075_s3tail60k_plateau_clip_smoothl1_seed23_20260611_090007_utc | B2_x | 0.005641 | 0.0214 | 0.03223 | 0.1103 |
| D66_grouped_width320_lr6e-4_dropout0.075_s3tail60k_plateau_clip_smoothl1_seed23_20260611_090007_utc | B2_y | 0.00557 | 0.02219 | 0.03161 | 0.1203 |
| D66_grouped_width320_lr6e-4_dropout0.075_s3tail60k_plateau_clip_smoothl1_seed23_20260611_090007_utc | A2_x | 0.002495 | 0.007781 | 0.07832 | 0.1971 |
| D66_grouped_width320_lr6e-4_dropout0.075_s3tail60k_plateau_clip_smoothl1_seed23_20260611_090007_utc | A2_y | 0.002457 | 0.00778 | 0.07747 | 0.2035 |
| D66_grouped_width320_lr6e-4_dropout0.075_s3tail60k_plateau_clip_smoothl1_seed23_20260611_090007_utc | S3_x | 0.007095 | 0.03524 | 1.274 | 6.95 |
| D66_grouped_width320_lr6e-4_dropout0.075_s3tail60k_plateau_clip_smoothl1_seed23_20260611_090007_utc | S3_y | 0.007145 | 0.0361 | 1.256 | 7.857 |
| D66_grouped_width320_lr6e-4_dropout0.075_s3tail60k_plateau_clip_smoothl1_seed23_20260611_090007_utc | A3_x | 0.003721 | 0.01653 | 0.6818 | 3.477 |
| D66_grouped_width320_lr6e-4_dropout0.075_s3tail60k_plateau_clip_smoothl1_seed23_20260611_090007_utc | A3_y | 0.003674 | 0.01522 | 0.6702 | 2.712 |


## Hard-regime label metrics

| run | split | label | n | normalized_mae | rmse | normalized_p95_abs_error |
| --- | --- | --- | --- | --- | --- | --- |
| D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_plateau_clip_smoothl1_20260610_071108_utc | train | coupled_A1_S3_random | 3500 | 0.001211 | 0.1468 | 0.003731 |
| D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_plateau_clip_smoothl1_20260610_071108_utc | train | coupled_A3_S3_random | 2000 | 0.001853 | 0.2301 | 0.005918 |
| D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_plateau_clip_smoothl1_20260610_071108_utc | train | coupled_full_random | 6814 | 0.01082 | 1.039 | 0.02943 |
| D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_plateau_clip_smoothl1_20260610_071108_utc | train | coupled_sparse_random | 5395 | 0.006776 | 0.7975 | 0.02343 |
| D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_plateau_clip_smoothl1_20260610_071108_utc | validation | coupled_full_random | 797 | 0.03262 | 5.367 | 0.1116 |
| D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_plateau_clip_smoothl1_20260610_071108_utc | validation | coupled_sparse_random | 395 | 0.01965 | 3.896 | 0.07789 |
| D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_plateau_clip_smoothl1_20260610_071108_utc | blind | coupled_full_random | 396 | 0.03262 | 5.765 | 0.1136 |
| D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_plateau_clip_smoothl1_20260610_071108_utc | blind | coupled_sparse_random | 185 | 0.01931 | 3.398 | 0.08181 |
| D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_plateau_clip_smoothl1_20260610_071108_utc | stress | coupled_A1_B2_S3_random | 1500 | 0.002731 | 0.2698 | 0.009913 |
| D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_plateau_clip_smoothl1_20260610_071108_utc | stress | coupled_full_random | 993 | 0.03248 | 5.348 | 0.1118 |
| D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_plateau_clip_smoothl1_20260610_071108_utc | stress | coupled_sparse_random | 525 | 0.01919 | 3.673 | 0.07609 |
| D66_grouped_width320_lr6e-4_dropout0.075_s3tail60k_plateau_clip_smoothl1_seed23_20260611_090007_utc | train | S3_high_random | 2500 | 0.0005428 | 0.07019 | 0.001818 |
| D66_grouped_width320_lr6e-4_dropout0.075_s3tail60k_plateau_clip_smoothl1_seed23_20260611_090007_utc | train | coupled_A1_B2_S3_random | 2500 | 0.002198 | 0.24 | 0.006761 |
| D66_grouped_width320_lr6e-4_dropout0.075_s3tail60k_plateau_clip_smoothl1_seed23_20260611_090007_utc | train | coupled_A1_S3_random | 4500 | 0.001099 | 0.1377 | 0.003413 |
| D66_grouped_width320_lr6e-4_dropout0.075_s3tail60k_plateau_clip_smoothl1_seed23_20260611_090007_utc | train | coupled_A3_S3_random | 5000 | 0.001557 | 0.215 | 0.005256 |
| D66_grouped_width320_lr6e-4_dropout0.075_s3tail60k_plateau_clip_smoothl1_seed23_20260611_090007_utc | train | coupled_B2_S3_random | 1500 | 0.001566 | 0.158 | 0.005028 |
| D66_grouped_width320_lr6e-4_dropout0.075_s3tail60k_plateau_clip_smoothl1_seed23_20260611_090007_utc | train | coupled_full_random | 7814 | 0.01159 | 1.213 | 0.0328 |
| D66_grouped_width320_lr6e-4_dropout0.075_s3tail60k_plateau_clip_smoothl1_seed23_20260611_090007_utc | train | coupled_sparse_random | 7395 | 0.006684 | 0.8695 | 0.02455 |
| D66_grouped_width320_lr6e-4_dropout0.075_s3tail60k_plateau_clip_smoothl1_seed23_20260611_090007_utc | validation | coupled_full_random | 797 | 0.02977 | 4.872 | 0.1036 |
| D66_grouped_width320_lr6e-4_dropout0.075_s3tail60k_plateau_clip_smoothl1_seed23_20260611_090007_utc | validation | coupled_sparse_random | 395 | 0.01796 | 3.475 | 0.07114 |
| D66_grouped_width320_lr6e-4_dropout0.075_s3tail60k_plateau_clip_smoothl1_seed23_20260611_090007_utc | blind | coupled_full_random | 396 | 0.02881 | 4.807 | 0.1013 |
| D66_grouped_width320_lr6e-4_dropout0.075_s3tail60k_plateau_clip_smoothl1_seed23_20260611_090007_utc | blind | coupled_sparse_random | 185 | 0.01805 | 3.16 | 0.07401 |
| D66_grouped_width320_lr6e-4_dropout0.075_s3tail60k_plateau_clip_smoothl1_seed23_20260611_090007_utc | stress | coupled_A1_B2_S3_random | 1500 | 0.001647 | 0.2027 | 0.005388 |
| D66_grouped_width320_lr6e-4_dropout0.075_s3tail60k_plateau_clip_smoothl1_seed23_20260611_090007_utc | stress | coupled_full_random | 993 | 0.02904 | 4.615 | 0.1013 |
| D66_grouped_width320_lr6e-4_dropout0.075_s3tail60k_plateau_clip_smoothl1_seed23_20260611_090007_utc | stress | coupled_sparse_random | 525 | 0.01729 | 3.211 | 0.06949 |


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

- `training_results/model_selection_reports/training_validation_loss_gap_audit_20260612_061052_utc/plots/audited_train_validation_history.png`
- `training_results/model_selection_reports/training_validation_loss_gap_audit_20260612_061052_utc/plots/validation_over_train_ratio.png`
- `training_results/model_selection_reports/training_validation_loss_gap_audit_20260612_061052_utc/plots/split_normalized_mae.png`
- `training_results/model_selection_reports/training_validation_loss_gap_audit_20260612_061052_utc/plots/enhanced_v3_targeted25k_s3_bin_counts.png`
- `training_results/model_selection_reports/training_validation_loss_gap_audit_20260612_061052_utc/plots/enhanced_v5_s3_tail60k_s3_bin_counts.png`
- `training_results/model_selection_reports/training_validation_loss_gap_audit_20260612_061052_utc/plots/enhanced_v6_benchmark_gap100k_s3_bin_counts.png`
