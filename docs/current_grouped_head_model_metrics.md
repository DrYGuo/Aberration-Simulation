# Current Grouped-Head Regression Model

Run: `D66_grouped_width320_lr6e-4_dropout0.075_v12benchmarkv2_500k_seed7_20260615_005333_utc`

Latest completed run as of this document is the v12 benchmark-v2 500K evaluation. The v13/1M workflow may be running, but no v13 metrics were available when this table was generated.

## Model

- Architecture: `grouped_heads` grouped-head residual MLP
- Feature count: `66`
- Hidden width: `320`
- Dropout: `0.075`
- Optimizer/loss: AdamW + `smooth_l1` component loss
- Dataset rows: `575000`
- Train / validation / blind / stress: `492556` / `26977` / `27370` / `28097`

## Validation Per-Coefficient Error

MAE, MSE, and RMSE are in the same physical units as each target coefficient; MSE is the squared unit. Normalized MAE is divided by the configured physical target scale.

| coefficient | validation MAE | validation MSE | validation RMSE | normalized MAE |
|---|---:|---:|---:|---:|
| `C1` | 0.6578 | 1.5476 | 1.2440 | 0.00658 |
| `C3` | 0.0042 | 0.0001 | 0.0078 | 0.00212 |
| `A1_x` | 0.3107 | 0.3911 | 0.6254 | 0.00518 |
| `A1_y` | 0.3086 | 0.3952 | 0.6287 | 0.00514 |
| `B2_x` | 0.0181 | 0.0016 | 0.0405 | 0.00605 |
| `B2_y` | 0.0179 | 0.0016 | 0.0403 | 0.00597 |
| `A2_x` | 0.0349 | 0.0053 | 0.0730 | 0.00218 |
| `A2_y` | 0.0353 | 0.0056 | 0.0745 | 0.00221 |
| `S3_x` | 0.9572 | 4.9573 | 2.2265 | 0.00957 |
| `S3_y` | 0.9569 | 4.9761 | 2.2307 | 0.00957 |
| `A3_x` | 0.4773 | 1.0009 | 1.0005 | 0.00477 |
| `A3_y` | 0.4819 | 1.0074 | 1.0037 | 0.00482 |

Source files:
- `training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v12benchmarkv2_500k_seed7_20260615_005333_utc/metrics_model_loop.json`
- `docs/current_grouped_head_model_metrics.csv`
