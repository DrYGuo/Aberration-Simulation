# C1 Feature Sensitivity - v8_c1diff_basic

- CSV: `training_results/feature_regression_enhanced/enhanced_v8_c1diff_basic_20260613_231456_utc/training_features_enhanced.csv`
- Rows: `100446`
- Audited features: `34`

## Top Overall Features

| feature | n | pearson r | slope | r2 |
|---|---:|---:|---:|---:|
| `under_Xigma_h2_imag` | 100446 | 0.0001 | 2.56097e-06 | 0.0000 |
| `over_Xigma_h2_real` | 100446 | 0.0002 | 3.62094e-06 | 0.0000 |
| `over_Xigma_h4_real` | 100446 | -0.0002 | -3.03443e-06 | 0.0000 |
| `over_Xigma_h2_imag` | 100446 | -0.0006 | -1.01773e-05 | 0.0000 |
| `over_Xigma_h4_imag` | 100446 | 0.0007 | 9.41765e-06 | 0.0000 |
| `under_Xigma_h4_real` | 100446 | 0.0008 | 1.09811e-05 | 0.0000 |
| `under_Xigma_h4_imag` | 100446 | -0.0009 | -1.30789e-05 | 0.0000 |
| `defocus_Mu_mean_under_minus_over` | 100446 | -0.0013 | -1.45971e-05 | 0.0000 |
| `defocus_Mu_mean_over_minus_under` | 100446 | 0.0013 | 1.45971e-05 | 0.0000 |
| `under_Xigma_h2_real` | 100446 | -0.0026 | -5.2745e-05 | 0.0000 |
| `defocus_Mu_mean_sum` | 100446 | -0.0028 | -0.00013913 | 0.0000 |
| `defocus_Mu_mean_norm_under_minus_over` | 100446 | 0.0044 | 3.93423e-05 | 0.0000 |
| `defocus_Mu_mean_norm_over_minus_under` | 100446 | -0.0044 | -3.93423e-05 | 0.0000 |
| `defocus_Rho_mean_norm_under_minus_over` | 100446 | 0.0068 | 7.88001e-05 | 0.0000 |
| `defocus_Rho_mean_norm_over_minus_under` | 100446 | -0.0068 | -7.88001e-05 | 0.0000 |

## Interpretation

Large absolute correlations or slopes indicate C1-sensitive features. If explicit defocus-difference features do not rank above the existing collapsed C1/defocus features, C1 error is less likely to be fixed by adding more rows alone.

Detailed JSON: `training_results/model_selection_reports/c1_feature_sensitivity_v8_c1diff_basic_20260613_231641_utc/c1_feature_sensitivity.json`
Overall CSV: `training_results/model_selection_reports/c1_feature_sensitivity_v8_c1diff_basic_20260613_231641_utc/c1_feature_sensitivity_overall.csv`
