# C1 Feature Sensitivity - v6_d66_fixedsplit_input

- CSV: `training_results/feature_regression_enhanced/enhanced_v6_benchmark_gap100k_20260613_230608_utc/training_features_enhanced.csv`
- Rows: `100446`
- Audited features: `19`

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
| `under_Xigma_h2_real` | 100446 | -0.0026 | -5.2745e-05 | 0.0000 |
| `under_Xigma_h1_real` | 100446 | -0.0825 | -0.00261358 | 0.0068 |
| `under_Xigma_h3_real` | 100446 | -0.0845 | -0.0026291 | 0.0071 |
| `under_Xigma_h3_imag` | 100446 | -0.1045 | -0.00955114 | 0.0109 |
| `under_Xigma_h1_imag` | 100446 | -0.1063 | -0.0291409 | 0.0113 |
| `under_Xigma_mean` | 100446 | -0.1064 | -0.0229666 | 0.0113 |
| `over_Xigma_h1_real` | 100446 | 0.1110 | 0.00211536 | 0.0123 |
| `over_Xigma_h3_real` | 100446 | 0.1192 | 0.00213219 | 0.0142 |

## Interpretation

Large absolute correlations or slopes indicate C1-sensitive features. If explicit defocus-difference features do not rank above the existing collapsed C1/defocus features, C1 error is less likely to be fixed by adding more rows alone.

Detailed JSON: `training_results/model_selection_reports/c1_feature_sensitivity_v6_d66_fixedsplit_input_20260614_000452_utc/c1_feature_sensitivity.json`
Overall CSV: `training_results/model_selection_reports/c1_feature_sensitivity_v6_d66_fixedsplit_input_20260614_000452_utc/c1_feature_sensitivity_overall.csv`
