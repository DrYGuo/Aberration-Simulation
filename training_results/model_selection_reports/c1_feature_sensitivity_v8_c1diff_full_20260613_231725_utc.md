# C1 Feature Sensitivity - v8_c1diff_full

- CSV: `training_results/feature_regression_enhanced/enhanced_v8_c1diff_full_20260613_231542_utc/training_features_enhanced.csv`
- Rows: `100446`
- Audited features: `80`

## Top Overall Features

| feature | n | pearson r | slope | r2 |
|---|---:|---:|---:|---:|
| `defocus_Mu_h3_under_minus_over_real` | 100446 | -0.0047 | -0.000127189 | 0.0000 |
| `defocus_Xigma_h4_norm_under_minus_over_imag` | 100446 | -0.0051 | -6.43564e-05 | 0.0000 |
| `defocus_Mu_h2_under_minus_over_imag` | 100446 | 0.0056 | 0.000106722 | 0.0000 |
| `defocus_Mu_h1_norm_under_minus_over_magnitude` | 100446 | 0.0061 | 7.15313e-05 | 0.0000 |
| `defocus_Xigma_h2_under_plus_over_real` | 100446 | -0.0064 | -4.9124e-05 | 0.0000 |
| `defocus_Rho_mean_norm_under_minus_over` | 100446 | 0.0068 | 7.88001e-05 | 0.0000 |
| `defocus_Rho_mean_norm_over_minus_under` | 100446 | -0.0068 | -7.88001e-05 | 0.0000 |
| `defocus_Rho_h1_norm_under_minus_over_magnitude` | 100446 | -0.0074 | -6.89874e-05 | 0.0001 |
| `defocus_Rho_h4_under_minus_over_real` | 100446 | 0.0080 | 0.00010066 | 0.0001 |
| `defocus_Mu_h4_norm_under_minus_over_magnitude` | 100446 | 0.0081 | 5.11243e-05 | 0.0001 |
| `defocus_Rho_h3_norm_under_minus_over_magnitude` | 100446 | -0.0108 | -9.54174e-05 | 0.0001 |
| `defocus_Xigma_h4_under_minus_over_magnitude` | 100446 | -0.0121 | -0.000381583 | 0.0001 |
| `defocus_Mu_h2_norm_under_minus_over_magnitude` | 100446 | 0.0130 | 7.50845e-05 | 0.0002 |
| `defocus_Xigma_h4_norm_under_minus_over_magnitude` | 100446 | 0.0139 | 0.000146452 | 0.0002 |
| `defocus_Xigma_h4_under_plus_over_magnitude` | 100446 | 0.0142 | 0.000147346 | 0.0002 |

## Interpretation

Large absolute correlations or slopes indicate C1-sensitive features. If explicit defocus-difference features do not rank above the existing collapsed C1/defocus features, C1 error is less likely to be fixed by adding more rows alone.

Detailed JSON: `training_results/model_selection_reports/c1_feature_sensitivity_v8_c1diff_full_20260613_231725_utc/c1_feature_sensitivity.json`
Overall CSV: `training_results/model_selection_reports/c1_feature_sensitivity_v8_c1diff_full_20260613_231725_utc/c1_feature_sensitivity_overall.csv`
