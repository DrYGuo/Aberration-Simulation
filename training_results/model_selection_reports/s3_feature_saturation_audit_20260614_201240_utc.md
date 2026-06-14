# S3 Feature Saturation Audit

- CSV: `/content/Aberration-Simulation/training_results/feature_regression_enhanced/enhanced_v3_targeted25k_20260614_200730_utc/training_features_enhanced.csv`
- Rows: 41446
- SHA256: `cc9bf7546193b1eeae4e33fa8f00b66d93185485bf71daeba4bb3cb4b67e257b`
- S3 vector scale: 90.7398
- High-S3 threshold: 63.5179
- High-S3 rows: 7934

## Available Current Features

| feature | n | Pearson | Spearman | high-S3 slope | high-S3 R2 | high/low mean ratio |
|---|---:|---:|---:|---:|---:|---:|
| Eq43_S3_value_magnitude | 41446 | 0.761 | 0.7909 | 0.03244 | 0.04608 | 4.006 |
| over_Mu_h2_magnitude | 41446 | -0.08502 | -0.08015 | -0.002317 | 6.168e-05 | 0.8103 |
| over_Rho_h2_magnitude | 41446 | 0.7701 | 0.7651 | 0.01747 | 0.05664 | 3.653 |
| over_Xigma_h2_magnitude | 41446 | 0.3798 | 0.4751 | 0.009441 | 0.01521 | 2.214 |
| under_Mu_h2_magnitude | 41446 | -0.05675 | -0.08626 | -0.004465 | 0.0001822 | 0.8816 |
| under_Rho_h2_magnitude | 41446 | 0.6602 | 0.6992 | 0.01678 | 0.0335 | 3.334 |
| under_Xigma_h2_magnitude | 41446 | 0.4945 | 0.56 | 0.01216 | 0.03119 | 2.356 |
| under_minus_over_Mu_h2_magnitude | 41446 | 0.1138 | 0.05246 | -0.001693 | 0.000288 | 1.292 |
| under_minus_over_Rho_h2_magnitude | 41446 | 0.761 | 0.7909 | 0.03244 | 0.04608 | 4.006 |
| under_minus_over_Xigma_h2_magnitude | 41446 | 0.4322 | 0.5292 | 0.02119 | 0.02133 | 2.302 |

## Requested Features Not Present In This CSV

- inner radial-band m=2 harmonic
- middle radial-band m=2 harmonic
- outer radial-band m=2 harmonic
- r^4-weighted m=2 moment
- r^6-weighted m=2 moment
- contour/radius r50 angular anisotropy
- contour/radius r80 angular anisotropy
- contour/radius r95 angular anisotropy

## Interpretation

This audit is diagnostic only. If high-S3 model slope remains compressed after v5 data expansion, compare the high-S3 feature slopes here against new radial-band or radially weighted m=2 descriptors before expanding directly to 100k rows.
