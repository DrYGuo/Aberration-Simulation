# S3 Feature Saturation Audit

- CSV: `/content/Aberration-Simulation/training_results/feature_regression_enhanced/enhanced_v5_s3_tail60k_20260612_050717_utc/training_features_enhanced.csv`
- Rows: 57446
- SHA256: `e073123a1fac3bca9dbfac0667128f01d89e4f7d336b4cf24fad12c58c43c98d`
- S3 vector scale: 90.7398
- High-S3 threshold: 63.5179
- High-S3 rows: 23929

## Available Current Features

| feature | n | Pearson | Spearman | high-S3 slope | high-S3 R2 | high/low mean ratio |
|---|---:|---:|---:|---:|---:|---:|
| Eq43_S3_value_magnitude | 57446 | 0.802 | 0.8264 | 0.03878 | 0.06608 | 4.78 |
| over_Mu_h2_magnitude | 57446 | -0.2141 | -0.2012 | -0.001494 | 3.879e-05 | 0.4341 |
| over_Rho_h2_magnitude | 57446 | 0.8109 | 0.8224 | 0.02039 | 0.08117 | 4.115 |
| over_Xigma_h2_magnitude | 57446 | 0.5151 | 0.5685 | 0.01195 | 0.02796 | 2.709 |
| under_Mu_h2_magnitude | 57446 | -0.211 | -0.2211 | -0.002581 | 9.168e-05 | 0.4602 |
| under_Rho_h2_magnitude | 57446 | 0.7409 | 0.7585 | 0.0193 | 0.0551 | 3.858 |
| under_Xigma_h2_magnitude | 57446 | 0.5951 | 0.6348 | 0.0138 | 0.05022 | 2.576 |
| under_minus_over_Mu_h2_magnitude | 57446 | -0.06799 | -0.111 | -0.0007486 | 9.352e-05 | 0.6391 |
| under_minus_over_Rho_h2_magnitude | 57446 | 0.802 | 0.8264 | 0.03878 | 0.06608 | 4.78 |
| under_minus_over_Xigma_h2_magnitude | 57446 | 0.5513 | 0.6092 | 0.02546 | 0.03614 | 2.675 |

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
