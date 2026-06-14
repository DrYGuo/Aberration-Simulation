# S3 Feature Saturation Audit

- CSV: `/content/Aberration-Simulation/training_results/feature_regression_enhanced/enhanced_v5_s3_tail60k_20260614_201244_utc/training_features_enhanced.csv`
- Rows: 57446
- SHA256: `962496dba426a6fa08f3a82793c11a9b8f2f62a829946dde63fb8e85dd06347c`
- S3 vector scale: 90.7398
- High-S3 threshold: 63.5179
- High-S3 rows: 23931

## Available Current Features

| feature | n | Pearson | Spearman | high-S3 slope | high-S3 R2 | high/low mean ratio |
|---|---:|---:|---:|---:|---:|---:|
| Eq43_S3_value_magnitude | 57446 | 0.8016 | 0.8249 | 0.03805 | 0.06413 | 4.77 |
| over_Mu_h2_magnitude | 57446 | -0.2142 | -0.2021 | 0.0001856 | 5.989e-07 | 0.4363 |
| over_Rho_h2_magnitude | 57446 | 0.8118 | 0.8224 | 0.02002 | 0.07912 | 4.117 |
| over_Xigma_h2_magnitude | 57446 | 0.5156 | 0.5691 | 0.0114 | 0.02543 | 2.72 |
| under_Mu_h2_magnitude | 57446 | -0.21 | -0.2219 | -0.0002811 | 1.078e-06 | 0.4625 |
| under_Rho_h2_magnitude | 57446 | 0.7371 | 0.7538 | 0.01938 | 0.05517 | 3.828 |
| under_Xigma_h2_magnitude | 57446 | 0.5975 | 0.6371 | 0.01328 | 0.04684 | 2.584 |
| under_minus_over_Mu_h2_magnitude | 57446 | -0.06603 | -0.1121 | 7.501e-05 | 9.239e-07 | 0.6399 |
| under_minus_over_Rho_h2_magnitude | 57446 | 0.8016 | 0.8249 | 0.03805 | 0.06413 | 4.77 |
| under_minus_over_Xigma_h2_magnitude | 57446 | 0.5535 | 0.6116 | 0.02432 | 0.03321 | 2.686 |

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
