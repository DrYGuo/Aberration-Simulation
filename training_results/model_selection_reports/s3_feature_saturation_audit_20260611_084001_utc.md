# S3 Feature Saturation Audit

- CSV: `/content/Aberration-Simulation/training_results/feature_regression_enhanced/enhanced_v3_targeted25k_20260611_072105_utc/training_features_enhanced.csv`
- Rows: 41446
- SHA256: `64fbc87578287c5929311081151240bfbc556b67a0b28b36b3b69202437e9207`
- S3 vector scale: 90.7398
- High-S3 threshold: 63.5179
- High-S3 rows: 7935

## Available Current Features

| feature | n | Pearson | Spearman | high-S3 slope | high-S3 R2 | high/low mean ratio |
|---|---:|---:|---:|---:|---:|---:|
| Eq43_S3_value_magnitude | 41446 | 0.7608 | 0.7927 | 0.03238 | 0.04562 | 3.998 |
| over_Mu_h2_magnitude | 41446 | -0.08365 | -0.0786 | -0.003459 | 0.000135 | 0.8155 |
| over_Rho_h2_magnitude | 41446 | 0.7672 | 0.7649 | 0.01736 | 0.05485 | 3.635 |
| over_Xigma_h2_magnitude | 41446 | 0.3819 | 0.4741 | 0.009446 | 0.01462 | 2.236 |
| under_Mu_h2_magnitude | 41446 | -0.05648 | -0.08552 | -0.005832 | 0.0003076 | 0.8857 |
| under_Rho_h2_magnitude | 41446 | 0.6639 | 0.705 | 0.01626 | 0.03182 | 3.341 |
| under_Xigma_h2_magnitude | 41446 | 0.4929 | 0.556 | 0.01236 | 0.03089 | 2.365 |
| under_minus_over_Mu_h2_magnitude | 41446 | 0.1133 | 0.05364 | -0.001259 | 0.0001587 | 1.295 |
| under_minus_over_Rho_h2_magnitude | 41446 | 0.7608 | 0.7927 | 0.03238 | 0.04562 | 3.998 |
| under_minus_over_Xigma_h2_magnitude | 41446 | 0.431 | 0.5253 | 0.02133 | 0.0206 | 2.313 |

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
