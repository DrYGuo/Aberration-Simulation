# Active Failed-Region Error Report

Created UTC: `2026-06-18T04:32:12.187398+00:00`

## Purpose

Quantify the coefficient uncertainty inside active-search failed subspaces and compare it with the v13 1M benchmark splits.

Important: active vector errors are Euclidean vector errors on deliberately selected failed probes. Benchmark vector rows use component-pair norm approximations and validation vector-magnitude diagnostics, so they are directionally comparable but not identical estimators.

## v13 Benchmark Context

- run: `D66_grouped_width320_lr6e-4_dropout0.075_v13_1m_d66_seed23_residual_nn_20260615_065556_utc`
- total rows: `1075000`
- train/validation/blind/stress: `992556` / `26977` / `27370` / `28097`

## Active Failed Regions: Local Error Levels

| group | metric | unit | MAE/mean | RMSE | median | p95 | max |
|---|---|---:|---:|---:|---:|---:|---:|
| all_active_top_failures | C1 | nm | 7.825 | 10.36 | 6.017 | 19.3 | 80.42 |
| all_active_top_failures | C3 | mm | 0.02633 | 0.03444 | 0.02145 | 0.06189 | 0.3588 |
| all_active_top_failures | A1 vector | nm | 17.41 | 22.12 | 12.3 | 44.31 | 65.56 |
| all_active_top_failures | B2 vector | um | 0.5579 | 0.7297 | 0.385 | 1.571 | 1.786 |
| all_active_top_failures | A2 vector | um | 0.5539 | 0.662 | 0.4684 | 1.228 | 3.385 |
| all_active_top_failures | S3 vector | um | 36.83 | 41.19 | 33.54 | 68.23 | 135.6 |
| all_active_top_failures | A3 vector | um | 24.75 | 33.82 | 16.21 | 75.13 | 104.1 |
| all_active_top_failures | overall mixed-unit abs |  | 9.336 | 11.06 | 6.294 | 21.99 | 28.91 |
| all_active_top_failures | weighted normalized abs |  | 0.1423 | 0.1691 | 0.09466 | 0.3424 | 0.4264 |
| all_active_top_failures | 12D NN distance |  | 1.048 | 1.055 | 1.072 | 1.203 | 1.318 |
| coverage_limited_sparse_failure | C1 | nm | 8.181 | 10.61 | 6.413 | 19.77 | 41.47 |
| coverage_limited_sparse_failure | C3 | mm | 0.02565 | 0.03184 | 0.02138 | 0.06035 | 0.1278 |
| coverage_limited_sparse_failure | A1 vector | nm | 19.41 | 24.41 | 13.37 | 45.07 | 65.56 |
| coverage_limited_sparse_failure | B2 vector | um | 0.623 | 0.8055 | 0.4381 | 1.597 | 1.786 |
| coverage_limited_sparse_failure | A2 vector | um | 0.5955 | 0.7002 | 0.511 | 1.237 | 2.901 |
| coverage_limited_sparse_failure | S3 vector | um | 39.39 | 43.73 | 39.01 | 68.87 | 107.5 |
| coverage_limited_sparse_failure | A3 vector | um | 28.12 | 37.67 | 18.31 | 77.35 | 104.1 |
| coverage_limited_sparse_failure | overall mixed-unit abs |  | 10.26 | 12.12 | 6.576 | 22.42 | 28.91 |
| coverage_limited_sparse_failure | weighted normalized abs |  | 0.1566 | 0.1857 | 0.09806 | 0.3472 | 0.4264 |
| coverage_limited_sparse_failure | 12D NN distance |  | 1.101 | 1.103 | 1.1 | 1.215 | 1.318 |
| mixed_failure | C1 | nm | 5.988 | 7.642 | 5.2 | 16.47 | 24.1 |
| mixed_failure | C3 | mm | 0.02322 | 0.02955 | 0.01909 | 0.056 | 0.1126 |
| mixed_failure | A1 vector | nm | 12.01 | 13.79 | 11.24 | 26.08 | 40.12 |
| mixed_failure | B2 vector | um | 0.3827 | 0.4465 | 0.3437 | 0.7819 | 1.319 |
| mixed_failure | A2 vector | um | 0.422 | 0.4946 | 0.4 | 0.845 | 1.799 |
| mixed_failure | S3 vector | um | 28.08 | 30.41 | 25.1 | 47.95 | 82.42 |
| mixed_failure | A3 vector | um | 16.01 | 19.81 | 12.91 | 41.12 | 82.26 |
| mixed_failure | overall mixed-unit abs |  | 6.553 | 6.965 | 5.781 | 11.57 | 21.06 |
| mixed_failure | weighted normalized abs |  | 0.09974 | 0.1055 | 0.08726 | 0.1753 | 0.3156 |
| mixed_failure | 12D NN distance |  | 0.9546 | 0.9566 | 0.9643 | 1.039 | 1.046 |
| dense_model_feature_loss_failure | C1 | nm | 8.154 | 12.28 | 5.704 | 17.79 | 80.42 |
| dense_model_feature_loss_failure | C3 | mm | 0.03706 | 0.0556 | 0.02869 | 0.09953 | 0.3588 |
| dense_model_feature_loss_failure | A1 vector | nm | 10.81 | 12.47 | 9.514 | 21.84 | 37.7 |
| dense_model_feature_loss_failure | B2 vector | um | 0.3424 | 0.4269 | 0.2745 | 0.8042 | 1.501 |
| dense_model_feature_loss_failure | A2 vector | um | 0.4505 | 0.5978 | 0.375 | 1.019 | 3.385 |
| dense_model_feature_loss_failure | S3 vector | um | 31.52 | 36.2 | 28.79 | 57.91 | 135.6 |
| dense_model_feature_loss_failure | A3 vector | um | 12.98 | 15.95 | 11.09 | 28.47 | 49.41 |
| dense_model_feature_loss_failure | overall mixed-unit abs |  | 6.77 | 7.381 | 5.791 | 11.22 | 28.3 |
| dense_model_feature_loss_failure | weighted normalized abs |  | 0.1011 | 0.1092 | 0.08671 | 0.1617 | 0.3959 |
| dense_model_feature_loss_failure | 12D NN distance |  | 0.7879 | 0.7936 | 0.7993 | 0.906 | 0.915 |

## v13 Benchmark Splits

| split | metric | unit | MAE | RMSE | p95 |
|---|---|---:|---:|---:|---:|
| validation | overall mixed-unit abs |  | 0.3289 | 1.071 | 1.577 |
| validation | overall normalized abs |  | 0.005084 |  | 0.02147 |
| validation | C1 | nm | 0.7277 | 1.515 | 2.804 |
| validation | C3 | mm | 0.004476 | 0.008508 | 0.01638 |
| validation | A1 component-pair norm | nm | 0.3845 | 0.8028 | 1.564 |
| validation | B2 component-pair norm | um | 0.02511 | 0.05214 | 0.1152 |
| validation | A2 component-pair norm | um | 0.05712 | 0.1138 | 0.2698 |
| validation | S3 component-pair norm | um | 1.197 | 3.036 | 5.019 |
| validation | A3 component-pair norm | um | 0.6094 | 1.26 | 2.654 |
| blind | overall mixed-unit abs |  | 0.3552 | 1.062 | 1.682 |
| blind | overall normalized abs |  | 0.005513 |  | 0.02311 |
| blind | C1 | nm | 0.7035 | 1.341 | 2.853 |
| blind | C3 | mm | 0.004485 | 0.007979 | 0.01739 |
| blind | A1 component-pair norm | nm | 0.4488 | 0.9061 | 1.772 |
| blind | B2 component-pair norm | um | 0.02729 | 0.05521 | 0.1214 |
| blind | A2 component-pair norm | um | 0.06229 | 0.123 | 0.2835 |
| blind | S3 component-pair norm | um | 1.296 | 3.001 | 5.499 |
| blind | A3 component-pair norm | um | 0.6788 | 1.376 | 2.927 |
| stress | overall mixed-unit abs |  | 0.3739 | 1.131 | 1.768 |
| stress | overall normalized abs |  | 0.005654 |  | 0.02413 |
| stress | C1 | nm | 0.6655 | 1.401 | 2.555 |
| stress | C3 | mm | 0.004332 | 0.008294 | 0.0162 |
| stress | A1 component-pair norm | nm | 0.4578 | 0.9096 | 1.827 |
| stress | B2 component-pair norm | um | 0.02776 | 0.05893 | 0.1281 |
| stress | A2 component-pair norm | um | 0.05187 | 0.1058 | 0.2241 |
| stress | S3 component-pair norm | um | 1.408 | 3.204 | 6.035 |
| stress | A3 component-pair norm | um | 0.7538 | 1.512 | 3.283 |

## Validation Vector Magnitude/Angle Diagnostics

| vector | unit | magnitude MAE | magnitude RMSE | bias | slope | R2 | mean angle err deg | p95 angle err deg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 | nm | 0.3498 | 0.6457 | 0.1597 | 0.9868 | 0.9971 | 1.24 | 4.896 |
| B2 | um | 0.02138 | 0.04175 | 0.002677 | 0.9922 | 0.9979 | 1.239 | 4.624 |
| A2 | um | 0.05424 | 0.1034 | -0.0008348 | 0.9935 | 0.9996 | 0.4693 | 1.435 |
| S3 | um | 1.063 | 2.492 | 0.01549 | 0.9809 | 0.9959 | 0.9188 | 4.3 |
| A3 | um | 0.5508 | 1.037 | 0.1065 | 0.9859 | 0.9984 | 1.054 | 4.013 |

## Failed-Region Counts

- top-failure rows analyzed: `1619`
- failure classes: `{'coverage_limited_sparse_failure': 1206, 'mixed_failure': 261, 'dense_model_feature_loss_failure': 152}`
- regimes: `{'coupled_full_random': 1603, 'coupled_sparse_random': 16}`
- A3-S3 angle categories: `{'aligned_or_anti_aligned': 650, 'anti_aligned': 197, 'oblique': 391, 'orthogonal': 381}`
- B2-S3 angle categories: `{'orthogonal': 253, 'aligned_or_anti_aligned': 97, 'anti_aligned': 221, 'oblique': 1048}`

## Highest-Priority Physical Cluster Centers

| rank | n | class | median err | median NN | C1 nm | C3 mm | A1x nm | A1y nm | B2x um | B2y um | S3x um | S3y um | A3x um | A3y um |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 276 | coverage_limited_sparse_failure | 0.3243 | 1.108 | -97.65 | 0.08988 | -23.53 | 52.94 | -0.2578 | 2.908 | 42.01 | 89.78 | 34.09 | 91.9 |
| 2 | 2 | dense_model_feature_loss_failure | 0.2789 | 0.9262 | 3.615 | 0.9691 | 33.44 | 38.12 | -2.294 | 1.776 | 67.86 | 21.29 | -18.98 | 79.99 |
| 3 | 9 | coverage_limited_sparse_failure | 0.2613 | 1.216 | -90.38 | 0.1182 | -4.98 | 52.68 | -2.753 | -0.6498 | 48.84 | 74.27 | 45.01 | 68.44 |
| 4 | 82 | coverage_limited_sparse_failure | 0.2531 | 1.051 | -95.77 | 0.1426 | -18.96 | 54.25 | -1.322 | 2.555 | 43.05 | 89.24 | 42.58 | 88.59 |
| 5 | 5 | coverage_limited_sparse_failure | 0.1919 | 1.034 | -88.6 | 0.1267 | 21.71 | -53.94 | 2.566 | -0.4076 | -4.853 | -73.03 | -33.32 | -92.94 |
| 6 | 4 | coverage_limited_sparse_failure | 0.1872 | 1.106 | -77.08 | 0.1309 | 51.78 | -28.44 | 2.058 | 1.912 | 70.38 | -27.08 | -82.48 | -30.3 |
| 7 | 1 | mixed_failure | 0.1663 | 0.909 | 81.33 | 1.246 | -21.11 | 30.28 | 2.945 | 0.5732 | -88.58 | 21.17 | -22.98 | -93.05 |
| 8 | 31 | coverage_limited_sparse_failure | 0.1076 | 1.029 | 87.71 | 1.775 | 18.17 | 50.94 | -1.327 | 1.307 | 74.18 | 38.42 | 14.26 | 62.35 |
| 9 | 57 | coverage_limited_sparse_failure | 0.1027 | 1.041 | -86.52 | 0.8279 | 38.34 | 16.39 | 1.168 | 1.91 | 75.48 | 29.15 | 10.53 | 75.09 |
| 10 | 43 | coverage_limited_sparse_failure | 0.09774 | 1.019 | -76.25 | 0.4557 | -28.5 | 8.219 | -1.641 | -0.3147 | -35.69 | 40.68 | -2.046 | 71.79 |
| 11 | 43 | mixed_failure | 0.09729 | 1.028 | -75.22 | 0.1744 | -31.46 | 22.41 | -2.279 | 0.2228 | 25.76 | 72.95 | 1.247 | 37.55 |
| 12 | 39 | coverage_limited_sparse_failure | 0.09621 | 1.02 | -77.53 | 0.4275 | -32.89 | 16.14 | -1.453 | -0.6315 | -2.955 | 79.49 | 0.838 | -73.27 |

## Interpretation

- Failed-region errors are tens of times larger than the ordinary v13 validation/blind/stress errors for S3/A3/A1 vector targets.
- Coverage-limited sparse failures are the largest group and have the highest median NN distance.
- Dense/mixed failures are smaller but non-negligible; they should remain a separate diagnostic group after v15.
- v15 should not train directly on the diagnostic probes. Convert these subspaces into a balanced expansion with jitter, angle balancing, and bridge/anchor controls.
