# Active Failed-Region Error Report

Created UTC: `2026-06-18T19:40:50.668804+00:00`

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
| all_active_top_failures | C1 | nm | 8.425 | 10.93 | 6.649 | 20.81 | 106.5 |
| all_active_top_failures | C3 | mm | 0.02832 | 0.03662 | 0.02436 | 0.06536 | 0.4939 |
| all_active_top_failures | A1 vector | nm | 13.6 | 17.61 | 9.876 | 41.18 | 65.56 |
| all_active_top_failures | B2 vector | um | 0.5123 | 0.6695 | 0.3514 | 1.482 | 1.786 |
| all_active_top_failures | A2 vector | um | 0.5476 | 0.6527 | 0.4695 | 1.159 | 4.095 |
| all_active_top_failures | S3 vector | um | 32.14 | 35.68 | 29.12 | 65.21 | 135.6 |
| all_active_top_failures | A3 vector | um | 22.51 | 30.73 | 14.15 | 70.29 | 104.1 |
| all_active_top_failures | overall mixed-unit abs |  | 8.024 | 9.301 | 5.998 | 20.23 | 28.91 |
| all_active_top_failures | weighted normalized abs |  | 0.1226 | 0.1425 | 0.08969 | 0.3108 | 0.4264 |
| all_active_top_failures | 12D NN distance |  | 1.025 | 1.033 | 1.043 | 1.205 | 1.318 |
| coverage_limited_sparse_failure | C1 | nm | 8.955 | 11.59 | 7.102 | 21.8 | 106.5 |
| coverage_limited_sparse_failure | C3 | mm | 0.02853 | 0.03662 | 0.02483 | 0.06406 | 0.4939 |
| coverage_limited_sparse_failure | A1 vector | nm | 15.11 | 19.64 | 10.57 | 42.98 | 65.56 |
| coverage_limited_sparse_failure | B2 vector | um | 0.5652 | 0.7353 | 0.3812 | 1.539 | 1.786 |
| coverage_limited_sparse_failure | A2 vector | um | 0.5828 | 0.689 | 0.511 | 1.198 | 4.095 |
| coverage_limited_sparse_failure | S3 vector | um | 33.61 | 37.37 | 30.45 | 66.74 | 118 |
| coverage_limited_sparse_failure | A3 vector | um | 25.57 | 34.45 | 15.48 | 74.26 | 104.1 |
| coverage_limited_sparse_failure | overall mixed-unit abs |  | 8.725 | 10.21 | 6.16 | 21.44 | 28.91 |
| coverage_limited_sparse_failure | weighted normalized abs |  | 0.1337 | 0.1568 | 0.09248 | 0.3291 | 0.4264 |
| coverage_limited_sparse_failure | 12D NN distance |  | 1.094 | 1.096 | 1.087 | 1.221 | 1.318 |
| mixed_failure | C1 | nm | 7.691 | 9.545 | 6.526 | 19.17 | 26.42 |
| mixed_failure | C3 | mm | 0.02555 | 0.03198 | 0.02176 | 0.06372 | 0.1282 |
| mixed_failure | A1 vector | nm | 11.44 | 13.44 | 9.813 | 26.56 | 40.99 |
| mixed_failure | B2 vector | um | 0.475 | 0.5815 | 0.3786 | 1.132 | 1.434 |
| mixed_failure | A2 vector | um | 0.5231 | 0.6108 | 0.4619 | 1.126 | 2.031 |
| mixed_failure | S3 vector | um | 29.17 | 31.46 | 27.89 | 47.12 | 120.8 |
| mixed_failure | A3 vector | um | 19.46 | 24.62 | 14.37 | 52.19 | 82.26 |
| mixed_failure | overall mixed-unit abs |  | 6.984 | 7.498 | 6.108 | 12.55 | 21.06 |
| mixed_failure | weighted normalized abs |  | 0.1072 | 0.115 | 0.09184 | 0.1976 | 0.3156 |
| mixed_failure | 12D NN distance |  | 0.9465 | 0.9475 | 0.9466 | 1.022 | 1.046 |
| dense_model_feature_loss_failure | C1 | nm | 6.799 | 9.236 | 5.51 | 15.98 | 80.42 |
| dense_model_feature_loss_failure | C3 | mm | 0.03131 | 0.04246 | 0.02582 | 0.07866 | 0.3588 |
| dense_model_feature_loss_failure | A1 vector | nm | 9.064 | 10.65 | 7.624 | 20.22 | 37.7 |
| dense_model_feature_loss_failure | B2 vector | um | 0.2971 | 0.3719 | 0.2373 | 0.7533 | 1.513 |
| dense_model_feature_loss_failure | A2 vector | um | 0.4041 | 0.5049 | 0.3438 | 0.9035 | 3.385 |
| dense_model_feature_loss_failure | S3 vector | um | 28.99 | 32.51 | 26.89 | 54.68 | 135.6 |
| dense_model_feature_loss_failure | A3 vector | um | 11.43 | 14.28 | 9.086 | 28.47 | 52.09 |
| dense_model_feature_loss_failure | overall mixed-unit abs |  | 5.973 | 6.378 | 5.382 | 10.59 | 28.3 |
| dense_model_feature_loss_failure | weighted normalized abs |  | 0.08889 | 0.09427 | 0.07931 | 0.1542 | 0.3959 |
| dense_model_feature_loss_failure | 12D NN distance |  | 0.7888 | 0.7934 | 0.8044 | 0.8881 | 0.915 |

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

- top-failure rows analyzed: `3988`
- failure classes: `{'coverage_limited_sparse_failure': 2688, 'mixed_failure': 772, 'dense_model_feature_loss_failure': 528}`
- regimes: `{'coupled_full_random': 3966, 'coupled_sparse_random': 22}`
- A3-S3 angle categories: `{'aligned_or_anti_aligned': 1060, 'anti_aligned': 586, 'oblique': 1483, 'orthogonal': 859}`
- B2-S3 angle categories: `{'orthogonal': 1170, 'aligned_or_anti_aligned': 196, 'anti_aligned': 296, 'oblique': 2326}`

## Highest-Priority Physical Cluster Centers

| rank | n | class | median err | median NN | C1 nm | C3 mm | A1x nm | A1y nm | B2x um | B2y um | S3x um | S3y um | A3x um | A3y um |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 276 | coverage_limited_sparse_failure | 0.3243 | 1.108 | -97.65 | 0.08988 | -23.53 | 52.94 | -0.2578 | 2.908 | 42.01 | 89.78 | 34.09 | 91.9 |
| 2 | 2 | dense_model_feature_loss_failure | 0.2789 | 0.9262 | 3.615 | 0.9691 | 33.44 | 38.12 | -2.294 | 1.776 | 67.86 | 21.29 | -18.98 | 79.99 |
| 3 | 9 | coverage_limited_sparse_failure | 0.2613 | 1.216 | -90.38 | 0.1182 | -4.98 | 52.68 | -2.753 | -0.6498 | 48.84 | 74.27 | 45.01 | 68.44 |
| 4 | 82 | coverage_limited_sparse_failure | 0.2531 | 1.051 | -95.77 | 0.1426 | -18.96 | 54.25 | -1.322 | 2.555 | 43.05 | 89.24 | 42.58 | 88.59 |
| 5 | 5 | coverage_limited_sparse_failure | 0.1919 | 1.034 | -88.6 | 0.1267 | 21.71 | -53.94 | 2.566 | -0.4076 | -4.853 | -73.03 | -33.32 | -92.94 |
| 6 | 4 | coverage_limited_sparse_failure | 0.1872 | 1.106 | -77.08 | 0.1309 | 51.78 | -28.44 | 2.058 | 1.912 | 70.38 | -27.08 | -82.48 | -30.3 |
| 7 | 346 | coverage_limited_sparse_failure | 0.1796 | 0.9981 | -70.12 | 0.2552 | 49.67 | -26.06 | 2.416 | 1.712 | 49.62 | -83.48 | 94.63 | -9.216 |
| 8 | 147 | coverage_limited_sparse_failure | 0.1749 | 1.068 | -88.46 | 0.167 | 51.46 | -16.47 | 2.272 | 1.749 | 27.63 | -91.91 | 93.04 | -7.413 |
| 9 | 1 | mixed_failure | 0.1663 | 0.909 | 81.33 | 1.246 | -21.11 | 30.28 | 2.945 | 0.5732 | -88.58 | 21.17 | -22.98 | -93.05 |
| 10 | 7 | coverage_limited_sparse_failure | 0.1657 | 1.058 | -44.98 | 0.2668 | 7.194 | -12.6 | 1.64 | -1.694 | 9.42 | -84.3 | 2.926 | 84.62 |
| 11 | 8 | coverage_limited_sparse_failure | 0.1495 | 1.106 | 93.3 | 1.963 | 47.8 | 17.36 | 2.082 | 1.091 | 54.03 | 46.16 | -10.73 | -0.258 |
| 12 | 13 | coverage_limited_sparse_failure | 0.1386 | 1.093 | -73.64 | 0.2255 | 20.56 | 33.29 | -1.14 | 2.108 | 52.47 | 28.23 | 38.01 | -11.08 |

## Interpretation

- Failed-region errors are tens of times larger than the ordinary v13 validation/blind/stress errors for S3/A3/A1 vector targets.
- Coverage-limited sparse failures are the largest group and have the highest median NN distance.
- Dense/mixed failures are smaller but non-negligible; they should remain a separate diagnostic group after v15.
- v15 should not train directly on the diagnostic probes. Convert these subspaces into a balanced expansion with jitter, angle balancing, and bridge/anchor controls.
