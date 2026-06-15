# Sampling Quality Report

Created UTC: `2026-06-15T00:53:20.926036+00:00`
Dataset: `training_results/feature_regression_enhanced/enhanced_v12_benchmark_v2_500k_eval_20260615_003057_utc/training_features_enhanced.csv`
Config: `configs/targeted_benchmark_v2_75k.json`

## Counts

- total rows: `575000`
- parent rows: `500000`
- new training-only rows: `75000`
- recommendation: **PASS_WITH_WARNINGS**

## Quota Check

| label | planned new | observed new | rel error | status |
|---|---:|---:|---:|---|
| `A1` | 0 | 0 | 0.0000 | PASS |
| `A1_B2_S3_grid` | 0 | 0 | 0.0000 | PASS |
| `A2` | 0 | 0 | 0.0000 | PASS |
| `A3` | 0 | 0 | 0.0000 | PASS |
| `A3_wide` | 0 | 0 | 0.0000 | PASS |
| `B2/C21` | 0 | 0 | 0.0000 | PASS |
| `C1` | 0 | 0 | 0.0000 | PASS |
| `C1_A1_C3_grid` | 0 | 0 | 0.0000 | PASS |
| `C3` | 0 | 0 | 0.0000 | PASS |
| `S3/C32` | 0 | 0 | 0.0000 | PASS |
| `S3_C32_wide` | 0 | 0 | 0.0000 | PASS |
| `S3_high_random` | 7000 | 7000 | 0.0000 | PASS |
| `coupled_A1_A2_B2_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_A1_B2_S3_random` | 7000 | 7000 | 0.0000 | PASS |
| `coupled_A1_B2_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_A1_S3_random` | 5000 | 5000 | 0.0000 | PASS |
| `coupled_A2_B2_random` | 6000 | 6000 | 0.0000 | PASS |
| `coupled_A3_S3_random` | 7000 | 7000 | 0.0000 | PASS |
| `coupled_B2_S3_random` | 7000 | 7000 | 0.0000 | PASS |
| `coupled_C1_A1_C3_A2_random` | 3000 | 3000 | 0.0000 | PASS |
| `coupled_C1_A1_C3_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_A1_S3_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_A1_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_A2_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_A3_S3_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_A3_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_B2_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_C3_A2_random` | 5000 | 5000 | 0.0000 | PASS |
| `coupled_C1_C3_S3_random` | 2000 | 2000 | 0.0000 | PASS |
| `coupled_C1_C3_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_S3_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C3_A3_S3_random` | 6000 | 6000 | 0.0000 | PASS |
| `coupled_C3_B2_random` | 2000 | 2000 | 0.0000 | PASS |
| `coupled_full_random` | 10000 | 10000 | 0.0000 | PASS |
| `coupled_sparse_random` | 8000 | 8000 | 0.0000 | PASS |

## Key Warnings

- Low new-row marginal entropy: new_training_only:C1

## Marginal Coverage Warnings

| subset | quantity | empty bins | entropy | max/median |
|---|---|---:|---:|---:|
| full | `A2_phase` | 0.000 | 0.502 | 43.76 |
| full | `A2_amp` | 0.000 | 0.505 | 43.30 |
| full | `A3_phase` | 0.000 | 0.595 | 30.99 |
| full | `A3_amp` | 0.000 | 0.604 | 30.00 |
| full | `C1` | 0.000 | 0.624 | 28.42 |
| full | `B2_phase` | 0.000 | 0.632 | 26.94 |
| full | `B2_amp` | 0.000 | 0.642 | 26.07 |
| full | `A1_phase` | 0.000 | 0.651 | 25.16 |
| full | `C3` | 0.000 | 0.656 | 24.69 |
| full | `A1_amp` | 0.000 | 0.660 | 24.31 |
| full | `S3_amp` | 0.000 | 0.827 | 21.82 |
| full | `S3_phase` | 0.000 | 0.899 | 7.73 |
| new_training_only | `C1` | 0.000 | 0.490 | 47.09 |
| new_training_only | `A3_phase` | 0.000 | 0.550 | 36.49 |
| new_training_only | `A3_amp` | 0.000 | 0.562 | 35.06 |
| new_training_only | `A2_phase` | 0.000 | 0.565 | 34.70 |
| new_training_only | `A2_amp` | 0.000 | 0.577 | 33.13 |
| new_training_only | `A1_phase` | 0.000 | 0.581 | 32.78 |
| new_training_only | `A1_amp` | 0.000 | 0.594 | 31.08 |
| new_training_only | `C3` | 0.000 | 0.634 | 26.81 |

## Pairwise Occupancy Warnings

| subset | pair | nonempty | entropy | max/median |
|---|---|---:|---:|---:|
| full | `S3_amp__A3_amp` | 1.000 | 0.706 | 263.76 |
| full | `S3_amp__B2_amp` | 1.000 | 0.732 | 204.04 |
| full | `S3_amp__A1_amp` | 1.000 | 0.741 | 195.46 |
| full | `S3_amp__C1` | 1.000 | 0.717 | 142.87 |
| full | `S3_phase__A3_phase` | 1.000 | 0.745 | 198.92 |
| full | `S3_phase__B2_phase` | 1.000 | 0.770 | 164.09 |
| full | `A3_amp__C3` | 1.000 | 0.629 | 395.87 |
| full | `B2_amp__A1_amp` | 1.000 | 0.648 | 349.41 |
| full | `C1__C3` | 1.000 | 0.631 | 365.95 |
| new_training_only | `S3_amp__A3_amp` | 1.000 | 0.670 | 305.66 |
| new_training_only | `S3_amp__B2_amp` | 1.000 | 0.747 | 175.72 |
| new_training_only | `S3_amp__A1_amp` | 1.000 | 0.693 | 260.18 |
| new_training_only | `S3_amp__C1` | 1.000 | 0.629 | 200.32 |
| new_training_only | `S3_phase__A3_phase` | 1.000 | 0.710 | 276.44 |
| new_training_only | `S3_phase__B2_phase` | 1.000 | 0.781 | 165.62 |
| new_training_only | `A3_amp__C3` | 1.000 | 0.597 | 459.93 |
| new_training_only | `B2_amp__A1_amp` | 1.000 | 0.647 | 369.28 |
| new_training_only | `C1__C3` | 1.000 | 0.537 | 500.39 |
| parent | `S3_amp__A3_amp` | 1.000 | 0.712 | 250.74 |
| parent | `S3_amp__B2_amp` | 1.000 | 0.730 | 202.83 |

## Relative-Angle Coverage

| pair | bin | count | fraction | mean dev | p95 dev | random entropy |
|---|---|---:|---:|---:|---:|---:|
| `A3_S3` | aligned | 3250 | 0.250 | 3.666862225241255 | 7.126136733203712 |  |
| `A3_S3` | orthogonal | 3250 | 0.250 | 3.77072706799341 | 7.15698977004989 |  |
| `A3_S3` | anti_aligned | 3250 | 0.250 | 3.7213112724594093 | 7.0991337695572865 |  |
| `A3_S3` | random | 3250 | 0.250 |  |  | 0.9992225968622633 |
| `B2_S3` | aligned | 3500 | 0.250 | 3.710886939283051 | 7.097049457593854 |  |
| `B2_S3` | orthogonal | 3500 | 0.250 | 3.760680128963058 | 7.124465633379898 |  |
| `B2_S3` | anti_aligned | 3500 | 0.250 | 3.725158718355399 | 7.021472413722852 |  |
| `B2_S3` | random | 3500 | 0.250 |  |  | 0.999582622225415 |
| `A1_S3` | aligned | 3000 | 0.250 | 3.747202221647468 | 7.114716858758074 |  |
| `A1_S3` | orthogonal | 3000 | 0.250 | 3.771263800548053 | 7.103331061563922 |  |
| `A1_S3` | anti_aligned | 3000 | 0.250 | 3.696311051146515 | 7.048162536828129 |  |
| `A1_S3` | random | 3000 | 0.250 |  |  | 0.998952494543148 |

## Nearest-Neighbor Coverage

| distribution | method | n | p5 | median | p95 | max |
|---|---|---:|---:|---:|---:|---:|
| `training_only_to_training_only_excluding_self` | sklearn | 75000 | 0.00924005564302206 | 0.15398266911506653 | 1.2118270039558412 | 1.596114158630371 |
| `parent_to_new_training_only` | sklearn | 500000 | 0.011523929797112942 | 0.21352753043174744 | 1.2368646740913392 | 1.6905447244644165 |
| `validation_to_v9_parent_train` | sklearn | 26977 | 0.004197763651609421 | 0.11444716900587082 | 0.9910881161689755 | 1.3242603540420532 |
| `validation_to_v11_full_train` | sklearn | 26977 | 0.004197763651609421 | 0.11444716900587082 | 0.9910881161689755 | 1.3242603540420532 |
| `blind_to_v9_parent_train` | sklearn | 27370 | 0.004380883881822228 | 0.12266122177243233 | 0.9841407448053359 | 1.3168811798095703 |
| `blind_to_v11_full_train` | sklearn | 27370 | 0.004380883881822228 | 0.12266122177243233 | 0.9841407448053359 | 1.3168811798095703 |
| `stress_to_v9_parent_train` | sklearn | 28097 | 0.004414715431630612 | 0.12782429158687592 | 1.0213456869125364 | 1.3494067192077637 |
| `stress_to_v11_full_train` | sklearn | 28097 | 0.004414715431630612 | 0.12782429158687592 | 1.0213456869125364 | 1.3494067192077637 |
