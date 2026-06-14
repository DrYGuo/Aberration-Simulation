# Sampling Quality Report

Created UTC: `2026-06-14T23:50:32.409466+00:00`
Dataset: `training_results/feature_regression_enhanced/enhanced_v11_gap500k_20260614_205607_utc/training_features_enhanced.csv`
Config: `configs/targeted_expansion_v11_500k.json`

## Counts

- total rows: `500000`
- parent rows: `16446`
- new training-only rows: `483554`
- recommendation: **FAIL**

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
| `S3_high_random` | 24000 | 33500 | 0.3958 | WARN |
| `coupled_A1_A2_B2_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_A1_B2_S3_random` | 24000 | 38000 | 0.5833 | WARN |
| `coupled_A1_B2_random` | 3000 | 9000 | 2.0000 | WARN |
| `coupled_A1_S3_random` | 16000 | 28500 | 0.7812 | WARN |
| `coupled_A2_B2_random` | 3000 | 8500 | 1.8333 | WARN |
| `coupled_A3_S3_random` | 24000 | 38500 | 0.6042 | WARN |
| `coupled_B2_S3_random` | 20000 | 29500 | 0.4750 | WARN |
| `coupled_C1_A1_C3_A2_random` | 0 | 10500 | 10500.0000 | WARN |
| `coupled_C1_A1_C3_random` | 0 | 5500 | 5500.0000 | WARN |
| `coupled_C1_A1_S3_random` | 8000 | 12000 | 0.5000 | WARN |
| `coupled_C1_A1_random` | 0 | 6000 | 6000.0000 | WARN |
| `coupled_C1_A2_random` | 0 | 5000 | 5000.0000 | WARN |
| `coupled_C1_A3_S3_random` | 6000 | 6000 | 0.0000 | PASS |
| `coupled_C1_A3_random` | 0 | 4000 | 4000.0000 | WARN |
| `coupled_C1_B2_random` | 0 | 2500 | 2500.0000 | WARN |
| `coupled_C1_C3_A2_random` | 0 | 9500 | 9500.0000 | WARN |
| `coupled_C1_C3_S3_random` | 7000 | 18000 | 1.5714 | WARN |
| `coupled_C1_C3_random` | 8000 | 18000 | 1.2500 | WARN |
| `coupled_C1_S3_random` | 8000 | 13500 | 0.6875 | WARN |
| `coupled_C3_A3_S3_random` | 22000 | 36500 | 0.6591 | WARN |
| `coupled_C3_B2_random` | 2000 | 7000 | 2.5000 | WARN |
| `coupled_full_random` | 40000 | 75554 | 0.8889 | WARN |
| `coupled_sparse_random` | 35000 | 68500 | 0.9571 | WARN |

## Key Warnings

- Regime quota mismatch >1% for: S3_high_random, coupled_A1_B2_S3_random, coupled_A1_B2_random, coupled_A1_S3_random, coupled_A2_B2_random, coupled_A3_S3_random, coupled_B2_S3_random, coupled_C1_A1_C3_A2_random, coupled_C1_A1_C3_random, coupled_C1_A1_S3_random, coupled_C1_A1_random, coupled_C1_A2_random, coupled_C1_A3_random, coupled_C1_B2_random, coupled_C1_C3_A2_random, coupled_C1_C3_S3_random, coupled_C1_C3_random, coupled_C1_S3_random, coupled_C3_A3_S3_random, coupled_C3_B2_random, coupled_full_random, coupled_sparse_random
- Low new-row marginal entropy: new_training_only:A2_amp, new_training_only:A2_phase

## Marginal Coverage Warnings

| subset | quantity | empty bins | entropy | max/median |
|---|---|---:|---:|---:|
| full | `A2_phase` | 0.000 | 0.492 | 45.39 |
| full | `A2_amp` | 0.000 | 0.493 | 45.18 |
| full | `A3_phase` | 0.000 | 0.602 | 30.19 |
| full | `A3_amp` | 0.000 | 0.610 | 29.33 |
| full | `B2_phase` | 0.000 | 0.625 | 27.69 |
| full | `B2_amp` | 0.000 | 0.633 | 26.92 |
| full | `C1` | 0.000 | 0.643 | 26.49 |
| full | `C3` | 0.000 | 0.660 | 24.40 |
| full | `A1_phase` | 0.000 | 0.661 | 24.26 |
| full | `A1_amp` | 0.000 | 0.670 | 23.43 |
| full | `S3_amp` | 0.000 | 0.830 | 21.01 |
| full | `S3_phase` | 0.000 | 0.901 | 7.65 |
| new_training_only | `A2_phase` | 0.000 | 0.475 | 48.57 |
| new_training_only | `A2_amp` | 0.000 | 0.481 | 47.46 |
| new_training_only | `A3_phase` | 0.000 | 0.598 | 30.65 |
| new_training_only | `A3_amp` | 0.000 | 0.609 | 29.45 |
| new_training_only | `B2_phase` | 0.000 | 0.618 | 28.39 |
| new_training_only | `B2_amp` | 0.000 | 0.630 | 27.20 |
| new_training_only | `C1` | 0.000 | 0.636 | 27.52 |
| new_training_only | `A1_phase` | 0.000 | 0.652 | 25.18 |

## Pairwise Occupancy Warnings

| subset | pair | nonempty | entropy | max/median |
|---|---|---:|---:|---:|
| full | `S3_amp__A3_amp` | 1.000 | 0.712 | 250.74 |
| full | `S3_amp__B2_amp` | 1.000 | 0.730 | 202.83 |
| full | `S3_amp__A1_amp` | 1.000 | 0.748 | 182.46 |
| full | `S3_amp__C1` | 1.000 | 0.729 | 132.78 |
| full | `S3_phase__A3_phase` | 1.000 | 0.750 | 190.56 |
| full | `S3_phase__B2_phase` | 1.000 | 0.767 | 165.74 |
| full | `A3_amp__C3` | 1.000 | 0.634 | 384.98 |
| full | `B2_amp__A1_amp` | 1.000 | 0.648 | 346.20 |
| full | `C1__C3` | 1.000 | 0.643 | 347.85 |
| new_training_only | `S3_amp__A3_amp` | 1.000 | 0.711 | 250.73 |
| new_training_only | `S3_amp__B2_amp` | 1.000 | 0.728 | 200.97 |
| new_training_only | `S3_amp__A1_amp` | 1.000 | 0.745 | 182.95 |
| new_training_only | `S3_amp__C1` | 1.000 | 0.725 | 133.14 |
| new_training_only | `S3_phase__A3_phase` | 1.000 | 0.750 | 191.84 |
| new_training_only | `S3_phase__B2_phase` | 1.000 | 0.766 | 165.89 |
| new_training_only | `A3_amp__C3` | 1.000 | 0.630 | 392.35 |
| new_training_only | `B2_amp__A1_amp` | 1.000 | 0.643 | 353.46 |
| new_training_only | `C1__C3` | 1.000 | 0.638 | 357.72 |
| parent | `S3_amp__C1` | 0.875 | 0.753 | 129.96 |
| parent | `C1__C3` | 0.875 | 0.758 | 114.59 |

## Relative-Angle Coverage

| pair | bin | count | fraction | mean dev | p95 dev | random entropy |
|---|---|---:|---:|---:|---:|---:|
| `A3_S3` | aligned | 13000 | 0.250 | 3.7292552607799947 | 7.1271384275683545 |  |
| `A3_S3` | orthogonal | 13000 | 0.250 | 3.7508289560224415 | 7.1392478775927914 |  |
| `A3_S3` | anti_aligned | 13000 | 0.250 | 3.7334500148501717 | 7.130372327948583 |  |
| `A3_S3` | random | 13000 | 0.250 |  |  | 0.999784161389619 |
| `B2_S3` | aligned | 11000 | 0.250 | 3.7546484909248368 | 7.109233423355459 |  |
| `B2_S3` | orthogonal | 11000 | 0.250 | 3.7301950344486396 | 7.121660716311283 |  |
| `B2_S3` | anti_aligned | 11000 | 0.250 | 3.7896439266842306 | 7.141002265010944 |  |
| `B2_S3` | random | 11000 | 0.250 |  |  | 0.9998001590534064 |
| `A1_S3` | aligned | 12000 | 0.250 | 3.7677115046169045 | 7.128298269023536 |  |
| `A1_S3` | orthogonal | 12000 | 0.250 | 3.734479177034992 | 7.1255136158760894 |  |
| `A1_S3` | anti_aligned | 12000 | 0.250 | 3.76309985689416 | 7.12430435092123 |  |
| `A1_S3` | random | 12000 | 0.250 |  |  | 0.9997694114749311 |

## Nearest-Neighbor Coverage

| distribution | method | n | p5 | median | p95 | max |
|---|---|---:|---:|---:|---:|---:|
| `training_only_to_training_only_excluding_self` | sklearn | 483554 | 0.003892135340720415 | 0.12224770337343216 | 1.0056780457496641 | 1.3473948240280151 |
| `parent_to_new_training_only` | sklearn | 16446 | 0.08047462813556194 | 0.5466833710670471 | 1.0925701558589935 | 1.3494067192077637 |
| `validation_to_v9_parent_train` | sklearn | 1977 | 0.14159666895866393 | 1.0168536901474 | 1.5131778478622435 | 1.8199361562728882 |
| `validation_to_v11_full_train` | sklearn | 1977 | 0.05865402892231941 | 0.7254406213760376 | 1.1095560789108276 | 1.3242603540420532 |
| `blind_to_v9_parent_train` | sklearn | 2370 | 0.24514900594949723 | 1.0163602828979492 | 1.4708006501197814 | 1.806520700454712 |
| `blind_to_v11_full_train` | sklearn | 2370 | 0.12268012613058091 | 0.6232757270336151 | 1.0497962892055512 | 1.2366487979888916 |
| `stress_to_v9_parent_train` | sklearn | 3097 | 0.49323295354843144 | 1.1400492191314697 | 1.5205440044403076 | 1.769875168800354 |
| `stress_to_v11_full_train` | sklearn | 3097 | 0.20003466308116913 | 0.6154933571815491 | 1.0975597381591797 | 1.3494067192077637 |
