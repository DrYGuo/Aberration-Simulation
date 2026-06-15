# Sampling Quality Report

Created UTC: `2026-06-15T00:05:10.580977+00:00`
Dataset: `training_results/feature_regression_enhanced/enhanced_v11_gap500k_20260614_205607_utc/training_features_enhanced.csv`
Config: `configs/targeted_expansion_v11_500k.json`

## Counts

- total rows: `500000`
- parent rows: `250000`
- new training-only rows: `250000`
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
| `S3_high_random` | 24000 | 24000 | 0.0000 | PASS |
| `coupled_A1_A2_B2_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_A1_B2_S3_random` | 24000 | 24000 | 0.0000 | PASS |
| `coupled_A1_B2_random` | 3000 | 3000 | 0.0000 | PASS |
| `coupled_A1_S3_random` | 16000 | 16000 | 0.0000 | PASS |
| `coupled_A2_B2_random` | 3000 | 3000 | 0.0000 | PASS |
| `coupled_A3_S3_random` | 24000 | 24000 | 0.0000 | PASS |
| `coupled_B2_S3_random` | 20000 | 20000 | 0.0000 | PASS |
| `coupled_C1_A1_C3_A2_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_A1_C3_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_A1_S3_random` | 8000 | 8000 | 0.0000 | PASS |
| `coupled_C1_A1_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_A2_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_A3_S3_random` | 6000 | 6000 | 0.0000 | PASS |
| `coupled_C1_A3_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_B2_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_C3_A2_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_C3_S3_random` | 7000 | 7000 | 0.0000 | PASS |
| `coupled_C1_C3_random` | 8000 | 8000 | 0.0000 | PASS |
| `coupled_C1_S3_random` | 8000 | 8000 | 0.0000 | PASS |
| `coupled_C3_A3_S3_random` | 22000 | 22000 | 0.0000 | PASS |
| `coupled_C3_B2_random` | 2000 | 2000 | 0.0000 | PASS |
| `coupled_full_random` | 40000 | 40000 | 0.0000 | PASS |
| `coupled_sparse_random` | 35000 | 35000 | 0.0000 | PASS |

## Key Warnings

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
| new_training_only | `A2_phase` | 0.000 | 0.407 | 63.52 |
| new_training_only | `A2_amp` | 0.000 | 0.409 | 62.82 |
| new_training_only | `C1` | 0.000 | 0.556 | 36.96 |
| new_training_only | `C3` | 0.000 | 0.591 | 31.49 |
| new_training_only | `A1_phase` | 0.000 | 0.631 | 27.09 |
| new_training_only | `B2_phase` | 0.000 | 0.635 | 26.64 |
| new_training_only | `A3_phase` | 0.000 | 0.636 | 26.61 |
| new_training_only | `A1_amp` | 0.000 | 0.644 | 25.89 |

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
| new_training_only | `S3_amp__A3_amp` | 1.000 | 0.744 | 124.84 |
| new_training_only | `S3_amp__B2_amp` | 1.000 | 0.744 | 92.95 |
| new_training_only | `S3_amp__A1_amp` | 1.000 | 0.743 | 111.47 |
| new_training_only | `S3_amp__C1` | 1.000 | 0.694 | 111.68 |
| new_training_only | `S3_phase__A3_phase` | 1.000 | 0.794 | 110.60 |
| new_training_only | `S3_phase__B2_phase` | 1.000 | 0.797 | 92.93 |
| new_training_only | `A3_amp__C3` | 1.000 | 0.611 | 398.89 |
| new_training_only | `B2_amp__A1_amp` | 1.000 | 0.637 | 349.48 |
| new_training_only | `C1__C3` | 1.000 | 0.567 | 515.54 |
| parent | `S3_amp__A3_amp` | 1.000 | 0.663 | 353.57 |
| parent | `S3_amp__B2_amp` | 1.000 | 0.698 | 303.30 |

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
| `training_only_to_training_only_excluding_self` | sklearn | 250000 | 0.004209400154650211 | 0.11965672299265862 | 1.0722783982753754 | 1.420972228050232 |
| `parent_to_new_training_only` | sklearn | 250000 | 0.006814606720581652 | 0.2247498854994774 | 1.0802785336971283 | 1.4526394605636597 |
| `validation_to_v9_parent_train` | sklearn | 1977 | 0.06675260663032533 | 0.770271897315979 | 1.1842637062072754 | 1.3798184394836426 |
| `validation_to_v11_full_train` | sklearn | 1977 | 0.05865402892231941 | 0.7254406213760376 | 1.1095560789108276 | 1.3242603540420532 |
| `blind_to_v9_parent_train` | sklearn | 2370 | 0.1247179675847292 | 0.6898728609085083 | 1.1107234299182889 | 1.4485557079315186 |
| `blind_to_v11_full_train` | sklearn | 2370 | 0.12268012613058091 | 0.6232757270336151 | 1.0497962892055512 | 1.2366487979888916 |
| `stress_to_v9_parent_train` | sklearn | 3097 | 0.2146588146686554 | 0.6528878808021545 | 1.1764578580856322 | 1.4153152704238892 |
| `stress_to_v11_full_train` | sklearn | 3097 | 0.20003466308116913 | 0.6154933571815491 | 1.0975597381591797 | 1.3494067192077637 |
