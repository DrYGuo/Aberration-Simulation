# Sampling Quality Report

Created UTC: `2026-06-15T04:27:20.681835+00:00`
Dataset: `training_results/feature_regression_enhanced/enhanced_v13_1m_spacefill_20260615_021848_utc/training_features_enhanced.csv`
Config: `configs/targeted_expansion_v13_1m.json`

## Counts

- total rows: `1075000`
- parent rows: `575000`
- new training-only rows: `500000`
- recommendation: **PASS**

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
| `S3_high_random` | 35000 | 35000 | 0.0000 | PASS |
| `coupled_A1_A2_B2_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_A1_B2_S3_random` | 35000 | 35000 | 0.0000 | PASS |
| `coupled_A1_B2_random` | 5000 | 5000 | 0.0000 | PASS |
| `coupled_A1_S3_random` | 20000 | 20000 | 0.0000 | PASS |
| `coupled_A2_B2_random` | 6000 | 6000 | 0.0000 | PASS |
| `coupled_A3_S3_random` | 35000 | 35000 | 0.0000 | PASS |
| `coupled_B2_S3_random` | 30000 | 30000 | 0.0000 | PASS |
| `coupled_C1_A1_C3_A2_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_A1_C3_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_A1_S3_random` | 8000 | 8000 | 0.0000 | PASS |
| `coupled_C1_A1_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_A2_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_A3_S3_random` | 8000 | 8000 | 0.0000 | PASS |
| `coupled_C1_A3_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_B2_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_C3_A2_random` | 10000 | 10000 | 0.0000 | PASS |
| `coupled_C1_C3_S3_random` | 8000 | 8000 | 0.0000 | PASS |
| `coupled_C1_C3_random` | 15000 | 15000 | 0.0000 | PASS |
| `coupled_C1_S3_random` | 10000 | 10000 | 0.0000 | PASS |
| `coupled_C3_A3_S3_random` | 30000 | 30000 | 0.0000 | PASS |
| `coupled_C3_B2_random` | 5000 | 5000 | 0.0000 | PASS |
| `coupled_full_random` | 160000 | 160000 | 0.0000 | PASS |
| `coupled_sparse_random` | 80000 | 80000 | 0.0000 | PASS |

## Key Warnings

- None.

## Marginal Coverage Warnings

| subset | quantity | empty bins | entropy | max/median |
|---|---|---:|---:|---:|
| full | `A2_phase` | 0.000 | 0.563 | 34.84 |
| full | `A2_amp` | 0.000 | 0.570 | 34.01 |
| full | `C1` | 0.000 | 0.663 | 24.51 |
| full | `A3_phase` | 0.000 | 0.667 | 23.74 |
| full | `A3_amp` | 0.000 | 0.678 | 22.70 |
| full | `A1_phase` | 0.000 | 0.689 | 21.70 |
| full | `B2_phase` | 0.000 | 0.692 | 21.48 |
| full | `C3` | 0.000 | 0.699 | 20.92 |
| full | `A1_amp` | 0.000 | 0.701 | 20.72 |
| full | `B2_amp` | 0.000 | 0.704 | 20.48 |
| full | `S3_amp` | 0.000 | 0.869 | 13.92 |
| full | `S3_phase` | 0.000 | 0.925 | 6.32 |
| new_training_only | `A2_phase` | 0.000 | 0.629 | 27.29 |
| new_training_only | `A2_amp` | 0.000 | 0.640 | 26.24 |
| new_training_only | `C1` | 0.000 | 0.705 | 20.76 |
| new_training_only | `A1_phase` | 0.000 | 0.732 | 18.35 |
| new_training_only | `A3_phase` | 0.000 | 0.741 | 17.61 |
| new_training_only | `C3` | 0.000 | 0.745 | 17.34 |
| new_training_only | `A1_amp` | 0.000 | 0.746 | 17.26 |
| new_training_only | `B2_phase` | 0.000 | 0.755 | 16.58 |

## Pairwise Occupancy Warnings

| subset | pair | nonempty | entropy | max/median |
|---|---|---:|---:|---:|
| full | `S3_amp__A3_amp` | 1.000 | 0.761 | 155.19 |
| full | `S3_amp__B2_amp` | 1.000 | 0.780 | 118.90 |
| full | `S3_amp__A1_amp` | 1.000 | 0.778 | 125.32 |
| full | `S3_amp__C1` | 1.000 | 0.755 | 91.51 |
| full | `S3_phase__A3_phase` | 1.000 | 0.793 | 135.98 |
| full | `S3_phase__B2_phase` | 1.000 | 0.811 | 111.44 |
| full | `A3_amp__C3` | 1.000 | 0.682 | 281.36 |
| full | `B2_amp__A1_amp` | 1.000 | 0.693 | 259.58 |
| full | `C1__C3` | 1.000 | 0.668 | 292.22 |
| new_training_only | `S3_amp__A3_amp` | 1.000 | 0.818 | 85.19 |
| new_training_only | `S3_amp__B2_amp` | 1.000 | 0.829 | 64.16 |
| new_training_only | `S3_amp__A1_amp` | 1.000 | 0.815 | 78.45 |
| new_training_only | `S3_amp__C1` | 1.000 | 0.793 | 57.03 |
| new_training_only | `S3_phase__A3_phase` | 1.000 | 0.844 | 85.35 |
| new_training_only | `S3_phase__B2_phase` | 1.000 | 0.855 | 70.35 |
| new_training_only | `A3_amp__C3` | 1.000 | 0.737 | 198.76 |
| new_training_only | `B2_amp__A1_amp` | 1.000 | 0.738 | 191.83 |
| new_training_only | `C1__C3` | 1.000 | 0.707 | 229.85 |
| parent | `S3_amp__A3_amp` | 1.000 | 0.706 | 263.76 |
| parent | `S3_amp__B2_amp` | 1.000 | 0.732 | 204.04 |

## Relative-Angle Coverage

| pair | bin | count | fraction | mean dev | p95 dev | random entropy |
|---|---|---:|---:|---:|---:|---:|
| `A3_S3` | aligned | 18250 | 0.250 | 3.7650067621927663 | 7.131979949110794 |  |
| `A3_S3` | orthogonal | 18250 | 0.250 | 3.7341019151421655 | 7.110508116000079 |  |
| `A3_S3` | anti_aligned | 18250 | 0.250 | 3.7458715938260374 | 7.124792577675415 |  |
| `A3_S3` | random | 18250 | 0.250 |  |  | 0.9997753738382774 |
| `B2_S3` | aligned | 16250 | 0.250 | 3.739582175600178 | 7.133187486327144 |  |
| `B2_S3` | orthogonal | 16250 | 0.250 | 3.738439740203538 | 7.133380969695945 |  |
| `B2_S3` | anti_aligned | 16250 | 0.250 | 3.704380957299819 | 7.1064590711303275 |  |
| `B2_S3` | random | 16250 | 0.250 |  |  | 0.9999271196245225 |
| `A1_S3` | aligned | 15750 | 0.250 | 3.753235355420422 | 7.116754574138502 |  |
| `A1_S3` | orthogonal | 15750 | 0.250 | 3.735009336345479 | 7.131086234486056 |  |
| `A1_S3` | anti_aligned | 15750 | 0.250 | 3.70628264267297 | 7.105076467168848 |  |
| `A1_S3` | random | 15750 | 0.250 |  |  | 0.999909670195823 |

## Nearest-Neighbor Coverage

| distribution | method | n | p5 | median | p95 | max |
|---|---|---:|---:|---:|---:|---:|
| `training_only_to_training_only_excluding_self` | sklearn | 500000 | 0.00404923006426543 | 0.1799224466085434 | 0.985162740945816 | 1.3023484945297241 |
| `parent_to_new_training_only` | sklearn | 575000 | 0.004000348784029484 | 0.13812056928873062 | 0.9444552004337309 | 1.2923314571380615 |
| `validation_to_v9_parent_train` | sklearn | 26977 | 0.004197763651609421 | 0.11444716900587082 | 0.9910881161689755 | 1.3242603540420532 |
| `validation_to_v11_full_train` | sklearn | 26977 | 0.0028726138174533845 | 0.09783343225717545 | 0.8923070549964904 | 1.1624983549118042 |
| `blind_to_v9_parent_train` | sklearn | 27370 | 0.004380883881822228 | 0.12266122177243233 | 0.9841407448053359 | 1.3168811798095703 |
| `blind_to_v11_full_train` | sklearn | 27370 | 0.0030496023944579067 | 0.10551239550113678 | 0.8895733922719955 | 1.1242228746414185 |
| `stress_to_v9_parent_train` | sklearn | 28097 | 0.004414715431630612 | 0.12782429158687592 | 1.0213456869125364 | 1.3494067192077637 |
| `stress_to_v11_full_train` | sklearn | 28097 | 0.003122456232085824 | 0.10983459651470184 | 0.9230253934860226 | 1.220363736152649 |
