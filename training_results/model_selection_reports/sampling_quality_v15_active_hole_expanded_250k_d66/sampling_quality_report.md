# Sampling Quality Report

Created UTC: `2026-06-19T07:58:33.539522+00:00`
Dataset: `training_results/feature_regression_enhanced/enhanced_v15_active_hole_expanded_250k_20260619_050058_utc/training_features_enhanced.csv`
Config: `colab_worker_logs/targeted_expansion_v15_active_hole_expanded_250k_runtime.json`

## Counts

- total rows: `1325000`
- parent rows: `1075000`
- new training-only rows: `250000`
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
| `S3_high_random` | 8000 | 8000 | 0.0000 | PASS |
| `active_failed_subspace_jitter` | 90000 | 90000 | 0.0000 | PASS |
| `coupled_A1_A2_B2_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_A1_B2_S3_random` | 25000 | 25000 | 0.0000 | PASS |
| `coupled_A1_B2_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_A1_S3_random` | 4000 | 4000 | 0.0000 | PASS |
| `coupled_A2_B2_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_A3_S3_random` | 22000 | 22000 | 0.0000 | PASS |
| `coupled_B2_S3_random` | 22000 | 22000 | 0.0000 | PASS |
| `coupled_C1_A1_C3_A2_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_A1_C3_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_A1_S3_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_A1_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_A2_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_A3_S3_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_A3_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_B2_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_C3_A2_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_C3_S3_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_C3_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_S3_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C3_A3_S3_random` | 16000 | 16000 | 0.0000 | PASS |
| `coupled_C3_B2_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_full_random` | 45000 | 45000 | 0.0000 | PASS |
| `coupled_sparse_random` | 18000 | 18000 | 0.0000 | PASS |

## Candidate Selection

| role | rows | fraction | median NN distance | p95 NN distance | max NN distance |
|---|---:|---:|---:|---:|---:|
| `bridge_anchor` | 62500 | 0.250 | 0.5579805374145508 | 0.6991774380207061 | 0.7247123718261719 |
| `far_nn` | 187500 | 0.750 | 0.7265536785125732 | 1.246124655008316 | 1.4169307947158813 |

## Key Warnings

- None.

## Marginal Coverage Warnings

| subset | quantity | empty bins | entropy | max/median |
|---|---|---:|---:|---:|
| full | `A2_phase` | 0.000 | 0.606 | 31.58 |
| full | `A2_amp` | 0.000 | 0.613 | 31.39 |
| full | `C1` | 0.000 | 0.678 | 26.10 |
| full | `A3_phase` | 0.000 | 0.712 | 21.15 |
| full | `C3` | 0.000 | 0.714 | 20.72 |
| full | `A1_phase` | 0.000 | 0.724 | 20.00 |
| full | `A3_amp` | 0.000 | 0.727 | 19.88 |
| full | `A1_amp` | 0.000 | 0.738 | 17.99 |
| full | `B2_phase` | 0.000 | 0.741 | 18.48 |
| full | `B2_amp` | 0.000 | 0.753 | 17.04 |
| full | `S3_amp` | 0.000 | 0.864 | 12.92 |
| full | `S3_phase` | 0.000 | 0.942 | 5.49 |
| new_training_only | `C1` | 0.000 | 0.662 | 41.42 |
| new_training_only | `A2_amp` | 0.000 | 0.735 | 20.77 |
| new_training_only | `A2_phase` | 0.000 | 0.744 | 20.07 |
| new_training_only | `C3` | 0.000 | 0.756 | 20.07 |
| new_training_only | `S3_amp` | 0.000 | 0.761 | 25.15 |
| new_training_only | `A1_phase` | 0.000 | 0.833 | 13.05 |
| new_training_only | `A1_amp` | 0.000 | 0.843 | 9.23 |
| new_training_only | `A3_phase` | 0.000 | 0.849 | 11.52 |

## Pairwise Occupancy Warnings

| subset | pair | nonempty | entropy | max/median |
|---|---|---:|---:|---:|
| full | `S3_amp__A3_amp` | 1.000 | 0.783 | 139.07 |
| full | `S3_amp__B2_amp` | 1.000 | 0.802 | 103.04 |
| full | `S3_amp__A1_amp` | 1.000 | 0.794 | 107.59 |
| full | `S3_amp__C1` | 1.000 | 0.761 | 79.51 |
| full | `S3_phase__A3_phase` | 1.000 | 0.820 | 115.50 |
| full | `S3_phase__B2_phase` | 1.000 | 0.838 | 94.81 |
| full | `A3_amp__C3` | 1.000 | 0.710 | 245.96 |
| full | `B2_amp__A1_amp` | 1.000 | 0.727 | 229.23 |
| full | `C1__C3` | 1.000 | 0.679 | 306.20 |
| new_training_only | `S3_amp__A3_amp` | 1.000 | 0.791 | 100.24 |
| new_training_only | `S3_amp__B2_amp` | 1.000 | 0.799 | 88.02 |
| new_training_only | `S3_amp__A1_amp` | 1.000 | 0.780 | 75.03 |
| new_training_only | `S3_amp__C1` | 1.000 | 0.689 | 98.39 |
| new_training_only | `S3_phase__A3_phase` | 1.000 | 0.872 | 33.11 |
| new_training_only | `S3_phase__B2_phase` | 1.000 | 0.887 | 34.54 |
| new_training_only | `A3_amp__C3` | 1.000 | 0.774 | 143.27 |
| new_training_only | `B2_amp__A1_amp` | 1.000 | 0.807 | 114.32 |
| new_training_only | `C1__C3` | 1.000 | 0.667 | 440.76 |
| parent | `S3_amp__A3_amp` | 1.000 | 0.761 | 155.19 |
| parent | `S3_amp__B2_amp` | 1.000 | 0.780 | 118.90 |

## Relative-Angle Coverage

| pair | bin | count | fraction | mean dev | p95 dev | random entropy |
|---|---|---:|---:|---:|---:|---:|
| `A3_S3` | aligned | 9040 | 0.238 | 3.817518352270786 | 7.146341495342407 |  |
| `A3_S3` | orthogonal | 9058 | 0.238 | 3.826980947067356 | 7.136343592576708 |  |
| `A3_S3` | anti_aligned | 9022 | 0.237 | 3.8326195716247633 | 7.159877869683664 |  |
| `A3_S3` | random | 10880 | 0.286 |  |  | 0.999134330141728 |
| `B2_S3` | aligned | 11004 | 0.234 | 3.8213118686739254 | 7.128575118462151 |  |
| `B2_S3` | orthogonal | 10985 | 0.234 | 3.818437924895303 | 7.151343140100629 |  |
| `B2_S3` | anti_aligned | 11054 | 0.235 | 3.7965172508382774 | 7.143874756366029 |  |
| `B2_S3` | random | 13957 | 0.297 |  |  | 0.9994805730457237 |
| `A1_S3` | aligned | 6717 | 0.232 | 3.828177816547861 | 7.143096150541129 |  |
| `A1_S3` | orthogonal | 6710 | 0.231 | 3.825875240488865 | 7.154501658630712 |  |
| `A1_S3` | anti_aligned | 6833 | 0.236 | 3.781775358346316 | 7.154787592892522 |  |
| `A1_S3` | random | 8740 | 0.301 |  |  | 0.9996417059013842 |

## Nearest-Neighbor Coverage

| distribution | method | n | p5 | median | p95 | max |
|---|---|---:|---:|---:|---:|---:|
| `training_only_to_training_only_excluding_self` | sklearn | 250000 | 0.04090832583606244 | 0.15071994066238403 | 1.0303212761878968 | 1.503586769104004 |
| `parent_to_new_training_only` | sklearn | 1075000 | 0.010752021335065367 | 0.32566089928150177 | 1.1315694391727444 | 1.5823336839675903 |
| `validation_to_v9_parent_train` | sklearn | 26977 | 0.0028726138174533845 | 0.09783343225717545 | 0.8923070549964904 | 1.1624983549118042 |
| `validation_to_v11_full_train` | sklearn | 26977 | 0.0027391603682190187 | 0.09446659684181213 | 0.8817047595977783 | 1.1624983549118042 |
| `blind_to_v9_parent_train` | sklearn | 27370 | 0.0030496023944579067 | 0.10551239550113678 | 0.8895733922719955 | 1.1242228746414185 |
| `blind_to_v11_full_train` | sklearn | 27370 | 0.0028677377267740667 | 0.10174888372421265 | 0.875978273153305 | 1.1242228746414185 |
| `stress_to_v9_parent_train` | sklearn | 28097 | 0.003122456232085824 | 0.10983459651470184 | 0.9230253934860226 | 1.220363736152649 |
| `stress_to_v11_full_train` | sklearn | 28097 | 0.002928482368588448 | 0.10546638071537018 | 0.9090770125389098 | 1.2202125787734985 |
