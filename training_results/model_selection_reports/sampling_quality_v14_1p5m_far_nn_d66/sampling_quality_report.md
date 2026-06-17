# Sampling Quality Report

Created UTC: `2026-06-17T07:25:46.136664+00:00`
Dataset: `training_results/feature_regression_enhanced/enhanced_v14_1p5m_far_nn_20260617_051304_utc/training_features_enhanced.csv`
Config: `configs/targeted_expansion_v14_1p5m_far_nn.json`

## Counts

- total rows: `1575000`
- parent rows: `1075000`
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
| `S3_high_random` | 6000 | 6000 | 0.0000 | PASS |
| `coupled_A1_A2_B2_random` | 35000 | 35000 | 0.0000 | PASS |
| `coupled_A1_B2_S3_random` | 20000 | 20000 | 0.0000 | PASS |
| `coupled_A1_B2_random` | 2000 | 2000 | 0.0000 | PASS |
| `coupled_A1_S3_random` | 5000 | 5000 | 0.0000 | PASS |
| `coupled_A2_B2_random` | 20000 | 20000 | 0.0000 | PASS |
| `coupled_A3_S3_random` | 13000 | 13000 | 0.0000 | PASS |
| `coupled_B2_S3_random` | 13000 | 13000 | 0.0000 | PASS |
| `coupled_C1_A1_C3_A2_random` | 45000 | 45000 | 0.0000 | PASS |
| `coupled_C1_A1_C3_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_A1_S3_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_A1_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_A2_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_A3_S3_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_A3_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_B2_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_C3_A2_random` | 20000 | 20000 | 0.0000 | PASS |
| `coupled_C1_C3_S3_random` | 0 | 0 | 0.0000 | PASS |
| `coupled_C1_C3_random` | 4000 | 4000 | 0.0000 | PASS |
| `coupled_C1_S3_random` | 3000 | 3000 | 0.0000 | PASS |
| `coupled_C3_A3_S3_random` | 18000 | 18000 | 0.0000 | PASS |
| `coupled_C3_B2_random` | 1000 | 1000 | 0.0000 | PASS |
| `coupled_full_random` | 210000 | 210000 | 0.0000 | PASS |
| `coupled_sparse_random` | 85000 | 85000 | 0.0000 | PASS |

## Candidate Selection

| role | rows | fraction | median NN distance | p95 NN distance | max NN distance |
|---|---:|---:|---:|---:|---:|
| `bridge_anchor` | 100000 | 0.200 | 0.36175745725631714 | 0.4522073999047279 | 0.4582562744617462 |
| `far_nn` | 400000 | 0.800 | 0.5234703719615936 | 0.8437438845634461 | 1.187024712562561 |

## Key Warnings

- None.

## Marginal Coverage Warnings

| subset | quantity | empty bins | entropy | max/median |
|---|---|---:|---:|---:|
| full | `A2_phase` | 0.000 | 0.697 | 21.11 |
| full | `A3_phase` | 0.000 | 0.706 | 20.32 |
| full | `A2_amp` | 0.000 | 0.708 | 19.41 |
| full | `C1` | 0.000 | 0.710 | 21.02 |
| full | `A3_amp` | 0.000 | 0.719 | 19.16 |
| full | `B2_phase` | 0.000 | 0.757 | 16.49 |
| full | `C3` | 0.000 | 0.760 | 16.37 |
| full | `A1_phase` | 0.000 | 0.764 | 16.03 |
| full | `B2_amp` | 0.000 | 0.771 | 15.20 |
| full | `A1_amp` | 0.000 | 0.778 | 14.57 |
| full | `S3_amp` | 0.000 | 0.868 | 13.17 |
| full | `S3_phase` | 0.000 | 0.904 | 7.47 |
| new_training_only | `A3_phase` | 0.000 | 0.784 | 14.62 |
| new_training_only | `A3_amp` | 0.000 | 0.796 | 13.29 |
| new_training_only | `C1` | 0.000 | 0.798 | 14.88 |
| new_training_only | `S3_amp` | 0.000 | 0.850 | 12.15 |
| new_training_only | `S3_phase` | 0.000 | 0.853 | 10.32 |
| new_training_only | `C3` | 0.000 | 0.873 | 9.37 |
| new_training_only | `B2_phase` | 0.000 | 0.876 | 9.04 |
| new_training_only | `B2_amp` | 0.000 | 0.886 | 7.68 |

## Pairwise Occupancy Warnings

| subset | pair | nonempty | entropy | max/median |
|---|---|---:|---:|---:|
| full | `S3_amp__A3_amp` | 1.000 | 0.779 | 129.17 |
| full | `S3_amp__B2_amp` | 1.000 | 0.816 | 88.80 |
| full | `S3_amp__A1_amp` | 1.000 | 0.819 | 83.57 |
| full | `S3_amp__C1` | 1.000 | 0.783 | 75.67 |
| full | `S3_phase__A3_phase` | 1.000 | 0.801 | 136.29 |
| full | `S3_phase__B2_phase` | 1.000 | 0.834 | 100.45 |
| full | `A3_amp__C3` | 1.000 | 0.732 | 208.45 |
| full | `B2_amp__A1_amp` | 1.000 | 0.765 | 170.87 |
| full | `C1__C3` | 1.000 | 0.719 | 228.28 |
| new_training_only | `S3_amp__A3_amp` | 1.000 | 0.795 | 132.90 |
| new_training_only | `S3_amp__B2_amp` | 1.000 | 0.865 | 73.83 |
| new_training_only | `S3_amp__A1_amp` | 1.000 | 0.876 | 53.20 |
| new_training_only | `S3_amp__C1` | 1.000 | 0.827 | 91.53 |
| new_training_only | `S3_phase__A3_phase` | 1.000 | 0.801 | 136.23 |
| new_training_only | `S3_phase__B2_phase` | 1.000 | 0.865 | 83.18 |
| new_training_only | `A3_amp__C3` | 1.000 | 0.826 | 108.76 |
| new_training_only | `B2_amp__A1_amp` | 1.000 | 0.890 | 59.36 |
| new_training_only | `C1__C3` | 1.000 | 0.816 | 130.80 |
| parent | `S3_amp__A3_amp` | 1.000 | 0.761 | 155.19 |
| parent | `S3_amp__B2_amp` | 1.000 | 0.780 | 118.90 |

## Relative-Angle Coverage

| pair | bin | count | fraction | mean dev | p95 dev | random entropy |
|---|---|---:|---:|---:|---:|---:|
| `A3_S3` | aligned | 5231 | 0.169 | 4.109579475071653 | 7.227578138280705 |  |
| `A3_S3` | orthogonal | 5413 | 0.175 | 4.032508591482363 | 7.233021640082642 |  |
| `A3_S3` | anti_aligned | 5402 | 0.174 | 4.0698119010203975 | 7.225690712202387 |  |
| `A3_S3` | random | 14954 | 0.482 |  |  | 0.9883179547935629 |
| `B2_S3` | aligned | 4585 | 0.139 | 4.264208238467786 | 7.242165599759511 |  |
| `B2_S3` | orthogonal | 4347 | 0.132 | 4.183687885253786 | 7.271290884891937 |  |
| `B2_S3` | anti_aligned | 4479 | 0.136 | 4.16816556662076 | 7.242960635450208 |  |
| `B2_S3` | random | 19589 | 0.594 |  |  | 0.9972347688749017 |
| `A1_S3` | aligned | 3210 | 0.128 | 4.258887839023627 | 7.235232525787691 |  |
| `A1_S3` | orthogonal | 3136 | 0.125 | 4.18222067237884 | 7.239302230482878 |  |
| `A1_S3` | anti_aligned | 3163 | 0.127 | 4.149970290024057 | 7.232206885111088 |  |
| `A1_S3` | random | 15491 | 0.620 |  |  | 0.9988429706722104 |

## Nearest-Neighbor Coverage

| distribution | method | n | p5 | median | p95 | max |
|---|---|---:|---:|---:|---:|---:|
| `training_only_to_training_only_excluding_self` | sklearn | 500000 | 0.060903215594589714 | 0.6265031099319458 | 0.977230703830719 | 1.2665579319000244 |
| `parent_to_new_training_only` | sklearn | 1075000 | 0.010878219455480577 | 0.2297818660736084 | 0.9593590736389159 | 1.3139522075653076 |
| `validation_to_v9_parent_train` | sklearn | 26977 | 0.0028726138174533845 | 0.09783343225717545 | 0.8923070549964904 | 1.1624983549118042 |
| `validation_to_v11_full_train` | sklearn | 26977 | 0.002746567130088806 | 0.091021828353405 | 0.852230930328369 | 1.149642825126648 |
| `blind_to_v9_parent_train` | sklearn | 27370 | 0.0030496023944579067 | 0.10551239550113678 | 0.8895733922719955 | 1.1242228746414185 |
| `blind_to_v11_full_train` | sklearn | 27370 | 0.002908771031070501 | 0.09707748144865036 | 0.840379899740219 | 1.1242228746414185 |
| `stress_to_v9_parent_train` | sklearn | 28097 | 0.003122456232085824 | 0.10983459651470184 | 0.9230253934860226 | 1.220363736152649 |
| `stress_to_v11_full_train` | sklearn | 28097 | 0.0029605096671730283 | 0.09998949617147446 | 0.8677197575569152 | 1.1557835340499878 |
