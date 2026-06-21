# Broad 12D Representative Validation v1

Created UTC: `2026-06-21T05:42:28.552147+00:00`

- rows: `100000`
- train on this: `false`
- role: frozen validation/benchmark design
- full design CSV: `training_results/model_selection_reports/broad_12d_representative_validation_v1_20260621_054222_utc/broad_12d_representative_validation_design.csv`
- compact GitHub sample: `training_results/model_selection_reports/broad_12d_representative_validation_v1_20260621_054222_utc/broad_12d_representative_validation_head1000.csv`

## Label Counts

| label | n | fraction |
|---|---:|---:|
| `S3_high_random` | 2500 | 0.025000 |
| `coupled_A1_A2_B2_random` | 3000 | 0.030000 |
| `coupled_A1_B2_S3_random` | 7000 | 0.070000 |
| `coupled_A1_S3_random` | 6000 | 0.060000 |
| `coupled_A3_S3_random` | 9000 | 0.090000 |
| `coupled_B2_S3_random` | 8000 | 0.080000 |
| `coupled_C1_C3_random` | 2500 | 0.025000 |
| `coupled_C3_A3_S3_random` | 6000 | 0.060000 |
| `coupled_full_random` | 42000 | 0.420000 |
| `coupled_sparse_random` | 14000 | 0.140000 |

## Notes

- Fresh frozen validation design for generalization scoring.
- Includes LHS space filling, coupled-full, coupled-sparse, high-vector couplings, relative-angle controls, and moderate bridge-like cases.
- Do not train on these exact rows.
