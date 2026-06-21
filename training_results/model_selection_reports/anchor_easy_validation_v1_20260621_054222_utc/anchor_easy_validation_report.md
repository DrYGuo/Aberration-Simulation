# Anchor/Easy Validation v1

Created UTC: `2026-06-21T05:42:28.698204+00:00`

- rows: `5000`
- train on this: `false`
- role: frozen validation/benchmark design
- full design CSV: `training_results/model_selection_reports/anchor_easy_validation_v1_20260621_054222_utc/anchor_easy_validation_design.csv`
- compact GitHub sample: `training_results/model_selection_reports/anchor_easy_validation_v1_20260621_054222_utc/anchor_easy_validation_head1000.csv`

## Label Counts

| label | n | fraction |
|---|---:|---:|
| `anchor_A1_sweep` | 600 | 0.120000 |
| `anchor_A2_sweep` | 500 | 0.100000 |
| `anchor_A3_sweep` | 800 | 0.160000 |
| `anchor_B2_sweep` | 600 | 0.120000 |
| `anchor_C1_sweep` | 600 | 0.120000 |
| `anchor_C3_sweep` | 400 | 0.080000 |
| `anchor_S3_sweep` | 800 | 0.160000 |
| `anchor_weak_full_coupled` | 500 | 0.100000 |
| `anchor_zero` | 200 | 0.040000 |

## Notes

- Frozen anchor/easy validation design to guard against over-specialized active-hole repair.
- Includes zero, one-coefficient sweeps, and weak fully coupled cases.
- Do not train on these exact rows.
