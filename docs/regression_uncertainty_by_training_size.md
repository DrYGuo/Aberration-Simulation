# Regression Uncertainty By Training Size

This table summarizes validation-split coefficient uncertainty for the promoted
D66 grouped-head model family as training data increased. Values are p95
absolute errors in the physical units shown in the first column.

For vector coefficients, the value is the worse of the x/y component p95
absolute errors. This gives a conservative single-number error-bar estimate for
presentation. The `500K + larger validation/test` column is labeled explicitly
because the validation/blind/stress benchmark was enlarged at v12; it should not
be interpreted as only a training-size effect.

| Coefficient | 25K | 100K | 250K | 500K | 500K + larger validation/test | 1M | Improvement vs 25K |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1 (nm) | 11.3 | 5.88 | 4.67 | 4.22 | 2.86 | 2.69 | 4.20x |
| A1 (nm) | 4.04 | 2.66 | 2.29 | 2.01 | 1.22 | 1.15 | 3.52x |
| B2 (um) | 0.281 | 0.187 | 0.162 | 0.143 | 0.0935 | 0.0838 | 3.36x |
| A2 (um) | 0.552 | 0.315 | 0.287 | 0.247 | 0.192 | 0.204 | 2.70x |
| C3 (mm) | 0.0598 | 0.0348 | 0.0283 | 0.0266 | 0.0176 | 0.0165 | 3.62x |
| S3 (um) | 17.2 | 10.5 | 8.69 | 7.03 | 4.01 | 3.68 | 4.67x |
| A3 (um) | 6.78 | 4.46 | 3.82 | 3.38 | 2.15 | 1.93 | 3.51x |

Reference RMSE values are included below for context. RMSE is in the same unit
as each coefficient, and for vector coefficients again uses the worse x/y
component.

| Coefficient | 25K | 100K | 250K | 500K | 500K + larger validation/test | 1M | Improvement vs 25K |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1 (nm) | 5.45 | 2.83 | 2.37 | 2.23 | 1.56 | 1.54 | 3.55x |
| A1 (nm) | 2.23 | 1.44 | 1.19 | 0.988 | 0.641 | 0.575 | 3.88x |
| B2 (um) | 0.139 | 0.0869 | 0.0769 | 0.0659 | 0.0413 | 0.0376 | 3.70x |
| A2 (um) | 0.258 | 0.154 | 0.131 | 0.113 | 0.0807 | 0.0866 | 2.97x |
| C3 (mm) | 0.0288 | 0.0157 | 0.0135 | 0.0130 | 0.00896 | 0.00874 | 3.30x |
| S3 (um) | 8.77 | 5.89 | 4.35 | 3.85 | 2.52 | 2.18 | 4.03x |
| A3 (um) | 3.77 | 2.25 | 2.03 | 1.72 | 1.01 | 0.910 | 4.15x |

Source runs:

| Column | Run |
|---|---|
| 25K | `D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_bin_diag_20260609_073514_utc` |
| 100K | `D66_grouped_width320_lr6e-4_dropout0.075_v6gap100k_seed23_20260612_051859_utc` |
| 250K | `D66_grouped_width320_lr6e-4_dropout0.075_v9gap250k_d66_seed23_20260614_062447_utc` |
| 500K | `D66_grouped_width320_lr6e-4_dropout0.075_v11gap500k_d66_seed7_20260614_223556_utc` |
| 500K + larger validation/test | `D66_grouped_width320_lr6e-4_dropout0.075_v12benchmarkv2_500k_seed7_20260615_005333_utc` |
| 1M | `D66_grouped_width320_lr6e-4_dropout0.075_v13_1m_d66_seed7_20260615_042743_utc` |
