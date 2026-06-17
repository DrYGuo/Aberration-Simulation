# Residual vs 12D Nearest-Neighbor Distance

Created UTC: `2026-06-17T09:30:55.472023+00:00`

This diagnostic asks whether the largest blind/stress residuals live far
from the training set in normalized 12D target space. If they do, the
remaining error is likely coverage-limited. If they do not, the bottleneck
is more likely feature sensitivity, model bias, loss weighting, or inverse
problem degeneracy.

- query splits: `blind, stress`
- train reference rows: `1492556`
- query rows: `55467`
- NN method: `sklearn_exact`
- Pearson corr(error, NN distance): `0.720779479054862`
- Spearman corr(error, NN distance): `0.7373989184633561`
- top 5% / all median NN distance ratio: `7.252768254869043`
- interpretation: **coverage_limited**

| Group | n | Median weighted error | Median NN distance | p95 weighted error | p95 NN distance |
|---|---:|---:|---:|---:|---:|
| all_blind_stress | 55467 | 0.0031274757348001003 | 0.07525456696748734 | 0.01992339231073856 | 0.5682427763938904 |
| top_1_percent_residuals | 555 | 0.04112968221306801 | 0.5620588660240173 | 0.07116668373346328 | 0.7509766697883605 |
| top_5_percent_residuals | 2774 | 0.02569564525038004 | 0.5458039343357086 | 0.04942095503211021 | 0.7062851667404173 |

Compact CSV outputs:

- `training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v14_1p5m_far_nn_d66_seed23_20260617_072619_utc/residual_vs_nn_distance_top_residuals.csv`
- `training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v14_1p5m_far_nn_d66_seed23_20260617_072619_utc/residual_vs_nn_distance_binned_summary.csv`
- `training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v14_1p5m_far_nn_d66_seed23_20260617_072619_utc/residual_vs_nn_distance_by_regime_quartile.csv`

The full row-level CSV is written in the run directory for Drive backup,
but may be too large for GitHub push policy.
