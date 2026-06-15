# Residual vs 12D Nearest-Neighbor Distance

Created UTC: `2026-06-15T08:33:11.645817+00:00`

This diagnostic asks whether the largest blind/stress residuals live far
from the training set in normalized 12D target space. If they do, the
remaining error is likely coverage-limited. If they do not, the bottleneck
is more likely feature sensitivity, model bias, loss weighting, or inverse
problem degeneracy.

- query splits: `blind, stress`
- train reference rows: `992556`
- query rows: `55467`
- NN method: `sklearn_exact`
- Pearson corr(error, NN distance): `0.7407441140299763`
- Spearman corr(error, NN distance): `0.7719476956440188`
- top 5% / all median NN distance ratio: `7.092152987106453`
- interpretation: **coverage_limited**

| Group | n | Median weighted error | Median NN distance | p95 weighted error | p95 NN distance |
|---|---:|---:|---:|---:|---:|
| all_blind_stress | 55467 | 0.0029412631411105394 | 0.08132940530776978 | 0.020849328860640497 | 0.5988839209079742 |
| top_1_percent_residuals | 555 | 0.044698506593704224 | 0.597533643245697 | 0.07366289719939231 | 0.8116495132446289 |
| top_5_percent_residuals | 2774 | 0.02706417441368103 | 0.5768005847930908 | 0.05278957784175872 | 0.7694437384605407 |

Compact CSV outputs:

- `training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v13_1m_d66_seed23_residual_nn_20260615_065556_utc/residual_vs_nn_distance_top_residuals.csv`
- `training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v13_1m_d66_seed23_residual_nn_20260615_065556_utc/residual_vs_nn_distance_binned_summary.csv`
- `training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v13_1m_d66_seed23_residual_nn_20260615_065556_utc/residual_vs_nn_distance_by_regime_quartile.csv`

The full row-level CSV is written in the run directory for Drive backup,
but may be too large for GitHub push policy.
