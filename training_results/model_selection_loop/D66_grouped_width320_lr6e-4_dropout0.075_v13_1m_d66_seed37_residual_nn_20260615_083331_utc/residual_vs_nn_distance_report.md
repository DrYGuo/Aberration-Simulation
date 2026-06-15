# Residual vs 12D Nearest-Neighbor Distance

Created UTC: `2026-06-15T10:10:52.106369+00:00`

This diagnostic asks whether the largest blind/stress residuals live far
from the training set in normalized 12D target space. If they do, the
remaining error is likely coverage-limited. If they do not, the bottleneck
is more likely feature sensitivity, model bias, loss weighting, or inverse
problem degeneracy.

- query splits: `blind, stress`
- train reference rows: `992556`
- query rows: `55467`
- NN method: `sklearn_exact`
- Pearson corr(error, NN distance): `0.7445857269670574`
- Spearman corr(error, NN distance): `0.7735352471219937`
- top 5% / all median NN distance ratio: `7.05776665266867`
- interpretation: **coverage_limited**

| Group | n | Median weighted error | Median NN distance | p95 weighted error | p95 NN distance |
|---|---:|---:|---:|---:|---:|
| all_blind_stress | 55467 | 0.0030103600583970547 | 0.08132940530776978 | 0.02126203645020722 | 0.5988839209079742 |
| top_1_percent_residuals | 555 | 0.04483802616596222 | 0.6089372038841248 | 0.07401129975914954 | 0.8168496310710905 |
| top_5_percent_residuals | 2774 | 0.02733448799699545 | 0.5740039646625519 | 0.05484841801226137 | 0.768213152885437 |

Compact CSV outputs:

- `training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v13_1m_d66_seed37_residual_nn_20260615_083331_utc/residual_vs_nn_distance_top_residuals.csv`
- `training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v13_1m_d66_seed37_residual_nn_20260615_083331_utc/residual_vs_nn_distance_binned_summary.csv`
- `training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v13_1m_d66_seed37_residual_nn_20260615_083331_utc/residual_vs_nn_distance_by_regime_quartile.csv`

The full row-level CSV is written in the run directory for Drive backup,
but may be too large for GitHub push policy.
