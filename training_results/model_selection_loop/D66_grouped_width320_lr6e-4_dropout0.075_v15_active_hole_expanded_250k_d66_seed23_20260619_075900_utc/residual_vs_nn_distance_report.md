# Residual vs 12D Nearest-Neighbor Distance

Created UTC: `2026-06-19T10:15:05.225097+00:00`

This diagnostic asks whether the largest blind/stress residuals live far
from the training set in normalized 12D target space. If they do, the
remaining error is likely coverage-limited. If they do not, the bottleneck
is more likely feature sensitivity, model bias, loss weighting, or inverse
problem degeneracy.

- query splits: `blind, stress`
- train reference rows: `1242556`
- query rows: `55467`
- NN method: `sklearn_exact`
- Pearson corr(error, NN distance): `0.7363559893790158`
- Spearman corr(error, NN distance): `0.775234809377701`
- top 5% / all median NN distance ratio: `7.2089940987229`
- interpretation: **coverage_limited**

| Group | n | Median weighted error | Median NN distance | p95 weighted error | p95 NN distance |
|---|---:|---:|---:|---:|---:|
| all_blind_stress | 55467 | 0.003302882192656398 | 0.07838713377714157 | 0.02212304528802633 | 0.5905344307422637 |
| top_1_percent_residuals | 555 | 0.04604135826230049 | 0.5905243754386902 | 0.07427594214677809 | 0.7964613437652588 |
| top_5_percent_residuals | 2774 | 0.02843543328344822 | 0.5650923848152161 | 0.054461971111595626 | 0.7562128514051437 |

Compact CSV outputs:

- `training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v15_active_hole_expanded_250k_d66_seed23_20260619_075900_utc/residual_vs_nn_distance_top_residuals.csv`
- `training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v15_active_hole_expanded_250k_d66_seed23_20260619_075900_utc/residual_vs_nn_distance_binned_summary.csv`
- `training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v15_active_hole_expanded_250k_d66_seed23_20260619_075900_utc/residual_vs_nn_distance_by_regime_quartile.csv`

The full row-level CSV is written in the run directory for Drive backup,
but may be too large for GitHub push policy.
