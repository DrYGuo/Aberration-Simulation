# Active 12D Hole Search v1

Created UTC: `2026-06-18T03:01:07.934620+00:00`

Purpose: freeze the v13 1M seed23 champion and actively probe holes in normalized 12D aberration space.

- selected probes: `3000`
- reference train rows: `992556`
- NN method: `sklearn`
- simulation status: `complete`
- inference status: `complete`

## Proposal Modes

| mode | selected | median NN distance | median score |
|---|---:|---:|---:|
| `far_nn` | 900 | 0.7367553114891052 | 0.5528350174427032 |
| `sobol_lhs` | 650 | 0.7489132583141327 | 0.5789717137813568 |
| `genetic_algorithm` | 800 | 1.2084516286849976 | 0.6107628643512726 |
| `residual_jitter` | 350 | 0.801971048116684 | 0.5674217045307159 |
| `bridge_anchor` | 300 | 0.2847375273704529 | 0.06439803913235664 |

## Evaluation

- median weighted error: `0.024750251322984695`
- p95 weighted error: `0.08157751709222794`
- median NN distance: `0.7541408538818359`
- p95 NN distance: `1.129651665687561`
- corr(error, NN distance): `0.6652966558520316`
- Spearman corr(error, NN distance): `0.7268325283147252`
- failure classes: `{'not_top_failure': 2700, 'coverage_limited_sparse_failure': 238, 'mixed_failure': 46, 'dense_model_feature_loss_failure': 16}`

Key compact artifacts:

- `active_hole_search_top_failures.csv`
- `active_hole_clusters.csv`
- `active_hole_regime_summary.csv`
- `active_hole_relative_angle_summary.csv`
- `active_hole_search_recommended_sampling_plan.json`

## Recommendation

- primary next move: **targeted_data_expansion**
- recommended future dataset: `v15_active_hole_targeted`
- this workflow remains diagnostic-only; it does not train a new model.
