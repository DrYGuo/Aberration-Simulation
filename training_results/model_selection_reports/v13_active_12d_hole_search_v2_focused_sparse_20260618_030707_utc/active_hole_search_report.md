# Active 12D Hole Search v1

Created UTC: `2026-06-18T03:12:30.491356+00:00`

Purpose: freeze the v13 1M seed23 champion and actively probe holes in normalized 12D aberration space.

- selected probes: `3790`
- reference train rows: `992556`
- NN method: `sklearn`
- simulation status: `complete`
- inference status: `complete`

## Proposal Modes

| mode | selected | median NN distance | median score |
|---|---:|---:|---:|
| `far_nn` | 1200 | 0.7845993638038635 | 0.583297997713089 |
| `sobol_lhs` | 600 | 0.8189472854137421 | 0.5625343024730682 |
| `genetic_algorithm` | 1190 | 1.246765911579132 | 0.6098838746547699 |
| `residual_jitter` | 600 | 0.8220810294151306 | 0.5821039378643036 |
| `bridge_anchor` | 200 | 0.5596419274806976 | 0.11828195676207542 |

## Evaluation

- median weighted error: `0.029183480888605118`
- p95 weighted error: `0.31026995182037354`
- median NN distance: `0.8035714626312256`
- p95 NN distance: `1.130496621131897`
- corr(error, NN distance): `0.630273650858079`
- Spearman corr(error, NN distance): `0.663024367989307`
- failure classes: `{'not_top_failure': 3411, 'coverage_limited_sparse_failure': 374, 'mixed_failure': 4, 'dense_model_feature_loss_failure': 1}`

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
