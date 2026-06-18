# Active 12D Hole Search v1

Created UTC: `2026-06-18T03:34:18.145193+00:00`

Purpose: freeze the v13 1M seed23 champion and actively probe holes in normalized 12D aberration space.

- selected probes: `4800`
- reference train rows: `992556`
- NN method: `sklearn`
- simulation status: `complete`
- inference status: `complete`

## Proposal Modes

| mode | selected | median NN distance | median score |
|---|---:|---:|---:|
| `high_amp_bridge` | 1800 | 0.8400843739509583 | 0.2736932933330536 |
| `high_amp_alignment` | 700 | 1.030620276927948 | 0.6095697283744812 |
| `residual_jitter` | 800 | 0.8160360455513 | 0.5450346767902374 |
| `genetic_algorithm` | 700 | 1.3230443000793457 | 0.6803770363330841 |
| `sobol_lhs` | 500 | 0.8116924464702606 | 0.5895706117153168 |
| `bridge_anchor` | 300 | 0.5626071989536285 | 0.11566895246505737 |

## Evaluation

- median weighted error: `0.032639533281326294`
- p95 weighted error: `0.08609282225370407`
- median NN distance: `0.880341649055481`
- p95 NN distance: `1.108024001121521`
- corr(error, NN distance): `0.5013705273650058`
- Spearman corr(error, NN distance): `0.5331108249107648`
- failure classes: `{'not_top_failure': 4320, 'mixed_failure': 83, 'dense_model_feature_loss_failure': 50, 'coverage_limited_sparse_failure': 347}`

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
