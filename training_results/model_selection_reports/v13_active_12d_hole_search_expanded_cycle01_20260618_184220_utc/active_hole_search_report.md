# Active 12D Hole Search v1

Created UTC: `2026-06-18T18:49:04.342019+00:00`

Purpose: freeze the v13 1M seed23 champion and actively probe holes in normalized 12D aberration space.

- selected probes: `5553`
- reference train rows: `992556`
- NN method: `sklearn`
- simulation status: `complete`
- inference status: `complete`

## Proposal Modes

| mode | selected | median NN distance | median score |
|---|---:|---:|---:|
| `genetic_algorithm` | 1553 | 1.2102525234222412 | 0.5893314480781555 |
| `high_amp_bridge` | 1200 | 0.8513986766338348 | 0.2948298901319504 |
| `high_amp_alignment` | 900 | 1.0291897058486938 | 0.5774522721767426 |
| `residual_jitter` | 900 | 0.7938489019870758 | 0.5486268401145935 |
| `far_nn` | 600 | 0.8187222182750702 | 0.5580897927284241 |
| `sobol_lhs` | 300 | 0.8481804430484772 | 0.5547771155834198 |
| `bridge_anchor` | 100 | 0.5625329911708832 | 0.12411387264728546 |

## Evaluation

- median weighted error: `0.03532499074935913`
- p95 weighted error: `0.17003022134304047`
- median NN distance: `0.8985945582389832`
- p95 NN distance: `1.1008195877075195`
- corr(error, NN distance): `0.4499748530030446`
- Spearman corr(error, NN distance): `0.5195697594750944`
- failure classes: `{'not_top_failure': 4886, 'coverage_limited_sparse_failure': 425, 'mixed_failure': 216, 'dense_model_feature_loss_failure': 26}`

Key compact artifacts:

- `active_hole_search_top_failures.csv`
- `active_hole_clusters.csv`
- `active_hole_regime_summary.csv`
- `active_hole_relative_angle_summary.csv`
- `active_hole_search_recommended_sampling_plan.json`

## Recommendation

- primary next move: **targeted_data_expansion**
- recommended future dataset: `v15_active_hole_expanded_250k`
- this workflow remains diagnostic-only; it does not train a new model.
