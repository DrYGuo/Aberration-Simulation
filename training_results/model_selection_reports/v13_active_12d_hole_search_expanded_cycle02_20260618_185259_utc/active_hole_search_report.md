# Active 12D Hole Search v1

Created UTC: `2026-06-18T18:59:44.931299+00:00`

Purpose: freeze the v13 1M seed23 champion and actively probe holes in normalized 12D aberration space.

- selected probes: `5658`
- reference train rows: `992556`
- NN method: `sklearn`
- simulation status: `complete`
- inference status: `complete`

## Proposal Modes

| mode | selected | median NN distance | median score |
|---|---:|---:|---:|
| `genetic_algorithm` | 1458 | 1.2866686582565308 | 0.6168636977672577 |
| `high_amp_alignment` | 1300 | 1.039264976978302 | 0.555081307888031 |
| `far_nn` | 900 | 0.8166715502738953 | 0.5714994668960571 |
| `high_amp_bridge` | 800 | 0.9059453010559082 | 0.3170087933540344 |
| `residual_jitter` | 700 | 0.8070911467075348 | 0.557874321937561 |
| `sobol_lhs` | 400 | 0.8604029417037964 | 0.6000595688819885 |
| `bridge_anchor` | 100 | 0.5850137174129486 | 0.12805236130952835 |

## Evaluation

- median weighted error: `0.03397234529256821`
- p95 weighted error: `0.08431403338909149`
- median NN distance: `0.9140455722808838`
- p95 NN distance: `1.2083457708358765`
- corr(error, NN distance): `0.5298915082908092`
- Spearman corr(error, NN distance): `0.5903710992816245`
- failure classes: `{'not_top_failure': 4866, 'coverage_limited_sparse_failure': 585, 'dense_model_feature_loss_failure': 68, 'mixed_failure': 139}`

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
