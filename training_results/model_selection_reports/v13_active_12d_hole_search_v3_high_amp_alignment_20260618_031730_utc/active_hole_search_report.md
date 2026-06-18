# Active 12D Hole Search v1

Created UTC: `2026-06-18T03:23:01.118539+00:00`

Purpose: freeze the v13 1M seed23 champion and actively probe holes in normalized 12D aberration space.

- selected probes: `4600`
- reference train rows: `992556`
- NN method: `sklearn`
- simulation status: `complete`
- inference status: `complete`

## Proposal Modes

| mode | selected | median NN distance | median score |
|---|---:|---:|---:|
| `high_amp_alignment` | 1600 | 1.0317403078079224 | 0.6080082058906555 |
| `genetic_algorithm` | 1000 | 1.2728731036186218 | 0.6410776376724243 |
| `far_nn` | 800 | 0.7834610044956207 | 0.5482123792171478 |
| `residual_jitter` | 600 | 0.8153824806213379 | 0.5775635242462158 |
| `sobol_lhs` | 400 | 0.8282006680965424 | 0.5903127193450928 |
| `bridge_anchor` | 200 | 0.5593081116676331 | 0.11726569011807442 |

## Evaluation

- median weighted error: `0.03292689472436905`
- p95 weighted error: `0.08708783984184265`
- median NN distance: `0.9154312014579773`
- p95 NN distance: `1.145148754119873`
- corr(error, NN distance): `0.44583980507085713`
- Spearman corr(error, NN distance): `0.5574908538821746`
- failure classes: `{'not_top_failure': 4140, 'coverage_limited_sparse_failure': 247, 'mixed_failure': 128, 'dense_model_feature_loss_failure': 85}`

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
