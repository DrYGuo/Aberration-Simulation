# Active 12D Hole Search v1

Created UTC: `2026-06-18T19:24:22.227596+00:00`

Purpose: freeze the v13 1M seed23 champion and actively probe holes in normalized 12D aberration space.

- selected probes: `6500`
- reference train rows: `992556`
- NN method: `sklearn`
- simulation status: `complete`
- inference status: `complete`

## Proposal Modes

| mode | selected | median NN distance | median score |
|---|---:|---:|---:|
| `high_amp_oblique` | 2200 | 0.9550653696060181 | 0.5242628157138824 |
| `high_amp_bridge` | 1600 | 0.8288266956806183 | 0.22948278486728668 |
| `residual_jitter` | 900 | 0.7540934383869171 | 0.4937986135482788 |
| `genetic_algorithm` | 900 | 1.0711472034454346 | 0.5141227841377258 |
| `bridge_anchor` | 600 | 0.5736766457557678 | 0.11584041267633438 |
| `sobol_lhs` | 300 | 0.8224335312843323 | 0.5155429840087891 |

## Evaluation

- median weighted error: `0.03029719740152359`
- p95 weighted error: `0.07102079689502716`
- median NN distance: `0.8772144317626953`
- p95 NN distance: `1.0638371706008911`
- corr(error, NN distance): `0.39358166245293297`
- Spearman corr(error, NN distance): `0.45215658485498467`
- failure classes: `{'not_top_failure': 5525, 'mixed_failure': 177, 'coverage_limited_sparse_failure': 514, 'dense_model_feature_loss_failure': 284}`

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
