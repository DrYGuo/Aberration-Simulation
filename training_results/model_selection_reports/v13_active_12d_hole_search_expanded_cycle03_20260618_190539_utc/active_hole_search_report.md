# Active 12D Hole Search v1

Created UTC: `2026-06-18T19:12:29.028294+00:00`

Purpose: freeze the v13 1M seed23 champion and actively probe holes in normalized 12D aberration space.

- selected probes: `5995`
- reference train rows: `992556`
- NN method: `sklearn`
- simulation status: `complete`
- inference status: `complete`

## Proposal Modes

| mode | selected | median NN distance | median score |
|---|---:|---:|---:|
| `genetic_algorithm` | 1495 | 1.2356129884719849 | 0.5780612230300903 |
| `high_amp_oblique` | 1700 | 0.9601123332977295 | 0.568723738193512 |
| `high_amp_alignment` | 900 | 1.0318385362625122 | 0.605900377035141 |
| `far_nn` | 900 | 0.8137480616569519 | 0.5689987242221832 |
| `residual_jitter` | 600 | 0.8059128522872925 | 0.5505677759647369 |
| `sobol_lhs` | 300 | 0.8690932393074036 | 0.6061245799064636 |
| `bridge_anchor` | 100 | 0.588074654340744 | 0.12664678692817688 |

## Evaluation

- median weighted error: `0.03452911972999573`
- p95 weighted error: `0.08175557851791382`
- median NN distance: `0.9092305898666382`
- p95 NN distance: `1.1144638061523438`
- corr(error, NN distance): `0.42015151592504657`
- Spearman corr(error, NN distance): `0.48197394410394173`
- failure classes: `{'not_top_failure': 5155, 'dense_model_feature_loss_failure': 165, 'coverage_limited_sparse_failure': 488, 'mixed_failure': 187}`

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
