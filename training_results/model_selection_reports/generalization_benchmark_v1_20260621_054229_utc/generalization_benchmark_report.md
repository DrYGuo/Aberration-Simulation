# Generalization Benchmark v1

Created UTC: `2026-06-21T05:42:30.112729+00:00`

Purpose: freeze a model-selection benchmark that prioritizes 12D generalization over narrow fixed-split wins.

- benchmark id: `generalization_benchmark_v1`
- train on this: `False`
- new-hole design dir: `training_results/model_selection_reports/v13_active_12d_generalization_benchmark_v1_new_holes_20260621_053803_utc`
- new-hole probe count: `5000`
- broad representative design dir: `training_results/model_selection_reports/broad_12d_representative_validation_v1_20260621_054222_utc`
- broad representative probe count: `100000`
- anchor/easy design dir: `training_results/model_selection_reports/anchor_easy_validation_v1_20260621_054222_utc`
- anchor/easy probe count: `5000`

## Components

| component | weight | status | rows | role |
|---|---:|---|---:|---|
| `broad_fixed_validation_blind_stress` | 0.2 | scored_for_v13_v15 |  | preserve broad fixed split generalization |
| `active_hole_repair` | 0.15 | scored_for_v13_v15 | 3988 | ensure previously discovered v13 holes are repaired |
| `new_hole_challenge` | 0.2 | frozen_design_pending_simulation | 5000 | held-out new holes not used by v15 training |
| `broad_12d_representative_validation` | 0.2 | frozen_design_pending_simulation | 100000 | large fresh representative validation design for the 12D coupled space |
| `hard_vector_diagnostics` | 0.15 | partially_scored_for_fixed_splits |  | guard A3/S3/B2/A1 vector magnitude and angle failures |
| `anchor_regression_guard` | 0.1 | frozen_design_pending_simulation | 5000 | prevent over-specialized active-hole repair from damaging easy/anchor regimes |
| `current_available_score_v13` |  | available_score=0.03580822013121358; promotable=True; gates= |  | current diagnostic score before new-hole challenge |
| `current_available_score_v15` |  | available_score=0.024439110976850743; promotable=False; gates=broad_blind_stress_regression;new_hole_challenge_missing_for_final_promotion |  | current diagnostic score before new-hole challenge |

## Interpretation

- v13 remains the production baseline until a candidate passes broad, active-hole, new-hole, vector, and anchor gates.
- v15 proves the active-hole errors are learnable, but it is not promoted because broad fixed benchmarks regress.
- The new-hole challenge, broad representative validation, and anchor/easy validation rows are intentionally held out from training.
- These frozen designs should be simulated/evaluated next for v13 and v15 before v16 training.
- v16 sampling should be decided from this benchmark, not from the old blind/stress split alone.

## Next Action

Simulate the frozen new-hole, broad representative, and anchor/easy probes and run v13/v15 inference. Then update benchmark-suite scoring so the full generalization suite is populated.
