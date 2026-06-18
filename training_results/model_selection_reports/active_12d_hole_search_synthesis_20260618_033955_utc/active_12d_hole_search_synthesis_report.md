# Active 12D Hole-Search Synthesis

Created UTC: `2026-06-18T03:39:55.611975+00:00`

## Cycle Comparison

| cycle | probes | median err | p95 err | median NN | Spearman err-NN | coverage | mixed | dense |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3000 | 0.02475 | 0.08158 | 0.75414 | 0.727 | 238 | 46 | 16 |
| 2 | 3790 | 0.02918 | 0.31027 | 0.80357 | 0.663 | 374 | 4 | 1 |
| 3 | 4600 | 0.03293 | 0.08709 | 0.91543 | 0.557 | 247 | 128 | 85 |
| 4 | 4800 | 0.03264 | 0.08609 | 0.88034 | 0.533 | 347 | 83 | 50 |

## Interpretation

- Aggregate coverage-limited failures: `1206`
- Aggregate mixed failures: `261`
- Aggregate dense/model-feature-loss failures: `152`
- The active search supports targeted expansion, but dense high-amplitude failures are real enough to track separately.

## Recommended Next Move

- Prepare `v15_active_hole_targeted` as a 250k targeted expansion.
- Keep v13 checkpoint frozen for active-search comparison.
- Run feature/model/loss diagnostics in parallel for high-amplitude A1/B2/S3/A3 cases.

See `v15_active_hole_targeted_sampling_plan.json` for the proposed sampling mix.
