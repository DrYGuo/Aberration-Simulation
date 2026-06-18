# Active 12D Hole-Search Synthesis

Created UTC: `2026-06-18T19:39:08.625061+00:00`

## Cycle Comparison

| cycle | probes | median err | p95 err | median NN | Spearman err-NN | coverage | mixed | dense |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3000 | 0.02475 | 0.08158 | 0.75414 | 0.727 | 238 | 46 | 16 |
| 2 | 5553 | 0.03532 | 0.17003 | 0.89859 | 0.520 | 425 | 216 | 26 |
| 3 | 5658 | 0.03397 | 0.08431 | 0.91405 | 0.590 | 585 | 139 | 68 |
| 4 | 5995 | 0.03453 | 0.08176 | 0.90923 | 0.482 | 488 | 187 | 165 |
| 5 | 6500 | 0.03030 | 0.07102 | 0.87721 | 0.452 | 514 | 177 | 284 |
| 6 | 3790 | 0.02918 | 0.31027 | 0.80357 | 0.663 | 374 | 4 | 1 |
| 7 | 4600 | 0.03293 | 0.08709 | 0.91543 | 0.557 | 247 | 128 | 85 |
| 8 | 4800 | 0.03264 | 0.08609 | 0.88034 | 0.533 | 347 | 83 | 50 |

## Interpretation

- Aggregate coverage-limited failures: `3218`
- Aggregate mixed failures: `980`
- Aggregate dense/model-feature-loss failures: `695`
- The active search supports targeted expansion, but dense high-amplitude failures are real enough to track separately.

## Recommended Next Move

- Prepare `v15_active_hole_targeted` as a 250k targeted expansion.
- Keep v13 checkpoint frozen for active-search comparison.
- Run feature/model/loss diagnostics in parallel for high-amplitude A1/B2/S3/A3 cases.

See `v15_active_hole_targeted_sampling_plan.json` for the proposed sampling mix.
