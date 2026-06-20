# Session Handoff: 2026-06-20

## Latest State

- GitHub head evaluated locally: `007d60f`.
- Latest completed Colab workflow: `v15-active-hole-retest`.
- Retest report:
  `training_results/model_selection_reports/v15_active_hole_retest_20260620_082522_utc/active_hole_retest_report.md`
- Drive backup:
  `/content/drive/MyDrive/Aberration-Simulation-Colab-Backups/v15_active_hole_retest_latest`

## Key Results

V15 repaired the active-hole probe set but worsened the old fixed
validation/blind/stress benchmark. Therefore:

- Keep v13 1M seed23 D66 grouped-head as the general baseline.
- Do not promote v15 as the general model.
- Use v15 evidence to design v16 with a balanced active-hole plus broad-space
  training distribution.

Matched active-hole RMSE:

| metric | v13 | v15 |
|---|---:|---:|
| weighted normalized abs | 0.05791 | 0.04434 |
| overall mixed-unit abs | 3.725 | 2.839 |
| S3 vector | 15.84 um | 13.49 um |
| A3 vector | 11.42 um | 7.27 um |
| B2 vector | 0.312 um | 0.250 um |

The old v13 top-failure subset (`n=3988`) had much larger RMSE:

| metric | v13 top-failure RMSE |
|---|---:|
| weighted normalized abs | 0.14255 |
| overall mixed-unit abs | 9.30051 |
| C1 | 10.93 nm |
| C3 | 0.0366 mm |
| A1 vector | 17.61 nm |
| B2 vector | 0.669 um |
| A2 vector | 0.653 um |
| S3 vector | 35.68 um |
| A3 vector | 30.73 um |

The exact v15 retest on that same top-3988 subset still needs a compact derived
report from Drive artifacts.

## Next Technical Task

Build benchmark-suite model selection:

1. broad representative benchmark;
2. frozen active-hole benchmark;
3. held-out new-hole challenge benchmark;
4. anchor/easy benchmark.

Then prepare a v16 balanced expansion:

- 35% known active-hole repair rows;
- 20% new active-search holes;
- 25% broad coupled-full/coupled-sparse Sobol/LHS rows;
- 15% benchmark-preserving anchors;
- 5% one-coefficient/Uno-success sweeps.

The next model-selection score should combine broad generalization and active
holes, instead of relying only on the old fixed validation/blind/stress split.

## Caution

Do not delete active-hole Drive folders that contain
`active_hole_search_probe_features.csv` until the v16 benchmark suite is frozen.
Do not push `.pt`, large CSVs, raw predictions, or full feature tables to GitHub.
