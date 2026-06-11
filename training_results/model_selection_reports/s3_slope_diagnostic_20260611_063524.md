# S3 Slope Diagnostic Report

Created UTC: 2026-06-11T06:35:24.664179+00:00

## Files Checked

- Yesterday baseline: `training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_bin_diag_20260609_073514_utc`
  - `selection_score.json`
  - `metrics_model_loop.json`
  - `vector_diagnostics.json`
  - `model_registry_model_loop.csv`
  - `run_manifest_model_loop.json`
- Today candidate: `training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_plateau_clip_smoothl1_20260610_071108_utc`
  - `selection_score.json`
  - `metrics_model_loop.json`
  - `vector_diagnostics.json`
  - `model_registry_model_loop.csv`
  - `run_manifest_model_loop.json`
- Optimizer batch summary: `training_results/model_selection_batches/v3_optimizer_stability_batch_20260610_065205_utc/batch_summary.csv`
- Optimizer batch manifest: `training_results/model_selection_batches/v3_optimizer_stability_batch_20260610_065205_utc/batch_manifest.json`

## Best-Run Verification

- Yesterday best confirmed as `D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_bin_diag_20260609_073514_utc` for the pre-optimizer v3 baseline.
- Today best confirmed as `D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_plateau_clip_smoothl1_20260610_071108_utc` within `v3_optimizer_stability_batch_20260610_065205_utc`.
- Batch best by weighted score: `D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_plateau_clip_smoothl1` with score `0.0556789`.
- Batch manifest status: `complete`.

## Comparison

| Metric | Yesterday baseline | Today SmoothL1 plateau/clip | Direction |
|---|---:|---:|---|
| weighted score | 0.059453 | 0.0556789 | lower is better; improved |
| hard-label MAE | 0.0281728 | 0.0263872 | lower is better; improved |
| true-hard-target MAE | 0.0294726 | 0.0276974 | lower is better; improved |
| overall normalized MAE | 0.0218805 | 0.0205724 | lower is better; improved |
| overall p95 | 0.0856746 | 0.0798316 | lower is better; improved |
| blind normalized MAE | 0.0157522 | 0.0150928 | lower is better; improved |
| stress normalized MAE | 0.0164777 | 0.0153577 | lower is better; improved |
| S3 high-bin magnitude MAE | 10.8393 | 9.54026 | lower is better; improved |
| S3 high-bin bias | -8.76654 | -7.68264 | closer to zero is better; improved |
| S3 high-bin RMSE | 18.387 | 16.7227 | lower is better; improved |
| S3 high-bin slope | 0.671123 | 0.720875 | higher is better; improved |
| S3 high-bin R2 | -1.14226 | -0.822726 | higher is better; improved |
| S3 high-bin mean angle deg | 5.70257 | 5.48412 | lower is better; improved |
| S3 high-bin p95 angle deg | 20.9317 | 23.4003 | lower is better; worse |
| B2 magnitude MAE | 0.0932718 | 0.0887067 | lower is better; improved |
| A3 magnitude MAE | 2.18905 | 2.15356 | lower is better; improved |
| n_train | 34002 | 34002 | same |
| n_validation | 1976 | 1976 | same |
| n_blind | 2370 | 2370 | same |
| n_stress | 3098 | 3098 | same |
| n_total | 41446 | 41446 | same |

## Validation-Set Identity Check

### Yesterday baseline

- split seed: `7`
- dataset rows: total `41446`, train `34002`, validation `1976`, blind `2370`, stress `3098`
- validation index hash: `cb5355cf950ca26f`
- S3 vector scale: `90.7398` from `training_split_true_magnitude_p95`
- S3 bin counts: near_zero `1155`, low `153`, medium `347`, high `321`

### Today SmoothL1

- split seed: `7`
- dataset rows: total `41446`, train `34002`, validation `1976`, blind `2370`, stress `3098`
- validation index hash: `cb5355cf950ca26f`
- S3 vector scale: `90.7398` from `training_split_true_magnitude_p95`
- S3 bin counts: near_zero `1155`, low `153`, medium `347`, high `321`

- Validation indices identical: `True`
- Total current data size is `41446` rows, so it is about `41k`, not `50k`.

## S3 Slope Discrepancy

There is no separate stress/blind slope being copied into `batch_summary.csv`. The `0.720875` value is the validation S3 high-magnitude-bin slope from `vector_diagnostics.json`.

Code path:

- `scripts/regression_diagnostics.py` computes `vector_diagnostics.json`.
- `vector_diagnostics()` sets top-level vector diagnostics on the validation split only: `split_indices["validation"]`.
- `_vector_magnitude_bin_summary()` defines high as `true_magnitude > 0.7 * vector_scale`.
- `vector_scale` is the 95th percentile of true S3 magnitude on the training split.
- `scripts/run_model_selection_batch.py::selection_summary()` reads `vector_pairs -> S3 -> magnitude_bins -> bins -> high -> magnitude -> magnitude_slope` and writes it to `batch_summary.csv` as `S3_high_magnitude_slope`.
- `scripts/run_model_selection_candidate.py` writes the same high-bin value into `model_registry_model_loop.csv` for current runs.

Relevant S3 values for today:

- Overall validation S3 magnitude slope: `0.866124`
- High-bin validation S3 magnitude slope: `0.720875`
- High-bin mean cosine similarity: `0.977058`
- High-bin median cosine similarity: `0.999559`

Therefore, values around `0.98-0.99` are consistent with directional cosine / orientation quality, not the high-bin magnitude slope. The high-S3 angle/cosine behavior is good, while high-S3 magnitude calibration remains compressed.

## Recommendation

Recommendation: **A. retrain the current best with 2-3 more seeds**, while keeping the same fixed validation/blind/stress protocol.

Reasoning:

- The `0.72` S3 high-bin slope is real, but it is a validation high-bin magnitude-slope diagnostic, not a reporting mismatch and not a stress/blind-only metric.
- Today's SmoothL1 plateau/clip candidate improves weighted score, hard-label MAE, true-hard-target MAE, overall MAE/p95, blind/stress MAE, B2/A3 magnitude MAE, and high-S3 magnitude MAE/bias/RMSE/slope/R2.
- The validation split and S3 bin definitions are identical between runs, so the improvement is comparable.
- High-S3 p95 angle worsened slightly despite mean angle improvement, so seed-repeat validation is needed before treating this as robust.
- Do not jump directly to 100k rows. If seed repeats confirm the candidate but the high-S3 slope remains around `0.72`, then generate a targeted S3-tail expansion to about 55k-60k total rows and retrain the confirmed champion configuration.
