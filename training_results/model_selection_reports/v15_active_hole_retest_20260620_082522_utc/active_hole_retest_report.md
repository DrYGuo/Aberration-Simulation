# V15 Active-Hole Retest

Created UTC: `2026-06-20T08:41:03.992040+00:00`

Purpose: retest the original active-hole probe feature sets with the v15 checkpoint and compare against the original v13 active-hole errors.

- dataset CSV: `training_results/feature_regression_enhanced/enhanced_v15_active_hole_expanded_250k_20260619_050058_utc/training_features_enhanced.csv`
- checkpoint run dir: `training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v15_active_hole_expanded_250k_d66_seed23_checkpoint_rebuild_20260620_060115_utc`
- active probe runs retested: `9`
- combined matched probes: `39896`

## Combined Error

| metric | v13 RMSE | v15 RMSE | v15-v13 RMSE | v15 worse fraction |
|---|---:|---:|---:|---:|
| `weighted_abs_error` | 0.05790723743632659 | 0.04407194585963116 | -0.013835291576695434 | 0.5219069580910367 |
| `overall_abs_error` | 3.7251196123059755 | 2.820734346683258 | -0.9043852656227176 | 0.5145628634449569 |
| `S3_vector_error` | 15.8425081172437 | 13.370740507459276 | -2.4717676097844237 | 0.49766893924202926 |
| `A3_vector_error` | 11.421229155511389 | 7.268981258177272 | -4.152247897334117 | 0.5037347102466413 |
| `B2_vector_error` | 0.312253985230997 | 0.24771594852356485 | -0.06453803670743216 | 0.5032083416883898 |

## Per Active-Search Run

| active run | n matched | v13 weighted RMSE | v15 weighted RMSE | v15 worse fraction |
|---|---:|---:|---:|---:|
| `v13_active_12d_hole_search_20260617_215931_utc` | 0 | None | 0.040328815317382936 | None |
| `v13_active_12d_hole_search_20260618_025610_utc` | 3000 | 0.04091578966031976 | 0.040328815317382936 | 0.5556666666666666 |
| `v13_active_12d_hole_search_expanded_cycle01_20260618_184220_utc` | 5553 | 0.07390489090135564 | 0.04070433201305713 | 0.45650999459751485 |
| `v13_active_12d_hole_search_expanded_cycle02_20260618_185259_utc` | 5658 | 0.04861499330359217 | 0.05233039783554485 | 0.543655001767409 |
| `v13_active_12d_hole_search_expanded_cycle03_20260618_190539_utc` | 5995 | 0.046627496176263404 | 0.0505440919370234 | 0.5994995829858215 |
| `v13_active_12d_hole_search_expanded_cycle04_20260618_191721_utc` | 6500 | 0.03975382404950471 | 0.03995130505500183 | 0.5432307692307692 |
| `v13_active_12d_hole_search_v2_focused_sparse_20260618_030707_utc` | 3790 | 0.10218377292730074 | 0.03875880339814863 | 0.49709762532981533 |
| `v13_active_12d_hole_search_v3_high_amp_alignment_20260618_031730_utc` | 4600 | 0.048164763245596674 | 0.04310939311177365 | 0.47282608695652173 |
| `v13_active_12d_hole_search_v4_dense_alignment_bridge_20260618_032829_utc` | 4800 | 0.048959256311778776 | 0.043139498391143706 | 0.49166666666666664 |

Interpretation rule:

- If v15 improves the active-hole RMSE while fixed validation/blind/stress got worse, then v15 repaired some searched holes but damaged the benchmark distribution.
- If v15 does not improve active-hole RMSE, then the 250k active-hole expansion failed its primary purpose.
