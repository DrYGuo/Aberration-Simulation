# v15 Active-Hole Postmortem

Created UTC: 2026-06-19T11:13:52.749064+00:00

## Conclusion

v15 did not repair the benchmark-visible coverage failures; it slightly improved 12D nearest-neighbor distances but worsened fixed validation/blind/stress errors.

- Original active-hole repair proven? **not_proven_from_current_artifacts**
- Benchmark effect: **damaged_fixed_benchmark_distribution**
- Coverage effect: **global NN distances improved slightly, but weighted residuals worsened**

The v15 data generation itself was valid: 250,000 new training-only rows were added with no benchmark leakage. But v15 worsened the fixed benchmark metrics and did not reduce the benchmark-visible sparse-tail residuals.

## Benchmark Metrics

| metric | v13 | v15 | delta_v15_minus_v13 | relative_change | v15_improved |
| --- | --- | --- | --- | --- | --- |
| selection_weighted_score | 0.012279222159309057 | 0.013165120231670824 | 0.0008858980723617672 | 0.07214610672143879 | False |
| true_hard_target_normalized_mae | 0.006564072240144015 | 0.0071319350972771645 | 0.0005678628571331498 | 0.08651075679214203 | False |
| validation_normalized_mae | 0.005084018688648939 | 0.005514475051313639 | 0.00043045636266469955 | 0.08466852484743558 | False |
| blind_normalized_mae | 0.005512593314051628 | 0.005967774894088507 | 0.0004551815800368786 | 0.08257122448641703 | False |
| stress_normalized_mae | 0.005654443055391312 | 0.006147522013634443 | 0.0004930789582431316 | 0.08720203801026845 | False |
| B2_magnitude_mae | 0.02138221636414528 | 0.023369701579213142 | 0.0019874852150678635 | 0.09295038368429259 | False |
| S3_magnitude_mae | 1.063392162322998 | 1.1215183734893799 | 0.058126211166381836 | 0.05466112430187951 | False |
| A3_magnitude_mae | 0.5508114099502563 | 0.5959453582763672 | 0.04513394832611084 | 0.08194083766381469 | False |
| S3_high_magnitude_mae | 0.8476534485816956 | 0.9030768275260925 | 0.05542337894439697 | 0.06538447880691345 | False |
| S3_high_magnitude_bias | -0.5059384703636169 | -0.4866616129875183 | 0.019276857376098633 | 0.038101189186591 | True |
| S3_high_magnitude_slope | 0.9721779204327441 | 0.9664945270450565 | -0.005683393387687641 | -0.005846042445767335 | False |
| S3_high_mean_abs_angle_error_deg | 0.3600972592830658 | 0.3743816614151001 | 0.014284402132034302 | 0.039668177870816834 | False |

## Residual-vs-NN

All blind/stress median weighted error changed from 0.0029412631411105394 to 0.003302882192656398.

Top 5% residual median weighted error changed from 0.02706417441368103 to 0.02843543328344822.

Top 5% median NN distance changed from 0.5768005847930908 to 0.5650923848152161.

Interpretation: the new rows moved some benchmark points slightly closer in 12D space, but the model's errors got worse. That is consistent with distribution/optimization damage, or the active-hole mixture pulling capacity away from the fixed benchmark, rather than a clean repair.

## Top-Residual Persistence

- v13 top residual rows: 2774
- v15 top residual rows: 2774
- persistent top residual rows: 2043 (0.736 of v13 top residuals)
- median error delta among persistent top residuals: 0.0014481619000434875
- fraction of persistent top residuals worse under v15: 0.6064610866372981

Limitation: only top-residual CSVs were pushed. Rows that dropped out of v15's top-residual list may have improved, but their exact v15 errors are not available locally without the full residual CSV or a rerun.

## Sampling

- sampling recommendation: PASS
- warnings: []
- new rows: 250000
- far-NN fraction: 0.75
- bridge/anchor fraction: 0.25
- benchmark leakage pass: True

## Original Active-Hole Source

- active top failure rows used to design v15: 3988
- active failure clusters: 62

The current artifacts show what v13 failed on and what v15 added, but they do not include a held-out post-v15 active-hole retest. That should be the next small inference-only job.

## Next Step

Run an inference-only v13-vs-v15 retest on the saved active-hole probe designs, or save full blind/stress residual CSVs, before deciding v16 sampling.
