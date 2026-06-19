# v15 Active-Hole Decision

Created UTC: 2026-06-19T10:24:12.007093+00:00

Recommendation: **do_not_promote_v15; keep v13 seed23 as champion and use v15 as coverage-diagnostic evidence**

| Metric | v13 | v15 | Delta | v15 improved? |
|---|---:|---:|---:|---|
| weighted_score | 0.012279222159309057 | 0.013165120231670824 | 0.0008858980723617672 | False |
| true_hard_target_normalized_mae | 0.006564072240144015 | 0.0071319350972771645 | 0.0005678628571331498 | False |
| overall_normalized_mae | 0.005976789630949497 | 0.0068692900240421295 | 0.0008925003930926323 | False |
| overall_normalized_p95 | None | 0.02279699221253395 | None | None |
| blind_normalized_mae | 0.005512593314051628 | 0.005967774894088507 | 0.0004551815800368786 | False |
| stress_normalized_mae | 0.005654443055391312 | 0.006147522013634443 | 0.0004930789582431316 | False |
| S3_high_magnitude_mae | 0.8476534485816956 | 0.9030768275260925 | 0.05542337894439697 | False |
| S3_high_magnitude_bias | -0.5059384703636169 | -0.4866616129875183 | 0.019276857376098633 | False |
| S3_high_magnitude_slope | 0.9721779204327441 | 0.9664945270450565 | -0.005683393387687641 | False |
| B2_magnitude_mae | 0.02138221636414528 | 0.023369701579213142 | 0.0019874852150678635 | False |
| A3_magnitude_mae | 0.5508114099502563 | 0.5959453582763672 | 0.04513394832611084 | False |

## Residual-vs-NN

Decision interpretation: coverage_limited
Pearson correlation: 0.7363559893790158
Spearman correlation: 0.775234809377701
Top-5%/all median NN-distance ratio: 7.2089940987229

## Next Step

Do not add another blind expansion immediately. Review v15 active-hole residual clusters; if the same sparse subspaces remain, design a more selective v16 hole-fill or revise features/loss for dense failures.
