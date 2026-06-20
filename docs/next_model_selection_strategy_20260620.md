# Next Model-Selection Strategy After v15 Active-Hole Retest

Date: 2026-06-20

## Current Conclusion

The v13 1M seed23 D66 grouped-head model remains the general baseline. The v15
active-hole-expanded model is not promoted because it worsened the old fixed
validation/blind/stress benchmark.

However, the v15 active-hole retest shows that the active-hole data did repair
searched failure regions:

| active-hole metric | v13 RMSE | v15 RMSE | change |
|---|---:|---:|---:|
| weighted normalized abs error | 0.05791 | 0.04434 | 23.4% better |
| overall mixed-unit abs error | 3.725 | 2.839 | 23.8% better |
| S3 vector error | 15.84 um | 13.49 um | 14.9% better |
| A3 vector error | 11.42 um | 7.27 um | 36.3% better |
| B2 vector error | 0.312 um | 0.250 um | 20.1% better |

This means active-hole expansion is scientifically useful, but pure active-hole
targeting can distort the training distribution enough to hurt general
benchmarks.

## Evaluation Of The Proposed Direction

The old fixed stress/hard tests are not enough. They are fixed finite samples
from a narrow set of regimes, so they can miss uncovered subspaces in the full
12D coefficient domain. The active 12D search found holes that were not
adequately represented by the previous stress benchmark.

The correct next step is not to replace the old stress benchmark entirely.
Instead, model selection should use a benchmark suite:

1. Broad representative benchmark:
   - fixed, large, non-training validation/blind/stress set;
   - sampled with space-filling/Sobol/LHS plus physically weighted coupled
     regimes;
   - intended to estimate generalization over the expected 12D use domain.

2. Active-hole benchmark:
   - frozen probe set from active search;
   - includes v13 discovered sparse and dense failures;
   - used as a hard diagnostic and weighted selection component.

3. New-hole challenge benchmark:
   - newly proposed holes not used in training;
   - prevents overfitting to the previous active-hole set.

4. Anchor/easy benchmark:
   - one-coefficient and low-coupling Uno-success regimes;
   - guards against damaging known-good regimes.

The training expansion should include holes, but not only holes. The v15 result
shows why: holes improved, benchmark balance worsened.

## Proposed v16 Direction

Use a balanced v16 dataset rather than a pure active-hole expansion.

Suggested new-row mix:

| component | fraction | purpose |
|---|---:|---|
| active-hole repair rows from known clusters | 35% | repair v13/v15 known sparse failures |
| new active-search candidate holes | 20% | explore uncovered nearby 12D subspaces |
| broad coupled-full/coupled-sparse Sobol/LHS rows | 25% | maintain full-space generalization |
| benchmark-preserving anchors | 15% | prevent regression on old validation/blind/stress-like distributions |
| one-coefficient and Uno-success sweeps | 5% | protect interpretable known regimes |

Do not promote a v16 model unless it improves active-hole metrics without
regressing the broad blind/stress benchmark beyond a configured tolerance.

## New Model-Selection Metric

Use a weighted suite score, not a single old validation score:

```text
score =
  0.35 * broad_blind_stress_score
  0.25 * active_hole_score
  0.15 * new_hole_challenge_score
  0.15 * hard_vector_score
  0.10 * anchor_regression_penalty
```

Where:

- `broad_blind_stress_score`: normalized MAE/p95 on broad validation/blind/stress.
- `active_hole_score`: weighted normalized RMSE/MAE on frozen active-hole probes.
- `new_hole_challenge_score`: same metrics on newly searched holes held out from
  training.
- `hard_vector_score`: S3/A3/B2 vector magnitude/angle-bin diagnostics.
- `anchor_regression_penalty`: penalty if C3/B2/A2/easy sweeps or Uno-success
  regimes regress beyond 5-10%.

Gates:

- Reject if broad blind/stress worsens by more than 5%.
- Reject if anchor/easy targets worsen by more than 5-10%.
- Reject if active-hole score does not improve over v13.
- Promote only if active-hole gains survive a held-out new-hole challenge set.

## Immediate Next Steps

1. Generate a compact v15-on-v13-top-3988-failure report from the full Drive
   retest artifacts.
2. Freeze a benchmark suite definition:
   - broad representative benchmark;
   - active-hole benchmark;
   - new-hole challenge benchmark;
   - anchors.
3. Implement the suite score JSON/config.
4. Prepare v16 balanced expansion using the mix above.
5. Train one controlled v16 candidate with the same architecture and seed before
   architecture changes.
6. Compare v13, v15, and v16 on the same suite.

## Artifact Policy

Keep large CSVs, checkpoints, raw predictions, and full probe-feature tables in
Google Drive. Push only compact benchmark-suite definitions, metrics JSON,
summary CSVs, reports, and small plots to GitHub.
