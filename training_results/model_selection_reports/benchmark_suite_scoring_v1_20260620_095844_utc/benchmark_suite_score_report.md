# Benchmark-Suite Score v1

Created UTC: `2026-06-20T09:58:44.160532+00:00`

new_hole_challenge_score is unavailable and excluded from available_suite_score normalization.

| model | available score | promotable now | gate failures | broad regression | active-hole change |
|---|---:|---|---|---:|---:|
| v13 | 0.03580822013121358 | True |  | 0.0 | 0.0 |
| v15 | 0.024439110976850743 | False | broad_blind_stress_regression;new_hole_challenge_missing_for_final_promotion | 0.06884830714710943 | -0.39906888704530913 |

Decision:

- Lower score is better.
- v15 repairs active holes, but broad benchmark regression exceeds the configured 5% gate.
- New-hole challenge is not frozen yet, so this suite is diagnostic rather than final.
- Current promoted model remains v13 until a v16 candidate passes all gates.
