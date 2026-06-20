# Active 12D Hole Search v1

Created UTC: `2026-06-17T22:03:01.038954+00:00`

Purpose: freeze the v13 1M seed23 champion and actively probe holes in normalized 12D aberration space.

- selected probes: `3000`
- reference train rows: `992556`
- NN method: `sklearn`
- simulation status: `complete`
- inference status: `skipped`

## Proposal Modes

| mode | selected | median NN distance | median score |
|---|---:|---:|---:|
| `far_nn` | 900 | 0.7367553114891052 | 0.5528350174427032 |
| `sobol_lhs` | 650 | 0.7489132583141327 | 0.5789717137813568 |
| `genetic_algorithm` | 800 | 1.2084516286849976 | 0.6107628643512726 |
| `residual_jitter` | 350 | 0.801971048116684 | 0.5674217045307159 |
| `bridge_anchor` | 300 | 0.2847375273704529 | 0.06439803913235664 |
