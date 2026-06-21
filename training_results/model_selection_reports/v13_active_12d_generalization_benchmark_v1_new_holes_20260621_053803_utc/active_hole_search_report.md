# Active 12D Hole Search v1

Created UTC: `2026-06-21T05:42:16.882762+00:00`

Purpose: freeze the v13 1M seed23 champion and actively probe holes in normalized 12D aberration space.

- selected probes: `5000`
- reference train rows: `992556`
- NN method: `sklearn`
- simulation status: `skipped`
- inference status: `skipped`

## Proposal Modes

| mode | selected | median NN distance | median score |
|---|---:|---:|---:|
| `genetic_algorithm` | 1200 | 1.1524163484573364 | 0.5445254445075989 |
| `far_nn` | 950 | 0.7812080085277557 | 0.5186393558979034 |
| `residual_jitter` | 850 | 0.7682390809059143 | 0.5073426961898804 |
| `high_amp_oblique` | 750 | 0.9609382152557373 | 0.5407376289367676 |
| `high_amp_bridge` | 500 | 0.8413056135177612 | 0.2003193497657776 |
| `bridge_anchor` | 500 | 0.5330030620098114 | 0.1101941205561161 |
| `sobol_lhs` | 250 | 0.8464331030845642 | 0.5479386150836945 |
