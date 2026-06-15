# Data-Scale Learning Curve

Created UTC: `2026-06-15T06:04:07.379013+00:00`

| dataset | best candidate | weighted | hard MAE | blind MAE | stress MAE | high-S3 MAE | high-S3 slope | B2 mag MAE | A3 mag MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v6_100k | `D66_grouped_width320_lr6e-4_dropout0.075_v6gap100k_seed23` | 0.03690 | 0.01850 | 0.01085 | 0.01018 | 6.274 | 0.799 | 0.0601 | 1.401 |
| v9_250k | `D66_grouped_width320_lr6e-4_dropout0.075_v9gap250k_d66_seed23` | 0.03051 | 0.01537 | 0.00969 | 0.00872 | 4.729 | 0.865 | 0.0553 | 1.208 |
| v13_1m | `D66_grouped_width320_lr6e-4_dropout0.075_v13_1m_d66_seed7` | 0.01258 | 0.00674 | 0.00565 | 0.00584 | 0.876 | 0.974 | 0.0220 | 0.559 |

## Incremental Improvements

- `v6_100k` -> `v9_250k`: weighted `17.3%`, hard `16.9%`, blind `10.7%`, stress `14.4%`.
- `v9_250k` -> `v13_1m`: weighted `58.8%`, hard `56.1%`, blind `41.7%`, stress `33.0%`.

Stopping heuristic: if doubling/near-doubling data improves weighted score or hard-regime metrics by less than about 3-5% across two seeds, inspect coverage/residual diagnostics before expanding again.
