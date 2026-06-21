# Generalization Benchmark v1 Evaluation

Created UTC: `2026-06-21T07:39:59.392246+00:00`

This run simulates/extracts features for frozen benchmark rows and evaluates saved v13/v15 checkpoints. It does not train.

## Overall Score

| model | populated score | promotable | notes |
|---|---:|---|---|
| `v13_1m_seed23` | 0.02344943569380438 | False | retains lower populated held-out benchmark score |
| `v15_active_hole_250k_seed23` | 0.02385107762764163 | False | benchmark populated; v16 decision requires review of broad/new-hole/anchor tradeoffs |

## Component Scores

| component | model | rows | weighted RMSE | weighted MAE | weighted p95 | S3 vector RMSE | A3 vector RMSE | B2 vector RMSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `new_hole_challenge` | `v13_1m_seed23` | 5000 | 0.043672474093203385 | 0.035959490722371264 | 0.0818874578922987 | 12.852086636670858 | 7.190044278289192 | 0.26054307926428416 |
| `broad_12d_representative_validation` | `v13_1m_seed23` | 100000 | 0.014029513638435673 | 0.009277901400147821 | 0.02874444015324113 | 4.71347625394852 | 2.0739049359936694 | 0.08093330943972576 |
| `anchor_regression_guard` | `v13_1m_seed23` | 4827 | 0.0018432030057437781 | 0.0013517540174375108 | 0.0040641200263053165 | 0.3340649517256729 | 0.16887870912459726 | 0.011302179439172957 |
| `new_hole_challenge` | `v15_active_hole_250k_seed23` | 5000 | 0.04394124349073588 | 0.036750052253203463 | 0.08190348334610463 | 12.94612256744215 | 7.823566717798725 | 0.26840909342942026 |
| `broad_12d_representative_validation` | `v15_active_hole_250k_seed23` | 100000 | 0.014567024191486469 | 0.009785487434244716 | 0.029656453989446125 | 4.84170655130466 | 2.1250717157003454 | 0.08480306866024503 |
| `anchor_regression_guard` | `v15_active_hole_250k_seed23` | 4827 | 0.0022388527737634612 | 0.0017153795450456596 | 0.004901586100459098 | 0.38226446667624586 | 0.2585102807596373 | 0.012323143237208699 |

## Interpretation

- Lower populated score is better.
- `new_hole_challenge`, `broad_12d_representative_validation`, and `anchor_regression_guard` are held out and must not be used for training.
- v16 sampling/training should only be designed after this score is reviewed.
