# V15 Retest On V13 Top Active-Hole Failures

Created UTC: `2026-06-20T09:58:43.913614+00:00`

Purpose: compare v13 and v15 on the exact rows selected as v13 active-hole top failures.

- active top-failure rows requested: `3988`
- matched retest rows: `3988`
- missing retest rows: `0`
- source retest dir: `training_results/model_selection_reports/v15_active_hole_retest_20260620_082522_utc`

| metric | unit | v13 RMSE | v15 RMSE | relative change | worse fraction |
|---|---:|---:|---:|---:|---:|
| weighted normalized abs |  | 0.1425484620539009 | 0.0763881207331423 | -0.4641252551412433 | 0.24573721163490472 |
| overall mixed-unit abs |  | 9.300506374551514 | 5.027303291431038 | -0.45945918544962416 | 0.24222668004012035 |
| C1 | nm | 10.933351382790441 | 6.824674608354882 | -0.3757929870344047 | 0.3415245737211635 |
| C3 | mm | 0.0366158578501019 | 0.02901354076088122 | -0.20762362363168077 | 0.3284854563691073 |
| A1 vector | nm | 17.606922463096947 | 9.092224831459784 | -0.48359942797973116 | 0.2815947843530592 |
| B2 vector | um | 0.6694771844689568 | 0.34289278425590364 | -0.48782005987568744 | 0.35255767301905716 |
| A2 vector | um | 0.652678169077467 | 0.49364572421020386 | -0.24366135164601688 | 0.365346038114343 |
| S3 vector | um | 35.67527597592862 | 23.391990306189285 | -0.34430807705670746 | 0.28309929789368105 |
| A3 vector | um | 30.72982586982065 | 13.130852785235097 | -0.5727000588659132 | 0.309679037111334 |

Interpretation:

- Negative relative change means v15 improved the v13 top-failure subset.
- This table is a harder population than the full active-hole probe set.
- Promotion still requires broad benchmark and anchor gates, not only this repair score.
