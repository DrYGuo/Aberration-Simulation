# Feature Regression Results

This folder is the default output location for temporary hybrid-regression
training results from `notebooks/uno_feature_regression.ipynb`.

The first regression notebook generates training coefficient combinations,
runs the GPU probe simulation, extracts Uno feature values, and then trains
from the generated `training_features.csv`. The current generator keeps the
deterministic anchor sweeps and adds 10,000 random coupled coefficient cases.
The feature/target interface is table-based so later large combinatorial
datasets can be saved as shards and loaded with a streaming data loader without
changing the feature definitions.

Generated files may include:

- `metrics.json`
- `training_features.csv`
- `normalization.json`
- `model_summary.txt`
- `history.csv`
- `predictions.csv`
- `hybrid_feature_regressor.pt`
- `training_history.png`
- `prediction_scatter.png`
- `feature_regression_results.zip`

Colab can also mirror this folder to Google Drive when Drive mounting is
enabled in the notebook.
