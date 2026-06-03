# Feature Regression Results

This folder is the default output location for temporary hybrid-regression
training results from `notebooks/uno_feature_regression.ipynb`.

The first regression notebook trains from the generated relationship CSV in
memory. The feature/target interface is table-based so later large
combinatorial datasets can be saved as shards and loaded with a streaming data
loader without changing the feature definitions.

Generated files may include:

- `metrics.json`
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
