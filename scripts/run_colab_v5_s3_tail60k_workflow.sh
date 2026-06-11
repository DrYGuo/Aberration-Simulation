#!/usr/bin/env bash
set -euo pipefail

V3_CSV=$(ls -td training_results/feature_regression_enhanced/enhanced_v3_targeted25k_*/training_features_enhanced.csv 2>/dev/null | head -1 || true)
if [ -z "$V3_CSV" ]; then
  V2_CSV=$(ls -td training_results/feature_regression_enhanced/enhanced_v2_coupled16k_stratified_dropout_*/training_features_enhanced.csv 2>/dev/null | head -1 || true)
  if [ -z "$V2_CSV" ]; then
    python3 scripts/run_notebook_headless.py notebooks/uno_feature_regression_enhanced_dataset_bootstrap.ipynb --output-dir colab_worker_logs --timeout 3600
    V2_CSV=$(ls -td training_results/feature_regression_enhanced/enhanced_v2_coupled16k_stratified_dropout_*/training_features_enhanced.csv | head -1)
  fi
  python3 scripts/generate_targeted_enhanced_dataset.py \
    --parent-csv "$V2_CSV" \
    --run-prefix enhanced_v3_targeted25k \
    --dataset-version enhanced_v3_targeted25k \
    --seed 31 \
    --batch-base-cases 256
  V3_CSV=$(ls -td training_results/feature_regression_enhanced/enhanced_v3_targeted25k_*/training_features_enhanced.csv | head -1)
fi

V5_CSV=$(ls -td training_results/feature_regression_enhanced/enhanced_v5_s3_tail60k_*/training_features_enhanced.csv 2>/dev/null | head -1 || true)
if [ -z "$V5_CSV" ]; then
  python3 scripts/audit_s3_feature_saturation.py \
    --csv-path "$V3_CSV" \
    --output-root training_results/model_selection_reports \
    --vector-scale 90.73979949951172 \
    --high-bin-fraction 0.7
  python3 scripts/generate_targeted_enhanced_dataset.py \
    --parent-csv "$V3_CSV" \
    --run-prefix enhanced_v5_s3_tail60k \
    --dataset-version enhanced_v5_s3_tail60k \
    --case-counts-json configs/targeted_expansion_s3_tail60k.json \
    --seed 53 \
    --batch-base-cases 256
  V5_CSV=$(ls -td training_results/feature_regression_enhanced/enhanced_v5_s3_tail60k_*/training_features_enhanced.csv | head -1)
  python3 scripts/audit_s3_feature_saturation.py \
    --csv-path "$V5_CSV" \
    --output-root training_results/model_selection_reports \
    --vector-scale 90.73979949951172 \
    --high-bin-fraction 0.7
fi

SEED7_DONE=$(ls training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_s3tail60k_plateau_clip_smoothl1_seed7_*/selection_score.json 2>/dev/null | head -1 || true)
SEED23_DONE=$(ls training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_s3tail60k_plateau_clip_smoothl1_seed23_*/selection_score.json 2>/dev/null | head -1 || true)
if [ -n "$SEED7_DONE" ] && [ -n "$SEED23_DONE" ]; then
  echo "v5 S3-tail seed7 and seed23 candidates already complete; no duplicate batch summary will be written."
  exit 0
fi

python3 scripts/run_model_selection_batch.py \
  --batch-config configs/model_selection_batch_v5_s3_tail60k.json \
  --output-root training_results/model_selection_loop \
  --summary-root training_results/model_selection_batches \
  --max-runtime-minutes 120
