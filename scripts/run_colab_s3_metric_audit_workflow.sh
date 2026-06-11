#!/usr/bin/env bash
set -euo pipefail

DONE_MARKER="colab_worker_logs/s3_metric_and_loss_gap_audit_done.json"
if [ -f "$DONE_MARKER" ]; then
  echo "S3 metric and loss-gap audit rerun already completed; marker exists at $DONE_MARKER"
  exit 0
fi

V3_CSV=$(ls -td training_results/feature_regression_enhanced/enhanced_v3_targeted25k_*/training_features_enhanced.csv 2>/dev/null | head -1 || true)
V5_CSV=$(ls -td training_results/feature_regression_enhanced/enhanced_v5_s3_tail60k_*/training_features_enhanced.csv 2>/dev/null | head -1 || true)

if [ -z "$V3_CSV" ] || [ -z "$V5_CSV" ]; then
  echo "Missing v3 or v5 cached CSV; running v5 setup workflow first."
  bash scripts/run_colab_v5_s3_tail60k_workflow.sh
  V3_CSV=$(ls -td training_results/feature_regression_enhanced/enhanced_v3_targeted25k_*/training_features_enhanced.csv | head -1)
  V5_CSV=$(ls -td training_results/feature_regression_enhanced/enhanced_v5_s3_tail60k_*/training_features_enhanced.csv | head -1)
fi

V3_CANDIDATE="D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_plateau_clip_smoothl1_s3metric_audit_seed7"
V5_CANDIDATE="D66_grouped_width320_lr6e-4_dropout0.075_s3tail60k_plateau_clip_smoothl1_s3metric_audit_seed23"

V3_DONE=$(ls training_results/model_selection_loop/${V3_CANDIDATE}_*/s3_magnitude_metric_audit_validation.csv 2>/dev/null | head -1 || true)
if [ -z "$V3_DONE" ]; then
  python3 scripts/run_model_selection_candidate.py \
    --family enhanced \
    --candidate-id "$V3_CANDIDATE" \
    --csv-path "$V3_CSV" \
    --output-root training_results/model_selection_loop \
    --architecture grouped_heads \
    --hidden-dim 320 \
    --dropout 0.075 \
    --learning-rate 0.0006 \
    --weight-decay 0.0001 \
    --residual-penalty 0.003 \
    --max-epochs 6000 \
    --eval-every 25 \
    --patience-epochs 1000 \
    --easy-regression-limit 0.10 \
    --torch-seed 7 \
    --component-loss-kind smooth_l1 \
    --component-smooth-l1-beta 0.25 \
    --grad-clip-norm 1.0 \
    --lr-scheduler plateau \
    --lr-plateau-factor 0.5 \
    --lr-plateau-patience-evals 8 \
    --min-learning-rate 0.00001 \
    --split-seed 7 \
    --baseline-metrics training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_bin_diag_20260609_073514_utc/metrics_model_loop.json \
    --selection-config experiments/model_selection_weights.json
fi

V5_DONE=$(ls training_results/model_selection_loop/${V5_CANDIDATE}_*/s3_magnitude_metric_audit_validation.csv 2>/dev/null | head -1 || true)
if [ -z "$V5_DONE" ]; then
  python3 scripts/run_model_selection_candidate.py \
    --family enhanced \
    --candidate-id "$V5_CANDIDATE" \
    --csv-path "$V5_CSV" \
    --output-root training_results/model_selection_loop \
    --architecture grouped_heads \
    --hidden-dim 320 \
    --dropout 0.075 \
    --learning-rate 0.0006 \
    --weight-decay 0.0001 \
    --residual-penalty 0.003 \
    --max-epochs 6000 \
    --eval-every 25 \
    --patience-epochs 1000 \
    --easy-regression-limit 0.10 \
    --torch-seed 23 \
    --component-loss-kind smooth_l1 \
    --component-smooth-l1-beta 0.25 \
    --grad-clip-norm 1.0 \
    --lr-scheduler plateau \
    --lr-plateau-factor 0.5 \
    --lr-plateau-patience-evals 8 \
    --min-learning-rate 0.00001 \
    --split-seed 7 \
    --baseline-metrics training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_plateau_clip_smoothl1_20260610_071108_utc/metrics_model_loop.json \
    --selection-config experiments/model_selection_weights.json
fi

V3_RUN=$(ls -td training_results/model_selection_loop/${V3_CANDIDATE}_* | head -1)
V5_RUN=$(ls -td training_results/model_selection_loop/${V5_CANDIDATE}_* | head -1)

python3 scripts/audit_s3_magnitude_metric.py \
  --run v3_smoothl1_audit="$V3_RUN" \
  --run v5_s3tail60k_seed23_audit="$V5_RUN"

python3 scripts/audit_training_validation_loss_gap.py \
  --include-default-runs \
  --run "$V3_RUN" \
  --run "$V5_RUN"

mkdir -p colab_worker_logs
python3 - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path

Path("colab_worker_logs").mkdir(exist_ok=True)
Path("colab_worker_logs/s3_metric_and_loss_gap_audit_done.json").write_text(
    json.dumps(
        {
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "workflow": "scripts/run_colab_s3_metric_audit_workflow.sh",
            "diagnostics": [
                "s3_magnitude_metric_audit",
                "training_validation_loss_gap_audit",
            ],
        },
        indent=2,
    )
    + "\n"
)
PY
