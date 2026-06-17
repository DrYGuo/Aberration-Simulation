#!/usr/bin/env bash
set -euo pipefail

DONE_MARKER="colab_worker_logs/v13_1m_seed23_checkpoint_rebuild_done.json"
PENDING_MARKER="colab_worker_logs/v13_1m_seed23_checkpoint_rebuild_pending.json"
DRIVE_BACKUP_RUN_NAME_FILE="colab_worker_logs/v13_seed23_checkpoint_drive_backup_run_name.txt"
DRIVE_BACKUP_ROOT="${ABERRATION_DRIVE_BACKUP_ROOT:-/content/drive/MyDrive/Aberration-Simulation-Colab-Backups}"
DRIVE_RESTORE_RETRIES="${ABERRATION_DRIVE_RESTORE_RETRIES:-3}"

mkdir -p colab_worker_logs

if [ -f "$DONE_MARKER" ]; then
  echo "v13 seed23 checkpoint rebuild already completed; marker exists at $DONE_MARKER"
  exit 0
fi

if [ -f "$DRIVE_BACKUP_RUN_NAME_FILE" ]; then
  DRIVE_BACKUP_RUN_NAME=$(cat "$DRIVE_BACKUP_RUN_NAME_FILE")
else
  DRIVE_BACKUP_RUN_NAME="v13_seed23_checkpoint_rebuild_$(date -u +%Y%m%d_%H%M%S_utc)"
  printf '%s\n' "$DRIVE_BACKUP_RUN_NAME" > "$DRIVE_BACKUP_RUN_NAME_FILE"
fi

restore_csv_folder_from_drive() {
  local label="$1"
  local local_glob="$2"
  local drive_glob="$3"
  local dest_parent="$4"
  local found
  found=$(ls -td $local_glob 2>/dev/null | head -1 || true)
  if [ -n "$found" ] && [ -f "$found" ]; then
    printf '%s\n' "$found"
    return 0
  fi

  echo "Missing cached $label CSV in the live runtime; trying Drive restore." >&2
  local drive_found
  drive_found=$(ls -td $drive_glob 2>/dev/null | head -1 || true)
  if [ -z "$drive_found" ]; then
    echo "Could not find $label CSV locally or in Drive backup root: $DRIVE_BACKUP_ROOT" >&2
    return 1
  fi

  mkdir -p "$dest_parent"
  local source_dir
  source_dir=$(dirname "$drive_found")
  local dest_dir
  dest_dir="$dest_parent/$(basename "$source_dir")"
  mkdir -p "$dest_dir"
  echo "Restoring $label CSV folder from Drive: $source_dir" >&2

  local required_files=(
    "training_features_enhanced.csv"
    "feature_columns_enhanced.json"
    "dataset_manifest.json"
    "dataset_recovery_manifest.json"
    "targeted25k_audit.json"
    "label_summary.csv"
    "new_targeted_label_summary.csv"
  )
  local name
  for name in "${required_files[@]}"; do
    if [ ! -f "$source_dir/$name" ]; then
      continue
    fi
    local attempt
    local copied="false"
    for attempt in $(seq 1 "$DRIVE_RESTORE_RETRIES"); do
      echo "Restoring $label file $name, attempt $attempt/$DRIVE_RESTORE_RETRIES" >&2
      rm -f "$dest_dir/$name"
      if command -v rsync >/dev/null 2>&1; then
        rsync -a --inplace --info=progress2 "$source_dir/$name" "$dest_dir/$name" >&2 && copied="true" || copied="false"
      else
        cp "$source_dir/$name" "$dest_dir/$name" && copied="true" || copied="false"
      fi
      if [ "$copied" = "true" ]; then
        local source_size
        local dest_size
        source_size=$(stat -c%s "$source_dir/$name" 2>/dev/null || stat -f%z "$source_dir/$name" 2>/dev/null || echo 0)
        dest_size=$(stat -c%s "$dest_dir/$name" 2>/dev/null || stat -f%z "$dest_dir/$name" 2>/dev/null || echo 0)
        if [ "$source_size" -eq "$dest_size" ] && [ "$dest_size" -gt 0 ]; then
          break
        fi
        echo "Restored $name size mismatch: source=$source_size bytes, dest=$dest_size bytes." >&2
        copied="false"
      fi
      sleep 10
    done
    if [ "$copied" != "true" ]; then
      echo "Failed to restore required $label file after retries: $source_dir/$name" >&2
      return 1
    fi
  done

  found=$(ls -td $local_glob 2>/dev/null | head -1 || true)
  if [ -z "$found" ]; then
    return 1
  fi
  printf '%s\n' "$found"
}

restore_file_from_drive() {
  local label="$1"
  local local_path="$2"
  local drive_glob="$3"
  if [ -f "$local_path" ]; then
    return 0
  fi
  local found
  found=$(ls -td $drive_glob 2>/dev/null | head -1 || true)
  if [ -z "$found" ]; then
    return 1
  fi
  echo "Restoring $label from Drive: $found"
  mkdir -p "$(dirname "$local_path")"
  cp "$found" "$local_path"
}

V2_SPLIT_MANIFEST="configs/benchmark_split_v12_v2_row_keys.json"
if [ ! -f "$V2_SPLIT_MANIFEST" ]; then
  if ! restore_file_from_drive "benchmark-v2 split manifest" "$V2_SPLIT_MANIFEST" "$DRIVE_BACKUP_ROOT/*/configs/benchmark_split_v12_v2_row_keys.json"; then
    echo "Missing frozen benchmark-v2 split manifest. Mount Drive first or restore $V2_SPLIT_MANIFEST."
    exit 1
  fi
fi

V13_CSV=$(restore_csv_folder_from_drive \
  "v13 1M" \
  "training_results/feature_regression_enhanced/enhanced_v13_1m_spacefill_*/training_features_enhanced.csv" \
  "$DRIVE_BACKUP_ROOT/*/training_results/feature_regression_enhanced/enhanced_v13_1m_spacefill_*/training_features_enhanced.csv" \
  "training_results/feature_regression_enhanced")
if [ -z "$V13_CSV" ]; then
  echo "Mount Google Drive first, or restore the v13 1M CSV folder before running this checkpoint rebuild."
  exit 1
fi

echo "v13 csv: $V13_CSV"
echo "benchmark-v2 split manifest: $V2_SPLIT_MANIFEST"
echo "drive backup run name: $DRIVE_BACKUP_RUN_NAME"

BASELINE_METRICS=""
for candidate in \
  training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v13_1m_d66_seed23_residual_nn_*/metrics_model_loop.json \
  training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v13_1m_d66_seed7_*/metrics_model_loop.json \
  training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v12benchmarkv2_500k_seed7_*/metrics_model_loop.json
do
  found=$(ls -td $candidate 2>/dev/null | head -1 || true)
  if [ -n "$found" ]; then
    BASELINE_METRICS="$found"
    break
  fi
done

BASELINE_ARGS=()
if [ -n "$BASELINE_METRICS" ]; then
  BASELINE_ARGS=(--baseline-metrics "$BASELINE_METRICS")
  echo "baseline metrics: $BASELINE_METRICS"
else
  echo "No baseline metrics visible; checkpoint rebuild will still train and save the model."
fi

python3 - "$PENDING_MARKER" "$V13_CSV" "$V2_SPLIT_MANIFEST" "$DRIVE_BACKUP_RUN_NAME" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "status": "running",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "workflow": "scripts/run_colab_v13_seed23_checkpoint_rebuild_workflow.sh",
            "dataset_csv": sys.argv[2],
            "benchmark_v2_split_manifest": sys.argv[3],
            "drive_backup_run_name": sys.argv[4],
            "purpose": "Retrain the exact v13 1M seed23 D66 champion configuration with --save-model so active 12D inference can use a frozen checkpoint.",
        },
        indent=2,
    )
    + "\n"
)
PY

python3 scripts/run_model_selection_candidate.py \
  --family enhanced \
  --candidate-id D66_grouped_width320_lr6e-4_dropout0.075_v13_1m_d66_seed23_checkpoint_rebuild \
  --output-root training_results/model_selection_loop \
  --architecture grouped_heads \
  --hidden-dim 320 \
  --dropout 0.075 \
  --learning-rate 0.0006 \
  --weight-decay 0.0001 \
  --residual-penalty 0.003 \
  --max-epochs 2000 \
  --eval-every 10 \
  --patience-epochs 300 \
  --easy-regression-limit 0.1 \
  --torch-seed 23 \
  --batch-size 65536 \
  --eval-batch-size 65536 \
  --predict-batch-size 65536 \
  --component-loss-kind smooth_l1 \
  --component-smooth-l1-beta 0.25 \
  --grad-clip-norm 1.0 \
  --lr-scheduler plateau \
  --lr-plateau-factor 0.5 \
  --lr-plateau-patience-evals 8 \
  --min-learning-rate 1e-05 \
  --shuffle-batches \
  --csv-path "$V13_CSV" \
  --selection-config experiments/model_selection_weights.json \
  --benchmark-split-manifest "$V2_SPLIT_MANIFEST" \
  --split-seed 7 \
  --save-model \
  "${BASELINE_ARGS[@]}"

RUN_DIR=$(ls -td training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v13_1m_d66_seed23_checkpoint_rebuild_* 2>/dev/null | head -1 || true)
if [ -z "$RUN_DIR" ] || [ ! -f "$RUN_DIR/model_loop_candidate.pt" ]; then
  echo "Checkpoint rebuild did not produce model_loop_candidate.pt."
  exit 1
fi

CHECKPOINT_PATH="$RUN_DIR/model_loop_candidate.pt"
CHECKPOINT_BYTES=$(stat -c%s "$CHECKPOINT_PATH" 2>/dev/null || stat -f%z "$CHECKPOINT_PATH" 2>/dev/null || echo 0)
echo "checkpoint path: $CHECKPOINT_PATH"
echo "checkpoint bytes: $CHECKPOINT_BYTES"

echo "Backing up checkpoint rebuild, including the .pt, to Google Drive."
python3 scripts/backup_colab_state_to_drive.py --run-name "$DRIVE_BACKUP_RUN_NAME"

DRIVE_CHECKPOINT_PATH="$DRIVE_BACKUP_ROOT/$DRIVE_BACKUP_RUN_NAME/$CHECKPOINT_PATH"

python3 - "$DONE_MARKER" "$PENDING_MARKER" "$V13_CSV" "$V2_SPLIT_MANIFEST" "$RUN_DIR" "$CHECKPOINT_PATH" "$CHECKPOINT_BYTES" "$DRIVE_BACKUP_RUN_NAME" "$DRIVE_CHECKPOINT_PATH" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

marker = Path(sys.argv[1])
pending = Path(sys.argv[2])
checkpoint_bytes = int(sys.argv[7])
marker.write_text(
    json.dumps(
        {
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "workflow": "scripts/run_colab_v13_seed23_checkpoint_rebuild_workflow.sh",
            "dataset_csv": sys.argv[3],
            "benchmark_v2_split_manifest": sys.argv[4],
            "run_dir": sys.argv[5],
            "checkpoint_path": sys.argv[6],
            "checkpoint_bytes": checkpoint_bytes,
            "drive_backup_run_name": sys.argv[8],
            "drive_checkpoint_path": sys.argv[9],
            "github_artifact_policy": "The .pt is intentionally blocked from GitHub and saved through the Drive backup.",
            "next_use": "Use this checkpoint for active 12D hole-search inference without retraining.",
        },
        indent=2,
    )
    + "\n"
)
if pending.exists():
    pending.unlink()
print("done marker:", marker)
print("checkpoint:", sys.argv[6])
print("drive checkpoint:", sys.argv[9])
PY
