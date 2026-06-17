#!/usr/bin/env bash
set -euo pipefail

DONE_MARKER="colab_worker_logs/active_12d_hole_search_v1_inference_repair_done.json"
PENDING_MARKER="colab_worker_logs/active_12d_hole_search_v1_inference_repair_pending.json"
CONFIG="configs/active_12d_hole_search_v1.json"
DRIVE_BACKUP_RUN_NAME_FILE="colab_worker_logs/active_12d_hole_search_v1_drive_backup_run_name.txt"
DRIVE_BACKUP_ROOT="${ABERRATION_DRIVE_BACKUP_ROOT:-/content/drive/MyDrive/Aberration-Simulation-Colab-Backups}"
ALLOW_CHECKPOINT_REBUILD="${ALLOW_CHECKPOINT_REBUILD:-0}"

if [ -f "$DONE_MARKER" ]; then
  echo "active 12D hole-search inference repair already completed; marker exists at $DONE_MARKER"
  exit 0
fi

mkdir -p colab_worker_logs training_results/model_selection_reports

if [ -f "$DRIVE_BACKUP_RUN_NAME_FILE" ]; then
  DRIVE_BACKUP_RUN_NAME=$(cat "$DRIVE_BACKUP_RUN_NAME_FILE")
else
  DRIVE_BACKUP_RUN_NAME="active_12d_hole_search_v1_$(date -u +%Y%m%d_%H%M%S_utc)"
  printf '%s\n' "$DRIVE_BACKUP_RUN_NAME" > "$DRIVE_BACKUP_RUN_NAME_FILE"
fi

restore_csv_folder_from_drive() {
  local label="$1"
  local local_glob="$2"
  local drive_glob="$3"
  local dest_parent="$4"
  local found
  found=$(ls -td $local_glob 2>/dev/null | head -1 || true)
  if [ -n "$found" ]; then
    printf '%s\n' "$found"
    return 0
  fi
  local drive_found
  drive_found=$(ls -td $drive_glob 2>/dev/null | head -1 || true)
  if [ -z "$drive_found" ]; then
    echo "Could not find $label locally or in Drive backup root: $DRIVE_BACKUP_ROOT" >&2
    return 1
  fi
  mkdir -p "$dest_parent"
  local source_dir
  source_dir=$(dirname "$drive_found")
  local dest_dir
  dest_dir="$dest_parent/$(basename "$source_dir")"
  mkdir -p "$dest_dir"
  echo "Restoring $label CSV folder from Drive: $source_dir" >&2
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --info=progress2 "$source_dir/" "$dest_dir/" >&2
  else
    cp -R "$source_dir/." "$dest_dir/"
  fi
  found=$(ls -td $local_glob 2>/dev/null | head -1 || true)
  if [ -z "$found" ]; then
    return 1
  fi
  printf '%s\n' "$found"
}

restore_file_from_drive() {
  local label="$1"
  local local_glob="$2"
  local drive_glob="$3"
  local dest_dir="$4"
  local found
  found=$(ls -td $local_glob 2>/dev/null | head -1 || true)
  if [ -n "$found" ]; then
    printf '%s\n' "$found"
    return 0
  fi
  local drive_found
  drive_found=$(ls -td $drive_glob 2>/dev/null | head -1 || true)
  if [ -z "$drive_found" ]; then
    echo "Could not find $label locally or in Drive backup root: $DRIVE_BACKUP_ROOT" >&2
    return 1
  fi
  mkdir -p "$dest_dir"
  local dest_file="$dest_dir/$(basename "$drive_found")"
  echo "Restoring $label from Drive: $drive_found" >&2
  cp "$drive_found" "$dest_file"
  printf '%s\n' "$dest_file"
}

restore_checkpoint_from_drive() {
  local label="$1"
  local local_run_dir="$2"
  local drive_glob="$3"
  if [ -n "$local_run_dir" ] && [ -f "$local_run_dir/model_loop_candidate.pt" ]; then
    printf '%s\n' "$local_run_dir/model_loop_candidate.pt"
    return 0
  fi
  local found
  found=$(ls -td $drive_glob 2>/dev/null | head -1 || true)
  if [ -z "$found" ]; then
    return 1
  fi
  if [ -z "$local_run_dir" ]; then
    local_run_dir="training_results/model_selection_loop/$(basename "$(dirname "$found")")"
  fi
  mkdir -p "$local_run_dir"
  echo "Restoring $label checkpoint from Drive: $found" >&2
  cp "$found" "$local_run_dir/model_loop_candidate.pt"
  printf '%s\n' "$local_run_dir/model_loop_candidate.pt"
}

restore_manifest_from_drive() {
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
  if ! restore_manifest_from_drive "benchmark-v2 split manifest" "$V2_SPLIT_MANIFEST" "$DRIVE_BACKUP_ROOT/*/configs/benchmark_split_v12_v2_row_keys.json"; then
    echo "Missing frozen benchmark-v2 split manifest. Restore/mount Drive before running active hole-search inference repair."
    exit 1
  fi
fi

V13_CSV=$(restore_csv_folder_from_drive \
  "v13 1M" \
  "training_results/feature_regression_enhanced/enhanced_v13_1m_spacefill_*/training_features_enhanced.csv" \
  "$DRIVE_BACKUP_ROOT/*/training_results/feature_regression_enhanced/enhanced_v13_1m_spacefill_*/training_features_enhanced.csv" \
  "training_results/feature_regression_enhanced")

PROBE_CSV=$(restore_file_from_drive \
  "active hole-search probe features" \
  "training_results/model_selection_reports/v13_active_12d_hole_search_*/active_hole_search_probe_features.csv" \
  "$DRIVE_BACKUP_ROOT/*/training_results/model_selection_reports/v13_active_12d_hole_search_*/active_hole_search_probe_features.csv" \
  "training_results/model_selection_reports/recovered_active_probe_features")

V13_RUN_DIR=$(ls -td training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v13_1m_d66_seed23_residual_nn_* 2>/dev/null | head -1 || true)
if [ -z "$V13_RUN_DIR" ]; then
  echo "Missing v13 seed23 metrics folder. The checkpoint-rebuild run can still proceed from the v13 CSV."
fi

CHECKPOINT_RUN_DIR=$(ls -td training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v13_1m_d66_seed23_checkpoint_rebuild_* 2>/dev/null | head -1 || true)
if [ -z "$CHECKPOINT_RUN_DIR" ] || [ ! -f "$CHECKPOINT_RUN_DIR/model_loop_candidate.pt" ]; then
  if restore_checkpoint_from_drive \
    "v13 seed23 champion" \
    "$V13_RUN_DIR" \
    "$DRIVE_BACKUP_ROOT/*/training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v13_1m_d66_seed23*/model_loop_candidate.pt" >/dev/null; then
    CHECKPOINT_RUN_DIR="$V13_RUN_DIR"
    echo "Using restored v13 seed23 checkpoint from Drive: $CHECKPOINT_RUN_DIR/model_loop_candidate.pt"
  fi
fi

if [ -z "$CHECKPOINT_RUN_DIR" ] || [ ! -f "$CHECKPOINT_RUN_DIR/model_loop_candidate.pt" ]; then
  if [ "$ALLOW_CHECKPOINT_REBUILD" != "1" ]; then
    echo "No v13 seed23 checkpoint was found locally or in Drive. Not rebuilding because ALLOW_CHECKPOINT_REBUILD is not 1."
    python3 - "$PENDING_MARKER" "$V13_CSV" "$PROBE_CSV" "${CHECKPOINT_RUN_DIR:-}" "$DRIVE_BACKUP_RUN_NAME" "$DRIVE_BACKUP_ROOT" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "status": "pending",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "workflow": "scripts/run_colab_active_12d_hole_search_v1_inference_repair_workflow.sh",
    "v13_csv": sys.argv[2],
    "probe_features_csv": sys.argv[3],
    "checkpoint_run_dir": sys.argv[4],
    "drive_backup_run_name": sys.argv[5],
    "drive_backup_root": sys.argv[6],
    "missing_artifact": "model_loop_candidate.pt",
    "note": "The active hole-search inference repair requires the frozen v13 checkpoint. No matching checkpoint was found in the mounted Drive backups, and checkpoint rebuild is disabled unless ALLOW_CHECKPOINT_REBUILD=1.",
}, indent=2) + "\n")
PY
    echo "Wrote pending marker: $PENDING_MARKER"
    exit 0
  fi
  echo "Rebuilding v13 seed23 checkpoint with --save-model. This trains the same frozen configuration and saves a checkpoint for active-probe inference."
  python3 scripts/run_model_selection_candidate.py \
    --family enhanced \
    --candidate-id D66_grouped_width320_lr6e-4_dropout0.075_v13_1m_d66_seed23_checkpoint_rebuild \
    --csv-path "$V13_CSV" \
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
    --min-learning-rate 0.00001 \
    --shuffle-batches \
    --selection-config experiments/model_selection_weights.json \
    --benchmark-split-manifest "$V2_SPLIT_MANIFEST" \
    --split-seed 7 \
    --save-model
  CHECKPOINT_RUN_DIR=$(ls -td training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v13_1m_d66_seed23_checkpoint_rebuild_* | head -1)
fi

if [ ! -f "$CHECKPOINT_RUN_DIR/model_loop_candidate.pt" ]; then
  python3 - "$PENDING_MARKER" "$V13_CSV" "$PROBE_CSV" "$CHECKPOINT_RUN_DIR" "$DRIVE_BACKUP_RUN_NAME" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "status": "pending",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "workflow": "scripts/run_colab_active_12d_hole_search_v1_inference_repair_workflow.sh",
    "v13_csv": sys.argv[2],
    "probe_features_csv": sys.argv[3],
    "checkpoint_run_dir": sys.argv[4],
    "drive_backup_run_name": sys.argv[5],
    "note": "Checkpoint rebuild did not produce model_loop_candidate.pt yet.",
}, indent=2) + "\n")
PY
  exit 0
fi

python3 scripts/run_active_12d_hole_search.py \
  --config "$CONFIG" \
  --v13-csv "$V13_CSV" \
  --v13-run-dir "$CHECKPOINT_RUN_DIR" \
  --benchmark-split-manifest "$V2_SPLIT_MANIFEST" \
  --probe-features-csv "$PROBE_CSV" \
  --output-root training_results/model_selection_reports

OUTPUT_DIR=$(ls -td training_results/model_selection_reports/v13_active_12d_hole_search_* 2>/dev/null | head -1 || true)
if [ -z "$OUTPUT_DIR" ]; then
  echo "Inference repair finished without an output directory."
  exit 1
fi

echo "Backing up active 12D hole-search inference repair state to Drive."
python3 scripts/backup_colab_state_to_drive.py --run-name "$DRIVE_BACKUP_RUN_NAME"

python3 - "$DONE_MARKER" "$V13_CSV" "$PROBE_CSV" "$CHECKPOINT_RUN_DIR" "$OUTPUT_DIR" "$DRIVE_BACKUP_RUN_NAME" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "status": "complete",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "workflow": "scripts/run_colab_active_12d_hole_search_v1_inference_repair_workflow.sh",
    "v13_csv": sys.argv[2],
    "probe_features_csv": sys.argv[3],
    "checkpoint_run_dir": sys.argv[4],
    "output_dir": sys.argv[5],
    "drive_backup_run_name": sys.argv[6],
    "diagnostics": [
        "checkpoint_rebuild_with_save_model",
        "active_probe_inference",
        "active_hole_search_summary",
        "top_failure_table",
        "hole_cluster_table",
        "nn_error_summary",
        "regime_summary",
        "relative_angle_summary",
        "recommended_sampling_plan",
        "drive_backup"
    ],
    "decision": "Inspect active hole-search inference results. If failures are sparse-NN dominated, steer a targeted v15 sampling plan; if dense failures dominate, steer feature/model/loss diagnostics.",
}, indent=2) + "\n")
PY
