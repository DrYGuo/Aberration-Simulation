#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/active_12d_hole_search_v1.json"
WORKFLOW_ID=$(python3 - "$CONFIG" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text())
print(config.get("workflow_id", "active_12d_hole_search"))
PY
)
DONE_MARKER="colab_worker_logs/${WORKFLOW_ID}_done.json"
PENDING_MARKER="colab_worker_logs/${WORKFLOW_ID}_pending.json"
DRIVE_BACKUP_RUN_NAME_FILE="colab_worker_logs/${WORKFLOW_ID}_drive_backup_run_name.txt"
DRIVE_BACKUP_ROOT="${ABERRATION_DRIVE_BACKUP_ROOT:-/content/drive/MyDrive/Aberration-Simulation-Colab-Backups}"

if [ -f "$DONE_MARKER" ]; then
  echo "active 12D hole search v1 already completed; marker exists at $DONE_MARKER"
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

restore_run_dir_from_drive() {
  local label="$1"
  local local_glob="$2"
  local drive_glob="$3"
  local dest_parent="$4"
  local found
  found=$(ls -td $local_glob 2>/dev/null | head -1 || true)
  if [ -n "$found" ] && [ -f "$found/model_loop_candidate.pt" ]; then
    printf '%s\n' "$found"
    return 0
  fi
  local drive_found
  drive_found=$(ls -td $drive_glob 2>/dev/null | head -1 || true)
  if [ -z "$drive_found" ]; then
    if [ -n "$found" ]; then
      echo "Using local $label run folder without checkpoint; inference will be skipped if checkpoint is unavailable: $found" >&2
      printf '%s\n' "$found"
      return 0
    fi
    echo "Could not find $label run folder locally or in Drive backup root: $DRIVE_BACKUP_ROOT" >&2
    return 1
  fi
  mkdir -p "$dest_parent"
  local dest_dir
  dest_dir="$dest_parent/$(basename "$drive_found")"
  mkdir -p "$dest_dir"
  echo "Restoring $label run folder from Drive: $drive_found" >&2
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --info=progress2 "$drive_found/" "$dest_dir/" >&2
  else
    cp -R "$drive_found/." "$dest_dir/"
  fi
  printf '%s\n' "$dest_dir"
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
    echo "Missing frozen benchmark-v2 split manifest. Restore/mount Drive before running active hole search."
    exit 1
  fi
fi

V13_CSV=$(restore_csv_folder_from_drive \
  "v13 1M" \
  "training_results/feature_regression_enhanced/enhanced_v13_1m_spacefill_*/training_features_enhanced.csv" \
  "$DRIVE_BACKUP_ROOT/*/training_results/feature_regression_enhanced/enhanced_v13_1m_spacefill_*/training_features_enhanced.csv" \
  "training_results/feature_regression_enhanced")

V13_RUN_DIR=$(restore_run_dir_from_drive \
  "v13 1M seed23 checkpoint rebuild" \
  "training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v13_1m_d66_seed23_checkpoint_rebuild_*" \
  "$DRIVE_BACKUP_ROOT/*/training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v13_1m_d66_seed23_checkpoint_rebuild_*" \
  "training_results/model_selection_loop" || true)

if [ -z "$V13_RUN_DIR" ]; then
  V13_RUN_DIR=$(restore_run_dir_from_drive \
    "v13 1M seed23 residual-NN metrics fallback" \
    "training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v13_1m_d66_seed23_residual_nn_*" \
    "$DRIVE_BACKUP_ROOT/*/training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v13_1m_d66_seed23_residual_nn_*" \
    "training_results/model_selection_loop")
fi

if [ ! -f "$V13_RUN_DIR/model_loop_candidate.pt" ]; then
  echo "The active 12D workflow requires the saved v13 checkpoint. Missing: $V13_RUN_DIR/model_loop_candidate.pt" >&2
  echo "Run scripts/run_colab_v13_seed23_checkpoint_rebuild_workflow.sh first, or mount the Drive backup containing v13_seed23_checkpoint_rebuild_*." >&2
  exit 1
fi

python3 scripts/run_active_12d_hole_search.py \
  --config "$CONFIG" \
  --v13-csv "$V13_CSV" \
  --v13-run-dir "$V13_RUN_DIR" \
  --benchmark-split-manifest "$V2_SPLIT_MANIFEST" \
  --output-root training_results/model_selection_reports

OUTPUT_DIR=$(ls -td training_results/model_selection_reports/v13_active_12d_hole_search_* 2>/dev/null | head -1 || true)
if [ -z "$OUTPUT_DIR" ]; then
  python3 - "$PENDING_MARKER" "$V13_CSV" "$V13_RUN_DIR" "$DRIVE_BACKUP_RUN_NAME" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "status": "pending",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "workflow": "scripts/run_colab_active_12d_hole_search_v1_workflow.sh",
    "v13_csv": sys.argv[2],
    "v13_run_dir": sys.argv[3],
    "drive_backup_run_name": sys.argv[4],
    "note": "Active hole-search command returned without a report directory.",
}, indent=2) + "\n")
PY
  exit 0
fi

echo "Backing up active 12D hole-search state to Drive."
python3 scripts/backup_colab_state_to_drive.py --run-name "$DRIVE_BACKUP_RUN_NAME"

python3 - "$DONE_MARKER" "$V13_CSV" "$V13_RUN_DIR" "$OUTPUT_DIR" "$DRIVE_BACKUP_RUN_NAME" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

marker = Path(sys.argv[1])
marker.write_text(json.dumps({
    "status": "complete",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "workflow": "scripts/run_colab_active_12d_hole_search_v1_workflow.sh",
    "v13_csv": sys.argv[2],
    "v13_run_dir": sys.argv[3],
    "output_dir": sys.argv[4],
    "drive_backup_run_name": sys.argv[5],
    "diagnostics": [
        "proposal_summary",
        "selected_probe_design",
        "simulation_manifest",
        "inference_manifest",
        "active_hole_search_summary",
        "top_failure_table",
        "hole_cluster_table",
        "nn_error_summary",
        "regime_summary",
        "relative_angle_summary",
        "recommended_sampling_plan",
        "drive_backup"
    ],
    "decision": "Use the active hole-search report to decide whether v15 should be targeted expansion, feature/model/loss work, architecture changes, or measurement-geometry changes.",
}, indent=2) + "\n")
print("done marker:", marker)
PY
