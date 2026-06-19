#!/usr/bin/env bash
set -euo pipefail

DONE_MARKER="colab_worker_logs/v15_active_hole_expanded_250k_d66_done.json"
SAMPLING_QUALITY_DIR="training_results/model_selection_reports/sampling_quality_v15_active_hole_expanded_250k_d66"
DRIVE_BACKUP_RUN_NAME_FILE="colab_worker_logs/v15_active_hole_expanded_250k_drive_backup_run_name.txt"
DRIVE_BACKUP_ROOT="${ABERRATION_DRIVE_BACKUP_ROOT:-/content/drive/MyDrive/Aberration-Simulation-Colab-Backups}"
DRIVE_RESTORE_RETRIES="${ABERRATION_DRIVE_RESTORE_RETRIES:-3}"

if [ -f "$DONE_MARKER" ]; then
  echo "v15 active-hole expanded 250k d66 workflow already completed; marker exists at $DONE_MARKER"
  exit 0
fi

mkdir -p colab_worker_logs

write_stage_marker() {
  local stage="$1"
  python3 - "$stage" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path("colab_worker_logs/v15_active_hole_expanded_250k_stage.json")
payload = {
    "stage": sys.argv[1],
    "updated_utc": datetime.now(timezone.utc).isoformat(),
    "workflow": "scripts/run_colab_v15_active_hole_expanded_250k_workflow.sh",
}
path.write_text(json.dumps(payload, indent=2) + "\n")
print("v15 stage:", sys.argv[1], flush=True)
PY
}

trap 'write_stage_marker failed' ERR
write_stage_marker "start"

if [ -f "$DRIVE_BACKUP_RUN_NAME_FILE" ]; then
  DRIVE_BACKUP_RUN_NAME=$(cat "$DRIVE_BACKUP_RUN_NAME_FILE")
else
  DRIVE_BACKUP_RUN_NAME="${ABERRATION_DRIVE_BACKUP_RUN_NAME:-v15_active_hole_expanded_250k_latest}"
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
  echo "Missing cached $label CSV locally; trying Drive restore." >&2
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
    "targeted25k_audit.json"
    "label_summary.csv"
    "new_targeted_label_summary.csv"
    "dataset_recovery_manifest.json"
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
      if command -v rsync >/dev/null 2>&1; then
        rsync -a "$source_dir/$name" "$dest_dir/$name" >&2 && copied="true" || copied="false"
      else
        cp "$source_dir/$name" "$dest_dir/$name" && copied="true" || copied="false"
      fi
      if [ "$copied" = "true" ]; then
        local source_size
        local dest_size
        source_size=$(stat -c%s "$source_dir/$name" 2>/dev/null || echo 0)
        dest_size=$(stat -c%s "$dest_dir/$name" 2>/dev/null || echo 0)
        if [ "$source_size" -eq "$dest_size" ] && [ "$dest_size" -gt 0 ]; then
          break
        fi
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
    echo "Missing $label locally and on Drive: $drive_glob" >&2
    return 1
  fi
  echo "Restoring $label from Drive: $found"
  mkdir -p "$(dirname "$local_path")"
  cp "$found" "$local_path"
}

incremental_expanded_drive_backup() {
  local stage="$1"
  shift || true
  echo "Incremental Drive backup for expanded active-hole v15 stage: $stage"
  python3 scripts/backup_colab_state_to_drive.py \
    --run-name "$DRIVE_BACKUP_RUN_NAME" \
    --no-default-includes \
    --include "$V15_DIR" \
    --include configs \
    --include experiments \
    --include colab_worker_logs \
    --include "$FAILED_REPORT_DIR" \
    "$@"
}

V2_SPLIT_MANIFEST="configs/benchmark_split_v12_v2_row_keys.json"
if [ ! -f "$V2_SPLIT_MANIFEST" ]; then
  restore_file_from_drive \
    "benchmark-v2 split manifest" \
    "$V2_SPLIT_MANIFEST" \
    "$DRIVE_BACKUP_ROOT/*/configs/benchmark_split_v12_v2_row_keys.json"
fi

V13_CSV=$(restore_csv_folder_from_drive \
  "v13 1M" \
  "training_results/feature_regression_enhanced/enhanced_v13_1m_spacefill_*/training_features_enhanced.csv" \
  "$DRIVE_BACKUP_ROOT/*/training_results/feature_regression_enhanced/enhanced_v13_1m_spacefill_*/training_features_enhanced.csv" \
  "training_results/feature_regression_enhanced")
if [ -z "$V13_CSV" ]; then
  echo "Mount Drive first, or restore the v13 CSV folder before running expanded v15."
  exit 1
fi

python3 scripts/restore_active_hole_results_from_drive.py || true

REPORT_JSON=$(python3 scripts/report_active_failed_region_errors.py)
FAILED_REPORT_DIR=$(python3 - "$REPORT_JSON" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
print(payload["output_dir"])
PY
)
FAILED_SPEC="$FAILED_REPORT_DIR/v15_failed_subspace_sampling_spec.json"
if [ ! -f "$FAILED_SPEC" ]; then
  echo "Missing failed-subspace spec after report generation: $FAILED_SPEC" >&2
  exit 1
fi
echo "failed-region report dir: $FAILED_REPORT_DIR"
echo "failed-region spec: $FAILED_SPEC"
write_stage_marker "failed_region_report_ready"

BASE_DATA_CONFIG="configs/targeted_expansion_v15_active_hole_expanded_250k.json"
RUNTIME_DATA_CONFIG="colab_worker_logs/targeted_expansion_v15_active_hole_expanded_250k_runtime.json"
python3 - "$BASE_DATA_CONFIG" "$RUNTIME_DATA_CONFIG" "$FAILED_SPEC" <<'PY'
import json
import sys
from pathlib import Path

base_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
spec_path = sys.argv[3]
config = json.loads(base_path.read_text())
config["sampling"]["active_failed_subspace_jitter"]["spec_path"] = spec_path
config["runtime_failed_subspace_spec_path"] = spec_path
config["runtime_note"] = "Generated by expanded active-hole workflow from latest failed-region report."
out_path.write_text(json.dumps(config, indent=2) + "\n")
print("runtime data config:", out_path)
PY
write_stage_marker "runtime_data_config_ready"

write_stage_marker "preflight_active_failed_subspace_sampler"
python3 scripts/preflight_active_failed_subspace_sampler.py \
  --config "$RUNTIME_DATA_CONFIG" \
  --count 8 \
  --seed 157

V15_CSV=$(ls -td training_results/feature_regression_enhanced/enhanced_v15_active_hole_expanded_250k_*/training_features_enhanced.csv 2>/dev/null | head -1 || true)
if [ -z "$V15_CSV" ]; then
  echo "No local completed v15 active-hole 250k CSV found; checking Drive only for a resumable completed v15 dataset."
  V15_DRIVE_CSV=$(ls -td "$DRIVE_BACKUP_ROOT"/*/training_results/feature_regression_enhanced/enhanced_v15_active_hole_expanded_250k_*/training_features_enhanced.csv 2>/dev/null | head -1 || true)
  if [ -n "$V15_DRIVE_CSV" ]; then
    V15_CSV=$(restore_csv_folder_from_drive \
      "v15 expanded active-hole 250k" \
      "training_results/feature_regression_enhanced/enhanced_v15_active_hole_expanded_250k_*/training_features_enhanced.csv" \
      "$DRIVE_BACKUP_ROOT/*/training_results/feature_regression_enhanced/enhanced_v15_active_hole_expanded_250k_*/training_features_enhanced.csv" \
      "training_results/feature_regression_enhanced" || true)
  else
    echo "No completed v15 active-hole 250k CSV found on Drive. This is expected before the first successful v15 generation; generating v15 dataset next."
  fi
fi
if [ -z "$V15_CSV" ]; then
  write_stage_marker "generating_v15_dataset"
  python3 scripts/generate_targeted_enhanced_dataset.py \
    --parent-csv "$V13_CSV" \
    --run-prefix enhanced_v15_active_hole_expanded_250k \
    --dataset-version enhanced_v15_active_hole_expanded_250k \
    --case-counts-json "$RUNTIME_DATA_CONFIG" \
    --seed 157 \
    --batch-base-cases 256 \
    --new-row-split-hint training_only
  V15_CSV=$(ls -td training_results/feature_regression_enhanced/enhanced_v15_active_hole_expanded_250k_*/training_features_enhanced.csv | head -1)
fi

V15_DIR=$(dirname "$V15_CSV")
write_stage_marker "v15_dataset_ready"
python3 scripts/write_dataset_recovery_manifest.py \
  --csv-path "$V15_CSV" \
  --config "$RUNTIME_DATA_CONFIG" \
  --output "$V15_DIR/dataset_recovery_manifest.json" \
  --dataset-version enhanced_v15_active_hole_expanded_250k \
  --expected-total-rows 1325000 \
  --expected-new-rows 250000

incremental_expanded_drive_backup "before_training"

write_stage_marker "sampling_quality"
python3 scripts/report_sampling_quality.py \
  --csv-path "$V15_CSV" \
  --config "$RUNTIME_DATA_CONFIG" \
  --benchmark-split-manifest "$V2_SPLIT_MANIFEST" \
  --output-dir "$SAMPLING_QUALITY_DIR"

incremental_expanded_drive_backup "after_sampling_quality" --include "$SAMPLING_QUALITY_DIR"

BASE_BATCH_CONFIG="configs/model_selection_batch_v15_active_hole_expanded_250k_d66.json"
RUNTIME_BATCH_CONFIG="colab_worker_logs/model_selection_batch_v15_active_hole_expanded_250k_d66_runtime.json"
write_stage_marker "runtime_batch_config"
python3 - "$BASE_BATCH_CONFIG" "$RUNTIME_BATCH_CONFIG" "$V15_CSV" "$V2_SPLIT_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

base_config = Path(sys.argv[1])
runtime_config = Path(sys.argv[2])
csv_path = sys.argv[3]
manifest_path = sys.argv[4]
config = json.loads(base_config.read_text())
config.setdefault("defaults", {})["csv_path"] = csv_path
config.setdefault("defaults", {})["benchmark_split_manifest"] = manifest_path
config["runtime_csv_path"] = csv_path
config["runtime_benchmark_split_manifest"] = manifest_path
config["runtime_note"] = "Expanded active-hole v15 training from latest active failed-region report."
runtime_config.write_text(json.dumps(config, indent=2) + "\n")
print("runtime batch config:", runtime_config)
print("v15 csv:", csv_path)
print("benchmark-v2 split manifest:", manifest_path)
PY

write_stage_marker "training"
python3 scripts/run_model_selection_batch.py \
  --batch-config "$RUNTIME_BATCH_CONFIG" \
  --output-root training_results/model_selection_loop \
  --summary-root training_results/model_selection_batches \
  --max-runtime-minutes 420

V15_RUN_DIR=$(ls -td training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v15_active_hole_expanded_250k_d66_seed23_* 2>/dev/null | head -1 || true)
V15_BATCH_DIR=$(ls -td training_results/model_selection_batches/v15_active_hole_expanded_250k_d66_* 2>/dev/null | head -1 || true)
EXTRA_BACKUP_INCLUDES=(--include "$SAMPLING_QUALITY_DIR")
if [ -n "$V15_RUN_DIR" ]; then
  EXTRA_BACKUP_INCLUDES+=(--include "$V15_RUN_DIR")
fi
if [ -n "$V15_BATCH_DIR" ]; then
  EXTRA_BACKUP_INCLUDES+=(--include "$V15_BATCH_DIR")
fi
incremental_expanded_drive_backup "after_training" "${EXTRA_BACKUP_INCLUDES[@]}"

python3 - "$DONE_MARKER" "$V15_CSV" "$DRIVE_BACKUP_RUN_NAME" "$FAILED_REPORT_DIR" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

marker = Path(sys.argv[1])
marker.write_text(json.dumps({
    "status": "complete",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "workflow": "scripts/run_colab_v15_active_hole_expanded_250k_workflow.sh",
    "v15_csv": sys.argv[2],
    "drive_backup_run_name": sys.argv[3],
    "failed_region_report_dir": sys.argv[4],
    "decision": "Compare expanded v15 against v13 seed23 and retest active failed regions.",
}, indent=2) + "\n")
print("done marker:", marker)
PY
write_stage_marker "complete"
