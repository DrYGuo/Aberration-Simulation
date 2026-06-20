#!/usr/bin/env bash
set -euo pipefail

DONE_MARKER="colab_worker_logs/v15_active_hole_retest_done.json"
STAGE_MARKER="colab_worker_logs/v15_active_hole_retest_stage.json"
DRIVE_BACKUP_ROOT="${ABERRATION_DRIVE_BACKUP_ROOT:-/content/drive/MyDrive/Aberration-Simulation-Colab-Backups}"
DRIVE_BACKUP_RUN_NAME="${ABERRATION_DRIVE_BACKUP_RUN_NAME:-v15_active_hole_retest_latest}"
V2_SPLIT_MANIFEST="configs/benchmark_split_v12_v2_row_keys.json"

mkdir -p colab_worker_logs

write_stage() {
  local stage="$1"
  python3 - "$STAGE_MARKER" "$stage" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "stage": sys.argv[2],
    "updated_utc": datetime.now(timezone.utc).isoformat(),
    "workflow": "scripts/run_colab_v15_active_hole_retest_workflow.sh",
}, indent=2) + "\n")
print("v15 active-hole retest stage:", sys.argv[2], flush=True)
PY
}

if [ -f "$DONE_MARKER" ]; then
  echo "v15 active-hole retest already completed; marker exists at $DONE_MARKER"
  exit 0
fi

write_stage "start"

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
  echo "Missing cached $label CSV locally; trying Drive restore." >&2
  local drive_found
  drive_found=$(ls -td $drive_glob 2>/dev/null | head -1 || true)
  if [ -z "$drive_found" ]; then
    echo "Could not find $label CSV locally or in Drive backup root: $DRIVE_BACKUP_ROOT" >&2
    return 1
  fi
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
    if [ -f "$source_dir/$name" ]; then
      if command -v rsync >/dev/null 2>&1; then
        rsync -a --inplace --info=progress2 "$source_dir/$name" "$dest_dir/$name" >&2
      else
        cp "$source_dir/$name" "$dest_dir/$name"
      fi
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
  mkdir -p "$(dirname "$local_path")"
  echo "Restoring $label from Drive: $found"
  cp "$found" "$local_path"
}

sync_dir_to_drive() {
  local source="$1"
  local dest="$DRIVE_BACKUP_ROOT/$DRIVE_BACKUP_RUN_NAME/$source"
  if [ ! -e "$source" ]; then
    echo "skip missing backup source: $source"
    return 0
  fi
  mkdir -p "$(dirname "$dest")"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "$source/" "$dest/"
  else
    mkdir -p "$dest"
    cp -R "$source/." "$dest/"
  fi
  echo "synced to Drive: $dest"
}

restore_active_probe_features_from_drive() {
  local restored=0
  mkdir -p training_results/model_selection_reports
  while IFS= read -r probe_csv; do
    [ -n "$probe_csv" ] || continue
    local source_dir
    source_dir=$(dirname "$probe_csv")
    local dest_dir
    dest_dir="training_results/model_selection_reports/$(basename "$source_dir")"
    mkdir -p "$dest_dir"
    echo "Restoring full active-hole probe folder from Drive: $source_dir"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a "$source_dir/" "$dest_dir/"
    else
      cp -R "$source_dir/." "$dest_dir/"
    fi
    restored=$((restored + 1))
  done < <(find "$DRIVE_BACKUP_ROOT" -path "*/training_results/model_selection_reports/v13_active_12d_hole_search*/active_hole_search_probe_features.csv" -type f 2>/dev/null | sort)
  echo "restored active-hole probe folders: $restored"
  test "$restored" -gt 0
}

write_stage "restore_v15_csv"
V15_CSV=$(restore_csv_folder_from_drive \
  "v15 active-hole expanded 250k" \
  "training_results/feature_regression_enhanced/enhanced_v15_active_hole_expanded_250k_*/training_features_enhanced.csv" \
  "$DRIVE_BACKUP_ROOT"/*/training_results/feature_regression_enhanced/enhanced_v15_active_hole_expanded_250k_*/training_features_enhanced.csv \
  "training_results/feature_regression_enhanced")
if [ -z "$V15_CSV" ]; then
  echo "Missing v15 active-hole expanded CSV. Keep Drive mounted and ensure v15_active_hole_expanded_250k_latest exists." >&2
  exit 1
fi

write_stage "restore_or_recreate_benchmark_v2_manifest"
if [ ! -f "$V2_SPLIT_MANIFEST" ]; then
  if ! restore_file_from_drive \
    "benchmark-v2 split manifest" \
    "$V2_SPLIT_MANIFEST" \
    "$DRIVE_BACKUP_ROOT"/*/configs/benchmark_split_v12_v2_row_keys.json; then
    echo "Missing frozen benchmark-v2 split manifest; recreating it from cached benchmark CSVs."
    V1_SPLIT_MANIFEST="configs/benchmark_split_v6_frozen_row_keys.json"
    if [ ! -f "$V1_SPLIT_MANIFEST" ]; then
      if ! restore_file_from_drive \
        "benchmark-v1 split manifest" \
        "$V1_SPLIT_MANIFEST" \
        "$DRIVE_BACKUP_ROOT"/*/configs/benchmark_split_v6_frozen_row_keys.json; then
        V6_CSV=$(restore_csv_folder_from_drive \
          "v6 benchmark-gap" \
          "training_results/feature_regression_enhanced/enhanced_v6_benchmark_gap100k_*/training_features_enhanced.csv" \
          "$DRIVE_BACKUP_ROOT"/*/training_results/feature_regression_enhanced/enhanced_v6_benchmark_gap100k_*/training_features_enhanced.csv \
          "training_results/feature_regression_enhanced")
        if [ -z "$V6_CSV" ]; then
          echo "Missing v6 cached CSV required to recreate benchmark-v1 split manifest." >&2
          exit 1
        fi
        python3 scripts/write_benchmark_split_manifest.py \
          --csv-path "$V6_CSV" \
          --output "$V1_SPLIT_MANIFEST" \
          --dataset-version enhanced_v6_benchmark_gap100k \
          --overwrite
      fi
    fi
    V12_CSV=$(restore_csv_folder_from_drive \
      "v12 benchmark-v2" \
      "training_results/feature_regression_enhanced/enhanced_v12_benchmark_v2_*/training_features_enhanced.csv" \
      "$DRIVE_BACKUP_ROOT"/*/training_results/feature_regression_enhanced/enhanced_v12_benchmark_v2_*/training_features_enhanced.csv \
      "training_results/feature_regression_enhanced" || true)
    if [ -z "$V12_CSV" ]; then
      echo "No separate v12 CSV found; using the restored v15 CSV to recreate the v12-v2 manifest from embedded v12 rows."
      V12_CSV="$V15_CSV"
    fi
    python3 scripts/write_benchmark_v2_split_manifest.py \
      --csv-path "$V12_CSV" \
      --base-manifest "$V1_SPLIT_MANIFEST" \
      --output "$V2_SPLIT_MANIFEST" \
      --new-dataset-version enhanced_v12_benchmark_v2 \
      --dataset-version enhanced_v12_benchmark_v2 \
      --overwrite
  fi
fi

write_stage "restore_active_probe_features"
if ! ls training_results/model_selection_reports/v13_active_12d_hole_search*/active_hole_search_probe_features.csv >/dev/null 2>&1; then
  restore_active_probe_features_from_drive
fi
PROBE_COUNT=$(ls training_results/model_selection_reports/v13_active_12d_hole_search*/active_hole_search_probe_features.csv 2>/dev/null | wc -l | tr -d ' ')
if [ "$PROBE_COUNT" -eq 0 ]; then
  echo "No active-hole probe feature CSVs are available after Drive restore." >&2
  exit 1
fi
echo "active-hole probe feature CSV count: $PROBE_COUNT"

write_stage "find_or_rebuild_v15_checkpoint"
V15_RUN_DIR=$(ls -td training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v15_active_hole_expanded_250k_d66_seed23_checkpoint_rebuild_* 2>/dev/null | head -1 || true)
if [ -z "$V15_RUN_DIR" ]; then
  V15_RUN_DIR=$(ls -td "$DRIVE_BACKUP_ROOT"/*/training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v15_active_hole_expanded_250k_d66_seed23_checkpoint_rebuild_* 2>/dev/null | head -1 || true)
  if [ -n "$V15_RUN_DIR" ]; then
    LOCAL_RUN_DIR="training_results/model_selection_loop/$(basename "$V15_RUN_DIR")"
    mkdir -p "$LOCAL_RUN_DIR"
    echo "Restoring v15 checkpoint rebuild run from Drive: $V15_RUN_DIR"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a "$V15_RUN_DIR/" "$LOCAL_RUN_DIR/"
    else
      cp -R "$V15_RUN_DIR/." "$LOCAL_RUN_DIR/"
    fi
    V15_RUN_DIR="$LOCAL_RUN_DIR"
  fi
fi

if [ -z "$V15_RUN_DIR" ] || [ ! -f "$V15_RUN_DIR/model_loop_candidate.pt" ]; then
  echo "No saved v15 checkpoint found. Rebuilding the exact v15 seed23 configuration once with --save-model."
  write_stage "rebuild_v15_checkpoint"
  python3 scripts/run_model_selection_candidate.py \
    --family enhanced \
    --candidate-id D66_grouped_width320_lr6e-4_dropout0.075_v15_active_hole_expanded_250k_d66_seed23_checkpoint_rebuild \
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
    --csv-path "$V15_CSV" \
    --selection-config experiments/model_selection_weights.json \
    --benchmark-split-manifest "$V2_SPLIT_MANIFEST" \
    --split-seed 7 \
    --save-model
  V15_RUN_DIR=$(ls -td training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v15_active_hole_expanded_250k_d66_seed23_checkpoint_rebuild_* 2>/dev/null | head -1 || true)
fi

if [ -z "$V15_RUN_DIR" ] || [ ! -f "$V15_RUN_DIR/model_loop_candidate.pt" ]; then
  echo "V15 checkpoint rebuild is missing model_loop_candidate.pt." >&2
  exit 1
fi
echo "v15 checkpoint run dir: $V15_RUN_DIR"

write_stage "run_retest"
RETEST_JSON=$(python3 scripts/retest_active_hole_with_checkpoint.py \
  --dataset-csv "$V15_CSV" \
  --run-dir "$V15_RUN_DIR" \
  --benchmark-split-manifest "$V2_SPLIT_MANIFEST" \
  --active-root training_results/model_selection_reports \
  --output-root training_results/model_selection_reports \
  --run-prefix v15_active_hole_retest)
echo "$RETEST_JSON"
RETEST_DIR=$(python3 - "$RETEST_JSON" <<'PY'
import json
import sys
print(json.loads(sys.argv[1])["output_dir"])
PY
)

write_stage "drive_sync"
sync_dir_to_drive "$V15_RUN_DIR"
sync_dir_to_drive "$RETEST_DIR"
sync_dir_to_drive "colab_worker_logs"

python3 - "$DONE_MARKER" "$V15_CSV" "$V15_RUN_DIR" "$RETEST_DIR" "$DRIVE_BACKUP_ROOT/$DRIVE_BACKUP_RUN_NAME" "$PROBE_COUNT" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "status": "complete",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "workflow": "scripts/run_colab_v15_active_hole_retest_workflow.sh",
    "v15_csv": sys.argv[2],
    "v15_checkpoint_run_dir": sys.argv[3],
    "retest_dir": sys.argv[4],
    "drive_backup_dir": sys.argv[5],
    "active_probe_feature_csv_count": int(sys.argv[6]),
    "decision": "Use active_hole_retest_report.md to decide whether v15 repaired original searched holes.",
}, indent=2) + "\n")
print("done marker:", sys.argv[1])
PY

write_stage "complete"
