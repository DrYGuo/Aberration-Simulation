#!/usr/bin/env bash
set -euo pipefail

DONE_MARKER="colab_worker_logs/generalization_benchmark_v1_done.json"
STAGE_MARKER="colab_worker_logs/generalization_benchmark_v1_stage.json"
ACTIVE_CONFIG="configs/active_12d_generalization_benchmark_v1.json"
BENCHMARK_CONFIG="configs/generalization_benchmark_v1.json"
DRIVE_BACKUP_ROOT="${ABERRATION_DRIVE_BACKUP_ROOT:-/content/drive/MyDrive/Aberration-Simulation-Colab-Backups}"
DRIVE_BACKUP_RUN_NAME="${ABERRATION_DRIVE_BACKUP_RUN_NAME:-generalization_benchmark_v1_latest}"

mkdir -p colab_worker_logs training_results/model_selection_reports

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
    "workflow": "scripts/run_colab_generalization_benchmark_v1_workflow.sh",
}, indent=2) + "\n")
print("generalization-benchmark stage:", sys.argv[2], flush=True)
PY
}

if [ -f "$DONE_MARKER" ]; then
  echo "generalization benchmark v1 already completed; marker exists at $DONE_MARKER"
  exit 0
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
    rsync -a "$source_dir/" "$dest_dir/" >&2
  else
    cp -R "$source_dir/." "$dest_dir/"
  fi
  found=$(ls -td $local_glob 2>/dev/null | head -1 || true)
  if [ -z "$found" ]; then
    return 1
  fi
  printf '%s\n' "$found"
}

restore_dir_from_drive() {
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
  local dest_dir
  dest_dir="$dest_parent/$(basename "$drive_found")"
  mkdir -p "$dest_dir"
  echo "Restoring $label from Drive: $drive_found" >&2
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "$drive_found/" "$dest_dir/" >&2
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
    echo "Could not find $label locally or in Drive backup root: $DRIVE_BACKUP_ROOT" >&2
    return 1
  fi
  echo "Restoring $label from Drive: $found" >&2
  mkdir -p "$(dirname "$local_path")"
  cp "$found" "$local_path"
}

sync_dir_to_drive() {
  local source="$1"
  local dest="$DRIVE_BACKUP_ROOT/$DRIVE_BACKUP_RUN_NAME/$source"
  if [ ! -e "$source" ]; then
    return 0
  fi
  mkdir -p "$(dirname "$dest")"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete "$source/" "$dest/"
  else
    rm -rf "$dest"
    mkdir -p "$dest"
    cp -R "$source/." "$dest/"
  fi
}

write_stage "restore_inputs"
V2_SPLIT_MANIFEST="configs/benchmark_split_v12_v2_row_keys.json"
restore_file_from_drive \
  "benchmark-v2 split manifest" \
  "$V2_SPLIT_MANIFEST" \
  "$DRIVE_BACKUP_ROOT/*/configs/benchmark_split_v12_v2_row_keys.json"
V13_CSV=$(restore_csv_folder_from_drive \
  "v13 1M" \
  "training_results/feature_regression_enhanced/enhanced_v13_1m_spacefill_*/training_features_enhanced.csv" \
  "$DRIVE_BACKUP_ROOT/*/training_results/feature_regression_enhanced/enhanced_v13_1m_spacefill_*/training_features_enhanced.csv" \
  "training_results/feature_regression_enhanced")
V13_RUN_DIR=$(restore_dir_from_drive \
  "v13 1M seed23 residual-NN metrics" \
  "training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v13_1m_d66_seed23_residual_nn_*" \
  "$DRIVE_BACKUP_ROOT/*/training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v13_1m_d66_seed23_residual_nn_*" \
  "training_results/model_selection_loop")
SCORE_SUMMARY=$(ls -td training_results/model_selection_reports/benchmark_suite_scoring_v1_*/benchmark_suite_score_summary.json 2>/dev/null | head -1 || true)
TOP_SUMMARY=$(ls -td training_results/model_selection_reports/v15_top_failed_region_retest_*/v15_top_failed_region_retest_summary.json 2>/dev/null | head -1 || true)

write_stage "new_hole_design"
python3 scripts/run_active_12d_hole_search.py \
  --config "$ACTIVE_CONFIG" \
  --v13-csv "$V13_CSV" \
  --v13-run-dir "$V13_RUN_DIR" \
  --benchmark-split-manifest "$V2_SPLIT_MANIFEST" \
  --output-root training_results/model_selection_reports \
  --proposal-only

NEW_HOLE_DIR=$(ls -td training_results/model_selection_reports/v13_active_12d_generalization_benchmark_v1_new_holes_* 2>/dev/null | head -1 || true)
if [ -z "$NEW_HOLE_DIR" ]; then
  echo "No new-hole design directory was produced." >&2
  exit 1
fi

write_stage "broad_and_anchor_designs"
DESIGN_JSON=$(python3 scripts/generate_generalization_benchmark_designs.py \
  --config "$BENCHMARK_CONFIG" \
  --output-root training_results/model_selection_reports \
  --broad-rows 100000 \
  --anchor-rows 5000)
echo "$DESIGN_JSON"
BROAD_DIR=$(python3 - "$DESIGN_JSON" <<'PY'
import json
import sys
print(json.loads(sys.argv[1])["broad_design_dir"])
PY
)
ANCHOR_DIR=$(python3 - "$DESIGN_JSON" <<'PY'
import json
import sys
print(json.loads(sys.argv[1])["anchor_design_dir"])
PY
)

write_stage "freeze_benchmark"
ARGS=(
  --config "$BENCHMARK_CONFIG"
  --new-hole-dir "$NEW_HOLE_DIR"
  --broad-dir "$BROAD_DIR"
  --anchor-dir "$ANCHOR_DIR"
  --output-root training_results/model_selection_reports
)
if [ -n "$SCORE_SUMMARY" ]; then
  ARGS+=(--benchmark-suite-summary "$SCORE_SUMMARY")
fi
if [ -n "$TOP_SUMMARY" ]; then
  ARGS+=(--top-failure-summary "$TOP_SUMMARY")
fi
BENCHMARK_JSON=$(python3 scripts/prepare_generalization_benchmark_v1.py "${ARGS[@]}")
echo "$BENCHMARK_JSON"
BENCHMARK_DIR=$(python3 - "$BENCHMARK_JSON" <<'PY'
import json
import sys
print(json.loads(sys.argv[1])["output_dir"])
PY
)

write_stage "drive_sync"
sync_dir_to_drive "$NEW_HOLE_DIR"
sync_dir_to_drive "$BROAD_DIR"
sync_dir_to_drive "$ANCHOR_DIR"
sync_dir_to_drive "$BENCHMARK_DIR"
sync_dir_to_drive "colab_worker_logs"

python3 - "$DONE_MARKER" "$NEW_HOLE_DIR" "$BROAD_DIR" "$ANCHOR_DIR" "$BENCHMARK_DIR" "$V13_CSV" "$V13_RUN_DIR" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "status": "complete",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "workflow": "scripts/run_colab_generalization_benchmark_v1_workflow.sh",
    "new_hole_design_dir": sys.argv[2],
    "broad_representative_design_dir": sys.argv[3],
    "anchor_easy_design_dir": sys.argv[4],
    "generalization_benchmark_dir": sys.argv[5],
    "v13_csv": sys.argv[6],
    "v13_run_dir": sys.argv[7],
    "training_launched": False,
    "simulation_launched": False,
    "next_step": "Simulate/evaluate the frozen new-hole, broad representative, and anchor/easy designs for v13 and v15, then populate generalization benchmark scores before v16 training.",
}, indent=2) + "\n")
print("done marker:", sys.argv[1])
PY

write_stage "complete"
