#!/usr/bin/env bash
set -euo pipefail

DONE_MARKER="colab_worker_logs/generalization_benchmark_v1_evaluation_done.json"
STAGE_MARKER="colab_worker_logs/generalization_benchmark_v1_evaluation_stage.json"
RUN_NAME_FILE="colab_worker_logs/generalization_benchmark_v1_evaluation_run_name.txt"
DRIVE_BACKUP_ROOT="${ABERRATION_DRIVE_BACKUP_ROOT:-/content/drive/MyDrive/Aberration-Simulation-Colab-Backups}"
DRIVE_BACKUP_RUN_NAME="${ABERRATION_DRIVE_BACKUP_RUN_NAME:-generalization_benchmark_v1_evaluation_latest}"
V2_SPLIT_MANIFEST="configs/benchmark_split_v12_v2_row_keys.json"

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
    "workflow": "scripts/run_colab_generalization_benchmark_v1_evaluation_workflow.sh",
}, indent=2) + "\n")
print("generalization-benchmark-evaluation stage:", sys.argv[2], flush=True)
PY
}

if [ -f "$DONE_MARKER" ]; then
  echo "generalization benchmark v1 evaluation already completed; marker exists at $DONE_MARKER"
  exit 0
fi

if [ -f "$RUN_NAME_FILE" ]; then
  RUN_NAME=$(cat "$RUN_NAME_FILE")
else
  RUN_NAME="generalization_benchmark_v1_evaluation_$(date -u +%Y%m%d_%H%M%S_utc)"
  printf '%s\n' "$RUN_NAME" > "$RUN_NAME_FILE"
fi
EVAL_DIR="training_results/model_selection_reports/$RUN_NAME"

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
  echo "Restoring $label from Drive: $found" >&2
  cp "$found" "$local_path"
}

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
  local drive_found
  drive_found=$(ls -td $drive_glob 2>/dev/null | head -1 || true)
  if [ -z "$drive_found" ]; then
    echo "Could not find $label CSV locally or in Drive backup root: $DRIVE_BACKUP_ROOT" >&2
    return 1
  fi
  local source_dir
  source_dir=$(dirname "$drive_found")
  local dest_dir="$dest_parent/$(basename "$source_dir")"
  mkdir -p "$dest_dir"
  echo "Restoring $label CSV folder from Drive: $source_dir" >&2
  local required_files=(
    "training_features_enhanced.csv"
    "feature_columns_enhanced.json"
    "dataset_manifest.json"
    "dataset_recovery_manifest.json"
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

restore_dir_from_drive() {
  local label="$1"
  local local_glob="$2"
  local drive_glob="$3"
  local dest_parent="$4"
  local found
  found=$(ls -td $local_glob 2>/dev/null | head -1 || true)
  if [ -n "$found" ]; then
    if [ -d "$found" ]; then
      printf '%s\n' "$found"
      return 0
    fi
    if [ -f "$found" ]; then
      dirname "$found"
      return 0
    fi
  fi
  local drive_found
  drive_found=$(ls -td $drive_glob 2>/dev/null | head -1 || true)
  if [ -z "$drive_found" ]; then
    echo "Could not find $label locally or in Drive backup root: $DRIVE_BACKUP_ROOT" >&2
    return 1
  fi
  local source_dir="$drive_found"
  if [ -f "$drive_found" ]; then
    source_dir=$(dirname "$drive_found")
  fi
  local dest_dir="$dest_parent/$(basename "$source_dir")"
  mkdir -p "$dest_dir"
  echo "Restoring $label from Drive: $source_dir" >&2
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "$source_dir/" "$dest_dir/" >&2
  else
    cp -R "$source_dir/." "$dest_dir/"
  fi
  printf '%s\n' "$dest_dir"
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
    rsync -a --delete "$source/" "$dest/"
  else
    rm -rf "$dest"
    mkdir -p "$dest"
    cp -R "$source/." "$dest/"
  fi
  echo "synced to Drive: $dest"
}

require_checkpoint() {
  local label="$1"
  local run_dir="$2"
  if [ ! -f "$run_dir/model_loop_candidate.pt" ]; then
    echo "Missing saved checkpoint for $label: $run_dir/model_loop_candidate.pt" >&2
    echo "This workflow is inference-only and will not rebuild/train checkpoints." >&2
    return 1
  fi
}

write_stage "restore_inputs"
restore_file_from_drive \
  "benchmark-v2 split manifest" \
  "$V2_SPLIT_MANIFEST" \
  "$DRIVE_BACKUP_ROOT/*/configs/benchmark_split_v12_v2_row_keys.json"

V13_CSV=$(restore_csv_folder_from_drive \
  "v13 1M" \
  "training_results/feature_regression_enhanced/enhanced_v13_1m_spacefill_*/training_features_enhanced.csv" \
  "$DRIVE_BACKUP_ROOT/*/training_results/feature_regression_enhanced/enhanced_v13_1m_spacefill_*/training_features_enhanced.csv" \
  "training_results/feature_regression_enhanced")

V15_CSV=$(restore_csv_folder_from_drive \
  "v15 active-hole expanded 250k" \
  "training_results/feature_regression_enhanced/enhanced_v15_active_hole_expanded_250k_*/training_features_enhanced.csv" \
  "$DRIVE_BACKUP_ROOT/*/training_results/feature_regression_enhanced/enhanced_v15_active_hole_expanded_250k_*/training_features_enhanced.csv" \
  "training_results/feature_regression_enhanced")

BENCHMARK_DIR=$(restore_dir_from_drive \
  "generalization benchmark v1 manifest" \
  "training_results/model_selection_reports/generalization_benchmark_v1_*/generalization_benchmark_manifest.json" \
  "$DRIVE_BACKUP_ROOT/*/training_results/model_selection_reports/generalization_benchmark_v1_*/generalization_benchmark_manifest.json" \
  "training_results/model_selection_reports")
BENCHMARK_MANIFEST="$BENCHMARK_DIR/generalization_benchmark_manifest.json"

NEW_HOLE_DIR=$(restore_dir_from_drive \
  "generalization benchmark new-hole design" \
  "training_results/model_selection_reports/v13_active_12d_generalization_benchmark_v1_new_holes_*/selected_probe_design.csv" \
  "$DRIVE_BACKUP_ROOT/*/training_results/model_selection_reports/v13_active_12d_generalization_benchmark_v1_new_holes_*/selected_probe_design.csv" \
  "training_results/model_selection_reports")
NEW_HOLE_DESIGN="$NEW_HOLE_DIR/selected_probe_design.csv"

BROAD_DIR=$(restore_dir_from_drive \
  "broad 12D representative validation design" \
  "training_results/model_selection_reports/broad_12d_representative_validation_v1_*/broad_12d_representative_validation_design.csv" \
  "$DRIVE_BACKUP_ROOT/*/training_results/model_selection_reports/broad_12d_representative_validation_v1_*/broad_12d_representative_validation_design.csv" \
  "training_results/model_selection_reports")
BROAD_DESIGN="$BROAD_DIR/broad_12d_representative_validation_design.csv"

ANCHOR_DIR=$(restore_dir_from_drive \
  "anchor/easy validation design" \
  "training_results/model_selection_reports/anchor_easy_validation_v1_*/anchor_easy_validation_design.csv" \
  "$DRIVE_BACKUP_ROOT/*/training_results/model_selection_reports/anchor_easy_validation_v1_*/anchor_easy_validation_design.csv" \
  "training_results/model_selection_reports")
ANCHOR_DESIGN="$ANCHOR_DIR/anchor_easy_validation_design.csv"

V13_RUN_DIR=$(restore_dir_from_drive \
  "v13 seed23 saved checkpoint run" \
  "training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v13_1m_d66_seed23_checkpoint_rebuild_*/model_loop_candidate.pt" \
  "$DRIVE_BACKUP_ROOT/*/training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v13_1m_d66_seed23_checkpoint_rebuild_*/model_loop_candidate.pt" \
  "training_results/model_selection_loop")
V15_RUN_DIR=$(restore_dir_from_drive \
  "v15 active-hole saved checkpoint run" \
  "training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v15_active_hole_expanded_250k_d66_seed23_checkpoint_rebuild_*/model_loop_candidate.pt" \
  "$DRIVE_BACKUP_ROOT/*/training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v15_active_hole_expanded_250k_d66_seed23_checkpoint_rebuild_*/model_loop_candidate.pt" \
  "training_results/model_selection_loop")

require_checkpoint "v13" "$V13_RUN_DIR"
require_checkpoint "v15" "$V15_RUN_DIR"

write_stage "preflight"
python3 - "$BENCHMARK_MANIFEST" "$NEW_HOLE_DESIGN" "$BROAD_DESIGN" "$ANCHOR_DESIGN" "$V13_CSV" "$V15_CSV" "$V13_RUN_DIR" "$V15_RUN_DIR" "$V2_SPLIT_MANIFEST" <<'PY'
import csv
import json
import sys
from pathlib import Path

paths = [Path(arg) for arg in sys.argv[1:]]
missing = [str(path) for path in paths if not path.exists()]
if missing:
    raise SystemExit("missing required evaluation input(s):\n- " + "\n- ".join(missing))
json.loads(paths[0].read_text())
json.loads(paths[8].read_text())
for csv_path in paths[1:6]:
    with csv_path.open(newline="") as handle:
        header = next(csv.reader(handle), None)
    if not header:
        raise SystemExit(f"CSV has no header: {csv_path}")
for run_dir in paths[6:8]:
    if not (run_dir / "model_loop_candidate.pt").exists():
        raise SystemExit(f"missing checkpoint: {run_dir / 'model_loop_candidate.pt'}")
print("evaluation preflight ok")
for path in paths:
    print(" ", path)
PY

write_stage "simulate_and_score"
python3 scripts/evaluate_generalization_benchmark_v1.py \
  --benchmark-manifest "$BENCHMARK_MANIFEST" \
  --new-hole-design-csv "$NEW_HOLE_DESIGN" \
  --broad-design-csv "$BROAD_DESIGN" \
  --anchor-design-csv "$ANCHOR_DESIGN" \
  --v13-dataset-csv "$V13_CSV" \
  --v13-run-dir "$V13_RUN_DIR" \
  --v15-dataset-csv "$V15_CSV" \
  --v15-run-dir "$V15_RUN_DIR" \
  --benchmark-split-manifest "$V2_SPLIT_MANIFEST" \
  --output-dir "$EVAL_DIR" \
  --batch-base-cases 192 \
  --predict-batch-size 65536

write_stage "drive_sync"
sync_dir_to_drive "$EVAL_DIR"
sync_dir_to_drive "colab_worker_logs"

python3 - "$DONE_MARKER" "$EVAL_DIR" "$DRIVE_BACKUP_ROOT/$DRIVE_BACKUP_RUN_NAME/$EVAL_DIR" "$BENCHMARK_MANIFEST" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "status": "complete",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "workflow": "scripts/run_colab_generalization_benchmark_v1_evaluation_workflow.sh",
    "evaluation_dir": sys.argv[2],
    "drive_evaluation_dir": sys.argv[3],
    "benchmark_manifest": sys.argv[4],
    "training_launched": False,
    "next_step": "Review populated generalization_benchmark_v1 scores for v13/v15, then design v16 sampling only if justified.",
}, indent=2) + "\n")
print("done marker:", sys.argv[1])
PY

write_stage "complete"
