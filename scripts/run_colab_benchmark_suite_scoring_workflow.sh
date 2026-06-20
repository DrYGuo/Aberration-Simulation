#!/usr/bin/env bash
set -euo pipefail

DONE_MARKER="colab_worker_logs/benchmark_suite_scoring_v1_done.json"
STAGE_MARKER="colab_worker_logs/benchmark_suite_scoring_v1_stage.json"
DRIVE_BACKUP_ROOT="${ABERRATION_DRIVE_BACKUP_ROOT:-/content/drive/MyDrive/Aberration-Simulation-Colab-Backups}"
DRIVE_BACKUP_RUN_NAME="${ABERRATION_DRIVE_BACKUP_RUN_NAME:-benchmark_suite_scoring_v1_latest}"

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
    "workflow": "scripts/run_colab_benchmark_suite_scoring_workflow.sh",
}, indent=2) + "\n")
print("benchmark-suite stage:", sys.argv[2], flush=True)
PY
}

if [ -f "$DONE_MARKER" ]; then
  echo "benchmark-suite scoring already completed; marker exists at $DONE_MARKER"
  exit 0
fi

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
    echo "Missing $label locally and on Drive: $drive_glob" >&2
    return 1
  fi
  mkdir -p "$dest_parent"
  local dest="$dest_parent/$(basename "$drive_found")"
  echo "Restoring $label from Drive: $drive_found" >&2
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "$drive_found/" "$dest/"
  else
    mkdir -p "$dest"
    cp -R "$drive_found/." "$dest/"
  fi
  printf '%s\n' "$dest"
}

sync_dir_to_drive() {
  local source="$1"
  local dest="$DRIVE_BACKUP_ROOT/$DRIVE_BACKUP_RUN_NAME/$source"
  if [ ! -e "$source" ]; then
    return 0
  fi
  mkdir -p "$(dirname "$dest")"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "$source/" "$dest/"
  else
    mkdir -p "$dest"
    cp -R "$source/." "$dest/"
  fi
}

write_stage "restore_inputs"
V13_RUN_DIR=$(restore_dir_from_drive \
  "v13 seed23 run metrics" \
  "training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v13_1m_d66_seed23_residual_nn_*" \
  "$DRIVE_BACKUP_ROOT"/*/training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v13_1m_d66_seed23_residual_nn_* \
  "training_results/model_selection_loop")
V15_RUN_DIR=$(restore_dir_from_drive \
  "v15 active-hole expanded run metrics" \
  "training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v15_active_hole_expanded_250k_d66_seed23_20260619_075900_utc" \
  "$DRIVE_BACKUP_ROOT"/*/training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v15_active_hole_expanded_250k_d66_seed23_20260619_075900_utc \
  "training_results/model_selection_loop")
RETEST_DIR=$(restore_dir_from_drive \
  "v15 active-hole retest full folder" \
  "training_results/model_selection_reports/v15_active_hole_retest_*" \
  "$DRIVE_BACKUP_ROOT"/v15_active_hole_retest_latest/training_results/model_selection_reports/v15_active_hole_retest_* \
  "training_results/model_selection_reports")
echo "retest dir: $RETEST_DIR"
if ! find "$RETEST_DIR" -path "*/active_hole_retest_probe_comparison.csv" -type f 2>/dev/null | grep -q .; then
  echo "Local retest folder is compact-only; restoring full per-run comparison CSVs from Drive."
  RETEST_DIR=$(restore_dir_from_drive \
    "v15 active-hole retest full folder" \
    "training_results/model_selection_reports/__force_drive_restore_no_local_match__" \
    "$DRIVE_BACKUP_ROOT"/v15_active_hole_retest_latest/training_results/model_selection_reports/v15_active_hole_retest_* \
    "training_results/model_selection_reports")
  echo "retest dir after force restore: $RETEST_DIR"
fi
if ! find "$RETEST_DIR" -path "*/active_hole_retest_probe_comparison.csv" -type f 2>/dev/null | grep -q .; then
  echo "No full retest comparison CSVs found under $RETEST_DIR." >&2
  echo "First files found in restored retest directory:" >&2
  find "$RETEST_DIR" -maxdepth 3 -type f 2>/dev/null | head -50 >&2 || true
  echo "The Drive folder v15_active_hole_retest_latest may be compact-only; rerun the v15 retest workflow if full per-probe CSVs are missing there." >&2
  exit 1
fi

write_stage "top_failure_retest_report"
TOP_JSON=$(python3 scripts/report_v15_top_failed_region_retest.py \
  --active-root training_results/model_selection_reports \
  --retest-dir "$RETEST_DIR" \
  --output-root training_results/model_selection_reports)
echo "$TOP_JSON"
TOP_DIR=$(python3 - "$TOP_JSON" <<'PY'
import json
import sys
print(json.loads(sys.argv[1])["output_dir"])
PY
)

write_stage "benchmark_suite_score"
SCORE_JSON=$(python3 scripts/score_benchmark_suite.py \
  --config configs/benchmark_suite_scoring_v1.json \
  --v13-run-dir "$V13_RUN_DIR" \
  --v15-run-dir "$V15_RUN_DIR" \
  --active-hole-retest-summary "$RETEST_DIR/active_hole_retest_summary.json" \
  --top-failure-retest-summary "$TOP_DIR/v15_top_failed_region_retest_summary.json" \
  --output-root training_results/model_selection_reports)
echo "$SCORE_JSON"
SCORE_DIR=$(python3 - "$SCORE_JSON" <<'PY'
import json
import sys
print(json.loads(sys.argv[1])["output_dir"])
PY
)

write_stage "drive_sync"
sync_dir_to_drive "$TOP_DIR"
sync_dir_to_drive "$SCORE_DIR"
sync_dir_to_drive "colab_worker_logs"

python3 - "$DONE_MARKER" "$TOP_DIR" "$SCORE_DIR" "$V13_RUN_DIR" "$V15_RUN_DIR" "$RETEST_DIR" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "status": "complete",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "workflow": "scripts/run_colab_benchmark_suite_scoring_workflow.sh",
    "top_failure_retest_dir": sys.argv[2],
    "benchmark_suite_score_dir": sys.argv[3],
    "v13_run_dir": sys.argv[4],
    "v15_run_dir": sys.argv[5],
    "retest_dir": sys.argv[6],
    "training_launched": False,
}, indent=2) + "\n")
print("done marker:", sys.argv[1])
PY

write_stage "complete"
