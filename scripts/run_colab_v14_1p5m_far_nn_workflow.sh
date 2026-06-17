#!/usr/bin/env bash
set -euo pipefail

DONE_MARKER="colab_worker_logs/v14_1p5m_far_nn_d66_done.json"
PENDING_MARKER="colab_worker_logs/v14_1p5m_far_nn_d66_pending.json"
SAMPLING_QUALITY_DIR="training_results/model_selection_reports/sampling_quality_v14_1p5m_far_nn_d66"
DRIVE_BACKUP_RUN_NAME_FILE="colab_worker_logs/v14_1p5m_far_nn_drive_backup_run_name.txt"
DRIVE_BACKUP_ROOT="${ABERRATION_DRIVE_BACKUP_ROOT:-/content/drive/MyDrive/Aberration-Simulation-Colab-Backups}"
DRIVE_RESTORE_RETRIES="${ABERRATION_DRIVE_RESTORE_RETRIES:-3}"

if [ -f "$DONE_MARKER" ]; then
  echo "v14 1.5M far-NN d66 workflow already completed; marker exists at $DONE_MARKER"
  exit 0
fi

mkdir -p colab_worker_logs

if [ -f "$DRIVE_BACKUP_RUN_NAME_FILE" ]; then
  DRIVE_BACKUP_RUN_NAME=$(cat "$DRIVE_BACKUP_RUN_NAME_FILE")
else
  DRIVE_BACKUP_RUN_NAME="v14_1p5m_far_nn_$(date -u +%Y%m%d_%H%M%S_utc)"
  printf '%s\n' "$DRIVE_BACKUP_RUN_NAME" > "$DRIVE_BACKUP_RUN_NAME_FILE"
fi

restore_csv_folder_from_drive() {
  local label="$1"
  local local_glob="$2"
  local drive_glob="$3"
  local dest_parent="$4"
  local found
  found=$(ls -td $local_glob 2>/dev/null | head -1 || true)
  local drive_found
  drive_found=$(ls -td $drive_glob 2>/dev/null | head -1 || true)
  if [ -n "$found" ]; then
    if [ -n "$drive_found" ]; then
      local local_size
      local drive_size
      local_size=$(stat -c%s "$found" 2>/dev/null || echo 0)
      drive_size=$(stat -c%s "$drive_found" 2>/dev/null || echo 0)
      if [ "$local_size" -lt "$drive_size" ]; then
        echo "Existing local $label CSV appears partial: local=$local_size bytes, drive=$drive_size bytes. Re-restoring." >&2
        rm -f "$found"
      else
        printf '%s\n' "$found"
        return 0
      fi
    else
      echo "Using local $label CSV because no Drive source is currently visible: $found" >&2
      printf '%s\n' "$found"
      return 0
    fi
  fi
  if [ -n "$found" ] && [ ! -f "$found" ]; then
    found=""
  fi
  found=$(ls -td $local_glob 2>/dev/null | head -1 || true)
  if [ -n "$found" ]; then
    printf '%s\n' "$found"
    return 0
  fi
  echo "Missing cached $label CSV in the live runtime; trying Drive restore." >&2
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
        source_size=$(stat -c%s "$source_dir/$name" 2>/dev/null || echo 0)
        dest_size=$(stat -c%s "$dest_dir/$name" 2>/dev/null || echo 0)
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
    echo "Missing frozen benchmark-v2 split manifest; recreating it from cached v12/v6 CSVs."
    V1_SPLIT_MANIFEST="configs/benchmark_split_v6_frozen_row_keys.json"
    if [ ! -f "$V1_SPLIT_MANIFEST" ]; then
      if ! restore_file_from_drive "benchmark-v1 split manifest" "$V1_SPLIT_MANIFEST" "$DRIVE_BACKUP_ROOT/*/configs/benchmark_split_v6_frozen_row_keys.json"; then
        V6_CSV=$(restore_csv_folder_from_drive \
          "v6 benchmark-gap" \
          "training_results/feature_regression_enhanced/enhanced_v6_benchmark_gap100k_*/training_features_enhanced.csv" \
          "$DRIVE_BACKUP_ROOT/*/training_results/feature_regression_enhanced/enhanced_v6_benchmark_gap100k_*/training_features_enhanced.csv" \
          "training_results/feature_regression_enhanced")
        if [ -z "$V6_CSV" ]; then
          echo "Missing v6 cached CSV required to recreate benchmark-v1 split manifest."
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
      "$DRIVE_BACKUP_ROOT/*/training_results/feature_regression_enhanced/enhanced_v12_benchmark_v2_*/training_features_enhanced.csv" \
      "training_results/feature_regression_enhanced")
    if [ -z "$V12_CSV" ]; then
      echo "Missing v12 cached CSV required to recreate benchmark-v2 split manifest."
      exit 1
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

V13_CSV=$(restore_csv_folder_from_drive \
  "v13 1M" \
  "training_results/feature_regression_enhanced/enhanced_v13_1m_spacefill_*/training_features_enhanced.csv" \
  "$DRIVE_BACKUP_ROOT/*/training_results/feature_regression_enhanced/enhanced_v13_1m_spacefill_*/training_features_enhanced.csv" \
  "training_results/feature_regression_enhanced")
if [ -z "$V13_CSV" ]; then
  echo "Mount Drive first, or restore the v13 CSV folder before running v14."
  exit 1
fi

V14_CSV=$(ls -td training_results/feature_regression_enhanced/enhanced_v14_1p5m_far_nn_*/training_features_enhanced.csv 2>/dev/null | head -1 || true)
if [ -z "$V14_CSV" ]; then
  python3 scripts/generate_targeted_enhanced_dataset.py \
    --parent-csv "$V13_CSV" \
    --run-prefix enhanced_v14_1p5m_far_nn \
    --dataset-version enhanced_v14_1p5m_far_nn \
    --case-counts-json configs/targeted_expansion_v14_1p5m_far_nn.json \
    --seed 141 \
    --batch-base-cases 256 \
    --new-row-split-hint training_only
  V14_CSV=$(ls -td training_results/feature_regression_enhanced/enhanced_v14_1p5m_far_nn_*/training_features_enhanced.csv | head -1)
fi

echo "Backing up v14 generated dataset and current Colab state to Drive before training."
python3 scripts/backup_colab_state_to_drive.py --run-name "$DRIVE_BACKUP_RUN_NAME"

python3 scripts/report_sampling_quality.py \
  --csv-path "$V14_CSV" \
  --config configs/targeted_expansion_v14_1p5m_far_nn.json \
  --benchmark-split-manifest "$V2_SPLIT_MANIFEST" \
  --output-dir "$SAMPLING_QUALITY_DIR"

BASE_CONFIG="configs/model_selection_batch_v14_1p5m_far_nn_d66.json"
RUNTIME_CONFIG="colab_worker_logs/model_selection_batch_v14_1p5m_far_nn_d66_runtime.json"
python3 - "$BASE_CONFIG" "$RUNTIME_CONFIG" "$V14_CSV" "$V2_SPLIT_MANIFEST" <<'PY'
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
config["runtime_note"] = (
    "Generated by run_colab_v14_1p5m_far_nn_workflow.sh. "
    "Runs one controlled v14 1.5M far-NN expansion model using the cached/generated v14 CSV "
    "and the frozen v12 benchmark-v2 validation/blind/stress split manifest."
)
runtime_config.write_text(json.dumps(config, indent=2) + "\n")
print("runtime batch config:", runtime_config)
print("v14 csv:", csv_path)
print("benchmark-v2 split manifest:", manifest_path)
PY

python3 scripts/run_model_selection_batch.py \
  --batch-config "$RUNTIME_CONFIG" \
  --output-root training_results/model_selection_loop \
  --summary-root training_results/model_selection_batches \
  --max-runtime-minutes 420

RUN_DIR=$(ls -td training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v14_1p5m_far_nn_d66_seed23_* 2>/dev/null | head -1 || true)
if [ -z "$RUN_DIR" ]; then
  python3 - "$PENDING_MARKER" "$V14_CSV" "$SAMPLING_QUALITY_DIR" "$DRIVE_BACKUP_RUN_NAME" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

marker = Path(sys.argv[1])
marker.write_text(
    json.dumps(
        {
            "status": "pending",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "workflow": "scripts/run_colab_v14_1p5m_far_nn_workflow.sh",
            "dataset_csv": sys.argv[2],
            "sampling_quality_dir": sys.argv[3],
            "drive_backup_run_name": sys.argv[4],
            "note": "v14 CSV and sampling-quality report exist, but training did not finish in this cycle. The next worker cycle will continue.",
        },
        indent=2,
    )
    + "\n"
)
PY
  echo "v14 1.5M far-NN training is not complete yet; leaving workflow pending."
  exit 0
fi

python3 scripts/audit_s3_magnitude_metric.py --run "v14_1p5m_far_nn_seed23=$RUN_DIR"
python3 scripts/audit_training_validation_loss_gap.py --include-default-runs --run "$RUN_DIR"

V14_BATCH_SUMMARY=$(ls -td training_results/model_selection_batches/v14_1p5m_far_nn_d66_*/batch_summary.csv 2>/dev/null | head -1 || true)
if [ -n "$V14_BATCH_SUMMARY" ]; then
  python3 scripts/report_data_scale_learning_curve.py \
    --include-default-batches \
    --batch-summary "v14_1p5m_far_nn=$V14_BATCH_SUMMARY"
fi

echo "Updating Drive backup after v14 training and audits."
python3 scripts/backup_colab_state_to_drive.py --run-name "$DRIVE_BACKUP_RUN_NAME"

python3 - "$DONE_MARKER" "$V14_CSV" "$V2_SPLIT_MANIFEST" "$RUN_DIR" "$SAMPLING_QUALITY_DIR" "$DRIVE_BACKUP_RUN_NAME" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

marker = Path(sys.argv[1])
marker.write_text(
    json.dumps(
        {
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "workflow": "scripts/run_colab_v14_1p5m_far_nn_workflow.sh",
            "dataset_csv": sys.argv[2],
            "benchmark_v2_split_manifest": sys.argv[3],
            "run_dir": sys.argv[4],
            "sampling_quality_dir": sys.argv[5],
            "drive_backup_run_name": sys.argv[6],
            "diagnostics": [
                "sampling_quality_v14_1p5m_far_nn_d66",
                "v14_1p5m_far_nn_d66_training",
                "residual_vs_12d_nearest_neighbor_distance",
                "vector_diagnostics",
                "s3_magnitude_metric_audit",
                "training_validation_loss_gap_audit",
                "data_scale_learning_curve",
                "drive_backup"
            ],
            "decision": "Compare v14 against v13 seed23. Expand beyond 1.5M only if residual-vs-NN remains coverage-limited after this targeted expansion.",
        },
        indent=2,
    )
    + "\n"
)
print("done marker:", marker)
PY
