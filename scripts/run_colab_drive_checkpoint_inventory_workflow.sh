#!/usr/bin/env bash
set -euo pipefail

DONE_MARKER="colab_worker_logs/drive_checkpoint_inventory_done.json"
DRIVE_BACKUP_ROOT="${ABERRATION_DRIVE_BACKUP_ROOT:-/content/drive/MyDrive/Aberration-Simulation-Colab-Backups}"

if [ -f "$DONE_MARKER" ]; then
  echo "Drive checkpoint inventory already completed; marker exists at $DONE_MARKER"
  exit 0
fi

mkdir -p colab_worker_logs

python3 scripts/inventory_drive_checkpoints.py \
  --drive-root "$DRIVE_BACKUP_ROOT" \
  --local-root training_results/model_selection_loop \
  --output-dir colab_worker_logs

python3 - "$DONE_MARKER" "$DRIVE_BACKUP_ROOT" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "status": "complete",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "workflow": "scripts/run_colab_drive_checkpoint_inventory_workflow.sh",
    "drive_backup_root": sys.argv[2],
    "note": "Checkpoint inventory only. No training, no simulation, no inference.",
}, indent=2) + "\n")
PY
