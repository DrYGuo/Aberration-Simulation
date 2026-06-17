#!/usr/bin/env bash
set -euo pipefail

DONE_MARKER="colab_worker_logs/drive_model_file_presence_done.json"
DRIVE_BACKUP_ROOT="${ABERRATION_DRIVE_BACKUP_ROOT:-/content/drive/MyDrive/Aberration-Simulation-Colab-Backups}"

mkdir -p colab_worker_logs

python3 scripts/check_drive_model_file_presence.py \
  --drive-root "$DRIVE_BACKUP_ROOT" \
  --output-dir colab_worker_logs

python3 - "$DONE_MARKER" "$DRIVE_BACKUP_ROOT" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "status": "complete",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "workflow": "scripts/run_colab_drive_model_file_presence_workflow.sh",
    "drive_backup_root": sys.argv[2],
    "note": "Google Drive model file presence check only. No training, no simulation, no inference.",
}, indent=2) + "\n")
PY
