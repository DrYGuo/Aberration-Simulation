#!/usr/bin/env bash
set -euo pipefail

mkdir -p colab_worker_logs

DONE_MARKER="colab_worker_logs/drive_backup_before_1m_done.json"
if [ -f "$DONE_MARKER" ]; then
  echo "Drive backup before 1M already completed; marker exists at $DONE_MARKER"
  exit 0
fi

RUN_NAME="before_1m_$(date -u +%Y%m%d_%H%M%S_utc)"
MANIFEST_OUTPUT=$(python3 scripts/backup_colab_state_to_drive.py --run-name "$RUN_NAME" | tee colab_worker_logs/drive_backup_before_1m.log)
BACKUP_DIR=$(printf '%s\n' "$MANIFEST_OUTPUT" | awk -F': ' '/drive backup directory:/ {print $2}' | tail -1)

python3 - "$DONE_MARKER" "$RUN_NAME" "$BACKUP_DIR" <<'PY'
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
            "workflow": "scripts/run_colab_drive_backup_workflow.sh",
            "run_name": sys.argv[2],
            "backup_dir": sys.argv[3],
            "note": "Large Colab training data and result folders were copied to Google Drive; GitHub receives only compact backup manifests/logs.",
        },
        indent=2,
    )
    + "\n"
)
print("done marker:", marker)
PY
