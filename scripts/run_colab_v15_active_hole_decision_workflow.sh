#!/usr/bin/env bash
set -euo pipefail

DONE_MARKER="colab_worker_logs/v15_active_hole_decision_done.json"

mkdir -p colab_worker_logs

if [ -f "$DONE_MARKER" ]; then
  echo "v15 active-hole decision report already completed; marker exists at $DONE_MARKER"
  exit 0
fi

REPORT_JSON=$(python3 scripts/report_v15_active_hole_decision.py)
REPORT_DIR=$(python3 - "$REPORT_JSON" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
print(payload["output_dir"])
PY
)

python3 - "$DONE_MARKER" "$REPORT_DIR" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "status": "complete",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "workflow": "scripts/run_colab_v15_active_hole_decision_workflow.sh",
    "report_dir": sys.argv[2],
}, indent=2) + "\n")
PY

echo "v15 active-hole decision report: $REPORT_DIR"
