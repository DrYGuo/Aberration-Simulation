#!/usr/bin/env bash
set -euo pipefail

mkdir -p colab_worker_logs

python3 - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path

Path("colab_worker_logs/noop_stop_done.json").write_text(json.dumps({
    "status": "complete",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "workflow": "scripts/run_colab_noop_stop_workflow.sh",
    "note": "Intentional no-training stop cycle. Used to let the Colab worker finish and disconnect while waiting for the next scientific decision.",
}, indent=2) + "\n")
PY

echo "No-op stop cycle complete. No training, simulation, or inference was run."
