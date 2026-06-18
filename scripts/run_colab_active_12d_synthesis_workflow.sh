#!/usr/bin/env bash
set -euo pipefail

DONE_MARKER="colab_worker_logs/active_12d_hole_search_synthesis_done.json"

mkdir -p colab_worker_logs training_results/model_selection_reports

if [ -f "$DONE_MARKER" ]; then
  echo "active 12D hole-search synthesis already completed; marker exists at $DONE_MARKER"
  exit 0
fi

python3 scripts/summarize_active_12d_hole_search_cycles.py \
  --search-root training_results/model_selection_reports \
  --output-root training_results/model_selection_reports

OUTPUT_DIR=$(ls -td training_results/model_selection_reports/active_12d_hole_search_synthesis_* 2>/dev/null | head -1 || true)
if [ -z "$OUTPUT_DIR" ]; then
  echo "Synthesis workflow did not produce an output directory." >&2
  exit 1
fi

python3 - "$DONE_MARKER" "$OUTPUT_DIR" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "workflow": "scripts/run_colab_active_12d_synthesis_workflow.sh",
            "output_dir": sys.argv[2],
            "decision": "Use this synthesis to prepare the next v15 targeted expansion and parallel high-amplitude feature/model diagnostics.",
        },
        indent=2,
    )
    + "\n"
)
print("done marker:", sys.argv[1])
print("synthesis output:", sys.argv[2])
PY
