#!/usr/bin/env bash
set -euo pipefail

mkdir -p colab_worker_logs

DRIVE_BACKUP_ROOT="${ABERRATION_DRIVE_BACKUP_ROOT:-/content/drive/MyDrive/Aberration-Simulation-Colab-Backups}"

python3 scripts/active_12d_preflight.py \
  --repo-root . \
  --drive-root "$DRIVE_BACKUP_ROOT" \
  --output-dir colab_worker_logs
