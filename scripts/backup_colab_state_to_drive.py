"""Back up Colab training data and compact results to Google Drive.

This is intentionally outside the GitHub artifact path. It copies large cached
CSV datasets and result folders that are too large or inappropriate for GitHub.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/Aberration-Simulation-Colab-Backups"
DEFAULT_INCLUDE_PATHS = [
    "training_results",
    "configs",
    "experiments",
    "colab_worker_logs",
    "MODEL_EVOLUTION.md",
    "CURRENT_STATE.md",
    "STABLE_CHECKPOINT.txt",
]
DEFAULT_EXCLUDE_PATTERNS = [
    "__pycache__",
    ".ipynb_checkpoints",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.zip",
    "*.npy",
    "*.npz",
    "*.h5",
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


def current_commit(repo_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def mount_google_drive(mount_point: Path) -> bool:
    if mount_point.exists() and any(mount_point.iterdir()):
        return True
    try:
        from google.colab import drive  # type: ignore
    except Exception:
        return False
    print(f"mounting Google Drive at {mount_point}", flush=True)
    try:
        drive.mount(str(mount_point), force_remount=False)
    except Exception as exc:
        print(
            "Google Drive mount could not be started from this Python process. "
            "In Colab/VS Code, run this in a notebook cell first:\n\n"
            "from google.colab import drive\n"
            "drive.mount('/content/drive')\n\n"
            "Then rerun the worker cell.",
            flush=True,
        )
        print(f"mount error: {type(exc).__name__}: {exc}", flush=True)
        return False
    return mount_point.exists() and any(mount_point.iterdir())


def path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def copy_with_rsync(source: Path, dest: Path) -> bool:
    if shutil.which("rsync") is None:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    source_arg = f"{source}/" if source.is_dir() else str(source)
    dest_arg = f"{dest}/" if source.is_dir() else str(dest)
    command = ["rsync", "-a"]
    for pattern in DEFAULT_EXCLUDE_PATTERNS:
        command.extend(["--exclude", pattern])
    command.extend([source_arg, dest_arg])
    print("$", " ".join(command), flush=True)
    result = subprocess.run(command)
    return result.returncode == 0


def copy_path(source: Path, dest: Path) -> None:
    if not source.exists():
        print(f"skip missing source: {source}", flush=True)
        return
    if copy_with_rsync(source, dest):
        return
    if source.is_dir():
        if dest.exists():
            shutil.rmtree(dest)
        print(f"copy directory {source} -> {dest}", flush=True)
        shutil.copytree(
            source,
            dest,
            ignore=shutil.ignore_patterns(*DEFAULT_EXCLUDE_PATTERNS),
        )
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"copy file {source} -> {dest}", flush=True)
        shutil.copy2(source, dest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--drive-root",
        type=Path,
        default=Path(os.environ.get("ABERRATION_DRIVE_BACKUP_ROOT", DEFAULT_DRIVE_ROOT)),
    )
    parser.add_argument("--run-name", default="")
    parser.add_argument("--include", action="append", default=[], help="Additional path to include.")
    parser.add_argument(
        "--no-default-includes",
        action="store_true",
        help=(
            "Do not copy the default broad backup set. Use this for incremental "
            "workflow backups that should update only explicitly listed folders."
        ),
    )
    parser.add_argument("--no-mount", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    drive_root = args.drive_root
    mount_point = Path("/content/drive")
    if not args.no_mount and str(drive_root).startswith("/content/drive"):
        if not mount_google_drive(mount_point):
            raise RuntimeError(
                "Google Drive is not mounted. Run the Drive mount cell in the notebook "
                "kernel first, then rerun this worker. For Shared Drives, mount Drive "
                "first and set ABERRATION_DRIVE_BACKUP_ROOT to a path under "
                "/content/drive/Shareddrives/<drive-name>/."
            )

    run_name = args.run_name or f"backup_{utc_stamp()}"
    backup_dir = drive_root / run_name
    backup_dir.mkdir(parents=True, exist_ok=True)

    include_paths = [*([] if args.no_default_includes else DEFAULT_INCLUDE_PATHS), *args.include]
    if not include_paths:
        raise RuntimeError("No backup paths requested. Provide --include or omit --no-default-includes.")
    copied: list[dict[str, Any]] = []
    for rel_text in include_paths:
        source = repo_root / rel_text
        dest = backup_dir / rel_text
        before = path_size(source)
        copy_path(source, dest)
        copied.append(
            {
                "source": str(source),
                "destination": str(dest),
                "exists": source.exists(),
                "bytes": before,
            }
        )

    manifest = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "git_commit": current_commit(repo_root),
        "backup_dir": str(backup_dir),
        "default_includes_used": not args.no_default_includes,
        "exclude_patterns": DEFAULT_EXCLUDE_PATTERNS,
        "copied": copied,
        "note": "Large Colab training CSVs and result folders are backed up here, not pushed to GitHub.",
    }
    manifest_path = repo_root / "colab_worker_logs" / f"drive_backup_{utc_stamp()}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (backup_dir / "drive_backup_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("drive backup directory:", backup_dir, flush=True)
    print("drive backup manifest:", manifest_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
