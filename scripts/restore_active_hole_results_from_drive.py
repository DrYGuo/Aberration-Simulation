#!/usr/bin/env python3
"""Restore lightweight active-hole search result folders from Google Drive backups."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


LIGHTWEIGHT_NAMES = {
    "active_hole_clusters.csv",
    "active_hole_nn_error_summary.csv",
    "active_hole_regime_summary.csv",
    "active_hole_relative_angle_summary.csv",
    "active_hole_search_recommended_sampling_plan.json",
    "active_hole_search_report.md",
    "active_hole_search_state.json",
    "active_hole_search_summary.json",
    "active_hole_search_top_failures.csv",
    "inference_manifest.json",
    "proposal_summary.json",
    "selected_probe_design.csv",
    "simulation_manifest.json",
}


def iter_active_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    candidates: list[Path] = []
    for path in root.rglob("active_hole_search_top_failures.csv"):
        run_dir = path.parent
        if run_dir.name.startswith("v13_active_12d_hole_search"):
            candidates.append(run_dir)
    return sorted(set(candidates))


def copy_lightweight_run(source: Path, output_root: Path) -> dict[str, object]:
    dest = output_root / source.name
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in LIGHTWEIGHT_NAMES:
        src = source / name
        if not src.exists():
            continue
        dst = dest / name
        shutil.copy2(src, dst)
        copied.append(name)
    plots = source / "plots"
    if plots.exists():
        (dest / "plots").mkdir(exist_ok=True)
        for src in sorted(plots.glob("*.png")):
            shutil.copy2(src, dest / "plots" / src.name)
            copied.append(f"plots/{src.name}")
    return {"source": str(source), "dest": str(dest), "copied_files": copied}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drive-root",
        type=Path,
        default=Path("/content/drive/MyDrive/Aberration-Simulation-Colab-Backups"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("training_results/model_selection_reports"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("colab_worker_logs/restored_active_hole_results_from_drive.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    restored = []
    for source in iter_active_dirs(args.drive_root):
        restored.append(copy_lightweight_run(source, args.output_root))

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "drive_root": str(args.drive_root),
        "output_root": str(args.output_root),
        "restored_count": len(restored),
        "restored": restored,
    }
    args.manifest.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
