"""Preflight checks for active 12D hole-search Colab runs.

The script only inventories expected inputs. It does not train, simulate, or
run inference.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/Aberration-Simulation-Colab-Backups"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


def describe_file(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def describe_matches(paths: list[Path], *, limit: int = 25) -> list[dict[str, Any]]:
    described: list[dict[str, Any]] = []
    for path in sorted(paths, key=lambda item: item.stat().st_mtime, reverse=True)[:limit]:
        described.append(describe_file(path))
    return described


def glob_files(root: Path, patterns: list[str]) -> list[Path]:
    matches: dict[str, Path] = {}
    if not root.exists():
        return []
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file():
                matches[str(path)] = path
    return list(matches.values())


def glob_dirs(root: Path, patterns: list[str]) -> list[Path]:
    matches: dict[str, Path] = {}
    if not root.exists():
        return []
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_dir():
                matches[str(path)] = path
    return list(matches.values())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--drive-root", type=Path, default=Path(DEFAULT_DRIVE_ROOT))
    parser.add_argument("--output-dir", type=Path, default=Path("colab_worker_logs"))
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    drive_root = args.drive_root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    v13_csv_patterns = [
        "training_results/feature_regression_enhanced/enhanced_v13_1m_spacefill_*/training_features_enhanced.csv",
    ]
    model_patterns = [
        "training_results/model_selection_loop/*v13*seed23*/model_loop_candidate.pt",
        "training_results/model_selection_loop/*v13*/model_loop_candidate.pt",
        "training_results/model_selection_loop/*/model_loop_candidate.pt",
    ]
    model_dir_patterns = [
        "training_results/model_selection_loop/*v13*seed23*",
        "training_results/model_selection_loop/*v13*",
        "training_results/model_selection_loop/*",
    ]
    probe_feature_patterns = [
        "training_results/model_selection_reports/v13_active_12d_hole_search_*/active_hole_search_probe_features.csv",
        "training_results/model_selection_reports/recovered_active_probe_features/active_hole_search_probe_features.csv",
    ]
    split_manifest_patterns = [
        "configs/benchmark_split_v12_v2_row_keys.json",
    ]

    local_v13_csv = glob_files(repo_root, v13_csv_patterns)
    local_models = glob_files(repo_root, model_patterns)
    local_model_dirs = glob_dirs(repo_root, model_dir_patterns)
    local_probe_features = glob_files(repo_root, probe_feature_patterns)
    local_split_manifests = glob_files(repo_root, split_manifest_patterns)

    drive_v13_csv = glob_files(drive_root, [f"*/{pattern}" for pattern in v13_csv_patterns])
    drive_models = glob_files(drive_root, [f"*/{pattern}" for pattern in model_patterns])
    drive_model_dirs = glob_dirs(drive_root, [f"*/{pattern}" for pattern in model_dir_patterns])
    drive_probe_features = glob_files(drive_root, [f"*/{pattern}" for pattern in probe_feature_patterns])
    drive_split_manifests = glob_files(drive_root, [f"*/{pattern}" for pattern in split_manifest_patterns])

    has_v13_csv = bool(local_v13_csv or drive_v13_csv)
    has_model = bool(local_models or drive_models)
    has_split_manifest = bool(local_split_manifests or drive_split_manifests)
    has_probe_features = bool(local_probe_features or drive_probe_features)

    payload = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "drive_root": str(drive_root),
        "drive_root_exists": drive_root.exists(),
        "checks": {
            "v13_csv": {
                "ready": has_v13_csv,
                "local": describe_matches(local_v13_csv),
                "drive": describe_matches(drive_v13_csv),
            },
            "v13_model_checkpoint": {
                "ready": has_model,
                "local": describe_matches(local_models),
                "drive": describe_matches(drive_models),
                "local_candidate_dirs": [str(path) for path in sorted(local_model_dirs, key=lambda item: item.stat().st_mtime, reverse=True)[:25]],
                "drive_candidate_dirs": [str(path) for path in sorted(drive_model_dirs, key=lambda item: item.stat().st_mtime, reverse=True)[:25]],
            },
            "active_probe_features": {
                "ready": has_probe_features,
                "local": describe_matches(local_probe_features),
                "drive": describe_matches(drive_probe_features),
                "note": "Optional. If absent, the active workflow may generate/simulate probes.",
            },
            "benchmark_split_manifest": {
                "ready": has_split_manifest,
                "local": describe_matches(local_split_manifests),
                "drive": describe_matches(drive_split_manifests),
            },
        },
        "ready_for_no_training_inference": bool(has_v13_csv and has_model and has_split_manifest),
        "ready_for_active_workflow_with_simulation": bool(has_v13_csv and has_split_manifest),
        "next_action": (
            "Run active inference/probe workflow without training."
            if has_v13_csv and has_model and has_split_manifest
            else "Do not run active inference yet; restore or explicitly rebuild the missing model checkpoint."
        ),
        "artifact_policy": "Compact JSON only. No training, simulation, inference, checkpoints, or large CSVs are created.",
    }

    output_path = output_dir / f"active_12d_preflight_{utc_stamp()}.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"active 12D preflight: {output_path}")
    print(f"ready_for_no_training_inference: {payload['ready_for_no_training_inference']}")
    print(f"ready_for_active_workflow_with_simulation: {payload['ready_for_active_workflow_with_simulation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
