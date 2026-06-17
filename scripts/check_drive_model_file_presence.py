"""Check mounted Google Drive for saved model files needed by active search."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


def file_payload(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "name": path.name,
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--drive-root",
        default="/content/drive/MyDrive/Aberration-Simulation-Colab-Backups",
    )
    parser.add_argument("--output-dir", default="colab_worker_logs")
    args = parser.parse_args()

    drive_root = Path(args.drive_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exact_patterns = [
        "*/training_results/model_selection_loop/"
        "D66_grouped_width320_lr6e-4_dropout0.075_v13_1m_d66_seed23_checkpoint_rebuild_*/"
        "model_loop_candidate.pt",
        "*/training_results/model_selection_loop/"
        "D66_grouped_width320_lr6e-4_dropout0.075_v13_1m_d66_seed23_residual_nn_*/"
        "model_loop_candidate.pt",
        "*/training_results/model_selection_loop/"
        "D66_grouped_width320_lr6e-4_dropout0.075_v13_1m_d66_seed23*/"
        "model_loop_candidate.pt",
    ]
    broad_patterns = [
        "*/training_results/model_selection_loop/*v13*seed23*/model_loop_candidate.pt",
        "*/training_results/model_selection_loop/*seed23*/model_loop_candidate.pt",
        "*/training_results/model_selection_loop/*/model_loop_candidate.pt",
    ]

    exact_matches: list[dict[str, object]] = []
    broad_matches: list[dict[str, object]] = []
    errors: list[str] = []

    if drive_root.exists():
        for pattern in exact_patterns:
            try:
                exact_matches.extend(file_payload(path) for path in sorted(drive_root.glob(pattern)))
            except OSError as exc:
                errors.append(f"{pattern}: {exc}")
        seen_exact = {item["path"] for item in exact_matches}
        for pattern in broad_patterns:
            try:
                for path in sorted(drive_root.glob(pattern)):
                    payload = file_payload(path)
                    if payload["path"] not in seen_exact:
                        broad_matches.append(payload)
            except OSError as exc:
                errors.append(f"{pattern}: {exc}")

    payload = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "drive_root": str(drive_root),
        "drive_root_exists": drive_root.exists(),
        "exact_patterns": exact_patterns,
        "broad_patterns": broad_patterns,
        "exact_match_count": len(exact_matches),
        "broad_match_count": len(broad_matches),
        "exact_matches": exact_matches,
        "broad_matches": broad_matches,
        "errors": errors,
        "decision": (
            "ready_for_active_inference"
            if exact_matches
            else "model_file_not_found_in_drive_root"
        ),
    }
    output_path = output_dir / f"drive_model_file_presence_{utc_stamp()}.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"drive model file presence: {output_path}")
    print(f"exact model_loop_candidate.pt matches: {len(exact_matches)}")
    print(f"broad model_loop_candidate.pt matches: {len(broad_matches)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
