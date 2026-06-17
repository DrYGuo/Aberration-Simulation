"""Inventory checkpoint-like files in Colab local/Drive backup storage.

This is intentionally lightweight and dependency-free. It is used when an
inference workflow expects a saved model checkpoint but the active Drive backup
restore does not find one at the known champion path.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt"}
NAME_MARKERS = ("model_loop_candidate", "checkpoint", "candidate.pt")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


def looks_like_checkpoint(path: Path) -> bool:
    name = path.name.lower()
    return path.suffix.lower() in CHECKPOINT_SUFFIXES or any(marker in name for marker in NAME_MARKERS)


def scan_root(root: Path, *, max_matches: int) -> dict[str, object]:
    matches: list[dict[str, object]] = []
    errors: list[str] = []
    visited_files = 0
    if not root.exists():
        return {
            "root": str(root),
            "exists": False,
            "visited_files": 0,
            "match_count": 0,
            "matches": [],
            "errors": [],
        }

    try:
        iterator = root.rglob("*")
        for path in iterator:
            if not path.is_file():
                continue
            visited_files += 1
            if not looks_like_checkpoint(path):
                continue
            try:
                stat = path.stat()
                size_bytes = stat.st_size
                modified_utc = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
            except OSError as exc:
                errors.append(f"{path}: {exc}")
                size_bytes = None
                modified_utc = None
            matches.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "size_bytes": size_bytes,
                    "modified_utc": modified_utc,
                    "is_v13_seed23_candidate": (
                        "v13_1m_d66_seed23" in str(path)
                        and path.name == "model_loop_candidate.pt"
                    ),
                }
            )
            if len(matches) >= max_matches:
                break
    except OSError as exc:
        errors.append(f"{root}: {exc}")

    return {
        "root": str(root),
        "exists": True,
        "visited_files": visited_files,
        "match_count": len(matches),
        "matches": matches,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--drive-root",
        default="/content/drive/MyDrive/Aberration-Simulation-Colab-Backups",
        help="Mounted Google Drive backup root to inspect.",
    )
    parser.add_argument(
        "--local-root",
        default="training_results/model_selection_loop",
        help="Local Colab model-selection folder to inspect.",
    )
    parser.add_argument(
        "--output-dir",
        default="colab_worker_logs",
        help="Directory for the compact JSON inventory.",
    )
    parser.add_argument("--max-matches-per-root", type=int, default=500)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scans = [
        scan_root(Path(args.drive_root), max_matches=args.max_matches_per_root),
        scan_root(Path(args.local_root), max_matches=args.max_matches_per_root),
    ]
    all_matches = [match for scan in scans for match in scan["matches"]]
    v13_seed23_matches = [match for match in all_matches if match["is_v13_seed23_candidate"]]

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "purpose": "Locate saved checkpoint artifacts for frozen v13 active 12D hole-search inference.",
        "scans": scans,
        "total_match_count": len(all_matches),
        "v13_seed23_model_loop_candidate_count": len(v13_seed23_matches),
        "v13_seed23_model_loop_candidate_matches": v13_seed23_matches,
        "decision_hint": (
            "Use a matching v13 seed23 model_loop_candidate.pt for active-probe inference."
            if v13_seed23_matches
            else "No v13 seed23 model_loop_candidate.pt found. Active inference needs an explicit one-time checkpoint rebuild or a separately uploaded checkpoint."
        ),
    }
    output_path = output_dir / f"drive_checkpoint_inventory_{utc_stamp()}.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"drive checkpoint inventory: {output_path}")
    print(f"v13 seed23 checkpoint matches: {len(v13_seed23_matches)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
