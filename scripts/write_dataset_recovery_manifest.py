"""Write a compact recovery manifest for an already generated feature CSV."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-total-rows", type=int)
    parser.add_argument("--expected-new-rows", type=int)
    parser.add_argument("--dataset-version", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    dataset_version = args.dataset_version or str(config.get("dataset_version", ""))
    if not dataset_version:
        raise RuntimeError("dataset version must be supplied or present in config")

    row_count = 0
    version_counts: Counter[str] = Counter()
    split_hint_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    regime_counts: Counter[str] = Counter()
    new_regime_counts: Counter[str] = Counter()
    fieldnames: list[str] = []
    with args.csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            row_count += 1
            version = str(row.get("dataset_version", ""))
            version_counts[version] += 1
            split_hint_counts[str(row.get("dataset_split_hint", ""))] += 1
            source_counts[str(row.get("dataset_source", ""))] += 1
            label = str(row.get("sweep_label", ""))
            regime_counts[label] += 1
            if version == dataset_version:
                new_regime_counts[label] += 1

    new_rows = int(version_counts.get(dataset_version, 0))
    errors: list[str] = []
    if args.expected_total_rows is not None and row_count != args.expected_total_rows:
        errors.append(f"expected_total_rows={args.expected_total_rows}, observed={row_count}")
    if args.expected_new_rows is not None and new_rows != args.expected_new_rows:
        errors.append(f"expected_new_rows={args.expected_new_rows}, observed={new_rows}")

    feature_columns_path = args.csv_path.parent / "feature_columns_enhanced.json"
    feature_count = None
    if feature_columns_path.exists():
        try:
            feature_columns = json.loads(feature_columns_path.read_text())
            feature_count = len(feature_columns)
        except (OSError, json.JSONDecodeError):
            feature_count = None

    payload: dict[str, Any] = {
        "status": "valid" if not errors else "invalid",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_type": "recovered_from_existing_csv",
        "csv_path": str(args.csv_path),
        "csv_sha256": file_sha256(args.csv_path),
        "csv_size_bytes": args.csv_path.stat().st_size,
        "config": str(args.config),
        "dataset_version": dataset_version,
        "row_count": row_count,
        "new_dataset_row_count": new_rows,
        "expected_total_rows": args.expected_total_rows,
        "expected_new_rows": args.expected_new_rows,
        "dataset_version_counts": dict(version_counts),
        "dataset_split_hint_counts": dict(split_hint_counts),
        "dataset_source_counts": dict(source_counts),
        "regime_counts_after_merge": dict(regime_counts),
        "new_rows_per_regime": dict(new_regime_counts),
        "feature_columns_path": str(feature_columns_path) if feature_columns_path.exists() else "",
        "feature_count": feature_count,
        "field_count": len(fieldnames),
        "fieldnames": fieldnames,
        "errors": errors,
        "note": (
            "This manifest was regenerated from an existing CSV after the original generator "
            "was interrupted before writing dataset_manifest.json. It is sufficient for "
            "row-count validation and reproducibility bookkeeping, but it does not include "
            "the original in-memory audit payload."
        ),
        "python": sys.version,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print("dataset recovery manifest:", args.output, flush=True)
    print("status:", payload["status"], flush=True)
    print("row_count:", row_count, flush=True)
    print("new_dataset_row_count:", new_rows, flush=True)
    if errors:
        raise RuntimeError("; ".join(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
