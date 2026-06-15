"""Write a frozen split manifest for a v12 benchmark-v2 appended dataset.

The intended input is a combined CSV containing an existing v11 500K dataset
plus newly appended rows with ``dataset_version=enhanced_v12_benchmark_v2``.
Existing training-only rows remain training. Existing unhinted benchmark-v1
rows keep their prior v1 split assignment. New v12 benchmark rows are assigned
only to validation/blind/stress, stratified by sweep label.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

from feature_regression_model import file_sha256
from run_model_selection_candidate import (
    DATASET_SPLIT_HINT_FIELD,
    FROZEN_BENCHMARK_ROW_KEY_FIELDS,
    TRAINING_ONLY_HINT,
    frozen_benchmark_row_key,
    load_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--new-dataset-version", default="enhanced_v12_benchmark_v2")
    parser.add_argument("--validation-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--blind-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--stress-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--dataset-version", default="enhanced_v12_benchmark_v2")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_base_key_to_split(path: Path) -> dict[str, str]:
    manifest = json.loads(path.read_text())
    output: dict[str, str] = {}
    for split_name, keys in manifest.get("splits", {}).items():
        for key in keys:
            key = str(key)
            previous = output.get(key)
            if previous is not None and previous != split_name:
                raise RuntimeError(f"base manifest assigns key to multiple splits: {key}")
            output[key] = str(split_name)
    return output


def assign_new_rows_by_label(
    new_items: list[tuple[int, dict[str, str], str]],
    *,
    validation_fraction: float,
    blind_fraction: float,
) -> dict[str, list[str]]:
    grouped: dict[str, list[tuple[int, dict[str, str], str]]] = defaultdict(list)
    for item in new_items:
        _, row, _ = item
        grouped[str(row.get("sweep_label", ""))].append(item)

    splits: dict[str, list[str]] = {"validation": [], "blind": [], "stress": []}
    for label in sorted(grouped):
        items = sorted(grouped[label], key=lambda item: item[2])
        n = len(items)
        n_validation = int(round(validation_fraction * n))
        n_blind = int(round(blind_fraction * n))
        if n_validation + n_blind > n:
            n_blind = max(0, n - n_validation)
        for position, (_, _, key) in enumerate(items):
            if position < n_validation:
                splits["validation"].append(key)
            elif position < n_validation + n_blind:
                splits["blind"].append(key)
            else:
                splits["stress"].append(key)
    return splits


def split_counts(splits: dict[str, list[str]]) -> dict[str, int]:
    return {name: len(values) for name, values in splits.items()}


def main() -> int:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        print(f"benchmark-v2 split manifest already exists: {args.output}", flush=True)
        return 0
    if abs(args.validation_fraction + args.blind_fraction + args.stress_fraction - 1.0) > 1e-6:
        raise RuntimeError("validation/blind/stress fractions must sum to 1.0 for v2 benchmark rows")

    rows = load_rows(args.csv_path)
    base_key_to_split = load_base_key_to_split(args.base_manifest)
    splits: dict[str, list[str]] = {"train": [], "validation": [], "blind": [], "stress": []}
    new_items: list[tuple[int, dict[str, str], str]] = []
    missing_base_keys: list[str] = []

    for index, row in enumerate(rows):
        hint = str(row.get(DATASET_SPLIT_HINT_FIELD, "")).strip()
        version = str(row.get("dataset_version", "")).strip()
        if hint == TRAINING_ONLY_HINT:
            continue
        key = frozen_benchmark_row_key(row, index)
        if version == args.new_dataset_version:
            new_items.append((index, row, key))
            continue
        base_split = base_key_to_split.get(key)
        if base_split is None:
            missing_base_keys.append(key)
            continue
        splits[base_split].append(key)

    if missing_base_keys:
        raise RuntimeError(
            "base manifest does not cover all pre-v12 unhinted rows; "
            f"missing {len(missing_base_keys)} keys, sample={missing_base_keys[:5]}"
        )
    if not new_items:
        raise RuntimeError(f"no rows found with dataset_version={args.new_dataset_version!r}")

    new_splits = assign_new_rows_by_label(
        new_items,
        validation_fraction=args.validation_fraction,
        blind_fraction=args.blind_fraction,
    )
    for split_name, keys in new_splits.items():
        splits[split_name].extend(keys)

    key_to_split: dict[str, str] = {}
    for split_name, keys in splits.items():
        for key in keys:
            previous = key_to_split.get(key)
            if previous is not None and previous != split_name:
                raise RuntimeError(
                    f"frozen benchmark manifest assigns key to multiple splits: {key}"
                )
            key_to_split[key] = split_name

    hint_counts: dict[str, int] = {}
    version_counts: dict[str, int] = {}
    for row in rows:
        hint = str(row.get(DATASET_SPLIT_HINT_FIELD, "")).strip() or "unhinted"
        version = str(row.get("dataset_version", "")).strip() or "unversioned"
        hint_counts[hint] = hint_counts.get(hint, 0) + 1
        version_counts[version] = version_counts.get(version, 0) + 1

    payload: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 3,
        "dataset_version": args.dataset_version,
        "new_dataset_version": args.new_dataset_version,
        "source_csv": str(args.csv_path),
        "source_csv_sha256": file_sha256(args.csv_path),
        "base_manifest": str(args.base_manifest),
        "validation_fraction": args.validation_fraction,
        "blind_fraction": args.blind_fraction,
        "stress_fraction": args.stress_fraction,
        "row_key_function": "run_model_selection_candidate.frozen_benchmark_row_key",
        "row_key_fields": ["row_index", *FROZEN_BENCHMARK_ROW_KEY_FIELDS],
        "policy": (
            "Existing training_only rows remain train. Existing unhinted rows "
            "keep the base manifest split. New benchmark-v2 rows are split only "
            "across validation, blind, and stress by sweep label. Training-only "
            "row keys are intentionally omitted because the model runner assigns "
            "them to train automatically."
        ),
        "split_counts": split_counts(splits),
        "new_benchmark_v2_split_counts": split_counts(new_splits),
        "source_dataset_split_hint_counts": hint_counts,
        "source_dataset_version_counts": version_counts,
        "splits": splits,
        "python": sys.version,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote benchmark-v2 split manifest:", args.output, flush=True)
    print("split counts:", payload["split_counts"], flush=True)
    print("new benchmark-v2 split counts:", payload["new_benchmark_v2_split_counts"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
