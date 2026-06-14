"""Write a frozen benchmark split manifest from a cached feature CSV.

The manifest stores stable row keys for train/validation/blind/stress benchmark
rows. Future datasets may append ``dataset_split_hint=training_only`` rows; those
rows remain training-only while unhinted benchmark rows are assigned from this
manifest.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

from feature_regression_model import TARGET_COLUMNS, file_sha256, target_from_row
from regression_diagnostics import discover_regime_column
from run_model_selection_candidate import (
    DATASET_SPLIT_HINT_FIELD,
    TRAINING_ONLY_HINT,
    four_way_benchmark_split,
    load_rows,
    stable_row_key,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("configs/benchmark_split_v6_frozen_row_keys.json"))
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--blind-fraction", type=float, default=0.10)
    parser.add_argument("--stress-fraction", type=float, default=0.20)
    parser.add_argument("--dataset-version", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        print(f"frozen split manifest already exists: {args.output}", flush=True)
        return 0

    rows = load_rows(args.csv_path)
    if not rows:
        raise RuntimeError(f"CSV is empty: {args.csv_path}")
    regime_column = discover_regime_column(rows)
    labels = np.asarray([row.get(regime_column, "") if regime_column else "" for row in rows])
    y = np.asarray([target_from_row(row) for row in rows], dtype=np.float32)
    split_indices = four_way_benchmark_split(
        rows,
        labels,
        y,
        validation_fraction=args.validation_fraction,
        blind_fraction=args.blind_fraction,
        stress_fraction=args.stress_fraction,
        seed=args.split_seed,
    )

    splits = {
        split_name: [stable_row_key(rows[int(index)], int(index)) for index in indices]
        for split_name, indices in split_indices.items()
    }
    key_to_split: dict[str, str] = {}
    for split_name, keys in splits.items():
        for key in keys:
            previous = key_to_split.get(key)
            if previous is not None and previous != split_name:
                raise RuntimeError(
                    "Stable benchmark key was assigned to multiple splits. "
                    f"key={key!r}, first_split={previous!r}, second_split={split_name!r}. "
                    "Add a stable unique row ID before freezing this dataset."
                )
            key_to_split[key] = split_name

    hint_counts: dict[str, int] = {}
    for row in rows:
        hint = str(row.get(DATASET_SPLIT_HINT_FIELD, "")).strip() or "unhinted_parent"
        hint_counts[hint] = hint_counts.get(hint, 0) + 1

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "dataset_version": args.dataset_version or "unknown",
        "source_csv": str(args.csv_path),
        "source_csv_sha256": file_sha256(args.csv_path),
        "split_seed": args.split_seed,
        "validation_fraction": args.validation_fraction,
        "blind_fraction": args.blind_fraction,
        "stress_fraction": args.stress_fraction,
        "row_key_function": "run_model_selection_candidate.stable_row_key",
        "row_key_fields": ["sweep_label", *TARGET_COLUMNS],
        "training_only_policy": (
            f"Rows with {DATASET_SPLIT_HINT_FIELD}={TRAINING_ONLY_HINT} are always assigned to train "
            "and are not required to appear in this manifest."
        ),
        "split_counts": {name: len(keys) for name, keys in splits.items()},
        "source_dataset_split_hint_counts": hint_counts,
        "splits": splits,
        "python": sys.version,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote frozen benchmark split manifest:", args.output, flush=True)
    print("split counts:", payload["split_counts"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
