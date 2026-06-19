#!/usr/bin/env python3
"""Preflight the active failed-subspace sampler before large Colab generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from generate_targeted_enhanced_dataset import generate_active_failed_subspace_cases


REQUIRED_CASE_KEYS = (
    "sweep_label",
    "sampling_method",
    "C1",
    "C3",
    "A1_amp",
    "B2_amp",
    "A2_amp",
    "S3_amp",
    "A3_amp",
    "sampling_relative_angle_bin",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=157)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    sampler_config = config.get("sampling", {}).get("active_failed_subspace_jitter", {})
    if not sampler_config.get("enabled", False):
        raise RuntimeError(f"active_failed_subspace_jitter is not enabled in {args.config}")
    spec_path = Path(sampler_config["spec_path"])
    if not spec_path.exists():
        raise RuntimeError(f"active failed-subspace spec does not exist: {spec_path}")

    rows = generate_active_failed_subspace_cases(
        max(1, int(args.count)),
        np.random.default_rng(int(args.seed)),
        sampler_config,
    )
    if not rows:
        raise RuntimeError("active failed-subspace preflight generated no rows")

    missing = sorted(key for key in REQUIRED_CASE_KEYS if key not in rows[0])
    if missing:
        raise RuntimeError(f"active failed-subspace generated row is missing keys: {missing}")

    summary = {
        "status": "ok",
        "config": str(args.config),
        "spec_path": str(spec_path),
        "generated_rows": len(rows),
        "first_row_label": rows[0].get("sweep_label"),
        "first_row_method": rows[0].get("sampling_method"),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
