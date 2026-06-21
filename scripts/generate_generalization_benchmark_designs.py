#!/usr/bin/env python3
"""Generate frozen coefficient-row designs for generalization benchmark v1.

This creates benchmark/test designs only. It does not simulate line profiles,
extract features, run model inference, or create training rows.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_targeted_enhanced_dataset import (
    BASELINE_PARAMETERS,
    COMBINATION_FIELDS,
    SAMPLING_METADATA_FIELDS,
    angle_category,
    generate_target_cases,
    phase_from_vector_angle,
    relative_angle_deg,
    write_csv,
)


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


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def normalize_counts(counts: dict[str, int], total: int) -> dict[str, int]:
    current = sum(counts.values())
    if current != total:
        raise ValueError(f"case counts sum to {current}, expected {total}: {counts}")
    return counts


def counts_from_fractions(fractions: dict[str, float], total: int) -> dict[str, int]:
    if total <= 0:
        raise ValueError(f"row count must be positive, got {total}")
    raw = {label: float(value) * total for label, value in fractions.items()}
    counts = {label: int(np.floor(value)) for label, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(fractions, key=lambda label: raw[label] - counts[label], reverse=True)
    for label in order[:remainder]:
        counts[label] += 1
    return normalize_counts(counts, total)


def add_benchmark_metadata(rows: list[dict[str, Any]], *, version: str, source: str, role: str) -> None:
    for index, row in enumerate(rows):
        row["benchmark_probe_id"] = f"{version}_{index:06d}"
        row["dataset_version"] = version
        row["dataset_source"] = source
        row["dataset_split_hint"] = "diagnostic_validation_only"
        row["train_on_this"] = False
        row["selection_primary_metric"] = True
        row.setdefault("sampling_candidate_role", role)
        row.setdefault("sampling_parent_nn_distance_12d", "")
        row.setdefault("sampling_relative_angle_bin", "")
        row.setdefault("sampling_method", "design")


def fieldnames() -> list[str]:
    return list(
        dict.fromkeys(
            [
                "benchmark_probe_id",
                *COMBINATION_FIELDS,
                *SAMPLING_METADATA_FIELDS,
                "dataset_version",
                "dataset_source",
                "dataset_split_hint",
                "train_on_this",
                "selection_primary_metric",
            ]
        )
    )


def write_head_sample(source: Path, dest: Path, limit: int = 1000) -> int:
    with source.open() as handle:
        reader = csv.DictReader(handle)
        rows = []
        for index, row in enumerate(reader):
            if index >= limit:
                break
            rows.append(row)
        names = list(reader.fieldnames or [])
    write_csv(dest, rows, names)
    return len(rows)


def count_rows(path: Path) -> int:
    with path.open() as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def row_float(row: dict[str, Any], name: str) -> float:
    try:
        return float(row.get(name, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def vector_amp(row: dict[str, Any], group: str) -> float:
    return row_float(row, f"{group}_amp")


def summarize_design(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(str(row.get("sweep_label", "")) for row in rows)
    methods = Counter(str(row.get("sampling_method", "")) for row in rows)
    angle_bins = Counter(str(row.get("sampling_relative_angle_bin", "")) for row in rows)
    amplitudes = {
        group: [vector_amp(row, group) for row in rows]
        for group in ("A1", "B2", "A2", "S3", "A3")
    }
    amp_summary = {
        group: {
            "median": float(np.median(values)) if values else None,
            "p90": float(np.quantile(values, 0.90)) if values else None,
            "nonzero_fraction": float(np.mean(np.asarray(values) > 1e-8)) if values else None,
        }
        for group, values in amplitudes.items()
    }
    rel_pairs = {
        "A1_S3": [],
        "B2_S3": [],
        "A3_S3": [],
    }
    for row in rows:
        for pair_name, left, right in (
            ("A1_S3", "A1", "S3"),
            ("B2_S3", "B2", "S3"),
            ("A3_S3", "A3", "S3"),
        ):
            if vector_amp(row, left) <= 1e-8 or vector_amp(row, right) <= 1e-8:
                continue
            rel_pairs[pair_name].append(
                angle_category(
                    relative_angle_deg(row, left, right)
                )
            )
    return {
        "row_count": len(rows),
        "sweep_label_counts": dict(labels),
        "sampling_method_counts": dict(methods),
        "sampling_relative_angle_bin_counts": dict(angle_bins),
        "amplitude_summary": amp_summary,
        "relative_angle_category_counts": {
            name: dict(Counter(values))
            for name, values in rel_pairs.items()
        },
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    counts = Counter(str(row.get("sweep_label", "")) for row in rows)
    output = [
        {
            "sweep_label": label,
            "n": count,
            "fraction": count / max(len(rows), 1),
        }
        for label, count in sorted(counts.items())
    ]
    write_csv(path, output, ["sweep_label", "n", "fraction"])


def broad_generation_config(total_rows: int) -> dict[str, Any]:
    counts = counts_from_fractions(
        {
            "coupled_full_random": 0.420,
            "coupled_sparse_random": 0.140,
            "coupled_A3_S3_random": 0.090,
            "coupled_B2_S3_random": 0.080,
            "coupled_A1_B2_S3_random": 0.070,
            "coupled_A1_S3_random": 0.060,
            "coupled_C3_A3_S3_random": 0.060,
            "coupled_A1_A2_B2_random": 0.030,
            "coupled_C1_C3_random": 0.025,
            "S3_high_random": 0.025,
        },
        total_rows,
    )
    return {
        "case_counts": counts,
        "sampling": {
            "space_filling": {
                "enabled": True,
                "method": "latin_hypercube",
                "labels": list(counts),
            },
            "relative_angles": {
                "enabled": True,
                "bins": ["aligned", "orthogonal", "anti_aligned", "random"],
                "label_pairs": {
                    "coupled_full_random": [["A1", "S3"], ["B2", "S3"], ["A3", "S3"]],
                    "coupled_A3_S3_random": [["A3", "S3"]],
                    "coupled_B2_S3_random": [["B2", "S3"]],
                    "coupled_A1_B2_S3_random": [["A1", "S3"], ["B2", "S3"]],
                    "coupled_A1_S3_random": [["A1", "S3"]],
                    "coupled_C3_A3_S3_random": [["A3", "S3"]],
                },
            },
            "s3_tail": {
                "enabled": True,
                "labels": ["S3_high_random", "coupled_A3_S3_random", "coupled_C3_A3_S3_random"],
                "s3_amp_min": 55.0,
                "s3_amp_max": 100.0,
                "force_s3_in_sparse": True,
            },
            "c1_balanced": {
                "enabled": True,
                "labels": ["coupled_full_random", "coupled_sparse_random", "coupled_C1_C3_random"],
                "magnitude_bins": [[0.0, 10.0], [10.0, 40.0], [40.0, 75.0], [75.0, 100.0]],
            },
        },
    }


def make_anchor_row(label: str) -> dict[str, Any]:
    row = dict(BASELINE_PARAMETERS)
    row["sweep_label"] = label
    row["sampling_method"] = "anchor_grid"
    row["sampling_relative_angle_bin"] = ""
    row["sampling_candidate_role"] = "anchor_easy"
    row["sampling_parent_nn_distance_12d"] = ""
    return row


def set_vector(row: dict[str, Any], group: str, amp: float, vector_angle: float) -> None:
    row[f"{group}_amp"] = float(amp)
    row[f"{group}_phase"] = phase_from_vector_angle(group, vector_angle)


def anchor_easy_rows(seed: int, total_rows: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    counts = counts_from_fractions(
        {
            "anchor_zero": 0.04,
            "anchor_C1_sweep": 0.12,
            "anchor_C3_sweep": 0.08,
            "anchor_A1_sweep": 0.12,
            "anchor_B2_sweep": 0.12,
            "anchor_A2_sweep": 0.10,
            "anchor_S3_sweep": 0.16,
            "anchor_A3_sweep": 0.16,
            "anchor_weak_full_coupled": 0.10,
        },
        total_rows,
    )
    for _ in range(counts["anchor_zero"]):
        rows.append(make_anchor_row("anchor_zero"))
    for i in range(counts["anchor_C1_sweep"]):
        row = make_anchor_row("anchor_C1_sweep")
        row["C1"] = float(np.linspace(-100.0, 100.0, counts["anchor_C1_sweep"])[i])
        rows.append(row)
    for i in range(counts["anchor_C3_sweep"]):
        row = make_anchor_row("anchor_C3_sweep")
        row["C3"] = float(np.linspace(0.05, 2.0, counts["anchor_C3_sweep"])[i])
        rows.append(row)
    for group, count, high, label in (
        ("A1", counts["anchor_A1_sweep"], 60.0, "anchor_A1_sweep"),
        ("B2", counts["anchor_B2_sweep"], 3.0, "anchor_B2_sweep"),
        ("A2", counts["anchor_A2_sweep"], 16.0, "anchor_A2_sweep"),
        ("S3", counts["anchor_S3_sweep"], 100.0, "anchor_S3_sweep"),
        ("A3", counts["anchor_A3_sweep"], 100.0, "anchor_A3_sweep"),
    ):
        amps = np.linspace(0.0, high, count)
        angles = (np.arange(count) * 137.50776405) % 360.0
        for amp, angle in zip(amps, angles):
            row = make_anchor_row(label)
            set_vector(row, group, float(amp), float(angle))
            rows.append(row)
    for _ in range(counts["anchor_weak_full_coupled"]):
        row = make_anchor_row("anchor_weak_full_coupled")
        row["sampling_method"] = "latin_hypercube_weak_anchor"
        row["C1"] = float(rng.uniform(-15.0, 15.0))
        row["C3"] = float(rng.uniform(0.05, 0.35))
        set_vector(row, "A1", float(rng.uniform(0.0, 12.0)), float(rng.uniform(0.0, 360.0)))
        set_vector(row, "B2", float(rng.uniform(0.0, 0.6)), float(rng.uniform(0.0, 360.0)))
        set_vector(row, "A2", float(rng.uniform(0.0, 3.2)), float(rng.uniform(0.0, 360.0)))
        set_vector(row, "S3", float(rng.uniform(0.0, 20.0)), float(rng.uniform(0.0, 360.0)))
        set_vector(row, "A3", float(rng.uniform(0.0, 20.0)), float(rng.uniform(0.0, 360.0)))
        rows.append(row)
    rng.shuffle(rows)
    return rows


def write_design_bundle(
    *,
    output_dir: Path,
    design_name: str,
    rows: list[dict[str, Any]],
    seed: int,
    full_csv_name: str,
    compact_prefix: str,
    notes: list[str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    names = fieldnames()
    full_csv = output_dir / full_csv_name
    write_csv(full_csv, rows, names)
    head_csv = output_dir / f"{compact_prefix}_head1000.csv"
    head_count = write_head_sample(full_csv, head_csv)
    summary = summarize_design(rows)
    summary_csv = output_dir / f"{compact_prefix}_label_summary.csv"
    write_summary_csv(summary_csv, rows)
    summary_json = output_dir / f"{compact_prefix}_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "design_name": design_name,
        "seed": seed,
        "row_count": len(rows),
        "train_on_this": False,
        "selection_primary_metric": True,
        "git_commit": current_commit(Path.cwd()),
        "python": sys.version,
        "platform": platform.platform(),
        "full_design_csv": str(full_csv),
        "github_compact_head_csv": str(head_csv),
        "github_compact_head_rows": head_count,
        "summary_json": str(summary_json),
        "summary_csv": str(summary_csv),
        "large_artifact_policy": "Full design CSV is Drive-backed. GitHub should receive manifest, summary, report, and head sample only.",
        "notes": notes,
    }
    manifest_path = output_dir / f"{compact_prefix}_manifest.json"
    manifest["manifest"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    report_path = output_dir / f"{compact_prefix}_report.md"
    report_path.write_text(
        "\n".join(
            [
                f"# {design_name}",
                "",
                f"Created UTC: `{manifest['created_utc']}`",
                "",
                f"- rows: `{len(rows)}`",
                "- train on this: `false`",
                "- role: frozen validation/benchmark design",
                f"- full design CSV: `{full_csv}`",
                f"- compact GitHub sample: `{head_csv}`",
                "",
                "## Label Counts",
                "",
                "| label | n | fraction |",
                "|---|---:|---:|",
                *[
                    f"| `{label}` | {count} | {count / max(len(rows), 1):.6f} |"
                    for label, count in sorted(summary["sweep_label_counts"].items())
                ],
                "",
                "## Notes",
                "",
                *[f"- {note}" for note in notes],
                "",
            ]
        )
    )
    manifest["report"] = str(report_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/generalization_benchmark_v1.json"))
    parser.add_argument("--output-root", type=Path, default=Path("training_results/model_selection_reports"))
    parser.add_argument("--broad-rows", type=int, default=100000)
    parser.add_argument("--anchor-rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=4101)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    read_json(args.config)
    stamp = utc_stamp()
    broad_dir = args.output_root / f"broad_12d_representative_validation_v1_{stamp}"
    anchor_dir = args.output_root / f"anchor_easy_validation_v1_{stamp}"

    broad_rows = generate_target_cases(args.seed, broad_generation_config(args.broad_rows))
    add_benchmark_metadata(
        broad_rows,
        version="broad_12d_representative_validation_v1",
        source="generalization_benchmark_v1",
        role="broad_representative_validation",
    )
    broad_manifest = write_design_bundle(
        output_dir=broad_dir,
        design_name="Broad 12D Representative Validation v1",
        rows=broad_rows,
        seed=args.seed,
        full_csv_name="broad_12d_representative_validation_design.csv",
        compact_prefix="broad_12d_representative_validation",
        notes=[
            "Fresh frozen validation design for generalization scoring.",
            "Includes LHS space filling, coupled-full, coupled-sparse, high-vector couplings, relative-angle controls, and moderate bridge-like cases.",
            "Do not train on these exact rows.",
        ],
    )

    anchor_rows = anchor_easy_rows(args.seed + 17, args.anchor_rows)
    add_benchmark_metadata(
        anchor_rows,
        version="anchor_easy_validation_v1",
        source="generalization_benchmark_v1",
        role="anchor_easy_validation",
    )
    anchor_manifest = write_design_bundle(
        output_dir=anchor_dir,
        design_name="Anchor/Easy Validation v1",
        rows=anchor_rows,
        seed=args.seed + 17,
        full_csv_name="anchor_easy_validation_design.csv",
        compact_prefix="anchor_easy_validation",
        notes=[
            "Frozen anchor/easy validation design to guard against over-specialized active-hole repair.",
            "Includes zero, one-coefficient sweeps, and weak fully coupled cases.",
            "Do not train on these exact rows.",
        ],
    )

    output = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config),
        "broad_design_dir": str(broad_dir),
        "anchor_design_dir": str(anchor_dir),
        "broad_manifest": broad_manifest,
        "anchor_manifest": anchor_manifest,
        "training_launched": False,
        "simulation_launched": False,
        "next_step": "Simulate/evaluate broad and anchor designs together with the 5K new-hole challenge for v13/v15 before v16 training.",
    }
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
