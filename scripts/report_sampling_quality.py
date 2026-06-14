"""Report sampling quality for enhanced feature-regression datasets."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_model_selection_candidate import (  # noqa: E402
    DATASET_SPLIT_HINT_FIELD,
    TRAINING_ONLY_HINT,
    frozen_benchmark_split,
    load_rows,
)


COEFFICIENT_RANGES = {
    "C1": (-100.0, 100.0),
    "C3": (0.0, 2.0),
    "A1_amp": (0.0, 60.0),
    "A1_phase": (0.0, 180.0),
    "A2_amp": (0.0, 16.0),
    "A2_phase": (0.0, 120.0),
    "B2_amp": (0.0, 3.0),
    "B2_phase": (0.0, 360.0),
    "A3_amp": (0.0, 100.0),
    "A3_phase": (0.0, 90.0),
    "S3_amp": (0.0, 100.0),
    "S3_phase": (0.0, 180.0),
}

PAIRWISE_RANGES = [
    ("S3_amp", "A3_amp"),
    ("S3_amp", "B2_amp"),
    ("S3_amp", "A1_amp"),
    ("S3_amp", "C1"),
    ("S3_phase", "A3_phase"),
    ("S3_phase", "B2_phase"),
    ("A3_amp", "C3"),
    ("B2_amp", "A1_amp"),
    ("C1", "C3"),
]

VECTOR_ORDERS = {"A1": 2, "B2": 1, "A2": 3, "S3": 2, "A3": 4}
VECTOR_PAIRS = [("A3", "S3"), ("B2", "S3"), ("A1", "S3")]
ANGLE_BINS = ("aligned", "orthogonal", "anti_aligned", "random")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


def row_float(row: dict[str, str], name: str, default: float = 0.0) -> float:
    value = row.get(name, default)
    if value in (None, ""):
        return float(default)
    return float(value)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalized_entropy(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total <= 0 or len(counts) <= 1:
        return 0.0
    probabilities = counts[counts > 0] / total
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return entropy / math.log(len(counts))


def split_parent_new(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hints = np.asarray([str(row.get(DATASET_SPLIT_HINT_FIELD, "")).strip() for row in rows], dtype=object)
    new_mask = hints == TRAINING_ONLY_HINT
    parent_mask = ~new_mask
    return np.ones(len(rows), dtype=bool), parent_mask, new_mask


def subset_indices(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(mask)


def regime_quota_rows(rows: list[dict[str, str]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _, parent_mask, new_mask = split_parent_new(rows)
    parent_rows = [row for row, is_parent in zip(rows, parent_mask) if bool(is_parent)]
    new_rows = [row for row, is_new in zip(rows, new_mask) if bool(is_new)]
    full_counts = Counter(str(row.get("sweep_label", "")) for row in rows)
    parent_counts = Counter(str(row.get("sweep_label", "")) for row in parent_rows)
    observed = Counter(str(row.get("sweep_label", "")) for row in new_rows)
    planned = {str(label): int(count) for label, count in config.get("case_counts", {}).items()}
    labels = sorted(set(planned).union(observed).union(parent_counts).union(full_counts))
    table: list[dict[str, Any]] = []
    failed = []
    for label in labels:
        planned_count = int(planned.get(label, 0))
        observed_count = int(observed.get(label, 0))
        absolute_error = observed_count - planned_count
        relative_error = abs(absolute_error) / max(planned_count, 1)
        status = "PASS" if relative_error <= 0.01 else "WARN"
        if status != "PASS":
            failed.append(label)
        table.append(
            {
                "sweep_label": label,
                "planned_count": planned_count,
                "parent_count": int(parent_counts.get(label, 0)),
                "observed_new_training_only_count": observed_count,
                "full_count": int(full_counts.get(label, 0)),
                "observed_count": observed_count,
                "absolute_error": absolute_error,
                "relative_error": relative_error,
                "status": status,
            }
        )
    summary = {
        "planned_total_new_rows": int(sum(planned.values())),
        "observed_total_new_rows": int(len(new_rows)),
        "quota_warning_labels": failed,
        "quota_pass": not failed and int(sum(planned.values())) == int(len(new_rows)),
    }
    return table, summary


def coefficient_marginal_rows(
    rows: list[dict[str, str]],
    masks: dict[str, np.ndarray],
    *,
    bin_count: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for subset_name, mask in masks.items():
        subset = [rows[int(index)] for index in subset_indices(mask)]
        for name, value_range in COEFFICIENT_RANGES.items():
            values = np.asarray([row_float(row, name) for row in subset], dtype=float)
            counts, _ = np.histogram(values, bins=bin_count, range=value_range)
            median = float(np.median(counts)) if len(counts) else 0.0
            output.append(
                {
                    "subset": subset_name,
                    "quantity": name,
                    "n": int(len(values)),
                    "bin_count": int(bin_count),
                    "empty_bin_fraction": float(np.mean(counts == 0)) if len(counts) else 1.0,
                    "min_bin_count": int(np.min(counts)) if len(counts) else 0,
                    "median_bin_count": median,
                    "max_bin_count": int(np.max(counts)) if len(counts) else 0,
                    "max_to_median_ratio": float(np.max(counts) / max(median, 1.0)) if len(counts) else 0.0,
                    "normalized_entropy": normalized_entropy(counts),
                }
            )
    return output


def pairwise_occupancy_rows(
    rows: list[dict[str, str]],
    masks: dict[str, np.ndarray],
    *,
    grid_bins: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for subset_name, mask in masks.items():
        subset = [rows[int(index)] for index in subset_indices(mask)]
        for left, right in PAIRWISE_RANGES:
            left_values = np.asarray([row_float(row, left) for row in subset], dtype=float)
            right_values = np.asarray([row_float(row, right) for row in subset], dtype=float)
            counts, _, _ = np.histogram2d(
                left_values,
                right_values,
                bins=grid_bins,
                range=[COEFFICIENT_RANGES[left], COEFFICIENT_RANGES[right]],
            )
            nonzero = counts[counts > 0]
            median_nonzero = float(np.median(nonzero)) if len(nonzero) else 0.0
            output.append(
                {
                    "subset": subset_name,
                    "pair": f"{left}__{right}",
                    "x": left,
                    "y": right,
                    "grid_shape": f"{grid_bins}x{grid_bins}",
                    "nonempty_cell_fraction": float(np.mean(counts > 0)),
                    "empty_cell_fraction": float(np.mean(counts == 0)),
                    "min_nonzero_cell_count": int(np.min(nonzero)) if len(nonzero) else 0,
                    "median_nonzero_cell_count": median_nonzero,
                    "max_cell_count": int(np.max(counts)) if counts.size else 0,
                    "max_to_median_nonzero_ratio": float(np.max(counts) / max(median_nonzero, 1.0)) if counts.size else 0.0,
                    "normalized_2d_entropy": normalized_entropy(counts.ravel()),
                }
            )
    return output


def vector_angle(row: dict[str, str], group: str) -> float:
    return (VECTOR_ORDERS[group] * row_float(row, f"{group}_phase")) % 360.0


def wrapped_delta(left_angle: float, right_angle: float) -> float:
    return (left_angle - right_angle + 180.0) % 360.0 - 180.0


def target_angle_deviation(delta: np.ndarray, angle_bin: str) -> np.ndarray:
    abs_delta = np.abs(delta)
    if angle_bin == "aligned":
        return abs_delta
    if angle_bin == "orthogonal":
        return np.abs(abs_delta - 90.0)
    if angle_bin == "anti_aligned":
        return np.abs(180.0 - abs_delta)
    raise ValueError(angle_bin)


def relative_angle_rows(rows: list[dict[str, str]], new_mask: np.ndarray, *, angular_bins: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    new_rows = [row for row, is_new in zip(rows, new_mask) if bool(is_new)]
    for left, right in VECTOR_PAIRS:
        active = [
            row
            for row in new_rows
            if row_float(row, f"{left}_amp") > 1e-8
            and row_float(row, f"{right}_amp") > 1e-8
            and str(row.get("sampling_relative_angle_bin", "")) in ANGLE_BINS
        ]
        total = max(len(active), 1)
        for angle_bin in ANGLE_BINS:
            selected = [row for row in active if str(row.get("sampling_relative_angle_bin", "")) == angle_bin]
            deltas = np.asarray([wrapped_delta(vector_angle(row, left), vector_angle(row, right)) for row in selected], dtype=float)
            row_data: dict[str, Any] = {
                "pair": f"{left}_{right}",
                "angle_bin": angle_bin,
                "row_count": int(len(selected)),
                "fraction": float(len(selected) / total),
            }
            if angle_bin == "random":
                counts, _ = np.histogram(deltas, bins=angular_bins, range=(-180.0, 180.0))
                row_data.update(
                    {
                        "mean_abs_deviation_deg": "",
                        "p95_abs_deviation_deg": "",
                        "random_angle_entropy": normalized_entropy(counts),
                    }
                )
            elif len(deltas):
                deviations = target_angle_deviation(deltas, angle_bin)
                row_data.update(
                    {
                        "mean_abs_deviation_deg": float(np.mean(deviations)),
                        "p95_abs_deviation_deg": float(np.percentile(deviations, 95)),
                        "random_angle_entropy": "",
                    }
                )
            else:
                row_data.update(
                    {
                        "mean_abs_deviation_deg": "",
                        "p95_abs_deviation_deg": "",
                        "random_angle_entropy": "",
                    }
                )
            output.append(row_data)
    return output


def normalized_target_matrix(rows: list[dict[str, str]]) -> np.ndarray:
    data: list[list[float]] = []
    for row in rows:
        values = [
            row_float(row, "C1") / 100.0,
            row_float(row, "C3") / 2.0,
        ]
        for group, amp_scale in [("A1", 60.0), ("A2", 16.0), ("B2", 3.0), ("A3", 100.0), ("S3", 100.0)]:
            amp = row_float(row, f"{group}_amp") / amp_scale
            angle = math.radians(vector_angle(row, group))
            values.extend([amp, math.sin(angle), math.cos(angle)])
        data.append(values)
    return np.asarray(data, dtype=np.float32)


def nearest_neighbor_distances(
    reference: np.ndarray,
    query: np.ndarray,
    *,
    exclude_self: bool = False,
    sample_size: int = 8000,
    seed: int = 123,
) -> tuple[np.ndarray, str]:
    if len(reference) == 0 or len(query) == 0:
        return np.asarray([], dtype=float), "empty"
    rng = np.random.default_rng(seed)
    query_used = query
    reference_used = reference
    mode = "sklearn"
    try:
        from sklearn.neighbors import NearestNeighbors

        k = 2 if exclude_self else 1
        model = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
        model.fit(reference_used)
        distances, _ = model.kneighbors(query_used)
        values = distances[:, 1] if exclude_self and distances.shape[1] > 1 else distances[:, 0]
        return values.astype(float), mode
    except Exception:
        mode = "numpy_sampled"
    if len(query_used) > sample_size:
        query_used = query_used[rng.choice(len(query_used), size=sample_size, replace=False)]
    if len(reference_used) > sample_size:
        reference_used = reference_used[rng.choice(len(reference_used), size=sample_size, replace=False)]
    chunks = []
    for start in range(0, len(query_used), 250):
        block = query_used[start : start + 250]
        distances = np.sqrt(np.sum((block[:, None, :] - reference_used[None, :, :]) ** 2, axis=2))
        if exclude_self and len(query_used) == len(reference_used):
            row_indices = np.arange(start, min(start + 250, len(query_used)))
            if np.max(row_indices) < distances.shape[1]:
                distances[np.arange(len(block)), row_indices] = np.inf
        chunks.append(np.min(distances, axis=1))
    return np.concatenate(chunks).astype(float), mode


def distribution_summary(name: str, values: np.ndarray, method: str) -> dict[str, Any]:
    if len(values) == 0:
        return {
            "distribution": name,
            "method": method,
            "n_query": 0,
            "p1": "",
            "p5": "",
            "median": "",
            "p95": "",
            "p99": "",
            "max": "",
        }
    return {
        "distribution": name,
        "method": method,
        "n_query": int(len(values)),
        "p1": float(np.percentile(values, 1)),
        "p5": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def nearest_neighbor_rows(
    rows: list[dict[str, str]],
    masks: dict[str, np.ndarray],
    split_indices: dict[str, np.ndarray] | None,
) -> list[dict[str, Any]]:
    matrix = normalized_target_matrix(rows)
    full_train_mask = np.zeros(len(rows), dtype=bool)
    if split_indices:
        full_train_mask[split_indices["train"]] = True
    else:
        full_train_mask = masks["parent"] | masks["new_training_only"]
    parent_train_mask = masks["parent"] & full_train_mask
    new_mask = masks["new_training_only"]

    output: list[dict[str, Any]] = []
    values, method = nearest_neighbor_distances(matrix[new_mask], matrix[new_mask], exclude_self=True)
    output.append(distribution_summary("training_only_to_training_only_excluding_self", values, method))

    values, method = nearest_neighbor_distances(matrix[new_mask], matrix[masks["parent"]], exclude_self=False)
    output.append(distribution_summary("parent_to_new_training_only", values, method))

    if split_indices:
        for split_name in ["validation", "blind", "stress"]:
            query = matrix[split_indices[split_name]]
            values_parent, method_parent = nearest_neighbor_distances(matrix[parent_train_mask], query, exclude_self=False)
            values_full, method_full = nearest_neighbor_distances(matrix[full_train_mask], query, exclude_self=False)
            output.append(distribution_summary(f"{split_name}_to_v9_parent_train", values_parent, method_parent))
            output.append(distribution_summary(f"{split_name}_to_v11_full_train", values_full, method_full))
    return output


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_split_indices(rows: list[dict[str, str]], manifest_path: Path | None) -> tuple[dict[str, np.ndarray] | None, dict[str, Any]]:
    if manifest_path is None or not manifest_path.exists():
        return None, {"status": "not_available"}
    split_indices = frozen_benchmark_split(rows, manifest_path=manifest_path)
    leakage = {}
    for split_name in ["validation", "blind", "stress"]:
        leakage[split_name] = int(
            sum(
                str(rows[int(index)].get(DATASET_SPLIT_HINT_FIELD, "")).strip() == TRAINING_ONLY_HINT
                for index in split_indices[split_name]
            )
        )
    return split_indices, {
        "status": "available",
        "split_counts": {name: int(len(values)) for name, values in split_indices.items()},
        "training_only_leakage_counts": leakage,
        "training_only_leakage_pass": all(count == 0 for count in leakage.values()),
    }


def warning_summary(
    quota_table: list[dict[str, Any]],
    marginal_table: list[dict[str, Any]],
    pairwise_table: list[dict[str, Any]],
    relative_table: list[dict[str, Any]],
    split_summary: dict[str, Any],
) -> tuple[list[str], str]:
    warnings = []
    bad_quotas = [row["sweep_label"] for row in quota_table if row["status"] != "PASS"]
    if bad_quotas:
        warnings.append(f"Regime quota mismatch >1% for: {', '.join(bad_quotas)}")
    if split_summary.get("status") == "available" and not split_summary.get("training_only_leakage_pass", False):
        warnings.append(f"Training-only leakage into benchmark splits: {split_summary.get('training_only_leakage_counts')}")
    low_entropy = [
        f"{row['subset']}:{row['quantity']}"
        for row in marginal_table
        if row["subset"] == "new_training_only" and float(row["normalized_entropy"]) < 0.55
    ][:10]
    if low_entropy:
        warnings.append(f"Low new-row marginal entropy: {', '.join(low_entropy)}")
    sparse_pairs = [
        f"{row['subset']}:{row['pair']}"
        for row in pairwise_table
        if row["subset"] == "new_training_only" and float(row["nonempty_cell_fraction"]) < 0.20
    ][:10]
    if sparse_pairs:
        warnings.append(f"Low new-row pairwise occupancy: {', '.join(sparse_pairs)}")
    unbalanced_angles = [
        f"{row['pair']}:{row['angle_bin']}"
        for row in relative_table
        if row["angle_bin"] in ANGLE_BINS and int(row["row_count"]) == 0
    ]
    if unbalanced_angles:
        warnings.append(f"Missing relative-angle bins: {', '.join(unbalanced_angles[:10])}")
    recommendation = "PASS" if not warnings else "PASS_WITH_WARNINGS"
    if bad_quotas or (split_summary.get("status") == "available" and not split_summary.get("training_only_leakage_pass", False)):
        recommendation = "FAIL"
    return warnings, recommendation


def write_markdown(
    path: Path,
    *,
    csv_path: Path,
    config_path: Path,
    summary: dict[str, Any],
    quota_table: list[dict[str, Any]],
    marginal_table: list[dict[str, Any]],
    pairwise_table: list[dict[str, Any]],
    relative_table: list[dict[str, Any]],
    nearest_table: list[dict[str, Any]],
) -> None:
    warnings = summary["warnings"]
    lines = [
        "# Sampling Quality Report",
        "",
        f"Created UTC: `{summary['created_utc']}`",
        f"Dataset: `{csv_path}`",
        f"Config: `{config_path}`",
        "",
        "## Counts",
        "",
        f"- total rows: `{summary['counts']['total_rows']}`",
        f"- parent rows: `{summary['counts']['parent_rows']}`",
        f"- new training-only rows: `{summary['counts']['new_training_only_rows']}`",
        f"- recommendation: **{summary['recommendation']}**",
        "",
        "## Quota Check",
        "",
        "| label | planned new | observed new | rel error | status |",
        "|---|---:|---:|---:|---|",
    ]
    for row in quota_table:
        lines.append(
            f"| `{row['sweep_label']}` | {row['planned_count']} | {row['observed_new_training_only_count']} | {float(row['relative_error']):.4f} | {row['status']} |"
        )
    lines.extend(["", "## Key Warnings", ""])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Marginal Coverage Warnings",
            "",
            "| subset | quantity | empty bins | entropy | max/median |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in sorted(marginal_table, key=lambda item: (item["subset"], item["normalized_entropy"]))[:20]:
        lines.append(
            f"| {row['subset']} | `{row['quantity']}` | {float(row['empty_bin_fraction']):.3f} | {float(row['normalized_entropy']):.3f} | {float(row['max_to_median_ratio']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Pairwise Occupancy Warnings",
            "",
            "| subset | pair | nonempty | entropy | max/median |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in sorted(pairwise_table, key=lambda item: (item["subset"], item["nonempty_cell_fraction"]))[:20]:
        lines.append(
            f"| {row['subset']} | `{row['pair']}` | {float(row['nonempty_cell_fraction']):.3f} | {float(row['normalized_2d_entropy']):.3f} | {float(row['max_to_median_nonzero_ratio']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Relative-Angle Coverage",
            "",
            "| pair | bin | count | fraction | mean dev | p95 dev | random entropy |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in relative_table:
        lines.append(
            f"| `{row['pair']}` | {row['angle_bin']} | {row['row_count']} | {float(row['fraction']):.3f} | {row['mean_abs_deviation_deg']} | {row['p95_abs_deviation_deg']} | {row['random_angle_entropy']} |"
        )
    lines.extend(
        [
            "",
            "## Nearest-Neighbor Coverage",
            "",
            "| distribution | method | n | p5 | median | p95 | max |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in nearest_table:
        lines.append(
            f"| `{row['distribution']}` | {row['method']} | {row['n_query']} | {row['p5']} | {row['median']} | {row['p95']} | {row['max']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def maybe_write_plots(
    plot_dir: Path,
    rows: list[dict[str, str]],
    masks: dict[str, np.ndarray],
    *,
    bin_count: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir.mkdir(parents=True, exist_ok=True)
    for quantity in ["S3_amp", "A3_amp", "B2_amp", "C1"]:
        fig, ax = plt.subplots(figsize=(5, 3.4))
        for subset_name, mask in masks.items():
            if subset_name == "full":
                continue
            subset = [rows[int(index)] for index in subset_indices(mask)]
            values = [row_float(row, quantity) for row in subset]
            ax.hist(
                values,
                bins=bin_count,
                range=COEFFICIENT_RANGES[quantity],
                alpha=0.5,
                label=subset_name,
            )
        ax.set_xlabel(quantity)
        ax.set_ylabel("rows")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{quantity}_parent_vs_new_hist.png", dpi=120)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--benchmark-split-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--pairwise-bins", type=int, default=16)
    parser.add_argument("--angular-bins", type=int, default=18)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_rows(args.csv_path)
    config = load_config(args.config)
    if not rows:
        raise RuntimeError(f"CSV is empty: {args.csv_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, parent_mask, new_mask = split_parent_new(rows)
    masks = {
        "full": np.ones(len(rows), dtype=bool),
        "parent": parent_mask,
        "new_training_only": new_mask,
    }

    split_indices, split_summary = load_split_indices(rows, args.benchmark_split_manifest)
    quota_table, quota_summary = regime_quota_rows(rows, config)
    marginal_table = coefficient_marginal_rows(rows, masks, bin_count=args.bins)
    pairwise_table = pairwise_occupancy_rows(rows, masks, grid_bins=args.pairwise_bins)
    relative_table = relative_angle_rows(rows, new_mask, angular_bins=args.angular_bins)
    nearest_table = nearest_neighbor_rows(rows, masks, split_indices)
    warnings, recommendation = warning_summary(
        quota_table,
        marginal_table,
        pairwise_table,
        relative_table,
        split_summary,
    )

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "csv_path": str(args.csv_path),
        "config": str(args.config),
        "benchmark_split_manifest": "" if args.benchmark_split_manifest is None else str(args.benchmark_split_manifest),
        "counts": {
            "total_rows": int(len(rows)),
            "parent_rows": int(np.sum(parent_mask)),
            "new_training_only_rows": int(np.sum(new_mask)),
            "dataset_versions": dict(Counter(str(row.get("dataset_version", "")) for row in rows)),
            "dataset_sources": dict(Counter(str(row.get("dataset_source", "")) for row in rows)),
            "split_hints": dict(Counter(str(row.get(DATASET_SPLIT_HINT_FIELD, "")) for row in rows)),
        },
        "quota": quota_summary,
        "split": split_summary,
        "warnings": warnings,
        "recommendation": recommendation,
    }

    write_csv(
        args.output_dir / "regime_count_summary.csv",
        quota_table,
        [
            "sweep_label",
            "planned_count",
            "parent_count",
            "observed_new_training_only_count",
            "full_count",
            "observed_count",
            "absolute_error",
            "relative_error",
            "status",
        ],
    )
    write_csv(
        args.output_dir / "coefficient_marginal_bin_summary.csv",
        marginal_table,
        [
            "subset",
            "quantity",
            "n",
            "bin_count",
            "empty_bin_fraction",
            "min_bin_count",
            "median_bin_count",
            "max_bin_count",
            "max_to_median_ratio",
            "normalized_entropy",
        ],
    )
    write_csv(
        args.output_dir / "pairwise_occupancy_summary.csv",
        pairwise_table,
        [
            "subset",
            "pair",
            "x",
            "y",
            "grid_shape",
            "nonempty_cell_fraction",
            "empty_cell_fraction",
            "min_nonzero_cell_count",
            "median_nonzero_cell_count",
            "max_cell_count",
            "max_to_median_nonzero_ratio",
            "normalized_2d_entropy",
        ],
    )
    write_csv(
        args.output_dir / "relative_angle_coverage_summary.csv",
        relative_table,
        [
            "pair",
            "angle_bin",
            "row_count",
            "fraction",
            "mean_abs_deviation_deg",
            "p95_abs_deviation_deg",
            "random_angle_entropy",
        ],
    )
    write_csv(
        args.output_dir / "nearest_neighbor_coverage_summary.csv",
        nearest_table,
        ["distribution", "method", "n_query", "p1", "p5", "median", "p95", "p99", "max"],
    )
    (args.output_dir / "sampling_quality_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown(
        args.output_dir / "sampling_quality_report.md",
        csv_path=args.csv_path,
        config_path=args.config,
        summary=summary,
        quota_table=quota_table,
        marginal_table=marginal_table,
        pairwise_table=pairwise_table,
        relative_table=relative_table,
        nearest_table=nearest_table,
    )
    maybe_write_plots(args.output_dir / "plots", rows, masks, bin_count=args.bins)
    print("sampling quality report:", args.output_dir / "sampling_quality_report.md")
    print("sampling quality summary:", args.output_dir / "sampling_quality_summary.json")
    print("recommendation:", recommendation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
