#!/usr/bin/env python3
"""Retest saved active-hole probe feature sets with a saved model checkpoint.

This is a postmortem diagnostic, not a training or expansion script. It uses an
existing feature CSV and checkpoint run directory, reruns inference on restored
active-hole probe features, and compares the errors with the original v13
active-hole summaries when available.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from generate_targeted_enhanced_dataset import target_scale_array
from run_active_12d_hole_search import (
    normalized_target_matrix,
    read_csv as read_active_csv,
    run_inference,
    summarize_evaluation,
    write_csv,
)
from run_model_selection_candidate import frozen_benchmark_split


ERROR_COLUMNS = (
    "weighted_abs_error",
    "overall_abs_error",
    "C1_abs_error",
    "C3_abs_error",
    "A1_vector_error",
    "B2_vector_error",
    "A2_vector_error",
    "S3_vector_error",
    "A3_vector_error",
)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def metric_stats(rows: list[dict[str, Any]], column: str) -> dict[str, float | int | None]:
    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row[column]))
        except Exception:
            continue
    if not values:
        return {"n": 0, "mae": None, "rmse": None, "median": None, "p95": None, "max": None}
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "mae": float(np.mean(arr)),
        "rmse": float(math.sqrt(float(np.mean(arr * arr)))),
        "median": float(np.median(arr)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def compare_rows(
    baseline_rows: list[dict[str, str]],
    retest_rows: list[dict[str, Any]],
    active_run_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline_by_id = {str(row.get("active_probe_id", "")): row for row in baseline_rows}
    comparison_rows: list[dict[str, Any]] = []
    for row in retest_rows:
        probe_id = str(row.get("active_probe_id", ""))
        old = baseline_by_id.get(probe_id)
        payload: dict[str, Any] = {
            "active_run": active_run_name,
            "active_probe_id": probe_id,
            "sweep_label": row.get("sweep_label", ""),
            "proposal_mode": row.get("proposal_mode", ""),
            "v15_failure_class": row.get("failure_class", ""),
        }
        for column in ERROR_COLUMNS:
            new_value = float(row.get(column, 0.0) or 0.0)
            payload[f"v15_{column}"] = new_value
            if old is not None and old.get(column, "") != "":
                old_value = float(old[column])
                payload[f"v13_{column}"] = old_value
                payload[f"delta_{column}"] = new_value - old_value
                payload[f"ratio_{column}"] = new_value / old_value if abs(old_value) > 1e-12 else ""
            else:
                payload[f"v13_{column}"] = ""
                payload[f"delta_{column}"] = ""
                payload[f"ratio_{column}"] = ""
        comparison_rows.append(payload)

    matched = [row for row in comparison_rows if row.get("v13_weighted_abs_error") != ""]
    summary = {
        "active_run": active_run_name,
        "n_v15": len(retest_rows),
        "n_v13_baseline": len(baseline_rows),
        "n_matched_probe_ids": len(matched),
        "v15": {column: metric_stats(retest_rows, column) for column in ERROR_COLUMNS},
        "v13": {column: metric_stats(baseline_rows, column) for column in ERROR_COLUMNS},
    }
    if matched:
        for column in ERROR_COLUMNS:
            deltas = [float(row[f"delta_{column}"]) for row in matched if row.get(f"delta_{column}") != ""]
            if deltas:
                arr = np.asarray(deltas, dtype=float)
                summary[f"delta_{column}"] = {
                    "mean": float(np.mean(arr)),
                    "median": float(np.median(arr)),
                    "worse_fraction": float(np.mean(arr > 0.0)),
                    "better_fraction": float(np.mean(arr < 0.0)),
                }
    return comparison_rows, summary


def combined_summary(
    baseline_rows: list[dict[str, str]],
    retest_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "active_run": "combined",
        "n_v15": len(retest_rows),
        "n_v13_baseline": len(baseline_rows),
        "n_matched_probe_ids": sum(1 for row in comparison_rows if row.get("v13_weighted_abs_error") != ""),
        "v15": {column: metric_stats(retest_rows, column) for column in ERROR_COLUMNS},
        "v13": {column: metric_stats(baseline_rows, column) for column in ERROR_COLUMNS},
    }
    for column in ERROR_COLUMNS:
        deltas = [float(row[f"delta_{column}"]) for row in comparison_rows if row.get(f"delta_{column}") != ""]
        if deltas:
            arr = np.asarray(deltas, dtype=float)
            summary[f"delta_{column}"] = {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "worse_fraction": float(np.mean(arr > 0.0)),
                "better_fraction": float(np.mean(arr < 0.0)),
            }
    return summary


def compact_run_row(summary: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "active_run": summary["active_run"],
        "n_v15": summary["n_v15"],
        "n_v13_baseline": summary["n_v13_baseline"],
        "n_matched_probe_ids": summary["n_matched_probe_ids"],
    }
    for prefix in ("v13", "v15"):
        metrics = summary.get(prefix, {})
        for column in ("weighted_abs_error", "overall_abs_error", "S3_vector_error", "A3_vector_error", "B2_vector_error"):
            stats = metrics.get(column, {})
            row[f"{prefix}_{column}_rmse"] = stats.get("rmse")
            row[f"{prefix}_{column}_mae"] = stats.get("mae")
            row[f"{prefix}_{column}_p95"] = stats.get("p95")
    for column in ("weighted_abs_error", "overall_abs_error", "S3_vector_error", "A3_vector_error", "B2_vector_error"):
        delta = summary.get(f"delta_{column}", {})
        row[f"delta_{column}_mean"] = delta.get("mean")
        row[f"delta_{column}_worse_fraction"] = delta.get("worse_fraction")
    return row


def find_probe_feature_dirs(active_root: Path) -> list[Path]:
    paths = sorted(active_root.glob("v13_active_12d_hole_search*/active_hole_search_probe_features.csv"))
    return [path.parent for path in paths]


def write_report(output_dir: Path, summary: dict[str, Any], run_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# V15 Active-Hole Retest",
        "",
        f"Created UTC: `{summary['created_utc']}`",
        "",
        "Purpose: retest the original active-hole probe feature sets with the v15 checkpoint and compare against the original v13 active-hole errors.",
        "",
        f"- dataset CSV: `{summary['dataset_csv']}`",
        f"- checkpoint run dir: `{summary['run_dir']}`",
        f"- active probe runs retested: `{summary['active_probe_run_count']}`",
        f"- combined matched probes: `{summary['combined']['n_matched_probe_ids']}`",
        "",
        "## Combined Error",
        "",
        "| metric | v13 RMSE | v15 RMSE | v15-v13 RMSE | v15 worse fraction |",
        "|---|---:|---:|---:|---:|",
    ]
    combined = summary["combined"]
    for column in ("weighted_abs_error", "overall_abs_error", "S3_vector_error", "A3_vector_error", "B2_vector_error"):
        v13_rmse = combined["v13"][column]["rmse"]
        v15_rmse = combined["v15"][column]["rmse"]
        delta = None if v13_rmse is None or v15_rmse is None else v15_rmse - v13_rmse
        worse = combined.get(f"delta_{column}", {}).get("worse_fraction")
        lines.append(f"| `{column}` | {v13_rmse} | {v15_rmse} | {delta} | {worse} |")
    lines.extend(
        [
            "",
            "## Per Active-Search Run",
            "",
            "| active run | n matched | v13 weighted RMSE | v15 weighted RMSE | v15 worse fraction |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in run_rows:
        lines.append(
            "| `{active_run}` | {n_matched_probe_ids} | {v13_weighted_abs_error_rmse} | "
            "{v15_weighted_abs_error_rmse} | {delta_weighted_abs_error_worse_fraction} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Interpretation rule:",
            "",
            "- If v15 improves the active-hole RMSE while fixed validation/blind/stress got worse, then v15 repaired some searched holes but damaged the benchmark distribution.",
            "- If v15 does not improve active-hole RMSE, then the 250k active-hole expansion failed its primary purpose.",
            "",
        ]
    )
    (output_dir / "active_hole_retest_report.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-csv", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--benchmark-split-manifest", type=Path, default=Path("configs/benchmark_split_v12_v2_row_keys.json"))
    parser.add_argument("--active-root", type=Path, default=Path("training_results/model_selection_reports"))
    parser.add_argument("--output-root", type=Path, default=Path("training_results/model_selection_reports"))
    parser.add_argument("--run-prefix", default="v15_active_hole_retest")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dirs = find_probe_feature_dirs(args.active_root)
    if not run_dirs:
        raise RuntimeError(
            "No active-hole probe feature CSVs found. Restore Drive folders containing "
            "v13_active_12d_hole_search*/active_hole_search_probe_features.csv first."
        )
    if not (args.run_dir / "model_loop_candidate.pt").exists():
        raise RuntimeError(f"missing checkpoint: {args.run_dir / 'model_loop_candidate.pt'}")

    rows, _ = read_active_csv(args.dataset_csv)
    if not rows:
        raise RuntimeError(f"dataset CSV is empty: {args.dataset_csv}")
    split_indices = frozen_benchmark_split(rows, manifest_path=args.benchmark_split_manifest)
    train_rows = [rows[int(index)] for index in split_indices["train"]]
    train_matrix = normalized_target_matrix(train_rows, target_scale_array())

    output_dir = args.output_root / f"{args.run_prefix}_{utc_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_v15_rows: list[dict[str, Any]] = []
    all_v13_rows: list[dict[str, str]] = []
    all_comparison_rows: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    run_summary_rows: list[dict[str, Any]] = []

    inference_config = {"proposal": {"reference_sample_size": 300000, "seed": 1507, "nearest_neighbor_chunk_size": 25000}, "classification": {"failure_top_fraction": 0.10, "coverage_limited_nn_quantile": 0.75, "dense_failure_nn_quantile": 0.50}, "inference": {"predict_batch_size": 65536}}

    for source_dir in run_dirs:
        probe_path = source_dir / "active_hole_search_probe_features.csv"
        probe_rows, _ = read_active_csv(probe_path)
        subdir = output_dir / source_dir.name
        subdir.mkdir(parents=True, exist_ok=True)
        y_pred, manifest = run_inference(rows, args.dataset_csv, args.run_dir, split_indices, probe_rows, inference_config, subdir)
        (subdir / "inference_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if y_pred is None:
            raise RuntimeError(f"inference failed for {source_dir.name}: {manifest}")
        evaluation_summary = summarize_evaluation(probe_rows, y_pred, train_matrix, inference_config, subdir)
        retest_rows = read_csv_rows(subdir / "active_hole_nn_error_summary.csv")
        baseline_path = source_dir / "active_hole_nn_error_summary.csv"
        baseline_rows = read_csv_rows(baseline_path) if baseline_path.exists() else []
        comparison_rows, run_summary = compare_rows(baseline_rows, retest_rows, source_dir.name)
        run_summary["evaluation_summary"] = evaluation_summary
        run_summaries.append(run_summary)
        row = compact_run_row(run_summary)
        run_summary_rows.append(row)
        write_csv(subdir / "active_hole_retest_probe_comparison.csv", comparison_rows, list(comparison_rows[0]) if comparison_rows else ["active_probe_id"])
        (subdir / "active_hole_retest_summary.json").write_text(json.dumps(run_summary, indent=2) + "\n")
        all_v15_rows.extend(retest_rows)
        all_v13_rows.extend(baseline_rows)
        all_comparison_rows.extend(comparison_rows)

    combined = combined_summary(all_v13_rows, all_v15_rows, all_comparison_rows)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_csv": str(args.dataset_csv),
        "run_dir": str(args.run_dir),
        "benchmark_split_manifest": str(args.benchmark_split_manifest),
        "active_root": str(args.active_root),
        "output_dir": str(output_dir),
        "active_probe_run_count": len(run_dirs),
        "split_counts": {name: int(len(values)) for name, values in split_indices.items()},
        "combined": combined,
        "runs": run_summaries,
    }
    (output_dir / "active_hole_retest_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(output_dir / "active_hole_retest_run_summary.csv", run_summary_rows, list(run_summary_rows[0]) if run_summary_rows else ["active_run"])
    compact_comparison_rows = sorted(all_comparison_rows, key=lambda row: float(row.get("v15_weighted_abs_error", 0.0) or 0.0), reverse=True)[:1000]
    write_csv(
        output_dir / "active_hole_retest_top1000_probe_comparison.csv",
        compact_comparison_rows,
        list(compact_comparison_rows[0]) if compact_comparison_rows else ["active_probe_id"],
    )
    write_report(output_dir, summary, run_summary_rows)
    print(json.dumps({"output_dir": str(output_dir), "active_probe_run_count": len(run_dirs)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
