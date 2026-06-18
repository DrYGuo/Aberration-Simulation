"""Compare active-search failed regions against v13 benchmark errors.

This is a lightweight reporting utility. It does not train a model, simulate
new probes, or create a new dataset. It summarizes the failed 12D subspaces
found by active hole-search and writes a machine-readable v15 subspace spec.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any


PHYSICAL_SCALES = {
    "C1": 100.0,
    "C3": 2.0,
    "A1_x": 60.0,
    "A1_y": 60.0,
    "B2_x": 3.0,
    "B2_y": 3.0,
    "A2_x": 16.0,
    "A2_y": 16.0,
    "S3_x": 100.0,
    "S3_y": 100.0,
    "A3_x": 100.0,
    "A3_y": 100.0,
}

TARGET_LABELS = {
    "C1_abs_error": ("C1", "nm"),
    "C3_abs_error": ("C3", "mm"),
    "A1_vector_error": ("A1 vector", "nm"),
    "B2_vector_error": ("B2 vector", "um"),
    "A2_vector_error": ("A2 vector", "um"),
    "S3_vector_error": ("S3 vector", "um"),
    "A3_vector_error": ("A3 vector", "um"),
}

VECTOR_PAIRS = {
    "A1": ("A1_x", "A1_y", "nm"),
    "B2": ("B2_x", "B2_y", "um"),
    "A2": ("A2_x", "A2_y", "um"),
    "S3": ("S3_x", "S3_y", "um"),
    "A3": ("A3_x", "A3_y", "um"),
}

CLUSTER_COLUMNS = (
    "C1",
    "C3",
    "A1_x",
    "A1_y",
    "B2_x",
    "B2_y",
    "A2_x",
    "A2_y",
    "S3_x",
    "S3_y",
    "A3_x",
    "A3_y",
)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def as_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_values(values: list[float]) -> dict[str, float | int]:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return {"n": 0}
    n = len(clean)

    def quantile(frac: float) -> float:
        index = min(n - 1, max(0, int(round(frac * (n - 1)))))
        return clean[index]

    return {
        "n": n,
        "mean_abs_or_mae": sum(clean) / n,
        "rmse": math.sqrt(sum(value * value for value in clean) / n),
        "median": quantile(0.5),
        "p95": quantile(0.95),
        "max": clean[-1],
    }


def summarize_active_failures(rows: list[dict[str, str]], group_name: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for column, (target, unit) in TARGET_LABELS.items():
        values = [value for row in rows if (value := as_float(row.get(column))) is not None]
        stats = summarize_values(values)
        output.append(
            {
                "group": group_name,
                "metric": target,
                "unit": unit,
                **stats,
            }
        )
    for column, label in (
        ("overall_abs_error", "overall mixed-unit abs"),
        ("weighted_abs_error", "weighted normalized abs"),
        ("nn_distance_12d", "12D NN distance"),
    ):
        values = [value for row in rows if (value := as_float(row.get(column))) is not None]
        output.append({"group": group_name, "metric": label, "unit": "", **summarize_values(values)})
    return output


def split_component_summary(metrics: dict[str, Any], split: str) -> list[dict[str, Any]]:
    split_data = metrics["splits"][split]
    rows: list[dict[str, Any]] = [
        {
            "split": split,
            "metric": "overall mixed-unit abs",
            "unit": "",
            "mae": split_data.get("overall_mae"),
            "rmse": split_data.get("overall_rmse"),
            "p95": split_data.get("overall_p95_abs_error"),
        },
        {
            "split": split,
            "metric": "overall normalized abs",
            "unit": "",
            "mae": split_data.get("overall_normalized_mae"),
            "rmse": "",
            "p95": split_data.get("overall_normalized_p95_abs_error"),
        },
    ]
    for target, unit in (("C1", "nm"), ("C3", "mm")):
        target_data = split_data["targets"][target]
        rows.append(
            {
                "split": split,
                "metric": target,
                "unit": unit,
                "mae": target_data.get("mae"),
                "rmse": target_data.get("rmse"),
                "p95": target_data.get("p95_abs_error"),
            }
        )
    for pair, (x_name, y_name, unit) in VECTOR_PAIRS.items():
        x_data = split_data["targets"][x_name]
        y_data = split_data["targets"][y_name]
        pair_mae_norm = math.sqrt(float(x_data["mae"]) ** 2 + float(y_data["mae"]) ** 2)
        pair_rmse_norm = math.sqrt(float(x_data["rmse"]) ** 2 + float(y_data["rmse"]) ** 2)
        pair_p95_norm = math.sqrt(float(x_data["p95_abs_error"]) ** 2 + float(y_data["p95_abs_error"]) ** 2)
        rows.append(
            {
                "split": split,
                "metric": f"{pair} component-pair norm",
                "unit": unit,
                "mae": pair_mae_norm,
                "rmse": pair_rmse_norm,
                "p95": pair_p95_norm,
            }
        )
    return rows


def vector_magnitude_summary(vector_diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair, (_, _, unit) in VECTOR_PAIRS.items():
        data = vector_diagnostics["vector_pairs"][pair]
        mag = data["magnitude"]
        angle = data.get("angle", {})
        rows.append(
            {
                "split": vector_diagnostics.get("split", "validation"),
                "metric": pair,
                "unit": unit,
                "magnitude_mae": mag.get("magnitude_mae"),
                "magnitude_rmse": mag.get("magnitude_rmse"),
                "magnitude_bias": mag.get("magnitude_bias"),
                "magnitude_slope": mag.get("magnitude_slope"),
                "magnitude_r2": mag.get("magnitude_r2"),
                "mean_abs_angle_error_deg": angle.get("mean_abs_angle_error_deg"),
                "p95_abs_angle_error_deg": angle.get("p95_abs_angle_error_deg"),
            }
        )
    return rows


def physical_from_cluster(row: dict[str, str], name: str) -> float | None:
    normalized = as_float(row.get(f"center_{name}_normalized"))
    if normalized is None:
        return None
    return normalized * PHYSICAL_SCALES[name]


def cluster_rows(run_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        for row in read_csv_rows(run_dir / "active_hole_clusters.csv"):
            output: dict[str, Any] = {
                "source_run": run_dir.name,
                "cluster_id": row.get("cluster_id", ""),
                "n": row.get("n", ""),
                "median_weighted_error": row.get("median_weighted_error", ""),
                "median_nn_distance_12d": row.get("median_nn_distance_12d", ""),
                "dominant_regime": row.get("dominant_regime", ""),
                "dominant_failure_class": row.get("dominant_failure_class", ""),
            }
            for name in CLUSTER_COLUMNS:
                value = physical_from_cluster(row, name)
                output[f"center_{name}"] = "" if value is None else value
            rows.append(output)
    rows.sort(
        key=lambda item: (
            -float(item["median_weighted_error"] or 0.0),
            -float(item["median_nn_distance_12d"] or 0.0),
        )
    )
    return rows


def write_markdown(
    path: Path,
    metrics: dict[str, Any],
    active_summary_rows: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]],
    vector_rows: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    top_failure_counts: dict[str, Any],
) -> None:
    lines = [
        "# Active Failed-Region Error Report",
        "",
        f"Created UTC: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Purpose",
        "",
        "Quantify the coefficient uncertainty inside active-search failed subspaces and compare it with the v13 1M benchmark splits.",
        "",
        "Important: active vector errors are Euclidean vector errors on deliberately selected failed probes. Benchmark vector rows use component-pair norm approximations and validation vector-magnitude diagnostics, so they are directionally comparable but not identical estimators.",
        "",
        "## v13 Benchmark Context",
        "",
        f"- run: `{metrics['run_name']}`",
        f"- total rows: `{metrics['n_samples']}`",
        f"- train/validation/blind/stress: `{metrics['n_train']}` / `{metrics['n_validation']}` / `{metrics['n_blind']}` / `{metrics['n_stress']}`",
        "",
        "## Active Failed Regions: Local Error Levels",
        "",
        "| group | metric | unit | MAE/mean | RMSE | median | p95 | max |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in active_summary_rows:
        lines.append(
            f"| {row['group']} | {row['metric']} | {row.get('unit', '')} | "
            f"{float(row.get('mean_abs_or_mae', 0.0)):.4g} | {float(row.get('rmse', 0.0)):.4g} | "
            f"{float(row.get('median', 0.0)):.4g} | {float(row.get('p95', 0.0)):.4g} | "
            f"{float(row.get('max', 0.0)):.4g} |"
        )

    lines.extend(
        [
            "",
            "## v13 Benchmark Splits",
            "",
            "| split | metric | unit | MAE | RMSE | p95 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in benchmark_rows:
        rmse = row.get("rmse", "")
        rmse_text = "" if rmse == "" else f"{float(rmse):.4g}"
        lines.append(
            f"| {row['split']} | {row['metric']} | {row.get('unit', '')} | "
            f"{float(row.get('mae', 0.0)):.4g} | {rmse_text} | {float(row.get('p95', 0.0)):.4g} |"
        )

    lines.extend(
        [
            "",
            "## Validation Vector Magnitude/Angle Diagnostics",
            "",
            "| vector | unit | magnitude MAE | magnitude RMSE | bias | slope | R2 | mean angle err deg | p95 angle err deg |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in vector_rows:
        lines.append(
            f"| {row['metric']} | {row['unit']} | {float(row['magnitude_mae']):.4g} | "
            f"{float(row['magnitude_rmse']):.4g} | {float(row['magnitude_bias']):.4g} | "
            f"{float(row['magnitude_slope']):.4g} | {float(row['magnitude_r2']):.4g} | "
            f"{float(row['mean_abs_angle_error_deg']):.4g} | {float(row['p95_abs_angle_error_deg']):.4g} |"
        )

    lines.extend(
        [
            "",
            "## Failed-Region Counts",
            "",
            f"- top-failure rows analyzed: `{top_failure_counts['n_top_failure_rows']}`",
            f"- failure classes: `{top_failure_counts['failure_classes']}`",
            f"- regimes: `{top_failure_counts['regimes']}`",
            f"- A3-S3 angle categories: `{top_failure_counts['a3_s3_angle_categories']}`",
            f"- B2-S3 angle categories: `{top_failure_counts['b2_s3_angle_categories']}`",
            "",
            "## Highest-Priority Physical Cluster Centers",
            "",
            "| rank | n | class | median err | median NN | C1 nm | C3 mm | A1x nm | A1y nm | B2x um | B2y um | S3x um | S3y um | A3x um | A3y um |",
            "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(clusters[:12], start=1):
        lines.append(
            f"| {rank} | {row['n']} | {row['dominant_failure_class']} | "
            f"{float(row['median_weighted_error']):.4g} | {float(row['median_nn_distance_12d']):.4g} | "
            f"{float(row['center_C1']):.4g} | {float(row['center_C3']):.4g} | "
            f"{float(row['center_A1_x']):.4g} | {float(row['center_A1_y']):.4g} | "
            f"{float(row['center_B2_x']):.4g} | {float(row['center_B2_y']):.4g} | "
            f"{float(row['center_S3_x']):.4g} | {float(row['center_S3_y']):.4g} | "
            f"{float(row['center_A3_x']):.4g} | {float(row['center_A3_y']):.4g} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Failed-region errors are tens of times larger than the ordinary v13 validation/blind/stress errors for S3/A3/A1 vector targets.",
            "- Coverage-limited sparse failures are the largest group and have the highest median NN distance.",
            "- Dense/mixed failures are smaller but non-negligible; they should remain a separate diagnostic group after v15.",
            "- v15 should not train directly on the diagnostic probes. Convert these subspaces into a balanced expansion with jitter, angle balancing, and bridge/anchor controls.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v13-run-dir",
        type=Path,
        default=Path(
            "training_results/model_selection_loop/"
            "D66_grouped_width320_lr6e-4_dropout0.075_v13_1m_d66_seed23_residual_nn_20260615_065556_utc"
        ),
    )
    parser.add_argument("--active-root", type=Path, default=Path("training_results/model_selection_reports"))
    parser.add_argument("--output-root", type=Path, default=Path("training_results/model_selection_reports"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = read_json(args.v13_run_dir / "metrics_model_loop.json")
    vectors = read_json(args.v13_run_dir / "vector_diagnostics.json")
    run_dirs = sorted(
        path
        for path in args.active_root.glob("v13_active_12d_hole_search*")
        if (path / "active_hole_search_top_failures.csv").exists()
    )
    if not run_dirs:
        raise RuntimeError(f"No active hole-search result dirs found under {args.active_root}")

    top_rows: list[dict[str, str]] = []
    for run_dir in run_dirs:
        for row in read_csv_rows(run_dir / "active_hole_search_top_failures.csv"):
            row["source_run"] = run_dir.name
            top_rows.append(row)

    active_summary_rows: list[dict[str, Any]] = []
    active_summary_rows.extend(summarize_active_failures(top_rows, "all_active_top_failures"))
    for group in (
        "coverage_limited_sparse_failure",
        "mixed_failure",
        "dense_model_feature_loss_failure",
    ):
        active_summary_rows.extend(
            summarize_active_failures(
                [row for row in top_rows if row.get("failure_class") == group],
                group,
            )
        )

    benchmark_rows: list[dict[str, Any]] = []
    for split in ("validation", "blind", "stress"):
        benchmark_rows.extend(split_component_summary(metrics, split))
    vector_rows = vector_magnitude_summary(vectors)
    clusters = cluster_rows(run_dirs)

    top_failure_counts = {
        "n_top_failure_rows": len(top_rows),
        "failure_classes": dict(Counter(row.get("failure_class", "") for row in top_rows)),
        "regimes": dict(Counter(row.get("sweep_label", "") for row in top_rows)),
        "proposal_modes": dict(Counter(row.get("proposal_mode", "") for row in top_rows)),
        "a3_s3_angle_categories": dict(Counter(row.get("relative_angle_A3_S3_category", "") for row in top_rows)),
        "b2_s3_angle_categories": dict(Counter(row.get("relative_angle_B2_S3_category", "") for row in top_rows)),
    }

    output_dir = args.output_root / f"active_failed_region_error_report_{utc_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "active_failed_region_error_summary.csv",
        active_summary_rows,
        ["group", "metric", "unit", "n", "mean_abs_or_mae", "rmse", "median", "p95", "max"],
    )
    write_csv(
        output_dir / "v13_benchmark_split_error_summary.csv",
        benchmark_rows,
        ["split", "metric", "unit", "mae", "rmse", "p95"],
    )
    write_csv(
        output_dir / "v13_validation_vector_magnitude_summary.csv",
        vector_rows,
        [
            "split",
            "metric",
            "unit",
            "magnitude_mae",
            "magnitude_rmse",
            "magnitude_bias",
            "magnitude_slope",
            "magnitude_r2",
            "mean_abs_angle_error_deg",
            "p95_abs_angle_error_deg",
        ],
    )
    cluster_fields = [
        "source_run",
        "cluster_id",
        "n",
        "median_weighted_error",
        "median_nn_distance_12d",
        "dominant_regime",
        "dominant_failure_class",
        *[f"center_{name}" for name in CLUSTER_COLUMNS],
    ]
    write_csv(output_dir / "active_failed_subspace_cluster_centers.csv", clusters, cluster_fields)

    subspace_spec = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_active_runs": [path.name for path in run_dirs],
        "v13_reference_run": str(args.v13_run_dir),
        "purpose": "Input specification for v15 targeted data expansion. Use these as subspace centers, not literal rows.",
        "do_not_train_directly_on_active_probe_rows": True,
        "recommended_new_rows": 250000,
        "sampling_mix": {
            "cluster_jitter_around_coverage_limited_centers": 0.30,
            "high_amp_A1_B2_S3_A3_angle_balanced": 0.25,
            "far_nn_coupled_full_random": 0.20,
            "bridge_moderate_nn_controls": 0.10,
            "coupled_sparse_random_controls": 0.10,
            "orthogonal_diagnostic_controls": 0.05,
        },
        "priority_failure_classes": [
            "coverage_limited_sparse_failure",
            "mixed_failure",
            "dense_model_feature_loss_failure",
        ],
        "priority_angle_pairs": {
            "A3_S3": ["aligned_or_anti_aligned", "anti_aligned", "orthogonal", "oblique"],
            "B2_S3": ["oblique", "orthogonal", "anti_aligned", "aligned_or_anti_aligned"],
        },
        "targeted_subspace_centers": clusters[:50],
        "success_criteria_after_v15": {
            "active_sparse_tail_error_should_decrease": True,
            "residual_vs_nn_spearman_should_drop_below": 0.45,
            "dense_high_amplitude_failures_must_be_reported_separately": True,
            "do_not_promote_if_A3_or_S3_blind_stress_regress_substantially": True,
        },
    }
    (output_dir / "v15_failed_subspace_sampling_spec.json").write_text(json.dumps(subspace_spec, indent=2) + "\n")
    summary = {
        "status": "complete",
        "output_dir": str(output_dir),
        "v13_reference_run": str(args.v13_run_dir),
        "source_active_runs": [path.name for path in run_dirs],
        "n_top_failure_rows": len(top_rows),
        "n_clusters": len(clusters),
        "files": {
            "markdown_report": str(output_dir / "active_failed_region_error_report.md"),
            "active_error_summary_csv": str(output_dir / "active_failed_region_error_summary.csv"),
            "benchmark_error_summary_csv": str(output_dir / "v13_benchmark_split_error_summary.csv"),
            "vector_summary_csv": str(output_dir / "v13_validation_vector_magnitude_summary.csv"),
            "cluster_centers_csv": str(output_dir / "active_failed_subspace_cluster_centers.csv"),
            "v15_sampling_spec": str(output_dir / "v15_failed_subspace_sampling_spec.json"),
        },
    }
    (output_dir / "active_failed_region_error_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown(
        output_dir / "active_failed_region_error_report.md",
        metrics,
        active_summary_rows,
        benchmark_rows,
        vector_rows,
        clusters,
        top_failure_counts,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
