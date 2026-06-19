#!/usr/bin/env python3
"""Postmortem the v15 active-hole expansion against v13."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import median
from typing import Any


V13_RUN = Path(
    "training_results/model_selection_loop/"
    "D66_grouped_width320_lr6e-4_dropout0.075_v13_1m_d66_seed23_residual_nn_20260615_065556_utc"
)
V15_RUN = Path(
    "training_results/model_selection_loop/"
    "D66_grouped_width320_lr6e-4_dropout0.075_v15_active_hole_expanded_250k_d66_seed23_20260619_075900_utc"
)
ACTIVE_FAILED_REPORT = Path(
    "training_results/model_selection_reports/active_failed_region_error_report_20260619_074229_utc"
)
SAMPLING_QUALITY = Path("training_results/model_selection_reports/sampling_quality_v15_active_hole_expanded_250k_d66")
OUTPUT_ROOT = Path("training_results/model_selection_reports")


PROMOTION_METRICS = (
    "selection_weighted_score",
    "true_hard_target_normalized_mae",
    "validation_normalized_mae",
    "blind_normalized_mae",
    "stress_normalized_mae",
    "B2_magnitude_mae",
    "S3_magnitude_mae",
    "A3_magnitude_mae",
    "S3_high_magnitude_mae",
    "S3_high_magnitude_bias",
    "S3_high_magnitude_slope",
    "S3_high_mean_abs_angle_error_deg",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_registry(run_dir: Path) -> dict[str, str]:
    rows = read_csv(run_dir / "model_registry_model_loop.csv")
    if not rows:
        raise RuntimeError(f"empty registry: {run_dir}")
    return rows[-1]


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compare_registries(v13: dict[str, str], v15: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    for key in PROMOTION_METRICS:
        old = to_float(v13.get(key))
        new = to_float(v15.get(key))
        lower_is_better = key != "S3_high_magnitude_slope"
        delta = None if old is None or new is None else new - old
        rel = None if old in (None, 0.0) or new is None else delta / abs(old)
        if delta is None:
            improved = None
        elif key.endswith("_bias"):
            improved = abs(new) < abs(old)
        else:
            improved = delta < 0 if lower_is_better else delta > 0
        rows.append(
            {
                "metric": key,
                "v13": old,
                "v15": new,
                "delta_v15_minus_v13": delta,
                "relative_change": rel,
                "lower_is_better": lower_is_better,
                "v15_improved": improved,
            }
        )
    return rows


def compare_residual_summary() -> dict[str, Any]:
    old = read_json(V13_RUN / "residual_vs_nn_distance_summary.json")
    new = read_json(V15_RUN / "residual_vs_nn_distance_summary.json")
    fields = (
        "correlation_weighted_abs_error_vs_nn_distance",
        "spearman_weighted_abs_error_vs_nn_distance",
        "top_5_to_all_median_nn_distance_ratio",
    )
    out: dict[str, Any] = {"summary_delta": {}}
    for key in fields:
        out["summary_delta"][key] = {
            "v13": old.get(key),
            "v15": new.get(key),
            "delta": None if old.get(key) is None or new.get(key) is None else new[key] - old[key],
        }
    for group in ("all_blind_stress", "top_1_percent_residuals", "top_5_percent_residuals"):
        out[group] = {}
        for key in ("median_weighted_abs_error", "median_nn_distance_12d", "p95_weighted_abs_error", "p95_nn_distance_12d"):
            old_value = old.get(group, {}).get(key)
            new_value = new.get(group, {}).get(key)
            out[group][key] = {
                "v13": old_value,
                "v15": new_value,
                "delta": None if old_value is None or new_value is None else new_value - old_value,
            }
    return out


def compare_binned_residuals() -> list[dict[str, Any]]:
    old_rows = {
        (row["split"], row["nn_distance_quartile"]): row
        for row in read_csv(V13_RUN / "residual_vs_nn_distance_binned_summary.csv")
    }
    new_rows = {
        (row["split"], row["nn_distance_quartile"]): row
        for row in read_csv(V15_RUN / "residual_vs_nn_distance_binned_summary.csv")
    }
    comparisons = []
    for key in sorted(set(old_rows) & set(new_rows)):
        old = old_rows[key]
        new = new_rows[key]
        comparisons.append(
            {
                "split": key[0],
                "nn_distance_quartile": key[1],
                "v13_median_weighted_abs_error": to_float(old["median_weighted_abs_error"]),
                "v15_median_weighted_abs_error": to_float(new["median_weighted_abs_error"]),
                "delta_median_weighted_abs_error": to_float(new["median_weighted_abs_error"])
                - to_float(old["median_weighted_abs_error"]),
                "v13_p95_weighted_abs_error": to_float(old["p95_weighted_abs_error"]),
                "v15_p95_weighted_abs_error": to_float(new["p95_weighted_abs_error"]),
                "delta_p95_weighted_abs_error": to_float(new["p95_weighted_abs_error"])
                - to_float(old["p95_weighted_abs_error"]),
                "v13_median_nn_distance_12d": to_float(old["median_nn_distance_12d"]),
                "v15_median_nn_distance_12d": to_float(new["median_nn_distance_12d"]),
            }
        )
    return comparisons


def top_residual_overlap() -> dict[str, Any]:
    old = read_csv(V13_RUN / "residual_vs_nn_distance_top_residuals.csv")
    new = read_csv(V15_RUN / "residual_vs_nn_distance_top_residuals.csv")
    old_by_key = {row["row_key"]: row for row in old}
    new_by_key = {row["row_key"]: row for row in new}
    old_keys = set(old_by_key)
    new_keys = set(new_by_key)
    common = sorted(old_keys & new_keys)

    common_deltas = []
    for key in common:
        old_error = to_float(old_by_key[key]["weighted_abs_error"])
        new_error = to_float(new_by_key[key]["weighted_abs_error"])
        if old_error is not None and new_error is not None:
            common_deltas.append(new_error - old_error)

    persistent_by_regime: dict[str, int] = {}
    for key in common:
        regime = new_by_key[key].get("sweep_label", "")
        persistent_by_regime[regime] = persistent_by_regime.get(regime, 0) + 1

    return {
        "v13_top_count": len(old),
        "v15_top_count": len(new),
        "persistent_top_residual_count": len(common),
        "persistent_top_residual_fraction_of_v13_top": len(common) / len(old) if old else None,
        "dropped_from_v15_top_count": len(old_keys - new_keys),
        "new_in_v15_top_count": len(new_keys - old_keys),
        "common_rows_median_delta_weighted_abs_error": median(common_deltas) if common_deltas else None,
        "common_rows_fraction_worse": (
            sum(1 for value in common_deltas if value > 0) / len(common_deltas) if common_deltas else None
        ),
        "persistent_top_residuals_by_regime": dict(sorted(persistent_by_regime.items(), key=lambda item: item[1], reverse=True)[:12]),
        "limitation": (
            "Only top-residual CSVs were pushed, not full residual_vs_nn_distance.csv; "
            "therefore dropped rows cannot be assigned exact v15 errors from local GitHub artifacts."
        ),
    }


def sampling_summary() -> dict[str, Any]:
    summary = read_json(SAMPLING_QUALITY / "sampling_quality_summary.json")
    candidate_rows = read_csv(SAMPLING_QUALITY / "candidate_selection_summary.csv")
    nn_rows = read_csv(SAMPLING_QUALITY / "nearest_neighbor_coverage_summary.csv")
    return {
        "recommendation": summary.get("recommendation"),
        "warnings": summary.get("warnings"),
        "counts": summary.get("counts"),
        "quota": summary.get("quota"),
        "candidate_selection": summary.get("candidate_selection"),
        "split": summary.get("split"),
        "candidate_selection_rows": candidate_rows,
        "nearest_neighbor_rows": nn_rows,
    }


def active_hole_source_summary() -> dict[str, Any]:
    summary = read_json(ACTIVE_FAILED_REPORT / "active_failed_region_error_summary.json")
    clusters = read_csv(ACTIVE_FAILED_REPORT / "active_failed_subspace_cluster_centers.csv")
    error_rows = read_csv(ACTIVE_FAILED_REPORT / "active_failed_region_error_summary.csv")
    top_clusters = []
    for row in clusters[:8]:
        top_clusters.append(
            {
                "source_run": row["source_run"],
                "cluster_id": row["cluster_id"],
                "n": to_float(row["n"]),
                "median_weighted_error": to_float(row["median_weighted_error"]),
                "median_nn_distance_12d": to_float(row["median_nn_distance_12d"]),
                "dominant_regime": row["dominant_regime"],
                "dominant_failure_class": row["dominant_failure_class"],
            }
        )
    return {
        "n_top_failure_rows": summary.get("n_top_failure_rows"),
        "n_clusters": summary.get("n_clusters"),
        "source_active_runs": summary.get("source_active_runs"),
        "top_clusters": top_clusters,
        "error_summary_first_rows": error_rows[:10],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return lines


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")
    output_dir = OUTPUT_ROOT / f"v15_active_hole_postmortem_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    v13_registry = read_registry(V13_RUN)
    v15_registry = read_registry(V15_RUN)
    metric_comparison = compare_registries(v13_registry, v15_registry)
    residual_summary = compare_residual_summary()
    binned = compare_binned_residuals()
    overlap = top_residual_overlap()
    sampling = sampling_summary()
    active_source = active_hole_source_summary()

    conclusion = {
        "headline": (
            "v15 did not repair the benchmark-visible coverage failures; it slightly improved 12D "
            "nearest-neighbor distances but worsened fixed validation/blind/stress errors."
        ),
        "did_250k_reduce_original_hole_cluster_errors": (
            "not_proven_from_current_artifacts"
        ),
        "why_not_proven": (
            "The run did not perform a post-v15 inference retest on the original active-hole probe set. "
            "GitHub has only top-residual summaries for fixed blind/stress, not full active-probe predictions."
        ),
        "benchmark_effect": "damaged_fixed_benchmark_distribution",
        "coverage_effect": "global NN distances improved slightly, but weighted residuals worsened",
        "next_required_test": (
            "Run an inference-only v13-vs-v15 retest on the saved active-hole probe designs, or save full "
            "blind/stress residual CSVs, before deciding v16 sampling."
        ),
    }

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "v13_run": str(V13_RUN),
        "v15_run": str(V15_RUN),
        "conclusion": conclusion,
        "metric_comparison": metric_comparison,
        "residual_vs_nn_distance": residual_summary,
        "top_residual_overlap": overlap,
        "sampling_quality": sampling,
        "active_hole_source": active_source,
    }
    (output_dir / "v15_active_hole_postmortem_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(output_dir / "metric_comparison.csv", metric_comparison)
    write_csv(output_dir / "residual_vs_nn_binned_comparison.csv", binned)

    report_lines = [
        "# v15 Active-Hole Postmortem",
        "",
        f"Created UTC: {summary['created_utc']}",
        "",
        "## Conclusion",
        "",
        conclusion["headline"],
        "",
        f"- Original active-hole repair proven? **{conclusion['did_250k_reduce_original_hole_cluster_errors']}**",
        f"- Benchmark effect: **{conclusion['benchmark_effect']}**",
        f"- Coverage effect: **{conclusion['coverage_effect']}**",
        "",
        "The v15 data generation itself was valid: 250,000 new training-only rows were added with no benchmark leakage. "
        "But v15 worsened the fixed benchmark metrics and did not reduce the benchmark-visible sparse-tail residuals.",
        "",
        "## Benchmark Metrics",
        "",
    ]
    report_lines.extend(
        markdown_table(
            metric_comparison,
            ["metric", "v13", "v15", "delta_v15_minus_v13", "relative_change", "v15_improved"],
        )
    )
    report_lines.extend(
        [
            "",
            "## Residual-vs-NN",
            "",
            f"All blind/stress median weighted error changed from "
            f"{residual_summary['all_blind_stress']['median_weighted_abs_error']['v13']} to "
            f"{residual_summary['all_blind_stress']['median_weighted_abs_error']['v15']}.",
            "",
            f"Top 5% residual median weighted error changed from "
            f"{residual_summary['top_5_percent_residuals']['median_weighted_abs_error']['v13']} to "
            f"{residual_summary['top_5_percent_residuals']['median_weighted_abs_error']['v15']}.",
            "",
            f"Top 5% median NN distance changed from "
            f"{residual_summary['top_5_percent_residuals']['median_nn_distance_12d']['v13']} to "
            f"{residual_summary['top_5_percent_residuals']['median_nn_distance_12d']['v15']}.",
            "",
            "Interpretation: the new rows moved some benchmark points slightly closer in 12D space, "
            "but the model's errors got worse. That is consistent with distribution/optimization damage, "
            "or the active-hole mixture pulling capacity away from the fixed benchmark, rather than a clean repair.",
            "",
            "## Top-Residual Persistence",
            "",
            f"- v13 top residual rows: {overlap['v13_top_count']}",
            f"- v15 top residual rows: {overlap['v15_top_count']}",
            f"- persistent top residual rows: {overlap['persistent_top_residual_count']} "
            f"({overlap['persistent_top_residual_fraction_of_v13_top']:.3f} of v13 top residuals)",
            f"- median error delta among persistent top residuals: {overlap['common_rows_median_delta_weighted_abs_error']}",
            f"- fraction of persistent top residuals worse under v15: {overlap['common_rows_fraction_worse']}",
            "",
            "Limitation: only top-residual CSVs were pushed. Rows that dropped out of v15's top-residual list may have improved, "
            "but their exact v15 errors are not available locally without the full residual CSV or a rerun.",
            "",
            "## Sampling",
            "",
            f"- sampling recommendation: {sampling['recommendation']}",
            f"- warnings: {sampling['warnings']}",
            f"- new rows: {sampling['candidate_selection']['new_rows']}",
            f"- far-NN fraction: {sampling['candidate_selection']['role_fractions'].get('far_nn')}",
            f"- bridge/anchor fraction: {sampling['candidate_selection']['role_fractions'].get('bridge_anchor')}",
            f"- benchmark leakage pass: {sampling['split'].get('training_only_leakage_pass')}",
            "",
            "## Original Active-Hole Source",
            "",
            f"- active top failure rows used to design v15: {active_source['n_top_failure_rows']}",
            f"- active failure clusters: {active_source['n_clusters']}",
            "",
            "The current artifacts show what v13 failed on and what v15 added, but they do not include a held-out post-v15 "
            "active-hole retest. That should be the next small inference-only job.",
            "",
            "## Next Step",
            "",
            conclusion["next_required_test"],
            "",
        ]
    )
    (output_dir / "v15_active_hole_postmortem_report.md").write_text("\n".join(report_lines))
    print(json.dumps({"output_dir": str(output_dir), "conclusion": conclusion}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
