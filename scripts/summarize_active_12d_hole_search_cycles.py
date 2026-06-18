"""Summarize active 12D hole-search cycles and write the next sampling plan."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


def summarize_run(path: Path) -> dict[str, Any]:
    summary = read_json(path / "active_hole_search_summary.json")
    evaluation = summary.get("evaluation_summary") or {}
    proposal = summary.get("proposal_summary") or {}
    classes = evaluation.get("failure_class_counts") or {}
    top_rows = read_csv_rows(path / "active_hole_search_top_failures.csv")
    top_by_mode: Counter[str] = Counter(row.get("proposal_mode", "") for row in top_rows)
    top_by_a3_s3: Counter[str] = Counter(row.get("relative_angle_A3_S3_category", "") for row in top_rows)
    top_by_b2_s3: Counter[str] = Counter(row.get("relative_angle_B2_S3_category", "") for row in top_rows)
    dense_rows = [row for row in top_rows if row.get("failure_class") == "dense_model_feature_loss_failure"]
    dense_by_mode: Counter[str] = Counter(row.get("proposal_mode", "") for row in dense_rows)
    return {
        "run_dir": str(path),
        "run_name": path.name,
        "workflow_id": summary.get("workflow_id"),
        "n_probes": evaluation.get("n_probes"),
        "proposal_mode_counts": proposal.get("proposal_mode_counts", {}),
        "regime_counts": proposal.get("regime_counts", {}),
        "weighted_error_median": (evaluation.get("weighted_error") or {}).get("median"),
        "weighted_error_p90": (evaluation.get("weighted_error") or {}).get("p90"),
        "weighted_error_p95": (evaluation.get("weighted_error") or {}).get("p95"),
        "weighted_error_max": (evaluation.get("weighted_error") or {}).get("max"),
        "nn_distance_median": (evaluation.get("nn_distance_12d") or {}).get("median"),
        "nn_distance_p75": (evaluation.get("nn_distance_12d") or {}).get("p75"),
        "nn_distance_p95": (evaluation.get("nn_distance_12d") or {}).get("p95"),
        "error_nn_corr": evaluation.get("correlation_weighted_error_vs_nn_distance"),
        "error_nn_spearman": evaluation.get("spearman_weighted_error_vs_nn_distance"),
        "not_top_failure": classes.get("not_top_failure", 0),
        "coverage_limited_sparse_failure": classes.get("coverage_limited_sparse_failure", 0),
        "mixed_failure": classes.get("mixed_failure", 0),
        "dense_model_feature_loss_failure": classes.get("dense_model_feature_loss_failure", 0),
        "top_failure_median_nn_distance": evaluation.get("top_failure_median_nn_distance"),
        "all_probe_median_nn_distance": evaluation.get("all_probe_median_nn_distance"),
        "top_failure_count": len(top_rows),
        "top_failure_by_mode": dict(top_by_mode),
        "top_failure_by_A3_S3_angle": dict(top_by_a3_s3),
        "top_failure_by_B2_S3_angle": dict(top_by_b2_s3),
        "dense_failure_by_mode": dict(dense_by_mode),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", type=Path, default=Path("training_results/model_selection_reports"))
    parser.add_argument("--output-root", type=Path, default=Path("training_results/model_selection_reports"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dirs = sorted(
        p
        for p in args.search_root.glob("v13_active_12d_hole_search*")
        if (p / "active_hole_search_summary.json").exists()
    )
    if not run_dirs:
        raise RuntimeError(f"No active hole-search summaries found under {args.search_root}")
    rows = [summarize_run(path) for path in run_dirs]
    out_dir = args.output_root / f"active_12d_hole_search_synthesis_{utc_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_rows = []
    for index, row in enumerate(rows, start=1):
        csv_rows.append(
            {
                "cycle": index,
                "run_name": row["run_name"],
                "n_probes": row["n_probes"],
                "weighted_error_median": row["weighted_error_median"],
                "weighted_error_p95": row["weighted_error_p95"],
                "weighted_error_max": row["weighted_error_max"],
                "nn_distance_median": row["nn_distance_median"],
                "nn_distance_p95": row["nn_distance_p95"],
                "error_nn_spearman": row["error_nn_spearman"],
                "coverage_limited_sparse_failure": row["coverage_limited_sparse_failure"],
                "mixed_failure": row["mixed_failure"],
                "dense_model_feature_loss_failure": row["dense_model_feature_loss_failure"],
                "top_failure_median_nn_distance": row["top_failure_median_nn_distance"],
            }
        )
    write_csv(
        out_dir / "active_12d_cycle_comparison.csv",
        csv_rows,
        list(csv_rows[0]),
    )

    latest = rows[-1]
    total_coverage = sum(int(row["coverage_limited_sparse_failure"]) for row in rows)
    total_dense = sum(int(row["dense_model_feature_loss_failure"]) for row in rows)
    total_mixed = sum(int(row["mixed_failure"]) for row in rows)
    top_modes: Counter[str] = Counter()
    dense_modes: Counter[str] = Counter()
    for row in rows:
        top_modes.update(row["top_failure_by_mode"])
        dense_modes.update(row["dense_failure_by_mode"])

    sampling_plan = {
        "plan_name": "v15_active_hole_targeted_sampling_plan",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "basis": [row["run_name"] for row in rows],
        "current_champion_to_freeze": "v13 1M seed23 D66 grouped-head checkpoint rebuild",
        "primary_recommendation": "targeted_data_expansion_plus_parallel_feature_model_audit",
        "recommended_new_rows": 250000,
        "escalate_to_500k_only_if": [
            "v15 250k targeted expansion reduces sparse-tail failures but residual-vs-NN correlation remains above 0.45",
            "high-amplitude coupled-full failures remain concentrated at NN distance above the v13 active-search median",
        ],
        "sampling_mix": {
            "high_amp_A1_B2_S3_A3_alignment_bridge": 0.30,
            "coverage_limited_sparse_cluster_jitter": 0.25,
            "far_nn_coupled_full_random": 0.20,
            "coupled_sparse_random_controls": 0.10,
            "sobol_lhs_global_balance": 0.10,
            "orthogonal_diagnostic_controls": 0.05,
        },
        "regime_mix": {
            "coupled_full_random": 0.78,
            "coupled_sparse_random": 0.10,
            "coupled_A1_B2_S3_random": 0.04,
            "coupled_C3_A3_S3_random": 0.03,
            "coupled_A3_S3_random": 0.025,
            "coupled_B2_S3_random": 0.025,
        },
        "parallel_non_data_work": [
            "Audit high-amplitude A1/B2/S3/A3 feature sensitivity and target scaling.",
            "Check whether A3/S3 high-amplitude vector errors are dominated by magnitude compression or direction.",
            "Test loss/architecture changes only after v15 targeted data establishes whether sparse failures shrink.",
        ],
        "do_not_do_next": [
            "Do not jump directly to another broad 1M expansion.",
            "Do not train on active diagnostic rows directly without converting them into a balanced planned dataset.",
            "Do not optimize against final blind test repeatedly.",
        ],
    }

    summary = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(out_dir),
        "n_cycles_summarized": len(rows),
        "cycle_rows": rows,
        "aggregate_failure_counts": {
            "coverage_limited_sparse_failure": total_coverage,
            "mixed_failure": total_mixed,
            "dense_model_feature_loss_failure": total_dense,
        },
        "top_failure_modes_all_cycles": dict(top_modes),
        "dense_failure_modes_all_cycles": dict(dense_modes),
        "interpretation": {
            "coverage_limited": total_coverage > total_dense,
            "dense_failures_present": total_dense > 0,
            "latest_cycle": latest["run_name"],
            "latest_cycle_dense_failures": latest["dense_model_feature_loss_failure"],
            "latest_cycle_coverage_failures": latest["coverage_limited_sparse_failure"],
        },
        "sampling_plan_path": str(out_dir / "v15_active_hole_targeted_sampling_plan.json"),
    }
    (out_dir / "active_12d_hole_search_synthesis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "v15_active_hole_targeted_sampling_plan.json").write_text(json.dumps(sampling_plan, indent=2) + "\n")

    lines = [
        "# Active 12D Hole-Search Synthesis",
        "",
        f"Created UTC: `{summary['created_utc']}`",
        "",
        "## Cycle Comparison",
        "",
        "| cycle | probes | median err | p95 err | median NN | Spearman err-NN | coverage | mixed | dense |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(rows, start=1):
        lines.append(
            f"| {index} | {row['n_probes']} | {row['weighted_error_median']:.5f} | "
            f"{row['weighted_error_p95']:.5f} | {row['nn_distance_median']:.5f} | "
            f"{row['error_nn_spearman']:.3f} | {row['coverage_limited_sparse_failure']} | "
            f"{row['mixed_failure']} | {row['dense_model_feature_loss_failure']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Aggregate coverage-limited failures: `{total_coverage}`",
            f"- Aggregate mixed failures: `{total_mixed}`",
            f"- Aggregate dense/model-feature-loss failures: `{total_dense}`",
            "- The active search supports targeted expansion, but dense high-amplitude failures are real enough to track separately.",
            "",
            "## Recommended Next Move",
            "",
            "- Prepare `v15_active_hole_targeted` as a 250k targeted expansion.",
            "- Keep v13 checkpoint frozen for active-search comparison.",
            "- Run feature/model/loss diagnostics in parallel for high-amplitude A1/B2/S3/A3 cases.",
            "",
            "See `v15_active_hole_targeted_sampling_plan.json` for the proposed sampling mix.",
        ]
    )
    (out_dir / "active_12d_hole_search_synthesis_report.md").write_text("\n".join(lines) + "\n")
    print("active 12D synthesis output:", out_dir, flush=True)
    print("active 12D synthesis status: complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
