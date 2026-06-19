#!/usr/bin/env python3
"""Summarize the v15 active-hole expansion against the v13 baseline."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


V13_GLOB = "training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v13_1m_d66_seed23_residual_nn_*/selection_score.json"
V15_GLOB = "training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_v15_active_hole_expanded_250k_d66_seed23_*/selection_score.json"
V15_BATCH_GLOB = "training_results/model_selection_batches/v15_active_hole_expanded_250k_d66_*/batch_summary.csv"
OUTPUT_ROOT = Path("training_results/model_selection_reports")


KEYS = (
    "weighted_score",
    "true_hard_target_normalized_mae",
    "overall_normalized_mae",
    "overall_normalized_p95",
    "blind_normalized_mae",
    "stress_normalized_mae",
    "S3_high_magnitude_mae",
    "S3_high_magnitude_bias",
    "S3_high_magnitude_slope",
    "B2_magnitude_mae",
    "A3_magnitude_mae",
)


def latest_path(pattern: str) -> Path:
    matches = sorted(Path().glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"no matches for {pattern}")
    return matches[0]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_batch_summary(path: Path) -> dict[str, str]:
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return {}
    return rows[-1]


def collect_run(selection_path: Path) -> dict[str, Any]:
    run_dir = selection_path.parent
    values: dict[str, Any] = {"run_dir": str(run_dir)}
    for filename in ("selection_score.json", "metrics_model_loop.json"):
        path = run_dir / filename
        if path.exists():
            values.update(read_json(path))
    registry = run_dir / "model_registry_model_loop.csv"
    if registry.exists():
        rows = list(csv.DictReader(registry.open()))
        if rows:
            for key, value in rows[-1].items():
                values.setdefault(key, value)
    return values


def as_float(data: dict[str, Any], key: str) -> float | None:
    value = data.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")
    output_dir = OUTPUT_ROOT / f"v15_active_hole_decision_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    v13 = collect_run(latest_path(V13_GLOB))
    v15 = collect_run(latest_path(V15_GLOB))
    batch = read_batch_summary(latest_path(V15_BATCH_GLOB))
    for key, value in batch.items():
        v15.setdefault(key, value)

    comparison = []
    v15_better_count = 0
    for key in KEYS:
        old = as_float(v13, key)
        new = as_float(v15, key)
        delta = None if old is None or new is None else new - old
        relative = None if old in (None, 0.0) or new is None else (new - old) / abs(old)
        lower_is_better = key not in {"S3_high_magnitude_slope"}
        improved = None
        if delta is not None:
            improved = delta < 0 if lower_is_better else delta > 0
            if improved:
                v15_better_count += 1
        comparison.append(
            {
                "metric": key,
                "v13": old,
                "v15": new,
                "delta_v15_minus_v13": delta,
                "relative_change": relative,
                "lower_is_better": lower_is_better,
                "v15_improved": improved,
            }
        )

    residual_summary_path = Path(v15["run_dir"]) / "residual_vs_nn_distance_summary.json"
    residual_summary = read_json(residual_summary_path) if residual_summary_path.exists() else {}
    promote = (
        as_float(v15, "weighted_score") is not None
        and as_float(v13, "weighted_score") is not None
        and as_float(v15, "weighted_score") < as_float(v13, "weighted_score")
        and as_float(v15, "blind_normalized_mae") <= as_float(v13, "blind_normalized_mae")
        and as_float(v15, "stress_normalized_mae") <= as_float(v13, "stress_normalized_mae")
    )
    recommendation = (
        "promote_v15"
        if promote
        else "do_not_promote_v15; keep v13 seed23 as champion and use v15 as coverage-diagnostic evidence"
    )

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "v13_run_dir": v13["run_dir"],
        "v15_run_dir": v15["run_dir"],
        "recommendation": recommendation,
        "v15_better_metric_count": v15_better_count,
        "n_compared_metrics": len(comparison),
        "comparison": comparison,
        "residual_vs_nn_distance": {
            "decision_interpretation": residual_summary.get("decision_interpretation"),
            "correlation_weighted_abs_error_vs_nn_distance": residual_summary.get(
                "correlation_weighted_abs_error_vs_nn_distance"
            ),
            "spearman_weighted_abs_error_vs_nn_distance": residual_summary.get(
                "spearman_weighted_abs_error_vs_nn_distance"
            ),
            "top_5_to_all_median_nn_distance_ratio": residual_summary.get(
                "top_5_to_all_median_nn_distance_ratio"
            ),
        },
        "next_step": (
            "Do not add another blind expansion immediately. Review v15 active-hole residual clusters; "
            "if the same sparse subspaces remain, design a more selective v16 hole-fill or revise "
            "features/loss for dense failures."
        ),
    }
    (output_dir / "v15_active_hole_decision_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# v15 Active-Hole Decision",
        "",
        f"Created UTC: {summary['created_utc']}",
        "",
        f"Recommendation: **{recommendation}**",
        "",
        "| Metric | v13 | v15 | Delta | v15 improved? |",
        "|---|---:|---:|---:|---|",
    ]
    for row in comparison:
        lines.append(
            f"| {row['metric']} | {row['v13']} | {row['v15']} | "
            f"{row['delta_v15_minus_v13']} | {row['v15_improved']} |"
        )
    lines.extend(
        [
            "",
            "## Residual-vs-NN",
            "",
            f"Decision interpretation: {summary['residual_vs_nn_distance']['decision_interpretation']}",
            f"Pearson correlation: {summary['residual_vs_nn_distance']['correlation_weighted_abs_error_vs_nn_distance']}",
            f"Spearman correlation: {summary['residual_vs_nn_distance']['spearman_weighted_abs_error_vs_nn_distance']}",
            f"Top-5%/all median NN-distance ratio: {summary['residual_vs_nn_distance']['top_5_to_all_median_nn_distance_ratio']}",
            "",
            "## Next Step",
            "",
            summary["next_step"],
            "",
        ]
    )
    (output_dir / "v15_active_hole_decision_report.md").write_text("\n".join(lines))
    print(json.dumps({"output_dir": str(output_dir), "recommendation": recommendation}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
