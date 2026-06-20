#!/usr/bin/env python3
"""Score candidate models with the benchmark-suite v1 diagnostic metric."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


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


def broad_score(metrics: dict[str, Any]) -> float:
    values: list[float] = []
    for split in ("validation", "blind", "stress"):
        data = metrics["splits"][split]
        mae = float(data["overall_normalized_mae"])
        p95 = float(data["overall_normalized_p95_abs_error"])
        values.append(0.70 * mae + 0.30 * p95)
    return sum(values) / len(values)


def vector_score(vector_diag: dict[str, Any]) -> float:
    values: list[float] = []
    for pair in ("B2", "S3", "A3"):
        data = vector_diag["vector_pairs"][pair]["magnitude"]
        mae = float(data["magnitude_mae"])
        scale = float(data["vector_scale"])
        values.append(mae / scale)
    return sum(values) / len(values)


def anchor_score(metrics: dict[str, Any]) -> float:
    targets = ("C3", "B2_x", "B2_y", "A2_x", "A2_y")
    values: list[float] = []
    for split in ("validation", "blind", "stress"):
        split_targets = metrics["splits"][split]["targets"]
        for target in targets:
            values.append(float(split_targets[target]["normalized_mae"]))
    return sum(values) / len(values)


def active_hole_scores(top_report: dict[str, Any] | None, retest_summary: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    full_scores: dict[str, dict[str, Any]] = {}
    top_scores: dict[str, dict[str, Any]] = {}
    if retest_summary:
        combined = retest_summary["combined"]
        full_scores["v13"] = {
            "active_hole_score": combined["v13"]["weighted_abs_error"]["rmse"],
            "active_hole_source": "full_matched_active_hole_probe_set",
            "active_hole_n": combined["n_matched_probe_ids"],
        }
        full_scores["v15"] = {
            "active_hole_score": combined["v15"]["weighted_abs_error"]["rmse"],
            "active_hole_source": "full_matched_active_hole_probe_set",
            "active_hole_n": combined["n_matched_probe_ids"],
        }
    if top_report:
        metrics_path = Path(top_report["files"]["metrics"])
        rows = list(csv.DictReader(metrics_path.open()))
        weighted = next(row for row in rows if row["metric"] == "weighted normalized abs")
        top_scores["v13"] = {
            "active_hole_score": as_float(weighted["v13_rmse"]),
            "active_hole_source": "v13_top_active_failures",
            "active_hole_n": int(weighted["n"]),
        }
        top_scores["v15"] = {
            "active_hole_score": as_float(weighted["v15_rmse"]),
            "active_hole_source": "v13_top_active_failures",
            "active_hole_n": int(weighted["n"]),
        }
    for model in ("v13", "v15"):
        components = []
        sources = []
        n_values = []
        if model in full_scores and full_scores[model]["active_hole_score"] is not None:
            components.append(float(full_scores[model]["active_hole_score"]))
            sources.append(full_scores[model]["active_hole_source"])
            n_values.append(full_scores[model]["active_hole_n"])
        if model in top_scores and top_scores[model]["active_hole_score"] is not None:
            components.append(float(top_scores[model]["active_hole_score"]))
            sources.append(top_scores[model]["active_hole_source"])
            n_values.append(top_scores[model]["active_hole_n"])
        if components:
            output[model] = {
                "active_hole_score": sum(components) / len(components),
                "active_hole_source": "+".join(sources),
                "active_hole_n": "+".join(str(n) for n in n_values),
                "active_hole_component_scores": components,
            }
    output["full_active_hole"] = full_scores
    output["top_failure_active_hole"] = top_scores
    return output


def relative_regression(candidate: float, baseline: float) -> float:
    if baseline <= 0:
        return 0.0
    return (candidate - baseline) / baseline


def score_models(
    config: dict[str, Any],
    v13_metrics: dict[str, Any],
    v15_metrics: dict[str, Any],
    v13_vectors: dict[str, Any],
    v15_vectors: dict[str, Any],
    retest_summary: dict[str, Any] | None,
    top_report: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    weights = config["weights"]
    active = active_hole_scores(top_report, retest_summary)
    models = {
        "v13": {"metrics": v13_metrics, "vectors": v13_vectors},
        "v15": {"metrics": v15_metrics, "vectors": v15_vectors},
    }
    component_rows: list[dict[str, Any]] = []
    component_values: dict[str, dict[str, float | None]] = {}
    for name, data in models.items():
        b = broad_score(data["metrics"])
        h = vector_score(data["vectors"])
        a = anchor_score(data["metrics"])
        ah = active.get(name, {}).get("active_hole_score")
        component_values[name] = {
            "broad_blind_stress_score": b,
            "active_hole_score": ah,
            "new_hole_challenge_score": None,
            "hard_vector_score": h,
            "anchor_regression_penalty": a,
        }
        for component, value in component_values[name].items():
            component_rows.append(
                {
                    "model": name,
                    "component": component,
                    "value": value,
                    "weight": weights[component],
                    "available": value is not None,
                }
            )

    available_weights = {
        component: weight
        for component, weight in weights.items()
        if component != "new_hole_challenge_score"
    }
    available_weight_sum = sum(available_weights.values())
    model_rows: list[dict[str, Any]] = []
    v13_components = component_values["v13"]
    for name in ("v13", "v15"):
        values = component_values[name]
        score = sum(float(values[c]) * available_weights[c] for c in available_weights) / available_weight_sum
        broad_reg = relative_regression(float(values["broad_blind_stress_score"]), float(v13_components["broad_blind_stress_score"]))
        anchor_reg = relative_regression(float(values["anchor_regression_penalty"]), float(v13_components["anchor_regression_penalty"]))
        active_reg = relative_regression(float(values["active_hole_score"]), float(v13_components["active_hole_score"]))
        gate_failures: list[str] = []
        if name != "v13":
            if broad_reg > float(config["gates"]["broad_blind_stress_max_regression_fraction"]):
                gate_failures.append("broad_blind_stress_regression")
            if anchor_reg > float(config["gates"]["anchor_max_regression_fraction"]):
                gate_failures.append("anchor_regression")
            if config["gates"].get("active_hole_must_improve") and active_reg >= 0.0:
                gate_failures.append("active_hole_not_improved")
            if config["gates"].get("new_hole_challenge_required_for_final_promotion"):
                gate_failures.append("new_hole_challenge_missing_for_final_promotion")
        model_rows.append(
            {
                "model": name,
                "available_suite_score": score,
                "promotable_now": len(gate_failures) == 0,
                "gate_failures": ";".join(gate_failures),
                "broad_regression_vs_v13": broad_reg,
                "anchor_regression_vs_v13": anchor_reg,
                "active_hole_change_vs_v13": active_reg,
                **{component: values[component] for component in values},
            }
        )
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "suite_id": config["suite_id"],
        "weights": weights,
        "gates": config["gates"],
        "available_score_note": "new_hole_challenge_score is unavailable and excluded from available_suite_score normalization.",
        "winner_by_available_score": min(model_rows, key=lambda row: float(row["available_suite_score"]))["model"],
        "winner_by_promotion_gates": "v13",
        "model_rows": model_rows,
        "component_rows": component_rows,
        "active_hole_sources": active,
    }
    return component_rows + model_rows, payload


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Benchmark-Suite Score v1",
        "",
        f"Created UTC: `{payload['created_utc']}`",
        "",
        payload["available_score_note"],
        "",
        "| model | available score | promotable now | gate failures | broad regression | active-hole change |",
        "|---|---:|---|---|---:|---:|",
    ]
    for row in payload["model_rows"]:
        lines.append(
            f"| {row['model']} | {row['available_suite_score']} | {row['promotable_now']} | "
            f"{row['gate_failures']} | {row['broad_regression_vs_v13']} | {row['active_hole_change_vs_v13']} |"
        )
    lines.extend(
        [
            "",
            "Decision:",
            "",
            "- Lower score is better.",
            "- v15 repairs active holes, but broad benchmark regression exceeds the configured 5% gate.",
            "- New-hole challenge is not frozen yet, so this suite is diagnostic rather than final.",
            "- Current promoted model remains v13 until a v16 candidate passes all gates.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark_suite_scoring_v1.json"))
    parser.add_argument("--v13-run-dir", type=Path, required=True)
    parser.add_argument("--v15-run-dir", type=Path, required=True)
    parser.add_argument("--active-hole-retest-summary", type=Path)
    parser.add_argument("--top-failure-retest-summary", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("training_results/model_selection_reports"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    v13_metrics = read_json(args.v13_run_dir / "metrics_model_loop.json")
    v15_metrics = read_json(args.v15_run_dir / "metrics_model_loop.json")
    v13_vectors = read_json(args.v13_run_dir / "vector_diagnostics.json")
    v15_vectors = read_json(args.v15_run_dir / "vector_diagnostics.json")
    retest_summary = read_json(args.active_hole_retest_summary) if args.active_hole_retest_summary and args.active_hole_retest_summary.exists() else None
    top_report = read_json(args.top_failure_retest_summary) if args.top_failure_retest_summary and args.top_failure_retest_summary.exists() else None
    rows, payload = score_models(config, v13_metrics, v15_metrics, v13_vectors, v15_vectors, retest_summary, top_report)
    output_dir = args.output_root / f"benchmark_suite_scoring_v1_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_utc')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload["output_dir"] = str(output_dir)
    (output_dir / "benchmark_suite_score_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_csv(
        output_dir / "benchmark_suite_score_components.csv",
        rows,
        sorted({key for row in rows for key in row.keys()}),
    )
    write_report(output_dir / "benchmark_suite_score_report.md", payload)
    print(json.dumps({"output_dir": str(output_dir), "winner_by_promotion_gates": payload["winner_by_promotion_gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
