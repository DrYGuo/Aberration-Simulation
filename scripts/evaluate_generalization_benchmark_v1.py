#!/usr/bin/env python3
"""Simulate and score frozen generalization-benchmark v1 designs.

This evaluates existing saved model checkpoints only. It does not train models
and it does not generate new benchmark designs.
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

from generate_targeted_enhanced_dataset import simulate_rows, target_scale_array
from run_active_12d_hole_search import (
    normalized_target_matrix,
    read_csv as read_rows,
    run_inference,
    summarize_evaluation,
    write_csv,
)
from run_model_selection_candidate import TARGET_COLUMNS, frozen_benchmark_split


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


COMPONENTS = (
    ("new_hole_challenge", "new_hole_design_csv"),
    ("broad_12d_representative_validation", "broad_design_csv"),
    ("anchor_regression_guard", "anchor_design_csv"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def metric_stats(values: list[float]) -> dict[str, float | int | None]:
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


def summarize_error_table(path: Path) -> dict[str, Any]:
    rows = read_csv_dicts(path)
    summary: dict[str, Any] = {
        "row_count": len(rows),
        "metrics": {},
        "by_regime": {},
        "by_failure_class": {},
    }
    for column in ERROR_COLUMNS:
        values = []
        for row in rows:
            try:
                values.append(float(row.get(column, "") or "nan"))
            except ValueError:
                continue
        summary["metrics"][column] = metric_stats([value for value in values if math.isfinite(value)])

    for group_key, output_key in (("sweep_label", "by_regime"), ("failure_class", "by_failure_class")):
        groups: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            groups.setdefault(str(row.get(group_key, "")), []).append(row)
        for label, group_rows in sorted(groups.items()):
            weighted = []
            nn_dist = []
            for row in group_rows:
                try:
                    weighted.append(float(row.get("weighted_abs_error", "") or "nan"))
                except ValueError:
                    pass
                try:
                    nn_dist.append(float(row.get("nn_distance_12d", "") or "nan"))
                except ValueError:
                    pass
            weighted = [value for value in weighted if math.isfinite(value)]
            nn_dist = [value for value in nn_dist if math.isfinite(value)]
            summary[output_key][label] = {
                "n": len(group_rows),
                "weighted_abs_error": metric_stats(weighted),
                "nn_distance_12d": metric_stats(nn_dist),
            }
    return summary


def load_or_simulate_component(
    *,
    component: str,
    design_csv: Path,
    output_dir: Path,
    batch_base_cases: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    component_dir = output_dir / component
    component_dir.mkdir(parents=True, exist_ok=True)
    feature_csv = component_dir / "benchmark_probe_features.csv"
    manifest_path = component_dir / "simulation_manifest.json"
    if feature_csv.exists():
        rows, _ = read_rows(feature_csv)
        manifest = {
            "status": "reused",
            "component": component,
            "feature_csv": str(feature_csv),
            "probe_rows": len(rows),
            "design_csv": str(design_csv),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        return rows, manifest

    design_rows, _ = read_rows(design_csv)
    if not design_rows:
        raise RuntimeError(f"empty benchmark design CSV for {component}: {design_csv}")
    probe_rows = simulate_rows(design_rows, batch_base_cases)
    for index, row in enumerate(probe_rows):
        row.setdefault("benchmark_component", component)
        row.setdefault("benchmark_probe_id", design_rows[index].get("benchmark_probe_id", f"{component}_{index:06d}"))
        row.setdefault("dataset_split_hint", "diagnostic_validation_only")
        row.setdefault("train_on_this", False)
    fieldnames = list(dict.fromkeys(name for row in probe_rows for name in row.keys()))
    write_csv(feature_csv, probe_rows, fieldnames)
    manifest = {
        "status": "complete",
        "component": component,
        "design_csv": str(design_csv),
        "feature_csv": str(feature_csv),
        "probe_rows": len(probe_rows),
        "batch_base_cases": batch_base_cases,
        "large_artifact_policy": "Drive backup only; not pushed to GitHub.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return probe_rows, manifest


def evaluate_model_on_component(
    *,
    model_name: str,
    dataset_csv: Path,
    run_dir: Path,
    split_manifest: Path,
    probe_rows: list[dict[str, Any]],
    component: str,
    output_dir: Path,
    predict_batch_size: int,
) -> dict[str, Any]:
    model_dir = output_dir / component / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    summary_path = model_dir / "component_score_summary.json"
    if summary_path.exists():
        return read_json(summary_path)

    dataset_rows, _ = read_rows(dataset_csv)
    if not dataset_rows:
        raise RuntimeError(f"empty training dataset CSV for {model_name}: {dataset_csv}")
    split_indices = frozen_benchmark_split(dataset_rows, manifest_path=split_manifest)
    train_rows = [dataset_rows[int(index)] for index in split_indices["train"]]
    train_matrix = normalized_target_matrix(train_rows, target_scale_array())
    inference_config = {
        "proposal": {
            "reference_sample_size": 300000,
            "seed": 1507,
            "nearest_neighbor_chunk_size": 25000,
        },
        "classification": {
            "failure_top_fraction": 0.10,
            "coverage_limited_nn_quantile": 0.75,
            "dense_failure_nn_quantile": 0.50,
        },
        "inference": {"predict_batch_size": predict_batch_size},
    }
    y_pred, inference_manifest = run_inference(
        dataset_rows,
        dataset_csv,
        run_dir,
        split_indices,
        probe_rows,
        inference_config,
        model_dir,
    )
    (model_dir / "inference_manifest.json").write_text(json.dumps(inference_manifest, indent=2) + "\n")
    if y_pred is None:
        raise RuntimeError(f"{model_name} inference failed on {component}: {inference_manifest}")
    evaluation_summary = summarize_evaluation(probe_rows, y_pred, train_matrix, inference_config, model_dir)
    error_summary = summarize_error_table(model_dir / "active_hole_nn_error_summary.csv")
    summary = {
        "status": "complete",
        "created_utc": utc_now(),
        "model": model_name,
        "component": component,
        "dataset_csv": str(dataset_csv),
        "run_dir": str(run_dir),
        "split_counts": {name: int(len(values)) for name, values in split_indices.items()},
        "inference_manifest": inference_manifest,
        "evaluation_summary": evaluation_summary,
        "error_summary": error_summary,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def component_score(summary: dict[str, Any]) -> float:
    metric = summary["error_summary"]["metrics"]["weighted_abs_error"]
    rmse = metric.get("rmse")
    if rmse is None:
        raise RuntimeError(f"missing weighted_abs_error RMSE for {summary.get('model')} {summary.get('component')}")
    return float(rmse)


def write_report(output_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Generalization Benchmark v1 Evaluation",
        "",
        f"Created UTC: `{payload['created_utc']}`",
        "",
        "This run simulates/extracts features for frozen benchmark rows and evaluates saved v13/v15 checkpoints. It does not train.",
        "",
        "## Overall Score",
        "",
        "| model | populated score | promotable | notes |",
        "|---|---:|---|---|",
    ]
    for model_name, score in payload["model_scores"].items():
        lines.append(
            f"| `{model_name}` | {score['weighted_score']} | {score['promotable']} | {score['notes']} |"
        )
    lines.extend(
        [
            "",
            "## Component Scores",
            "",
            "| component | model | rows | weighted RMSE | weighted MAE | weighted p95 | S3 vector RMSE | A3 vector RMSE | B2 vector RMSE |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["component_rows"]:
        lines.append(
            "| `{component}` | `{model}` | {rows} | {weighted_rmse} | {weighted_mae} | {weighted_p95} | "
            "{s3_rmse} | {a3_rmse} | {b2_rmse} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Lower populated score is better.",
            "- `new_hole_challenge`, `broad_12d_representative_validation`, and `anchor_regression_guard` are held out and must not be used for training.",
            "- v16 sampling/training should only be designed after this score is reviewed.",
            "",
        ]
    )
    (output_dir / "generalization_benchmark_v1_evaluation_report.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--new-hole-design-csv", type=Path, required=True)
    parser.add_argument("--broad-design-csv", type=Path, required=True)
    parser.add_argument("--anchor-design-csv", type=Path, required=True)
    parser.add_argument("--v13-dataset-csv", type=Path, required=True)
    parser.add_argument("--v13-run-dir", type=Path, required=True)
    parser.add_argument("--v15-dataset-csv", type=Path, required=True)
    parser.add_argument("--v15-run-dir", type=Path, required=True)
    parser.add_argument("--benchmark-split-manifest", type=Path, default=Path("configs/benchmark_split_v12_v2_row_keys.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-base-cases", type=int, default=192)
    parser.add_argument("--predict-batch-size", type=int, default=65536)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_manifest = read_json(args.benchmark_manifest)
    design_paths = {
        "new_hole_challenge": args.new_hole_design_csv,
        "broad_12d_representative_validation": args.broad_design_csv,
        "anchor_regression_guard": args.anchor_design_csv,
    }
    models = {
        "v13_1m_seed23": (args.v13_dataset_csv, args.v13_run_dir),
        "v15_active_hole_250k_seed23": (args.v15_dataset_csv, args.v15_run_dir),
    }
    status_path = args.output_dir / "generalization_benchmark_v1_evaluation_status.json"
    status_path.write_text(
        json.dumps(
            {
                "status": "started",
                "created_utc": utc_now(),
                "benchmark_manifest": str(args.benchmark_manifest),
                "output_dir": str(args.output_dir),
                "training_launched": False,
            },
            indent=2,
        )
        + "\n"
    )

    component_summaries: dict[str, dict[str, Any]] = {}
    simulation_manifests: dict[str, Any] = {}
    for component, design_path in design_paths.items():
        probe_rows, simulation_manifest = load_or_simulate_component(
            component=component,
            design_csv=design_path,
            output_dir=args.output_dir,
            batch_base_cases=args.batch_base_cases,
        )
        simulation_manifests[component] = simulation_manifest
        component_summaries[component] = {}
        for model_name, (dataset_csv, run_dir) in models.items():
            component_summaries[component][model_name] = evaluate_model_on_component(
                model_name=model_name,
                dataset_csv=dataset_csv,
                run_dir=run_dir,
                split_manifest=args.benchmark_split_manifest,
                probe_rows=probe_rows,
                component=component,
                output_dir=args.output_dir,
                predict_batch_size=args.predict_batch_size,
            )

    config_components = benchmark_manifest.get("components", [])
    weights = {
        str(row.get("component")): float(row.get("weight") or 0.0)
        for row in config_components
        if row.get("weight") not in ("", None)
    }
    scored_components = {
        "new_hole_challenge": weights.get("new_hole_challenge", 0.20),
        "broad_12d_representative_validation": weights.get("broad_12d_representative_validation", 0.20),
        "anchor_regression_guard": weights.get("anchor_regression_guard", 0.10),
    }
    normalizer = sum(scored_components.values())
    model_scores: dict[str, Any] = {}
    component_rows: list[dict[str, Any]] = []
    for model_name in models:
        weighted_score = 0.0
        for component, weight in scored_components.items():
            summary = component_summaries[component][model_name]
            score = component_score(summary)
            weighted_score += weight * score / max(normalizer, 1e-12)
            metrics = summary["error_summary"]["metrics"]
            component_rows.append(
                {
                    "component": component,
                    "model": model_name,
                    "rows": summary["error_summary"]["row_count"],
                    "weighted_rmse": metrics["weighted_abs_error"]["rmse"],
                    "weighted_mae": metrics["weighted_abs_error"]["mae"],
                    "weighted_p95": metrics["weighted_abs_error"]["p95"],
                    "s3_rmse": metrics["S3_vector_error"]["rmse"],
                    "a3_rmse": metrics["A3_vector_error"]["rmse"],
                    "b2_rmse": metrics["B2_vector_error"]["rmse"],
                }
            )
        model_scores[model_name] = {
            "weighted_score": float(weighted_score),
            "promotable": False,
            "notes": "benchmark populated; v16 decision requires review of broad/new-hole/anchor tradeoffs",
        }
    if model_scores["v15_active_hole_250k_seed23"]["weighted_score"] < model_scores["v13_1m_seed23"]["weighted_score"]:
        model_scores["v15_active_hole_250k_seed23"]["notes"] = "beats v13 on populated held-out benchmark components; still check fixed split regression before promotion"
    else:
        model_scores["v13_1m_seed23"]["notes"] = "retains lower populated held-out benchmark score"

    payload = {
        "status": "complete",
        "created_utc": utc_now(),
        "benchmark_manifest": str(args.benchmark_manifest),
        "output_dir": str(args.output_dir),
        "training_launched": False,
        "benchmark_components_scored": list(scored_components),
        "component_weights_used": scored_components,
        "simulation_manifests": simulation_manifests,
        "model_scores": model_scores,
        "component_rows": component_rows,
        "component_summaries": component_summaries,
        "large_artifact_policy": "Feature CSVs and per-probe error CSVs are Drive-backed subdirectory artifacts; GitHub should receive compact top-level summaries/reports only.",
    }
    (args.output_dir / "generalization_benchmark_v1_evaluation_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_csv(args.output_dir / "generalization_benchmark_v1_component_scores.csv", component_rows, list(component_rows[0]))
    write_report(args.output_dir, payload)
    status_path.write_text(json.dumps({"status": "complete", "finished_utc": utc_now(), "output_dir": str(args.output_dir)}, indent=2) + "\n")
    print(json.dumps({"status": "complete", "output_dir": str(args.output_dir), "model_scores": model_scores}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
