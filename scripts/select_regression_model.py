"""Rank regression runs with a hard-regime weighted selection metric."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


HARD_LABELS = (
    "coupled_full_random",
    "coupled_sparse_random",
)
HARD_TARGETS = (
    "S3_x",
    "S3_y",
    "A3_x",
    "A3_y",
)
EASY_TARGETS = (
    "C3",
    "A1_x",
    "A1_y",
    "B2_x",
    "B2_y",
    "A2_x",
    "A2_y",
)
DEFAULT_TARGET_PHYSICAL_SCALES = {
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
DEFAULT_SCORE_WEIGHTS = {
    "hard_label_normalized_mae": 0.30,
    "hard_label_normalized_p95": 0.20,
    "hard_target_normalized_mae": 0.25,
    "hard_target_normalized_p95": 0.15,
    "overall_normalized_mae": 0.05,
    "overall_normalized_p95": 0.05,
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def nested_metric(data: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def target_metric(metrics: dict[str, Any], target: str, metric: str) -> float:
    return nested_metric(metrics, "test_targets", target, metric)


def label_metric(metrics: dict[str, Any], label: str, metric: str) -> float:
    return nested_metric(metrics, "labels", label, "test", metric)


def split_metrics(metrics: dict[str, Any], split: str) -> dict[str, Any]:
    splits = metrics.get("splits")
    if isinstance(splits, dict) and split in splits and isinstance(splits[split], dict):
        return splits[split]
    return metrics


def split_target_metric(
    metrics: dict[str, Any],
    split: str,
    target: str,
    metric: str,
    target_scales: dict[str, float] | None = None,
) -> float:
    data = split_metrics(metrics, split)
    target_scales = target_scales or DEFAULT_TARGET_PHYSICAL_SCALES
    if "targets" in data:
        value = nested_metric(data, "targets", target, metric, default=float("nan"))
        if value == value:
            return value
    if metric.startswith("normalized_"):
        raw_metric = metric.replace("normalized_", "")
        raw_value = nested_metric(data, "targets", target, raw_metric, default=float("nan"))
        if raw_value != raw_value:
            raw_value = nested_metric(metrics, "test_targets", target, raw_metric, default=float("nan"))
        if raw_value != raw_value:
            return float("nan")
        return raw_value / max(float(target_scales.get(target, 1.0)), 1e-12)
    value = nested_metric(metrics, "test_targets", target, metric, default=float("nan"))
    return value


def split_label_metric(
    metrics: dict[str, Any],
    split: str,
    label: str,
    metric: str,
    target_scales: dict[str, float] | None = None,
) -> float:
    data = split_metrics(metrics, split)
    target_scales = target_scales or DEFAULT_TARGET_PHYSICAL_SCALES
    if "labels" in data:
        value = nested_metric(data, "labels", label, metric, default=float("nan"))
        if value == value:
            return value
    if metric.startswith("normalized_"):
        raw_metric = metric.replace("normalized_", "")
        raw_value = nested_metric(data, "labels", label, raw_metric, default=float("nan"))
        if raw_value != raw_value:
            raw_value = nested_metric(metrics, "labels", label, "test", raw_metric, default=float("nan"))
        if raw_value != raw_value:
            return float("nan")
        mean_scale = mean([float(target_scales.get(target, 1.0)) for target in target_scales])
        return raw_value / max(mean_scale, 1e-12)
    value = nested_metric(metrics, "labels", label, "test", metric, default=float("nan"))
    return value


def split_overall_metric(metrics: dict[str, Any], split: str, metric: str) -> float:
    data = split_metrics(metrics, split)
    return float(data.get(metric, metrics.get(metric, 0.0)) or 0.0)


def mean(values: list[float]) -> float:
    finite = [value for value in values if value == value and value not in (float("inf"), float("-inf"))]
    return sum(finite) / len(finite) if finite else 0.0


def relative_change(value: float, baseline: float) -> float:
    if baseline <= 0:
        return 0.0 if value <= baseline else float("inf")
    return (value - baseline) / baseline


def run_name_for(metrics_path: Path, metrics: dict[str, Any]) -> str:
    manifest_name = metrics.get("run_name")
    if manifest_name:
        return str(manifest_name)
    return metrics_path.parent.name


def score_run(
    metrics_path: Path,
    metrics: dict[str, Any],
    *,
    baseline: dict[str, Any] | None,
    easy_regression_limit: float,
    selection_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selection_config = selection_config or {}
    split = str(selection_config.get("selection_split", "validation"))
    hard_labels = tuple(selection_config.get("hard_labels", HARD_LABELS))
    hard_targets = tuple(selection_config.get("hard_targets", HARD_TARGETS))
    easy_targets = tuple(selection_config.get("easy_targets", EASY_TARGETS))
    target_scales = {**DEFAULT_TARGET_PHYSICAL_SCALES, **selection_config.get("target_physical_scales", {})}
    weights = {**DEFAULT_SCORE_WEIGHTS, **selection_config.get("score_weights", {})}

    hard_label_norm_mae = mean([split_label_metric(metrics, split, label, "normalized_mae", target_scales) for label in hard_labels])
    hard_label_norm_p95 = mean([split_label_metric(metrics, split, label, "normalized_p95_abs_error", target_scales) for label in hard_labels])
    hard_target_norm_mae = mean([split_target_metric(metrics, split, target, "normalized_mae", target_scales) for target in hard_targets])
    hard_target_norm_p95 = mean([split_target_metric(metrics, split, target, "normalized_p95_abs_error", target_scales) for target in hard_targets])
    overall_norm_mae = split_overall_metric(metrics, split, "overall_normalized_mae")
    overall_norm_p95 = split_overall_metric(metrics, split, "overall_normalized_p95_abs_error")
    if overall_norm_mae == 0.0:
        overall_norm_mae = mean(
            [split_target_metric(metrics, split, target, "normalized_mae", target_scales) for target in target_scales]
        )
    if overall_norm_p95 == 0.0:
        overall_norm_p95 = mean(
            [split_target_metric(metrics, split, target, "normalized_p95_abs_error", target_scales) for target in target_scales]
        )
    easy_target_norm_mae = mean([split_target_metric(metrics, split, target, "normalized_mae", target_scales) for target in easy_targets])

    weighted_score = (
        weights["hard_label_normalized_mae"] * hard_label_norm_mae
        + weights["hard_label_normalized_p95"] * hard_label_norm_p95
        + weights["hard_target_normalized_mae"] * hard_target_norm_mae
        + weights["hard_target_normalized_p95"] * hard_target_norm_p95
        + weights["overall_normalized_mae"] * overall_norm_mae
        + weights["overall_normalized_p95"] * overall_norm_p95
    )

    easy_regression = 0.0
    rejected = False
    rejection_reasons: list[str] = []
    if baseline is not None:
        baseline_easy_mae = mean(
            [split_target_metric(baseline, split, target, "normalized_mae", target_scales) for target in easy_targets]
        )
        easy_regression = relative_change(easy_target_norm_mae, baseline_easy_mae)
        if easy_regression > easy_regression_limit:
            rejected = True
            rejection_reasons.append(
                "easy_target_normalized_mae_regression_{:.1f}_percent".format(easy_regression * 100)
            )

    return {
        "metrics_path": str(metrics_path),
        "run_name": run_name_for(metrics_path, metrics),
        "weighted_score": weighted_score,
        "selection_split": split,
        "rejected": rejected,
        "rejection_reasons": rejection_reasons,
        "components": {
            "hard_label_normalized_mae": hard_label_norm_mae,
            "hard_label_normalized_p95": hard_label_norm_p95,
            "hard_target_normalized_mae": hard_target_norm_mae,
            "hard_target_normalized_p95": hard_target_norm_p95,
            "overall_normalized_mae": overall_norm_mae,
            "overall_normalized_p95": overall_norm_p95,
            "easy_target_normalized_mae": easy_target_norm_mae,
            "easy_target_normalized_mae_relative_change": easy_regression,
        },
        "selection_config": selection_config,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="+", type=Path, help="metrics*.json files to rank.")
    parser.add_argument("--baseline", type=Path, help="Baseline metrics*.json for easy-target guard.")
    parser.add_argument("--selection-config", type=Path, help="Selection weights and target physical scales JSON.")
    parser.add_argument(
        "--easy-regression-limit",
        type=float,
        default=0.10,
        help="Reject runs whose easy-target MAE is worse than baseline by this fraction.",
    )
    parser.add_argument("--output", type=Path, help="Optional output JSON path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline = load_json(args.baseline) if args.baseline else None
    selection_config = load_json(args.selection_config) if args.selection_config else None
    ranked = [
        score_run(
            path,
            load_json(path),
            baseline=baseline,
            easy_regression_limit=args.easy_regression_limit,
            selection_config=selection_config,
        )
        for path in args.metrics
    ]
    ranked.sort(key=lambda row: (row["rejected"], row["weighted_score"]))
    payload = {
        "selection_policy": {
            "hard_labels": (selection_config or {}).get("hard_labels", HARD_LABELS),
            "hard_targets": (selection_config or {}).get("hard_targets", HARD_TARGETS),
            "easy_targets": (selection_config or {}).get("easy_targets", EASY_TARGETS),
            "easy_regression_limit": args.easy_regression_limit,
            "lower_weighted_score_is_better": True,
            "selection_config_path": None if args.selection_config is None else str(args.selection_config),
        },
        "ranked_runs": ranked,
        "selected_run": next((row for row in ranked if not row["rejected"]), None),
    }
    text = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
