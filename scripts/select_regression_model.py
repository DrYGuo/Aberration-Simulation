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


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


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
) -> dict[str, Any]:
    hard_label_mae = mean([label_metric(metrics, label, "mae") for label in HARD_LABELS])
    hard_label_rmse = mean([label_metric(metrics, label, "rmse") for label in HARD_LABELS])
    hard_target_mae = mean([target_metric(metrics, target, "mae") for target in HARD_TARGETS])
    hard_target_rmse = mean([target_metric(metrics, target, "rmse") for target in HARD_TARGETS])
    overall_mae = float(metrics.get("overall_mae", 0.0))
    overall_rmse = float(metrics.get("overall_rmse", 0.0))
    easy_target_mae = mean([target_metric(metrics, target, "mae") for target in EASY_TARGETS])

    weighted_score = (
        0.30 * hard_label_mae
        + 0.25 * hard_label_rmse
        + 0.20 * hard_target_mae
        + 0.15 * hard_target_rmse
        + 0.05 * overall_mae
        + 0.05 * overall_rmse
    )

    easy_regression = 0.0
    rejected = False
    rejection_reasons: list[str] = []
    if baseline is not None:
        baseline_easy_mae = mean(
            [target_metric(baseline, target, "mae") for target in EASY_TARGETS]
        )
        easy_regression = relative_change(easy_target_mae, baseline_easy_mae)
        if easy_regression > easy_regression_limit:
            rejected = True
            rejection_reasons.append(
                "easy_target_mae_regression_{:.1f}_percent".format(easy_regression * 100)
            )

    return {
        "metrics_path": str(metrics_path),
        "run_name": run_name_for(metrics_path, metrics),
        "weighted_score": weighted_score,
        "rejected": rejected,
        "rejection_reasons": rejection_reasons,
        "components": {
            "hard_label_mae": hard_label_mae,
            "hard_label_rmse": hard_label_rmse,
            "hard_target_mae": hard_target_mae,
            "hard_target_rmse": hard_target_rmse,
            "overall_mae": overall_mae,
            "overall_rmse": overall_rmse,
            "easy_target_mae": easy_target_mae,
            "easy_target_mae_relative_change": easy_regression,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="+", type=Path, help="metrics*.json files to rank.")
    parser.add_argument("--baseline", type=Path, help="Baseline metrics*.json for easy-target guard.")
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
    ranked = [
        score_run(
            path,
            load_json(path),
            baseline=baseline,
            easy_regression_limit=args.easy_regression_limit,
        )
        for path in args.metrics
    ]
    ranked.sort(key=lambda row: (row["rejected"], row["weighted_score"]))
    payload = {
        "selection_policy": {
            "hard_labels": HARD_LABELS,
            "hard_targets": HARD_TARGETS,
            "easy_targets": EASY_TARGETS,
            "easy_regression_limit": args.easy_regression_limit,
            "lower_weighted_score_is_better": True,
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
