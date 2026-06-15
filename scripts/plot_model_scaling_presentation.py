"""Create presentation plots for model performance versus training data size."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "docs"

TARGET_COLUMNS = [
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
]

MILESTONES = [
    {
        "label": "25K",
        "title": "v3 25K",
        "batch_summary": "training_results/model_selection_batches/v3_smoothl1_seed_repeat_batch_20260611_072634_utc/batch_summary.csv",
        "training_rows_k": 25,
    },
    {
        "label": "60K",
        "title": "v5 60K",
        "batch_summary": "training_results/model_selection_batches/v5_s3_tail60k_smoothl1_retrain_batch_20260611_084342_utc/batch_summary.csv",
        "training_rows_k": 60,
    },
    {
        "label": "100K",
        "title": "v6 100K",
        "batch_summary": "training_results/model_selection_batches/v6_benchmark_gap100k_smoothl1_20260614_202505_utc/batch_summary.csv",
        "training_rows_k": 100,
    },
    {
        "label": "250K",
        "title": "v9 250K",
        "batch_summary": "training_results/model_selection_batches/v9_gap250k_d66_20260614_062446_utc/batch_summary.csv",
        "training_rows_k": 250,
    },
    {
        "label": "500K",
        "title": "v11 500K",
        "batch_summary": "training_results/model_selection_batches/v11_gap500k_d66_20260614_214817_utc/batch_summary.csv",
        "training_rows_k": 500,
    },
    {
        "label": "500K + benchmark-v2",
        "title": "v12 500K + benchmark-v2",
        "batch_summary": "training_results/model_selection_batches/v12_benchmark_v2_500k_20260615_005333_utc/batch_summary.csv",
        "training_rows_k": 575,
    },
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def best_batch_row(path: Path) -> dict[str, str]:
    rows = read_csv_rows(path)
    if not rows:
        raise RuntimeError(f"empty batch summary: {path}")

    def key(row: dict[str, str]) -> tuple[int, float]:
        rejected = str(row.get("rejected", "")).lower() == "true"
        score = as_float(row.get("weighted_score"))
        return (1 if rejected else 0, score if score is not None else float("inf"))

    return min(rows, key=key)


def load_metrics_from_selection(selection_path: Path) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text())
    metrics_path = REPO_ROOT / selection["metrics_path"]
    return json.loads(metrics_path.read_text())


def collect_data() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in MILESTONES:
        batch_path = REPO_ROOT / item["batch_summary"]
        batch_row = best_batch_row(batch_path)
        selection_path = REPO_ROOT / str(batch_row["selection_score_path"])
        metrics = load_metrics_from_selection(selection_path)
        record: dict[str, Any] = {
            "label": item["label"],
            "title": item["title"],
            "training_rows_k": item["training_rows_k"],
            "candidate_id": batch_row.get("candidate_id", ""),
            "weighted_score": as_float(batch_row.get("weighted_score")),
            "true_hard_target_normalized_mae": as_float(batch_row.get("true_hard_target_normalized_mae")),
            "overall_normalized_mae": as_float(batch_row.get("overall_normalized_mae")),
            "blind_normalized_mae": as_float(batch_row.get("blind_normalized_mae")),
            "stress_normalized_mae": as_float(batch_row.get("stress_normalized_mae")),
            "S3_high_magnitude_mae_normalized": (
                as_float(batch_row.get("S3_high_magnitude_mae")) / 100.0
                if as_float(batch_row.get("S3_high_magnitude_mae")) is not None
                else None
            ),
        }
        targets = metrics.get("targets", {})
        for target in TARGET_COLUMNS:
            target_metrics = targets.get(target, {})
            record[f"{target}_normalized_mae"] = as_float(target_metrics.get("normalized_mae"))
        records.append(record)
    return records


def write_plot_data(records: list[dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "title",
        "training_rows_k",
        "candidate_id",
        "weighted_score",
        "true_hard_target_normalized_mae",
        "overall_normalized_mae",
        "blind_normalized_mae",
        "stress_normalized_mae",
        "S3_high_magnitude_mae_normalized",
        *[f"{target}_normalized_mae" for target in TARGET_COLUMNS],
    ]
    with (OUTPUT_DIR / "model_scaling_plot_data.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({name: record.get(name, "") for name in fieldnames})


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.26,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_key_metrics(records: list[dict[str, Any]]) -> None:
    x = list(range(len(records)))
    labels = [record["label"] for record in records]
    series = [
        ("Weighted score", "weighted_score", "#1f77b4", "o"),
        ("True-hard normalized MAE", "true_hard_target_normalized_mae", "#d62728", "s"),
        ("Overall normalized MAE", "overall_normalized_mae", "#2ca02c", "^"),
        ("Blind normalized MAE", "blind_normalized_mae", "#9467bd", "D"),
        ("Stress normalized MAE", "stress_normalized_mae", "#8c564b", "v"),
        ("High-S3 magnitude MAE / 100", "S3_high_magnitude_mae_normalized", "#ff7f0e", "P"),
    ]

    fig, ax = plt.subplots(figsize=(10.8, 6.2), constrained_layout=True)
    for name, key, color, marker in series:
        y = [record.get(key) for record in records]
        ax.plot(x, y, label=name, color=color, marker=marker, linewidth=2.3, markersize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_xlabel("Training data scale")
    ax.set_ylabel("Error metric (lower is better)")
    ax.set_title("Regression performance improves with training data scale")
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        title="Metric",
        borderaxespad=0.0,
    )
    ax.text(
        0.01,
        0.965,
        "v12: same 500K training scale, larger benchmark-v2 held-out set",
        transform=ax.transAxes,
        fontsize=8.8,
        color="0.28",
        va="top",
    )
    for ext in ("png", "svg"):
        fig.savefig(OUTPUT_DIR / f"model_scaling_key_metrics.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_coefficient_errors(records: list[dict[str, Any]]) -> None:
    x = list(range(len(records)))
    labels = [record["label"] for record in records]
    colors = {
        "C1": "#1f77b4",
        "C3": "#17becf",
        "A1_x": "#2ca02c",
        "A1_y": "#98df8a",
        "B2_x": "#9467bd",
        "B2_y": "#c5b0d5",
        "A2_x": "#8c564b",
        "A2_y": "#c49c94",
        "S3_x": "#d62728",
        "S3_y": "#ff9896",
        "A3_x": "#ff7f0e",
        "A3_y": "#ffbb78",
    }
    markers = {
        "C1": "o",
        "C3": "s",
        "A1_x": "^",
        "A1_y": "v",
        "B2_x": "D",
        "B2_y": "P",
        "A2_x": "X",
        "A2_y": "*",
        "S3_x": "<",
        "S3_y": ">",
        "A3_x": "h",
        "A3_y": "H",
    }

    fig, ax = plt.subplots(figsize=(11.5, 6.7), constrained_layout=True)
    for target in TARGET_COLUMNS:
        y = [record.get(f"{target}_normalized_mae") for record in records]
        ax.plot(
            x,
            y,
            label=target,
            color=colors[target],
            marker=markers[target],
            linewidth=2.0,
            markersize=5.7,
            alpha=0.95,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Training data scale")
    ax.set_ylabel("Validation normalized MAE")
    ax.set_title("Per-coefficient prediction error versus training data scale")
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        frameon=True,
        title="Coefficient",
        borderaxespad=0.0,
    )
    ax.text(
        0.01,
        0.02,
        "Normalized by physical target scale; lower is better.",
        transform=ax.transAxes,
        fontsize=8.8,
        color="0.28",
    )
    for ext in ("png", "svg"):
        fig.savefig(OUTPUT_DIR / f"model_scaling_per_coefficient_errors.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    setup_style()
    records = collect_data()
    write_plot_data(records)
    plot_key_metrics(records)
    plot_coefficient_errors(records)
    print("wrote:", OUTPUT_DIR / "model_scaling_key_metrics.png")
    print("wrote:", OUTPUT_DIR / "model_scaling_key_metrics.svg")
    print("wrote:", OUTPUT_DIR / "model_scaling_per_coefficient_errors.png")
    print("wrote:", OUTPUT_DIR / "model_scaling_per_coefficient_errors.svg")
    print("wrote:", OUTPUT_DIR / "model_scaling_plot_data.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
