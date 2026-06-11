"""Audit train/validation loss gaps from compact model-selection artifacts.

This is intentionally diagnostic-only. It does not train a model and does not
require raw predictions. When per-row prediction artifacts are absent, the
report marks source-level loss decomposition as unavailable instead of
pretending it can be reconstructed from aggregate metrics.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any


DEFAULT_RUNS = [
    "training_results/model_selection_loop/D66_width128_lr4e-4_dropout0.05_20260606_102435_utc",
    "training_results/model_selection_loop/D66_grouped_width192_lr6e-4_dropout0.075_targeted25k_20260607_050604_utc",
    "training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_bin_diag_20260609_073514_utc",
    "training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_plateau_clip_smoothl1_20260610_071108_utc",
    "training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_plateau_clip_smoothl1_seed11_20260611_072635_utc",
    "training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_plateau_clip_smoothl1_seed23_20260611_073754_utc",
    "training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_plateau_clip_smoothl1_seed37_20260611_074903_utc",
    "training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_s3tail60k_plateau_clip_smoothl1_seed7_20260611_084343_utc",
    "training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_s3tail60k_plateau_clip_smoothl1_seed23_20260611_090007_utc",
]

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

HARMONIC_TARGETS = [
    ("A1_amp", "A1_phase", 2),
    ("B2_amp", "B2_phase", 1),
    ("A2_amp", "A2_phase", 3),
    ("S3_amp", "S3_phase", 2),
    ("A3_amp", "A3_phase", 4),
]

DATASET_SPLIT_HINT_FIELD = "dataset_split_hint"
TRAINING_ONLY_HINT = "training_only"

DATASET_CSV_HINTS = {
    "enhanced_v3_targeted25k": "Downloads from Colab/feature_regression_enhanced/enhanced_v3_targeted25k_20260611/training_features_enhanced.csv",
    "enhanced_v5_s3_tail60k": "Downloads from Colab/feature_regression_enhanced/enhanced_v5_s3_tail60k_20260611/training_features_enhanced.csv",
}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def row_float(row: dict[str, str], name: str, default: float = 0.0) -> float:
    value = row.get(name, "")
    return float(default) if value in ("", None) else float(value)


def target_from_row(row: dict[str, str]) -> list[float]:
    target = [row_float(row, "C1"), row_float(row, "C3")]
    for amp_field, phase_field, order in HARMONIC_TARGETS:
        amp = row_float(row, amp_field)
        theta = math.radians(order * row_float(row, phase_field))
        target.extend([amp * math.cos(theta), amp * math.sin(theta)])
    return target


def stable_row_key(row: dict[str, str], index: int) -> str:
    fields = ["sweep_label", *TARGET_COLUMNS]
    payload: dict[str, Any] = {"row_index_fallback": index}
    for field in fields:
        payload[field] = row.get(field, "")
    if any(row.get(field, "") for field in fields):
        payload.pop("row_index_fallback")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def stable_unit_interval(text: str, seed: int, salt: str) -> float:
    digest = hashlib.sha256(f"{seed}:{salt}:{text}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def four_way_split(
    rows: list[dict[str, str]],
    labels: list[str],
    y: list[list[float]],
    *,
    validation_fraction: float,
    blind_fraction: float,
    stress_fraction: float,
    seed: int,
) -> dict[str, list[int]]:
    stress_labels = {
        "coupled_full_random",
        "coupled_sparse_random",
        "coupled_C1_C3_random",
        "coupled_A1_B2_random",
        "coupled_A1_B2_S3_random",
        "coupled_A2_B2_random",
        "coupled_C3_B2_random",
        "coupled_A1_S3_random",
        "coupled_C3_A3_S3_random",
        "S3",
        "A3",
    }
    scales = [DEFAULT_TARGET_PHYSICAL_SCALES[name] for name in TARGET_COLUMNS]
    max_normalized_abs = [max(abs(value) / scale for value, scale in zip(row_y, scales)) for row_y in y]
    stress_threshold = quantile(max_normalized_abs, 0.90)

    splits: dict[str, list[int]] = {"train": [], "validation": [], "blind": [], "stress": []}
    for index, row in enumerate(rows):
        if str(row.get(DATASET_SPLIT_HINT_FIELD, "")).strip() == TRAINING_ONLY_HINT:
            splits["train"].append(index)
            continue

        key = stable_row_key(row, index)
        label = str(labels[index])
        stress_candidate = label in stress_labels or max_normalized_abs[index] >= stress_threshold
        if stress_candidate and stable_unit_interval(key, seed, "stress") < stress_fraction:
            splits["stress"].append(index)
            continue

        value = stable_unit_interval(key, seed, "selection")
        if value < blind_fraction:
            splits["blind"].append(index)
        elif value < blind_fraction + validation_fraction:
            splits["validation"].append(index)
        else:
            splits["train"].append(index)
    return splits


def read_history(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    history: list[dict[str, float]] = []
    for row in rows:
        validation_key = "validation_loss" if row.get("validation_loss", "") != "" else "test_loss"
        item = {
            "epoch": float(row["epoch"]),
            "train_loss": float(row["train_loss"]),
            "validation_loss": float(row[validation_key]),
        }
        for optional in ["train_s3_magnitude_loss", "train_total_objective_loss", "learning_rate"]:
            if row.get(optional, "") != "":
                item[optional] = float(row[optional])
        history.append(item)
    return history


def summarize_history(history: list[dict[str, float]]) -> dict[str, Any]:
    if not history:
        return {}
    best = min(history, key=lambda row: row["validation_loss"])
    final = history[-1]
    min_train = min(row["train_loss"] for row in history)
    min_validation = min(row["validation_loss"] for row in history)

    def ratio(numerator: float, denominator: float) -> float | None:
        if abs(denominator) < 1e-12:
            return None
        return float(numerator / denominator)

    return {
        "first_epoch": int(history[0]["epoch"]),
        "last_epoch": int(final["epoch"]),
        "best_epoch": int(best["epoch"]),
        "best_train_loss": best["train_loss"],
        "best_validation_loss": best["validation_loss"],
        "final_train_loss": final["train_loss"],
        "final_validation_loss": final["validation_loss"],
        "min_train_loss": min_train,
        "min_validation_loss": min_validation,
        "train_over_validation_at_best": ratio(best["train_loss"], best["validation_loss"]),
        "validation_over_train_at_best": ratio(best["validation_loss"], best["train_loss"]),
        "train_over_validation_at_final": ratio(final["train_loss"], final["validation_loss"]),
        "validation_over_train_at_final": ratio(final["validation_loss"], final["train_loss"]),
        "final_train_total_objective_loss": final.get("train_total_objective_loss"),
        "final_train_s3_magnitude_loss": final.get("train_s3_magnitude_loss"),
    }


def run_summary(run_dir: Path) -> dict[str, Any]:
    manifest = load_json(run_dir / "run_manifest_model_loop.json")
    metrics = load_json(run_dir / "metrics_model_loop.json")
    selection = load_json(run_dir / "selection_score.json")
    history = read_history(run_dir / "training_history_summary.csv")
    dataset = manifest.get("dataset", metrics.get("dataset", {}))
    model = manifest.get("model", {})
    training_config = metrics.get("training_config", {})
    candidate = manifest.get("candidate", {})
    history_summary = summarize_history(history)
    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "dataset_version": dataset.get("dataset_version") or training_config.get("dataset_version"),
        "total_rows": dataset.get("n_rows") or metrics.get("n_samples"),
        "train_rows": dataset.get("n_train") or metrics.get("n_train"),
        "validation_rows": dataset.get("n_validation") or metrics.get("n_validation"),
        "blind_rows": dataset.get("n_blind") or metrics.get("n_blind"),
        "stress_rows": dataset.get("n_stress") or metrics.get("n_stress"),
        "training_only_rows": dataset.get("training_only_new_rows") or training_config.get("training_only_new_rows"),
        "split_seed": training_config.get("split_seed") or candidate.get("split_seed"),
        "torch_seed": model.get("torch_seed") or training_config.get("torch_seed"),
        "component_loss_kind": model.get("component_loss_kind")
        or training_config.get("component_loss_kind")
        or candidate.get("component_loss_kind")
        or "mse",
        "residual_penalty": training_config.get("residual_penalty") or candidate.get("residual_penalty"),
        "s3_magnitude_loss": training_config.get("s3_magnitude_loss", {}),
        "weighted_score": selection.get("weighted_score"),
        "overall_normalized_mae": metrics.get("overall_normalized_mae"),
        "overall_normalized_p95_abs_error": metrics.get("overall_normalized_p95_abs_error"),
        "blind_normalized_mae": metrics.get("splits", {}).get("blind", {}).get("overall_normalized_mae"),
        "stress_normalized_mae": metrics.get("splits", {}).get("stress", {}).get("overall_normalized_mae"),
        "history": history_summary,
        "history_points": history,
        "metrics": metrics,
    }


def dataset_csv_for_version(dataset_version: str | None) -> Path | None:
    if not dataset_version:
        return None
    hinted = DATASET_CSV_HINTS.get(dataset_version)
    if hinted and Path(hinted).exists():
        return Path(hinted)
    training_results_candidates = sorted(
        Path("training_results/feature_regression_enhanced").glob(
            f"*{dataset_version}*/training_features_enhanced.csv"
        )
    )
    if training_results_candidates:
        return training_results_candidates[-1]
    candidates = sorted(Path("Downloads from Colab").glob(f"**/*{dataset_version}*/training_features_enhanced.csv"))
    return candidates[-1] if candidates else None


def summarize_dataset_csv(csv_path: Path, split_seed: int) -> dict[str, Any]:
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    y = [target_from_row(row) for row in rows]
    labels = [row.get("sweep_label", "") for row in rows]
    splits = four_way_split(
        rows,
        labels,
        y,
        validation_fraction=0.2,
        blind_fraction=0.1,
        stress_fraction=0.2,
        seed=split_seed,
    )
    s3_x = TARGET_COLUMNS.index("S3_x")
    s3_y = TARGET_COLUMNS.index("S3_y")
    train_s3_mag = [math.hypot(y[index][s3_x], y[index][s3_y]) for index in splits["train"]]
    vector_scale = quantile(train_s3_mag, 0.95) if train_s3_mag else 1.0

    def source_for(row: dict[str, str]) -> str:
        source = str(row.get("dataset_source", "")).strip()
        version = str(row.get("dataset_version", "")).strip()
        if source:
            return source
        if version:
            return version
        return "unknown"

    def s3_bin(value: float) -> str:
        if value <= 0.1 * vector_scale:
            return "near_zero"
        if value <= 0.3 * vector_scale:
            return "low"
        if value <= 0.7 * vector_scale:
            return "medium"
        return "high"

    split_summaries: dict[str, Any] = {}
    for split_name, indices in splits.items():
        source_counts: dict[str, int] = {}
        label_counts: dict[str, int] = {}
        hint_counts: dict[str, int] = {}
        bin_counts: dict[str, int] = {"near_zero": 0, "low": 0, "medium": 0, "high": 0}
        for index in indices:
            row = rows[index]
            source_counts[source_for(row)] = source_counts.get(source_for(row), 0) + 1
            label = str(labels[index])
            label_counts[label] = label_counts.get(label, 0) + 1
            hint = str(row.get(DATASET_SPLIT_HINT_FIELD, "")).strip() or "unhinted_parent"
            hint_counts[hint] = hint_counts.get(hint, 0) + 1
            bin_counts[s3_bin(math.hypot(y[index][s3_x], y[index][s3_y]))] += 1
        split_summaries[split_name] = {
            "n": int(len(indices)),
            "dataset_source_counts": dict(sorted(source_counts.items())),
            "dataset_split_hint_counts": dict(sorted(hint_counts.items())),
            "sweep_label_counts": dict(sorted(label_counts.items())),
            "s3_magnitude_bin_counts": bin_counts,
        }

    return {
        "csv_path": str(csv_path),
        "n_rows": int(len(rows)),
        "split_seed": split_seed,
        "s3_vector_scale_training_p95": vector_scale,
        "splits": split_summaries,
    }


def target_table(metrics: dict[str, Any], split_a: str = "train", split_b: str = "validation") -> list[dict[str, Any]]:
    splits = metrics.get("splits", {})
    a_targets = splits.get(split_a, {}).get("targets", {})
    b_targets = splits.get(split_b, {}).get("targets", {})
    rows = []
    for target in TARGET_COLUMNS:
        a = a_targets.get(target, {})
        b = b_targets.get(target, {})
        rows.append(
            {
                "target": target,
                f"{split_a}_normalized_mae": a.get("normalized_mae"),
                f"{split_b}_normalized_mae": b.get("normalized_mae"),
                f"{split_a}_rmse": a.get("rmse"),
                f"{split_b}_rmse": b.get("rmse"),
            }
        )
    return rows


def label_table(metrics: dict[str, Any], split_name: str) -> list[dict[str, Any]]:
    labels = metrics.get("splits", {}).get(split_name, {}).get("labels", {})
    return [
        {
            "split": split_name,
            "label": label,
            "n": values.get("n"),
            "normalized_mae": values.get("normalized_mae"),
            "rmse": values.get("rmse"),
            "normalized_p95_abs_error": values.get("normalized_p95_abs_error"),
        }
        for label, values in sorted(labels.items())
    ]


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows available._\n"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column)) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def make_plots(output_dir: Path, summaries: list[dict[str, Any]], dataset_summaries: dict[str, Any]) -> list[str]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        for summary in summaries:
            source = Path(summary["run_dir"]) / "training_history_model_loop.png"
            if source.exists():
                target = plot_dir / f"{summary['run_name']}_training_history_model_loop.png"
                shutil.copy2(source, target)
                outputs.append(str(target))
        return outputs

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for summary in summaries:
        history = summary["history_points"]
        if not history:
            continue
        label = summary["run_name"].replace("D66_grouped_width320_lr6e-4_dropout0.075_", "")
        epochs = [row["epoch"] for row in history]
        ax.plot(epochs, [row["train_loss"] for row in history], linewidth=1.1, label=f"{label} train")
        ax.plot(epochs, [row["validation_loss"] for row in history], linewidth=1.1, linestyle="--", label=f"{label} val")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("weighted scaled MSE")
    ax.set_title("History curves use weighted scaled MSE")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    path = plot_dir / "audited_train_validation_history.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    outputs.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 4))
    for summary in summaries:
        history = summary["history_points"]
        if not history:
            continue
        label = summary["run_name"].replace("D66_grouped_width320_lr6e-4_dropout0.075_", "")
        epochs = [row["epoch"] for row in history]
        ratios = [row["validation_loss"] / max(row["train_loss"], 1e-12) for row in history]
        ax.plot(epochs, ratios, linewidth=1.2, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation_loss / train_loss")
    ax.set_title("History loss ratio")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6)
    fig.tight_layout()
    path = plot_dir / "validation_over_train_ratio.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    outputs.append(str(path))

    rows = []
    for summary in summaries:
        metrics = summary["metrics"]
        if not metrics.get("splits"):
            continue
        for split in ["train", "validation", "blind", "stress"]:
            split_metrics = metrics["splits"].get(split, {})
            rows.append((summary["run_name"][:28], split, split_metrics.get("overall_normalized_mae")))
    if rows:
        labels = [f"{run}\n{split}" for run, split, _ in rows]
        values = [value or 0.0 for _, _, value in rows]
        fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.35), 4))
        ax.bar(range(len(rows)), values)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=6)
        ax.set_ylabel("overall normalized MAE")
        ax.set_title("Split metrics from metrics_model_loop.json")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = plot_dir / "split_normalized_mae.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        outputs.append(str(path))

    for dataset_version, dataset_summary in dataset_summaries.items():
        splits = dataset_summary.get("splits", {})
        if not splits:
            continue
        split_names = ["train", "validation", "blind", "stress"]
        bin_names = ["near_zero", "low", "medium", "high"]
        values = [
            [splits.get(split, {}).get("s3_magnitude_bin_counts", {}).get(bin_name, 0) for bin_name in bin_names]
            for split in split_names
        ]
        fig, ax = plt.subplots(figsize=(6.5, 4))
        bottom = [0.0 for _ in split_names]
        for i, bin_name in enumerate(bin_names):
            bar_values = [row[i] for row in values]
            ax.bar(split_names, bar_values, bottom=bottom, label=bin_name)
            bottom = [old + new for old, new in zip(bottom, bar_values)]
        ax.set_ylabel("rows")
        ax.set_title(f"S3 magnitude bins: {dataset_version}")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = plot_dir / f"{dataset_version}_s3_bin_counts.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        outputs.append(str(path))

    return outputs


def generate_report(output_root: Path, run_dirs: list[Path]) -> tuple[Path, Path]:
    stamp = utc_stamp()
    report_dir = output_root / f"training_validation_loss_gap_audit_{stamp}"
    report_dir.mkdir(parents=True, exist_ok=True)
    summaries = [run_summary(run_dir) for run_dir in run_dirs if run_dir.exists()]

    dataset_summaries: dict[str, Any] = {}
    for summary in summaries:
        dataset_version = summary.get("dataset_version")
        split_seed = int(summary.get("split_seed") or 7)
        if not dataset_version or dataset_version in dataset_summaries:
            continue
        csv_path = dataset_csv_for_version(dataset_version)
        if csv_path:
            dataset_summaries[dataset_version] = summarize_dataset_csv(csv_path, split_seed)

    plots = make_plots(report_dir, summaries, dataset_summaries)

    history_rows = []
    for summary in summaries:
        history = summary.get("history", {})
        history_rows.append(
            {
                "run": summary["run_name"],
                "dataset": summary.get("dataset_version"),
                "train_rows": summary.get("train_rows"),
                "validation_rows": summary.get("validation_rows"),
                "training_only": summary.get("training_only_rows"),
                "loss_kind": summary.get("component_loss_kind"),
                "best_epoch": history.get("best_epoch"),
                "best_val_mse": history.get("best_validation_loss"),
                "final_train_mse": history.get("final_train_loss"),
                "final_val_mse": history.get("final_validation_loss"),
                "val/train final": history.get("validation_over_train_at_final"),
                "weighted_score": summary.get("weighted_score"),
                "overall_norm_mae": summary.get("overall_normalized_mae"),
            }
        )

    rows_for_gap = [row for row in history_rows if row.get("val/train final") is not None]
    first_large = next((row for row in rows_for_gap if float(row["val/train final"]) >= 8.0), None)
    v3 = next((s for s in summaries if "targeted25k_plateau_clip_smoothl1_20260610_071108" in s["run_name"]), None)
    v5 = next((s for s in summaries if "s3tail60k_plateau_clip_smoothl1_seed23" in s["run_name"]), None)

    per_target_rows = []
    for summary in [s for s in [v3, v5] if s]:
        for row in target_table(summary["metrics"]):
            row = {"run": summary["run_name"], **row}
            per_target_rows.append(row)

    label_rows = []
    for summary in [s for s in [v3, v5] if s]:
        for split in ["train", "validation", "blind", "stress"]:
            for row in label_table(summary["metrics"], split):
                if row["label"] in {
                    "coupled_full_random",
                    "coupled_sparse_random",
                    "S3_high_random",
                    "coupled_A1_S3_random",
                    "coupled_B2_S3_random",
                    "coupled_A3_S3_random",
                    "coupled_A1_B2_S3_random",
                }:
                    label_rows.append({"run": summary["run_name"], **row})

    dataset_rows = []
    for dataset_version, dataset_summary in dataset_summaries.items():
        for split, split_summary in dataset_summary["splits"].items():
            dataset_rows.append(
                {
                    "dataset": dataset_version,
                    "split": split,
                    "n": split_summary["n"],
                    "training_only": split_summary["dataset_split_hint_counts"].get(TRAINING_ONLY_HINT, 0),
                    "unhinted_parent": split_summary["dataset_split_hint_counts"].get("unhinted_parent", 0),
                    "s3_near_zero": split_summary["s3_magnitude_bin_counts"].get("near_zero", 0),
                    "s3_low": split_summary["s3_magnitude_bin_counts"].get("low", 0),
                    "s3_medium": split_summary["s3_magnitude_bin_counts"].get("medium", 0),
                    "s3_high": split_summary["s3_magnitude_bin_counts"].get("high", 0),
                    "sources": json.dumps(split_summary["dataset_source_counts"], sort_keys=True),
                }
            )

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "classification": {
            "primary": "C_loss_logging_mismatch_plus_D_split_benchmark_distribution_difference",
            "secondary": "F_real_generalization_gap_on_parent_validation_benchmark",
            "not_supported_by_current_artifacts": "B_MSE_tail_effect and top-row outlier dominance cannot be tested without raw train predictions.",
        },
        "source_code_findings": {
            "plot_history": "training_history_model_loop.png plots history train_loss and validation_loss on log y-axis with label weighted scaled MSE.",
            "train_loss": "Eval-mode weighted_mse(model(x_train), y_train, target_weights) on standardized targets.",
            "validation_loss": "Eval-mode weighted_mse(model(x_validation), y_validation, target_weights) on standardized targets.",
            "optimized_objective": "For SmoothL1 runs, training optimizes SmoothL1 component loss plus optional S3 magnitude loss plus residual penalty, not the plotted weighted MSE.",
            "scalers": "x_scaler and y_scaler are fit on train_index only; all splits use the same frozen scaler and target weights.",
        },
        "first_large_gap_run": first_large,
        "history_rows": history_rows,
        "dataset_summaries": dataset_summaries,
        "per_target_rows": per_target_rows,
        "label_rows": label_rows,
        "plots": plots,
        "limitations": [
            "Raw per-row train predictions are not present in audited historical run folders.",
            "Model checkpoints were not pushed, so top training outliers and exact source-level MSE cannot be reconstructed without rerunning.",
            "Existing aggregate metrics provide split/label/target MAE/RMSE, not source-separated parent-vs-training-only losses.",
        ],
    }
    json_path = report_dir / "training_validation_loss_gap_audit.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    md = [
        "# Training-Validation Loss Gap Audit",
        "",
        f"Created UTC: {payload['created_utc']}",
        "",
        "## Summary conclusion",
        "",
        "The large visual gap is real in the stored history curves, but the plotted curves are not the same objective optimized by SmoothL1 runs. The PNG plots eval-mode weighted scaled MSE for train and validation on a log y-axis. SmoothL1 runs optimize SmoothL1 component loss plus residual penalty, with optional S3 magnitude loss. That makes the plot label technically correct but incomplete for SmoothL1 interpretation.",
        "",
        "The gap first becomes large after the targeted25k training-only rows enter the workflow. Later v4b and SmoothL1 variants increase the validation/train MSE ratio further, but the step change is already visible before SmoothL1. The v5 S3-tail run improves validation/blind/stress normalized MAE and weighted score while keeping a large MSE-history gap, so the current model comparison remains valid under the stored selection metrics.",
        "",
        "Classification: `C_loss_logging_mismatch_plus_D_split_benchmark_distribution_difference`, with a real validation/generalization gap on parent benchmark rows. The audit cannot prove `B_MSE_tail_effect` or identify top training outlier rows from existing compact artifacts because raw train predictions/checkpoints were intentionally not pushed.",
        "",
        "## Source-code verification",
        "",
        "- `plot_history()` plots `train_loss` and `validation_loss` with y-axis label `weighted scaled MSE` and `log` scaling.",
        "- `train_loss` is eval-mode `weighted_mse(train_pred_eval, y_train, target_weights)` on standardized targets.",
        "- `validation_loss` is eval-mode `weighted_mse(model(x_validation), y_validation, target_weights)` on standardized targets.",
        "- SmoothL1 runs optimize `weighted_component_loss(..., loss_kind=smooth_l1) + S3 magnitude loss + residual penalty`, not the plotted MSE.",
        "- `x_scaler` and `y_scaler` are fit on `train_index` only; the same transformed target scale and target weights are used for train and validation loss.",
        "",
        "## History gap table",
        "",
        markdown_table(
            history_rows,
            [
                "run",
                "dataset",
                "train_rows",
                "validation_rows",
                "training_only",
                "loss_kind",
                "best_epoch",
                "best_val_mse",
                "final_train_mse",
                "final_val_mse",
                "val/train final",
                "weighted_score",
                "overall_norm_mae",
            ],
        ),
        "",
        "## Dataset split/source composition",
        "",
        markdown_table(
            dataset_rows,
            [
                "dataset",
                "split",
                "n",
                "training_only",
                "unhinted_parent",
                "s3_near_zero",
                "s3_low",
                "s3_medium",
                "s3_high",
                "sources",
            ],
        ),
        "",
        "## Per-target train vs validation metrics",
        "",
        "These are physical-unit RMSE and physical-scale normalized MAE from `metrics_model_loop.json`, not the standardized weighted MSE used by the history PNG.",
        "",
        markdown_table(
            per_target_rows,
            [
                "run",
                "target",
                "train_normalized_mae",
                "validation_normalized_mae",
                "train_rmse",
                "validation_rmse",
            ],
        ),
        "",
        "## Hard-regime label metrics",
        "",
        markdown_table(
            label_rows,
            ["run", "split", "label", "n", "normalized_mae", "rmse", "normalized_p95_abs_error"],
        ),
        "",
        "## Requested decomposition status",
        "",
        "- `train parent only` vs `validation parent`: unavailable for historical runs without per-row predictions or checkpoints.",
        "- `training_only`, `coupled`, and `high-S3` fraction of training MSE: unavailable for historical runs without per-row train predictions.",
        "- top 20 training rows by squared error: unavailable for historical runs without per-row train predictions.",
        "- benchmark row drift: split counts are stable in manifests, and training-only rows are explicitly excluded from validation/blind/stress by policy.",
        "",
        "## Interpretation",
        "",
        "The gap first appears when training-only targeted/coupled rows are appended and the split policy changes to parent benchmark rows plus training-only append rows. Because the curve is train MSE much lower than validation MSE, the stored history does not support the explanation that hard training-only rows are raising train MSE. Instead, the model fits the enlarged training set very closely while validation remains a harder parent benchmark distribution. Coupled-full and coupled-sparse validation/stress labels remain the dominant difficult regimes in aggregate metrics.",
        "",
        "The log y-axis visually amplifies the separation after train MSE becomes very small. For SmoothL1 runs, the plot also hides the objective actually optimized during training, so users can easily compare a plotted MSE curve against optimizer behavior incorrectly.",
        "",
        "## Recommendation",
        "",
        "Do not use this audit alone to justify feature engineering or a jump to 100k rows. The immediate fix should be logging: future training-history output should show `train_all_weighted_mse`, `validation_weighted_mse`, `train_total_objective`, and, when per-row predictions are available, `train_parent`, `train_training_only`, `blind`, and `stress` curves. For source-level loss decomposition, either save compact split/source loss summaries during each run or rerun the audited candidates with the diagnostic CSV enabled.",
        "",
        "## Plots",
        "",
    ]
    md.extend([f"- `{plot}`" for plot in plots])
    report_path = output_root / f"training_validation_loss_gap_audit_{stamp}.md"
    report_path.write_text("\n".join(md) + "\n")
    return report_path, json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=Path, default=None, help="Run folder to audit; may be repeated.")
    parser.add_argument(
        "--include-default-runs",
        action="store_true",
        help="When --run is supplied, also include the standard historical comparison runs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("training_results/model_selection_reports"),
        help="Directory for the markdown report and JSON payload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = []
    if args.include_default_runs or not args.run:
        run_dirs.extend(Path(path) for path in DEFAULT_RUNS)
    if args.run:
        run_dirs.extend(args.run)
    report_path, json_path = generate_report(args.output_root, run_dirs)
    print("report:", report_path)
    print("json:", json_path)


if __name__ == "__main__":
    main()
