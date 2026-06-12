"""Run one model-only regression candidate from an existing feature CSV.

This script intentionally does not generate simulations. It is the Colab worker
entry point for architecture and hyperparameter selection using cached feature
tables.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import platform
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import numpy as np

from feature_regression_model import TARGET_COLUMNS, Standardizer, file_sha256, target_from_row
from regression_diagnostics import discover_regime_column, per_target_diagnostics, regime_breakdown, vector_diagnostics
from select_regression_model import score_run


TARGET_WEIGHTS = {
    "C1": 1.5,
    "C3": 0.8,
    "A1_x": 1.1,
    "A1_y": 1.1,
    "B2_x": 0.8,
    "B2_y": 0.8,
    "A2_x": 0.9,
    "A2_y": 0.9,
    "S3_x": 1.5,
    "S3_y": 1.5,
    "A3_x": 1.7,
    "A3_y": 1.7,
}


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
DATASET_SPLIT_HINT_FIELD = "dataset_split_hint"
TRAINING_ONLY_HINT = "training_only"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


def current_commit(repo_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode:
        return None
    return result.stdout.strip()


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open() as handle:
        return list(csv.DictReader(handle))


def row_float(row: dict[str, str], name: str, default: float = 0.0) -> float:
    value = row.get(name, default)
    if value in (None, ""):
        return float(default)
    return float(value)


def find_latest_csv(search_root: Path, filename: str) -> Path:
    matches = sorted(search_root.glob(f"**/{filename}"), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} under {search_root}. "
            "This model-loop runner uses existing cached feature CSVs only."
        )
    return matches[-1]


def run_dataset_bootstrap(notebook: Path, timeout_seconds: int, output_dir: Path) -> None:
    command = [
        sys.executable,
        "scripts/run_notebook_headless.py",
        str(notebook),
        "--output-dir",
        str(output_dir),
        "--timeout",
        str(timeout_seconds),
    ]
    print("$", " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="" if line.endswith("\n") else "\n", flush=True)
    returncode = int(process.wait())
    if returncode:
        raise RuntimeError(f"dataset bootstrap failed with exit {returncode}")


def ensure_csv_available(args: argparse.Namespace, filename: str) -> Path:
    if args.csv_path:
        return args.csv_path
    try:
        return find_latest_csv(args.search_root, filename)
    except FileNotFoundError:
        if not args.bootstrap_if_missing:
            raise
    run_dataset_bootstrap(
        args.bootstrap_notebook,
        args.bootstrap_timeout,
        args.bootstrap_output_dir,
    )
    return find_latest_csv(args.search_root, filename)


def find_feature_columns(csv_path: Path, family: str) -> list[str]:
    candidates = []
    if family == "enhanced":
        candidates = [
            csv_path.parent / "feature_columns_enhanced.json",
            csv_path.parent / "feature_columns.json",
        ]
    elif family == "raw_angles":
        candidates = [
            csv_path.parent / "feature_columns_raw_angles.json",
            csv_path.parent / "feature_columns_enhanced.json",
            csv_path.parent / "feature_columns.json",
        ]
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text())
            if isinstance(data, dict) and "features" in data:
                return list(data["features"])
            return list(data)
    raise FileNotFoundError(
        f"No feature_columns*.json found beside {csv_path}. "
        "Refusing to infer features from CSV headers because target columns are also present."
    )


def prepare_dataset(csv_path: Path, feature_columns: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, str]]]:
    rows = load_rows(csv_path)
    regime_column = discover_regime_column(rows)
    X = np.asarray(
        [[row_float(row, name) for name in feature_columns] for row in rows],
        dtype=np.float32,
    )
    y = np.asarray([target_from_row(row) for row in rows], dtype=np.float32)
    labels = np.asarray([row.get(regime_column, "") if regime_column else "" for row in rows])
    return X, y, labels, rows


def stratified_train_test_split(labels: np.ndarray, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for label in sorted(set(labels)):
        indices = np.where(labels == label)[0]
        shuffled = rng.permutation(indices)
        n_test = max(1, int(round(test_fraction * len(shuffled)))) if len(shuffled) > 1 else 1
        test_parts.append(shuffled[:n_test])
        train_parts.append(shuffled[n_test:])
    train_index = np.concatenate([part for part in train_parts if len(part)])
    test_index = np.concatenate([part for part in test_parts if len(part)])
    return rng.permutation(train_index), rng.permutation(test_index)


def stable_row_key(row: dict[str, str], index: int) -> str:
    fields = ["sweep_label", *TARGET_COLUMNS]
    payload = {"row_index_fallback": index}
    for field in fields:
        payload[field] = row.get(field, "")
    if any(row.get(field, "") for field in fields):
        payload.pop("row_index_fallback")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def stable_unit_interval(text: str, seed: int, salt: str) -> float:
    digest = hashlib.sha256(f"{seed}:{salt}:{text}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def four_way_benchmark_split(
    rows: list[dict[str, str]],
    labels: np.ndarray,
    y: np.ndarray,
    *,
    validation_fraction: float,
    blind_fraction: float,
    stress_fraction: float,
    seed: int,
) -> dict[str, np.ndarray]:
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
    target_abs = np.abs(y)
    scales = np.asarray([DEFAULT_TARGET_PHYSICAL_SCALES[name] for name in TARGET_COLUMNS], dtype=np.float32)
    normalized_abs = target_abs / scales[None, :]
    stress_threshold = float(np.quantile(np.max(normalized_abs, axis=1), 0.90))

    splits: dict[str, list[int]] = {"train": [], "validation": [], "blind": [], "stress": []}
    for index, row in enumerate(rows):
        if str(row.get(DATASET_SPLIT_HINT_FIELD, "")).strip() == TRAINING_ONLY_HINT:
            splits["train"].append(index)
            continue

        key = stable_row_key(row, index)
        label = str(labels[index])
        stress_candidate = label in stress_labels or float(np.max(normalized_abs[index])) >= stress_threshold
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

    for name, values in splits.items():
        if not values:
            raise RuntimeError(f"split {name!r} is empty; cannot run model selection safely")
    return {name: np.asarray(values, dtype=np.int64) for name, values in splits.items()}


def dataset_version_summary(rows: list[dict[str, str]], csv_path: Path) -> dict[str, Any]:
    versions = sorted({str(row.get("dataset_version", "")).strip() for row in rows if str(row.get("dataset_version", "")).strip()})
    sources = sorted({str(row.get("dataset_source", "")).strip() for row in rows if str(row.get("dataset_source", "")).strip()})
    training_only_count = sum(
        1 for row in rows if str(row.get(DATASET_SPLIT_HINT_FIELD, "")).strip() == TRAINING_ONLY_HINT
    )
    has_training_only_rows = training_only_count > 0
    if has_training_only_rows:
        split_policy = "stable_hash_parent_benchmark_with_training_only_append_rows"
        benchmark_provenance = (
            "Rows marked dataset_split_hint=training_only are assigned only to training. "
            "Validation, blind, and stress benchmarks are drawn from unhinted parent rows."
        )
    else:
        split_policy = "stable_hash_disjoint_benchmark_split"
        benchmark_provenance = "All rows are eligible for stable-hash train/validation/blind/stress assignment."
    preferred_versions = [
        version
        for version in versions
        if version and version not in {"parent_cached_dataset", "unversioned_cached_csv"}
    ]
    return {
        "dataset_version": preferred_versions[-1] if preferred_versions else versions[-1] if versions else "unversioned_cached_csv",
        "dataset_versions_present": versions,
        "dataset_sources_present": sources,
        "csv_path": str(csv_path),
        "csv_sha256": file_sha256(csv_path),
        "n_rows": int(len(rows)),
        "training_only_new_rows": int(training_only_count),
        "training_only_rows_respected": has_training_only_rows,
        "split_policy": split_policy,
        "validation_provenance": benchmark_provenance,
        "blind_provenance": benchmark_provenance,
        "stress_provenance": benchmark_provenance,
    }


def import_torch():
    import torch
    import torch.nn as nn
    return torch, nn


def build_residual_model(input_dim: int, output_dim: int, hidden_dim: int, dropout: float):
    torch, nn = import_torch()

    class ResidualRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(input_dim, output_dim)
            self.residual = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)

        def forward(self, x):
            return self.linear(x) + self.residual(x)

    return ResidualRegressor()


def build_grouped_head_model(input_dim: int, output_dim: int, hidden_dim: int, dropout: float):
    torch, nn = import_torch()
    if output_dim != len(TARGET_COLUMNS):
        raise ValueError(f"grouped_heads expects {len(TARGET_COLUMNS)} targets, got {output_dim}")

    class GroupedHeadRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(input_dim, output_dim)
            self.trunk = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            )
            self.scalar_head = self._head(2)
            self.low_order_head = self._head(6)
            self.high_order_head = self._head(4)

        def _head(self, out_dim):
            head_hidden = max(32, hidden_dim // 2)
            head = nn.Sequential(
                nn.Linear(hidden_dim, head_hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, out_dim),
            )
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            return head

        def residual(self, x):
            z = self.trunk(x)
            return torch.cat(
                [
                    self.scalar_head(z),
                    self.low_order_head(z),
                    self.high_order_head(z),
                ],
                dim=1,
            )

        def forward(self, x):
            return self.linear(x) + self.residual(x)

    return GroupedHeadRegressor()


def build_model(input_dim: int, output_dim: int, hidden_dim: int, dropout: float, architecture: str):
    if architecture == "residual_mlp":
        return build_residual_model(input_dim, output_dim, hidden_dim, dropout)
    if architecture == "grouped_heads":
        return build_grouped_head_model(input_dim, output_dim, hidden_dim, dropout)
    raise ValueError(f"unknown architecture: {architecture}")


def weighted_component_loss(pred, target, target_weights, *, loss_kind: str, smooth_l1_beta: float):
    torch, _ = import_torch()
    if loss_kind == "mse":
        per_component = (pred - target) ** 2
    elif loss_kind == "smooth_l1":
        per_component = torch.nn.functional.smooth_l1_loss(
            pred,
            target,
            reduction="none",
            beta=smooth_l1_beta,
        )
    else:
        raise ValueError(f"unknown component loss: {loss_kind}")
    return torch.mean(per_component * target_weights[None, :])


def weighted_mse(pred, target, target_weights):
    return weighted_component_loss(
        pred,
        target,
        target_weights,
        loss_kind="mse",
        smooth_l1_beta=1.0,
    )


def s3_magnitude_loss(
    pred_scaled,
    target_scaled,
    *,
    y_mean,
    y_std,
    vector_scale: float,
    loss_weight: float,
    low_bin_weight: float,
    medium_bin_weight: float,
    high_bin_weight: float,
    use_smooth_l1: bool,
):
    torch, _ = import_torch()
    if loss_weight <= 0:
        return torch.zeros((), dtype=pred_scaled.dtype, device=pred_scaled.device)

    s3_x_index = TARGET_COLUMNS.index("S3_x")
    s3_y_index = TARGET_COLUMNS.index("S3_y")
    pred_phys = pred_scaled * y_std[None, :] + y_mean[None, :]
    target_phys = target_scaled * y_std[None, :] + y_mean[None, :]
    pred_mag = torch.sqrt(pred_phys[:, s3_x_index] ** 2 + pred_phys[:, s3_y_index] ** 2 + 1e-8)
    true_mag = torch.sqrt(target_phys[:, s3_x_index] ** 2 + target_phys[:, s3_y_index] ** 2 + 1e-8)
    scale = max(float(vector_scale), 1e-8)
    normalized_error = (pred_mag - true_mag) / scale
    if use_smooth_l1:
        per_sample = torch.nn.functional.smooth_l1_loss(
            normalized_error,
            torch.zeros_like(normalized_error),
            reduction="none",
        )
    else:
        per_sample = torch.abs(normalized_error)

    bin_weights = torch.zeros_like(true_mag)
    bin_weights = torch.where(
        (true_mag > 0.1 * scale) & (true_mag <= 0.3 * scale),
        torch.full_like(bin_weights, float(low_bin_weight)),
        bin_weights,
    )
    bin_weights = torch.where(
        (true_mag > 0.3 * scale) & (true_mag <= 0.7 * scale),
        torch.full_like(bin_weights, float(medium_bin_weight)),
        bin_weights,
    )
    bin_weights = torch.where(
        true_mag > 0.7 * scale,
        torch.full_like(bin_weights, float(high_bin_weight)),
        bin_weights,
    )
    active = bin_weights > 0
    if int(torch.sum(active).detach().cpu()) == 0:
        return torch.zeros((), dtype=pred_scaled.dtype, device=pred_scaled.device)
    return float(loss_weight) * torch.mean(per_sample[active] * bin_weights[active])


def summarize_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray,
    split_indices: dict[str, np.ndarray],
    training_config: dict[str, Any],
    target_physical_scales: dict[str, float],
) -> dict[str, Any]:
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    scale_array = np.asarray([target_physical_scales[name] for name in TARGET_COLUMNS], dtype=np.float32)
    normalized_abs_errors = abs_errors / scale_array[None, :]

    def split_summary(indices: np.ndarray) -> dict[str, Any]:
        split_errors = errors[indices]
        split_abs = abs_errors[indices]
        split_norm_abs = normalized_abs_errors[indices]
        data: dict[str, Any] = {
            "n": int(len(indices)),
            "overall_mae": float(split_abs.mean()),
            "overall_rmse": float(np.sqrt(np.mean(split_errors**2))),
            "overall_p95_abs_error": float(np.quantile(split_abs, 0.95)),
            "overall_normalized_mae": float(split_norm_abs.mean()),
            "overall_normalized_p95_abs_error": float(np.quantile(split_norm_abs, 0.95)),
            "targets": {},
            "labels": {},
        }
        for i, name in enumerate(TARGET_COLUMNS):
            target_errors = split_errors[:, i]
            target_abs = np.abs(target_errors)
            data["targets"][name] = {
                "mae": float(np.mean(target_abs)),
                "rmse": float(np.sqrt(np.mean(target_errors**2))),
                "p95_abs_error": float(np.quantile(target_abs, 0.95)),
                "bias": float(np.mean(target_errors)),
                "normalized_mae": float(np.mean(target_abs) / target_physical_scales[name]),
                "normalized_p95_abs_error": float(np.quantile(target_abs, 0.95) / target_physical_scales[name]),
            }
        for label in sorted(set(labels[indices])):
            mask_indices = indices[labels[indices] == label]
            label_errors = errors[mask_indices]
            label_abs = abs_errors[mask_indices]
            label_norm_abs = normalized_abs_errors[mask_indices]
            data["labels"][label] = {
                "n": int(len(mask_indices)),
                "mae": float(np.mean(label_abs)),
                "rmse": float(np.sqrt(np.mean(label_errors**2))),
                "p95_abs_error": float(np.quantile(label_abs, 0.95)),
                "normalized_mae": float(np.mean(label_norm_abs)),
                "normalized_p95_abs_error": float(np.quantile(label_norm_abs, 0.95)),
            }
        return data

    metrics: dict[str, Any] = {
        "n_samples": int(len(y_true)),
        "n_train": int(len(split_indices["train"])),
        "n_validation": int(len(split_indices["validation"])),
        "n_blind": int(len(split_indices["blind"])),
        "n_stress": int(len(split_indices["stress"])),
        "overall_mae": float(abs_errors.mean()),
        "overall_rmse": float(np.sqrt(np.mean(errors**2))),
        "overall_normalized_mae": float(normalized_abs_errors.mean()),
        "overall_normalized_p95_abs_error": float(np.quantile(normalized_abs_errors, 0.95)),
        "targets": {},
        "test_targets": {},
        "labels": {},
        "splits": {},
        "training_config": training_config,
        "target_physical_scales": target_physical_scales,
    }
    for i, name in enumerate(TARGET_COLUMNS):
        target_errors = errors[:, i]
        metrics["targets"][name] = {
            "mae": float(np.mean(np.abs(target_errors))),
            "rmse": float(np.sqrt(np.mean(target_errors**2))),
            "p95_abs_error": float(np.quantile(np.abs(target_errors), 0.95)),
            "bias": float(np.mean(target_errors)),
            "normalized_mae": float(np.mean(np.abs(target_errors)) / target_physical_scales[name]),
            "normalized_p95_abs_error": float(np.quantile(np.abs(target_errors), 0.95) / target_physical_scales[name]),
        }
    for split_name, indices in split_indices.items():
        metrics["splits"][split_name] = split_summary(indices)
    metrics["test_targets"] = metrics["splits"]["validation"]["targets"]
    for label in sorted(set(labels)):
        for split_name, indices in [("all", np.arange(len(labels))), ("test", split_indices["validation"])]:
            mask_indices = indices[labels[indices] == label]
            if len(mask_indices) == 0:
                continue
            label_errors = errors[mask_indices]
            label_norm_abs = normalized_abs_errors[mask_indices]
            metrics["labels"].setdefault(label, {})[split_name] = {
                "n": int(len(mask_indices)),
                "mae": float(np.mean(np.abs(label_errors))),
                "rmse": float(np.sqrt(np.mean(label_errors**2))),
                "p95_abs_error": float(np.quantile(np.abs(label_errors), 0.95)),
                "normalized_mae": float(np.mean(label_norm_abs)),
                "normalized_p95_abs_error": float(np.quantile(label_norm_abs, 0.95)),
            }
    return metrics


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_s3_magnitude_metric_audit_csv(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rows: list[dict[str, str]],
    target_columns: list[str],
    validation_index: np.ndarray,
    vector_diag: dict[str, Any],
) -> None:
    s3_x_index = target_columns.index("S3_x")
    s3_y_index = target_columns.index("S3_y")
    s3_diag = vector_diag["vector_pairs"]["S3"]
    vector_scale = float(s3_diag["magnitude"]["vector_scale"])
    high_threshold = 0.7 * max(vector_scale, 1e-8)
    audit_rows: list[dict[str, Any]] = []
    for validation_position, row_index in enumerate(validation_index):
        row_i = int(row_index)
        true_x = float(y_true[row_i, s3_x_index])
        true_y = float(y_true[row_i, s3_y_index])
        pred_x = float(y_pred[row_i, s3_x_index])
        pred_y = float(y_pred[row_i, s3_y_index])
        true_mag = float(np.hypot(true_x, true_y))
        pred_mag = float(np.hypot(pred_x, pred_y))
        true_angle = float(np.rad2deg(np.arctan2(true_y, true_x)))
        pred_angle = float(np.rad2deg(np.arctan2(pred_y, pred_x)))
        angle_error = float((pred_angle - true_angle + 180.0) % 360.0 - 180.0)
        audit_rows.append(
            {
                "validation_position": validation_position,
                "row_index": row_i,
                "sweep_label": rows[row_i].get("sweep_label", ""),
                "dataset_version": rows[row_i].get("dataset_version", ""),
                "true_s3_x": true_x,
                "true_s3_y": true_y,
                "pred_s3_x": pred_x,
                "pred_s3_y": pred_y,
                "true_s3_magnitude": true_mag,
                "pred_s3_magnitude": pred_mag,
                "magnitude_error": pred_mag - true_mag,
                "vector_residual_magnitude": float(np.hypot(pred_x - true_x, pred_y - true_y)),
                "true_s3_angle_deg": true_angle,
                "pred_s3_angle_deg": pred_angle,
                "angle_error_deg": angle_error,
                "vector_scale": vector_scale,
                "high_s3_threshold": high_threshold,
                "high_s3_bin": true_mag > high_threshold,
            }
        )
    write_csv(
        path,
        audit_rows,
        [
            "validation_position",
            "row_index",
            "sweep_label",
            "dataset_version",
            "true_s3_x",
            "true_s3_y",
            "pred_s3_x",
            "pred_s3_y",
            "true_s3_magnitude",
            "pred_s3_magnitude",
            "magnitude_error",
            "vector_residual_magnitude",
            "true_s3_angle_deg",
            "pred_s3_angle_deg",
            "angle_error_deg",
            "vector_scale",
            "high_s3_threshold",
            "high_s3_bin",
        ],
    )


def write_scale_summary(path: Path, names: list[str], data: np.ndarray) -> None:
    rows = []
    for i, name in enumerate(names):
        column = data[:, i]
        rows.append(
            {
                "name": name,
                "mean": float(np.mean(column)),
                "std": float(np.std(column)),
                "min": float(np.min(column)),
                "max": float(np.max(column)),
                "p01": float(np.quantile(column, 0.01)),
                "p99": float(np.quantile(column, 0.99)),
                "near_constant": bool(np.std(column) < 1e-8),
            }
        )
    write_csv(path, rows, ["name", "mean", "std", "min", "max", "p01", "p99", "near_constant"])


def finite_history_values(history: list[dict[str, float]], key: str) -> tuple[list[float], list[float]]:
    epochs: list[float] = []
    values: list[float] = []
    for row in history:
        value = row.get(key)
        if value is None:
            continue
        if not np.isfinite(float(value)):
            continue
        epochs.append(float(row["epoch"]))
        values.append(float(value))
    return epochs, values


def plot_history(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    mse_curves = [
        ("train_weighted_mse", "train all"),
        ("train_parent_weighted_mse", "train parent"),
        ("train_training_only_weighted_mse", "train training_only"),
        ("validation_weighted_mse", "validation"),
        ("blind_weighted_mse", "blind"),
        ("stress_weighted_mse", "stress"),
    ]
    for key, label in mse_curves:
        epochs, values = finite_history_values(history, key)
        if values:
            axes[0].plot(epochs, values, label=label)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("weighted scaled MSE")
    axes[0].set_title("Eval-mode weighted MSE by split/source")
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(alpha=0.3)

    objective_curves = [
        ("train_total_objective", "train total objective"),
        ("train_component_smoothl1", "train component SmoothL1"),
        ("validation_component_smoothl1", "validation component SmoothL1"),
        ("train_s3_magnitude_loss", "train S3 magnitude loss"),
    ]
    for key, label in objective_curves:
        epochs, values = finite_history_values(history, key)
        if values:
            axes[1].plot(epochs, values, label=label)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("loss")
    axes[1].set_title("Objective diagnostics")
    axes[1].legend(fontsize=8, ncol=2)
    axes[1].grid(alpha=0.3)

    ratio_epochs = []
    ratios = []
    for row in history:
        train_value = row.get("train_weighted_mse", row.get("train_loss"))
        validation_value = row.get("validation_weighted_mse", row.get("validation_loss", row.get("test_loss")))
        if train_value is None or validation_value is None or float(train_value) <= 0:
            continue
        ratio_epochs.append(float(row["epoch"]))
        ratios.append(float(validation_value) / float(train_value))
    if ratios:
        axes[2].plot(ratio_epochs, ratios, label="validation/train weighted MSE")
        axes[2].legend(fontsize=8)
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("ratio")
    axes[2].set_title("Gap diagnostic")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_scatter(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray,
    test_index: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    ncols = 4
    nrows = int(math.ceil(len(TARGET_COLUMNS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 8))
    axes = axes.ravel()
    test_labels = labels[test_index]
    unique_labels = sorted(set(test_labels))
    color_map = plt.get_cmap("tab20")
    colors = {
        label: color_map(i % color_map.N)
        for i, label in enumerate(unique_labels)
    }
    for i, name in enumerate(TARGET_COLUMNS):
        ax = axes[i]
        true = y_true[test_index, i]
        pred = y_pred[test_index, i]
        for label in unique_labels:
            mask = test_labels == label
            ax.scatter(
                true[mask],
                pred[mask],
                s=6,
                alpha=0.55,
                color=colors[label],
                label=label if i == 0 else None,
            )
        low = float(min(true.min(), pred.min()))
        high = float(max(true.max(), pred.max()))
        ax.plot([low, high], [low, high], "k--", linewidth=0.8)
        ax.set_title(name, fontsize=9)
    for j in range(len(TARGET_COLUMNS), len(axes)):
        axes[j].axis("off")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            legend_labels,
            loc="center right",
            fontsize=6,
            markerscale=1.5,
            frameon=True,
        )
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_vector_diagnostics(
    output_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_columns: list[str],
    validation_index: np.ndarray,
    vector_diag: dict[str, Any],
) -> None:
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots" / "vector_diagnostics"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for pair_name in ("B2", "S3", "A3"):
        x_name = f"{pair_name}_x"
        y_name = f"{pair_name}_y"
        x_index = target_columns.index(x_name)
        y_index = target_columns.index(y_name)
        true_x = y_true[validation_index, x_index]
        true_y = y_true[validation_index, y_index]
        pred_x = y_pred[validation_index, x_index]
        pred_y = y_pred[validation_index, y_index]
        true_magnitude = np.sqrt(true_x**2 + true_y**2)
        pred_magnitude = np.sqrt(pred_x**2 + pred_y**2)
        true_angle = np.rad2deg(np.arctan2(true_y, true_x))
        pred_angle = np.rad2deg(np.arctan2(pred_y, pred_x))
        angle_error = (pred_angle - true_angle + 180.0) % 360.0 - 180.0
        threshold = float(
            vector_diag["vector_pairs"][pair_name]["angle"]["angle_threshold"]
        )
        angle_mask = true_magnitude > threshold

        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        ax.scatter(true_magnitude, pred_magnitude, s=7, alpha=0.5)
        high = float(max(np.max(true_magnitude) if len(true_magnitude) else 0.0, np.max(pred_magnitude) if len(pred_magnitude) else 0.0))
        ax.plot([0.0, high], [0.0, high], "k--", linewidth=0.8)
        ax.set_xlabel("true magnitude")
        ax.set_ylabel("pred magnitude")
        ax.set_title(f"{pair_name} magnitude", fontsize=10)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{pair_name}_pred_vs_true_magnitude.png", dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        ax.scatter(true_magnitude[angle_mask], angle_error[angle_mask], s=7, alpha=0.5)
        ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8)
        ax.set_xlabel("true magnitude")
        ax.set_ylabel("angle error deg")
        ax.set_title(f"{pair_name} angle error", fontsize=10)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{pair_name}_angle_error_vs_true_magnitude.png", dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        ax.scatter(true_angle[angle_mask], pred_angle[angle_mask], s=7, alpha=0.5)
        ax.plot([-180.0, 180.0], [-180.0, 180.0], "k--", linewidth=0.8)
        ax.set_xlim(-180.0, 180.0)
        ax.set_ylim(-180.0, 180.0)
        ax.set_xlabel("true angle deg")
        ax.set_ylabel("pred angle deg")
        ax.set_title(f"{pair_name} angle", fontsize=10)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{pair_name}_pred_vs_true_angle.png", dpi=120)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["enhanced", "raw_angles"], required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--csv-path", type=Path)
    parser.add_argument("--search-root", type=Path, default=Path("training_results"))
    parser.add_argument("--output-root", type=Path, default=Path("training_results/model_selection_loop"))
    parser.add_argument("--architecture", choices=["residual_mlp", "grouped_heads"], default="residual_mlp")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--residual-penalty", type=float, default=3e-3)
    parser.add_argument("--torch-seed", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Use full-batch training when <= 0; otherwise train with shuffled mini-batches.",
    )
    parser.add_argument("--shuffle-batches", action="store_true")
    parser.add_argument("--component-loss-kind", choices=["mse", "smooth_l1"], default="mse")
    parser.add_argument("--component-smooth-l1-beta", type=float, default=0.25)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--lr-scheduler", choices=["none", "plateau"], default="none")
    parser.add_argument("--lr-plateau-factor", type=float, default=0.5)
    parser.add_argument("--lr-plateau-patience-evals", type=int, default=8)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--s3-magnitude-loss-weight", type=float, default=0.0)
    parser.add_argument("--s3-magnitude-low-bin-weight", type=float, default=1.0)
    parser.add_argument("--s3-magnitude-medium-bin-weight", type=float, default=2.0)
    parser.add_argument("--s3-magnitude-high-bin-weight", type=float, default=4.0)
    parser.add_argument("--s3-magnitude-loss-kind", choices=["smooth_l1", "mae"], default="smooth_l1")
    parser.add_argument("--max-epochs", type=int, default=6000)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--patience-epochs", type=int, default=1000)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--easy-regression-limit", type=float, default=0.10)
    parser.add_argument("--baseline-metrics", type=Path)
    parser.add_argument("--selection-config", type=Path, default=Path("experiments/model_selection_weights.json"))
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--blind-fraction", type=float, default=0.10)
    parser.add_argument("--stress-fraction", type=float, default=0.20)
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument(
        "--bootstrap-if-missing",
        action="store_true",
        help="Generate the cached feature CSV once if it is absent.",
    )
    parser.add_argument(
        "--bootstrap-notebook",
        type=Path,
        default=Path("notebooks/uno_feature_regression_enhanced_dataset_bootstrap.ipynb"),
        help="Dataset-only notebook used when --bootstrap-if-missing is set.",
    )
    parser.add_argument(
        "--bootstrap-timeout",
        type=int,
        default=3600,
        help="Timeout in seconds for dataset bootstrap notebook execution.",
    )
    parser.add_argument(
        "--bootstrap-output-dir",
        type=Path,
        default=Path("colab_worker_logs"),
        help="Where to write the executed bootstrap notebook manifest.",
    )
    return parser.parse_args()


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    values = vars(args).copy()
    for key, value in list(values.items()):
        if isinstance(value, Path):
            values[key] = str(value)
    return values


def default_baseline_metrics(csv_path: Path, family: str) -> Path | None:
    names = ["metrics_enhanced.json"] if family == "enhanced" else ["metrics_raw_angles.json", "metrics_enhanced.json"]
    for name in names:
        path = csv_path.parent / name
        if path.exists():
            return path
    return None


def write_preflight_failure(output_root: Path, candidate_id: str, message: str, details: dict[str, Any]) -> Path:
    output_dir = output_root / f"{candidate_id}_preflight_failure_{utc_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "preflight_failure",
        "candidate_id": candidate_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "message": message,
        "details": details,
    }
    path = output_dir / "preflight_failure_model_loop.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print("preflight failure:", path)
    print(message)
    return path


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd()
    filename = "training_features_enhanced.csv" if args.family == "enhanced" else "training_features_raw_angles.csv"
    try:
        csv_path = ensure_csv_available(args, filename)
    except FileNotFoundError as exc:
        write_preflight_failure(
            args.output_root,
            args.candidate_id,
            str(exc),
            {
                "family": args.family,
                "search_root": str(args.search_root),
                "required_filename": filename,
                "uses_existing_cached_csv_only": True,
            },
        )
        return 2
    csv_path = csv_path.resolve()
    try:
        feature_columns = find_feature_columns(csv_path, args.family)
    except FileNotFoundError as exc:
        write_preflight_failure(
            args.output_root,
            args.candidate_id,
            str(exc),
            {
                "family": args.family,
                "csv_path": str(csv_path),
                "uses_existing_cached_csv_only": True,
            },
        )
        return 2

    run_name = f"{args.candidate_id}_{utc_stamp()}"
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("candidate:", args.candidate_id)
    print("family:", args.family)
    print("source CSV:", csv_path)
    print("features:", len(feature_columns))
    print("output:", output_dir)

    X, y, labels, rows = prepare_dataset(csv_path, feature_columns)
    dataset_info = dataset_version_summary(rows, csv_path)
    selection_config = (
        json.loads(args.selection_config.read_text())
        if args.selection_config and args.selection_config.exists()
        else {}
    )
    target_physical_scales = {
        **DEFAULT_TARGET_PHYSICAL_SCALES,
        **selection_config.get("target_physical_scales", {}),
    }
    split_indices = four_way_benchmark_split(
        rows,
        labels,
        y,
        validation_fraction=args.validation_fraction,
        blind_fraction=args.blind_fraction,
        stress_fraction=args.stress_fraction,
        seed=args.split_seed,
    )
    train_index = split_indices["train"]
    validation_index = split_indices["validation"]

    x_scaler = Standardizer().fit(X[train_index])
    y_scaler = Standardizer().fit(y[train_index])
    Xn = x_scaler.transform(X).astype(np.float32)
    yn = y_scaler.transform(y).astype(np.float32)

    torch, _ = import_torch()
    if args.torch_seed is not None:
        random.seed(args.torch_seed)
        np.random.seed(args.torch_seed)
        torch.manual_seed(args.torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.torch_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(Xn.shape[1], yn.shape[1], args.hidden_dim, args.dropout, args.architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = None
    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_plateau_factor,
            patience=args.lr_plateau_patience_evals,
            min_lr=args.min_learning_rate,
        )
    target_weights = torch.tensor(
        [TARGET_WEIGHTS[name] for name in TARGET_COLUMNS],
        dtype=torch.float32,
        device=device,
    )
    y_mean = torch.tensor(y_scaler.mean, dtype=torch.float32, device=device)
    y_std = torch.tensor(y_scaler.std, dtype=torch.float32, device=device)
    s3_x_index = TARGET_COLUMNS.index("S3_x")
    s3_y_index = TARGET_COLUMNS.index("S3_y")
    s3_train_magnitude = np.sqrt(y[train_index, s3_x_index] ** 2 + y[train_index, s3_y_index] ** 2)
    s3_vector_scale = float(np.quantile(s3_train_magnitude, 0.95)) if len(s3_train_magnitude) else 1e-8

    x_train = torch.tensor(Xn[train_index], device=device)
    y_train = torch.tensor(yn[train_index], device=device)
    x_validation = torch.tensor(Xn[validation_index], device=device)
    y_validation = torch.tensor(yn[validation_index], device=device)
    split_tensors: dict[str, tuple[Any, Any]] = {
        "train": (x_train, y_train),
        "validation": (x_validation, y_validation),
        "blind": (
            torch.tensor(Xn[split_indices["blind"]], device=device),
            torch.tensor(yn[split_indices["blind"]], device=device),
        ),
        "stress": (
            torch.tensor(Xn[split_indices["stress"]], device=device),
            torch.tensor(yn[split_indices["stress"]], device=device),
        ),
    }
    train_parent_positions = np.asarray(
        [
            position
            for position, row_index in enumerate(train_index)
            if str(rows[int(row_index)].get(DATASET_SPLIT_HINT_FIELD, "")).strip() != TRAINING_ONLY_HINT
        ],
        dtype=np.int64,
    )
    train_training_only_positions = np.asarray(
        [
            position
            for position, row_index in enumerate(train_index)
            if str(rows[int(row_index)].get(DATASET_SPLIT_HINT_FIELD, "")).strip() == TRAINING_ONLY_HINT
        ],
        dtype=np.int64,
    )
    train_source_tensors: dict[str, tuple[Any, Any] | None] = {
        "parent": (
            (x_train[torch.tensor(train_parent_positions, device=device)], y_train[torch.tensor(train_parent_positions, device=device)])
            if len(train_parent_positions)
            else None
        ),
        "training_only": (
            (
                x_train[torch.tensor(train_training_only_positions, device=device)],
                y_train[torch.tensor(train_training_only_positions, device=device)],
            )
            if len(train_training_only_positions)
            else None
        ),
    }

    history: list[dict[str, float]] = []
    best_state = None
    best_epoch = None
    best_validation_loss = float("inf")
    epochs_since_best = 0

    n_train_rows = int(x_train.shape[0])
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        epoch_total_loss = 0.0
        if args.batch_size and args.batch_size > 0:
            if args.shuffle_batches:
                order = torch.randperm(n_train_rows, device=device)
            else:
                order = torch.arange(n_train_rows, device=device)
            batch_size = int(args.batch_size)
            batches = [order[start : start + batch_size] for start in range(0, n_train_rows, batch_size)]
        else:
            batches = [None]

        for batch in batches:
            if batch is None:
                xb = x_train
                yb = y_train
                batch_n = n_train_rows
            else:
                xb = x_train[batch]
                yb = y_train[batch]
                batch_n = int(batch.shape[0])
            optimizer.zero_grad(set_to_none=True)
            pred_batch = model(xb)
            residual = model.residual(xb)
            train_component_loss = weighted_component_loss(
                pred_batch,
                yb,
                target_weights,
                loss_kind=args.component_loss_kind,
                smooth_l1_beta=args.component_smooth_l1_beta,
            )
            train_s3_magnitude_loss = s3_magnitude_loss(
                pred_batch,
                yb,
                y_mean=y_mean,
                y_std=y_std,
                vector_scale=s3_vector_scale,
                loss_weight=args.s3_magnitude_loss_weight,
                low_bin_weight=args.s3_magnitude_low_bin_weight,
                medium_bin_weight=args.s3_magnitude_medium_bin_weight,
                high_bin_weight=args.s3_magnitude_high_bin_weight,
                use_smooth_l1=args.s3_magnitude_loss_kind == "smooth_l1",
            )
            loss = (
                train_component_loss
                + train_s3_magnitude_loss
                + args.residual_penalty * torch.mean(residual**2)
            )
            loss.backward()
            if args.grad_clip_norm and args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            epoch_total_loss += float(loss.detach().cpu()) * batch_n

        if epoch % args.eval_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                train_pred_eval = model(x_train)
                train_weighted_mse = float(weighted_mse(train_pred_eval, y_train, target_weights).detach().cpu())
                train_component_loss_eval = float(
                    weighted_component_loss(
                        train_pred_eval,
                        y_train,
                        target_weights,
                        loss_kind=args.component_loss_kind,
                        smooth_l1_beta=args.component_smooth_l1_beta,
                    )
                    .detach()
                    .cpu()
                )
                train_component_smoothl1 = float(
                    weighted_component_loss(
                        train_pred_eval,
                        y_train,
                        target_weights,
                        loss_kind="smooth_l1",
                        smooth_l1_beta=args.component_smooth_l1_beta,
                    )
                    .detach()
                    .cpu()
                )
                train_s3_mag_loss_eval = float(
                    s3_magnitude_loss(
                        train_pred_eval,
                        y_train,
                        y_mean=y_mean,
                        y_std=y_std,
                        vector_scale=s3_vector_scale,
                        loss_weight=args.s3_magnitude_loss_weight,
                        low_bin_weight=args.s3_magnitude_low_bin_weight,
                        medium_bin_weight=args.s3_magnitude_medium_bin_weight,
                        high_bin_weight=args.s3_magnitude_high_bin_weight,
                        use_smooth_l1=args.s3_magnitude_loss_kind == "smooth_l1",
                    )
                    .detach()
                    .cpu()
                )
                train_residual_penalty_eval = float(
                    (args.residual_penalty * torch.mean(model.residual(x_train) ** 2)).detach().cpu()
                )
                train_total_objective = (
                    train_component_loss_eval
                    + train_s3_mag_loss_eval
                    + train_residual_penalty_eval
                )
                validation_pred_eval = model(x_validation)
                validation_weighted_mse = float(
                    weighted_mse(validation_pred_eval, y_validation, target_weights).detach().cpu()
                )
                validation_component_loss_eval = float(
                    weighted_component_loss(
                        validation_pred_eval,
                        y_validation,
                        target_weights,
                        loss_kind=args.component_loss_kind,
                        smooth_l1_beta=args.component_smooth_l1_beta,
                    )
                    .detach()
                    .cpu()
                )
                validation_component_smoothl1 = float(
                    weighted_component_loss(
                        validation_pred_eval,
                        y_validation,
                        target_weights,
                        loss_kind="smooth_l1",
                        smooth_l1_beta=args.component_smooth_l1_beta,
                    )
                    .detach()
                    .cpu()
                )
                split_weighted_mse: dict[str, float] = {
                    "train": train_weighted_mse,
                    "validation": validation_weighted_mse,
                }
                for split_name in ["blind", "stress"]:
                    xs, ys = split_tensors[split_name]
                    split_weighted_mse[split_name] = float(weighted_mse(model(xs), ys, target_weights).detach().cpu())
                train_source_weighted_mse: dict[str, float | None] = {"parent": None, "training_only": None}
                for source_name, tensors in train_source_tensors.items():
                    if tensors is None:
                        continue
                    xs, ys = tensors
                    train_source_weighted_mse[source_name] = float(
                        weighted_mse(model(xs), ys, target_weights).detach().cpu()
                    )
                current_lr = float(optimizer.param_groups[0]["lr"])
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_weighted_mse,
                    "validation_loss": validation_weighted_mse,
                    "train_weighted_mse": train_weighted_mse,
                    "validation_weighted_mse": validation_weighted_mse,
                    "blind_weighted_mse": split_weighted_mse["blind"],
                    "stress_weighted_mse": split_weighted_mse["stress"],
                    "train_parent_weighted_mse": train_source_weighted_mse["parent"],
                    "train_training_only_weighted_mse": train_source_weighted_mse["training_only"],
                    "train_component_loss": train_component_loss_eval,
                    "validation_component_loss": validation_component_loss_eval,
                    "train_component_smoothl1": train_component_smoothl1,
                    "validation_component_smoothl1": validation_component_smoothl1,
                    "train_s3_magnitude_loss": train_s3_mag_loss_eval,
                    "train_residual_penalty": train_residual_penalty_eval,
                    "train_total_objective": train_total_objective,
                    "train_total_objective_loss": train_total_objective,
                    "train_epoch_objective_loss": epoch_total_loss / max(n_train_rows, 1),
                    "learning_rate": current_lr,
                }
            )
            print(
                f"epoch {epoch:5d} train_mse={train_weighted_mse:.6f} "
                f"val_mse={validation_weighted_mse:.6f} blind_mse={split_weighted_mse['blind']:.6f} "
                f"stress_mse={split_weighted_mse['stress']:.6f} objective={train_total_objective:.6f} "
                f"lr={current_lr:.3g}"
            )
            if validation_weighted_mse < best_validation_loss:
                best_validation_loss = validation_weighted_mse
                best_epoch = epoch
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                epochs_since_best = 0
            else:
                epochs_since_best += args.eval_every
            if scheduler is not None:
                scheduler.step(validation_weighted_mse)
            if epochs_since_best >= args.patience_epochs:
                print("early stopping at epoch", epoch)
                break

    if best_state is not None:
        model.load_state_dict({name: value.to(device) for name, value in best_state.items()})

    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.tensor(Xn, device=device)).detach().cpu().numpy()
    y_pred = y_scaler.inverse_transform(pred_scaled)

    training_config = {
        "candidate_id": args.candidate_id,
        "family": args.family,
        "architecture": args.architecture,
        "max_epochs": args.max_epochs,
        "eval_every": args.eval_every,
        "patience_epochs": args.patience_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "residual_penalty": args.residual_penalty,
        "torch_seed": args.torch_seed,
        "batch_size": args.batch_size,
        "shuffle_batches": args.shuffle_batches,
        "component_loss_kind": args.component_loss_kind,
        "component_smooth_l1_beta": args.component_smooth_l1_beta,
        "grad_clip_norm": args.grad_clip_norm,
        "lr_scheduler": args.lr_scheduler,
        "lr_plateau_factor": args.lr_plateau_factor,
        "lr_plateau_patience_evals": args.lr_plateau_patience_evals,
        "min_learning_rate": args.min_learning_rate,
        "s3_magnitude_loss": {
            "enabled": bool(args.s3_magnitude_loss_weight > 0),
            "loss_weight": args.s3_magnitude_loss_weight,
            "loss_kind": args.s3_magnitude_loss_kind,
            "vector_scale": s3_vector_scale,
            "scale_source": "training_split_true_S3_magnitude_p95",
            "near_zero_bin_weight": 0.0,
            "low_bin_weight": args.s3_magnitude_low_bin_weight,
            "medium_bin_weight": args.s3_magnitude_medium_bin_weight,
            "high_bin_weight": args.s3_magnitude_high_bin_weight,
            "bin_policy": {
                "near_zero": "true_magnitude <= 0.1 * vector_scale",
                "low": "0.1 * vector_scale < true_magnitude <= 0.3 * vector_scale",
                "medium": "0.3 * vector_scale < true_magnitude <= 0.7 * vector_scale",
                "high": "true_magnitude > 0.7 * vector_scale",
            },
            "angle_loss_enabled": False,
        },
        "hidden_dim": args.hidden_dim,
        "dropout_probability": args.dropout,
        "split_strategy": dataset_info["split_policy"],
        "evaluation_protocol": "train_validation_blind_stress_disjoint",
        "split_seed": args.split_seed,
        "validation_fraction": args.validation_fraction,
        "blind_fraction": args.blind_fraction,
        "stress_fraction": args.stress_fraction,
        "best_epoch": best_epoch,
        "best_validation_weighted_mse_scaled": best_validation_loss,
        "feature_count": len(feature_columns),
        "scaler": "standard_train_split",
        "dataset_version": dataset_info["dataset_version"],
        "dataset_versions_present": dataset_info["dataset_versions_present"],
        "training_only_new_rows": dataset_info["training_only_new_rows"],
        "training_only_rows_respected": dataset_info["training_only_rows_respected"],
        "validation_provenance": dataset_info["validation_provenance"],
        "blind_provenance": dataset_info["blind_provenance"],
        "stress_provenance": dataset_info["stress_provenance"],
    }
    metrics = summarize_predictions(y, y_pred, labels, split_indices, training_config, target_physical_scales)
    metrics["run_name"] = run_name
    metrics["dataset"] = {
        **dataset_info,
        "n_train": int(len(train_index)),
        "n_validation": int(len(split_indices["validation"])),
        "n_blind": int(len(split_indices["blind"])),
        "n_stress": int(len(split_indices["stress"])),
    }
    metrics_path = output_dir / "metrics_model_loop.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    diagnostics = {
        split_name: per_target_diagnostics(
            y[indices],
            y_pred[indices],
            TARGET_COLUMNS,
            target_physical_scales,
            split_name=split_name,
        )
        for split_name, indices in split_indices.items()
    }
    diagnostics["validation_regime_breakdown"] = regime_breakdown(
        y,
        y_pred,
        rows,
        validation_index,
        TARGET_COLUMNS,
        target_physical_scales,
    )
    (output_dir / "per_target_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    vector_diag = vector_diagnostics(
        y,
        y_pred,
        rows,
        TARGET_COLUMNS,
        split_indices,
    )
    vector_diagnostics_path = output_dir / "vector_diagnostics.json"
    vector_diagnostics_path.write_text(json.dumps(vector_diag, indent=2) + "\n")
    plot_vector_diagnostics(output_dir, y, y_pred, TARGET_COLUMNS, validation_index, vector_diag)
    s3_magnitude_metric_audit_path = output_dir / "s3_magnitude_metric_audit_validation.csv"
    write_s3_magnitude_metric_audit_csv(
        s3_magnitude_metric_audit_path,
        y,
        y_pred,
        rows,
        TARGET_COLUMNS,
        validation_index,
        vector_diag,
    )

    baseline_path = args.baseline_metrics or default_baseline_metrics(csv_path, args.family)
    baseline = json.loads(baseline_path.read_text()) if baseline_path and baseline_path.exists() else None
    selection = score_run(
        metrics_path,
        metrics,
        baseline=baseline,
        easy_regression_limit=args.easy_regression_limit,
        selection_config=selection_config,
    )
    (output_dir / "selection_score.json").write_text(json.dumps(selection, indent=2) + "\n")

    write_scale_summary(output_dir / "feature_scale_summary.csv", feature_columns, X)
    write_scale_summary(output_dir / "target_scale_summary.csv", TARGET_COLUMNS, y)
    write_csv(
        output_dir / "training_history_summary.csv",
        history,
        [
            "epoch",
            "train_loss",
            "validation_loss",
            "train_weighted_mse",
            "validation_weighted_mse",
            "blind_weighted_mse",
            "stress_weighted_mse",
            "train_parent_weighted_mse",
            "train_training_only_weighted_mse",
            "train_component_loss",
            "validation_component_loss",
            "train_component_smoothl1",
            "validation_component_smoothl1",
            "train_s3_magnitude_loss",
            "train_residual_penalty",
            "train_total_objective",
            "train_total_objective_loss",
            "train_epoch_objective_loss",
            "learning_rate",
        ],
    )
    plot_history(output_dir / "training_history_model_loop.png", history)
    plot_scatter(output_dir / "prediction_scatter_model_loop.png", y, y_pred, labels, validation_index)

    if args.save_model:
        torch.save(model.state_dict(), output_dir / "model_loop_candidate.pt")

    manifest = {
        "run_name": run_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_commit(repo_root),
        "python": sys.version,
        "platform": platform.platform(),
        "device": device,
        "candidate": jsonable_args(args),
        "baseline_metrics_path": None if baseline_path is None else str(baseline_path),
        "selection_config_path": None if args.selection_config is None else str(args.selection_config),
        "dataset": {
            **dataset_info,
            "n_rows": int(len(rows)),
            "n_train": int(len(train_index)),
            "n_validation": int(len(split_indices["validation"])),
            "n_blind": int(len(split_indices["blind"])),
            "n_stress": int(len(split_indices["stress"])),
        },
        "split_indices": {name: values.tolist() for name, values in split_indices.items()},
        "model": {
            "type": f"standardized_linear_plus_{args.architecture}",
            "input_dim": int(Xn.shape[1]),
            "output_dim": int(yn.shape[1]),
            "hidden_dim": args.hidden_dim,
            "dropout_probability": args.dropout,
            "torch_seed": args.torch_seed,
            "batch_size": args.batch_size,
            "shuffle_batches": args.shuffle_batches,
            "component_loss_kind": args.component_loss_kind,
            "component_smooth_l1_beta": args.component_smooth_l1_beta,
            "grad_clip_norm": args.grad_clip_norm,
            "lr_scheduler": args.lr_scheduler,
        },
        "feature_columns": feature_columns,
        "target_columns": TARGET_COLUMNS,
        "output_dir": str(output_dir),
        "metrics_path": str(metrics_path),
        "selection_score_path": str(output_dir / "selection_score.json"),
        "per_target_diagnostics_path": str(output_dir / "per_target_diagnostics.json"),
        "vector_diagnostics_path": str(vector_diagnostics_path),
        "s3_magnitude_metric_audit_path": str(s3_magnitude_metric_audit_path),
    }
    (output_dir / "run_manifest_model_loop.json").write_text(json.dumps(manifest, indent=2) + "\n")

    registry_path = output_dir / "model_registry_model_loop.csv"
    registry_row = {
        "run_name": run_name,
        "candidate_id": args.candidate_id,
        "family": args.family,
        "architecture": args.architecture,
        "input_dim": Xn.shape[1],
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "torch_seed": args.torch_seed,
        "batch_size": args.batch_size,
        "shuffle_batches": args.shuffle_batches,
        "component_loss_kind": args.component_loss_kind,
        "grad_clip_norm": args.grad_clip_norm,
        "lr_scheduler": args.lr_scheduler,
        "best_epoch": best_epoch,
        "best_validation_weighted_mse_scaled": best_validation_loss,
        "overall_mae": metrics["overall_mae"],
        "overall_rmse": metrics["overall_rmse"],
        "validation_normalized_mae": metrics["splits"]["validation"]["overall_normalized_mae"],
        "blind_normalized_mae": metrics["splits"]["blind"]["overall_normalized_mae"],
        "stress_normalized_mae": metrics["splits"]["stress"]["overall_normalized_mae"],
        "selection_weighted_score": selection["weighted_score"],
        "selection_rejected": selection["rejected"],
        "source_csv_sha256": manifest["dataset"]["csv_sha256"],
        "dataset_version": dataset_info["dataset_version"],
        "training_only_new_rows": dataset_info["training_only_new_rows"],
        "true_hard_target_normalized_mae": selection["components"].get("true_hard_target_normalized_mae"),
    }
    for pair_name in ("B2", "S3", "A3"):
        pair_diag = vector_diag["vector_pairs"][pair_name]
        registry_row[f"{pair_name}_mean_abs_angle_error_deg"] = pair_diag["angle"]["mean_abs_angle_error_deg"]
        registry_row[f"{pair_name}_magnitude_mae"] = pair_diag["magnitude"]["magnitude_mae"]
        registry_row[f"{pair_name}_mean_cosine_similarity"] = pair_diag["directional_cosine"]["mean_cosine_similarity"]
    s3_high = vector_diag["vector_pairs"]["S3"].get("magnitude_bins", {}).get("bins", {}).get("high", {})
    s3_high_magnitude = s3_high.get("magnitude", {})
    s3_high_angle = s3_high.get("angle", {})
    registry_row["S3_high_bin_n"] = s3_high.get("n")
    registry_row["S3_high_magnitude_mae"] = s3_high_magnitude.get("magnitude_mae")
    registry_row["S3_high_magnitude_bias"] = s3_high_magnitude.get("magnitude_bias")
    registry_row["S3_high_magnitude_slope"] = s3_high_magnitude.get("magnitude_slope")
    registry_row["S3_high_mean_abs_angle_error_deg"] = s3_high_angle.get("mean_abs_angle_error_deg")
    registry_row["S3_high_p95_abs_angle_error_deg"] = s3_high_angle.get("p95_abs_angle_error_deg")
    write_csv(registry_path, [registry_row], list(registry_row))

    print("metrics:", metrics_path)
    print("selection:", output_dir / "selection_score.json")
    print("weighted_score:", selection["weighted_score"])
    print("rejected:", selection["rejected"], selection["rejection_reasons"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
