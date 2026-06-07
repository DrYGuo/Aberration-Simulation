"""Compact diagnostics for aberration coefficient regression results."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


REGIME_COLUMN_CANDIDATES = ("regime_name", "regime", "sweep_label")
VECTOR_TARGET_PAIRS = {
    "A1": ("A1_x", "A1_y"),
    "B2": ("B2_x", "B2_y"),
    "A2": ("A2_x", "A2_y"),
    "S3": ("S3_x", "S3_y"),
    "A3": ("A3_x", "A3_y"),
}


def discover_regime_column(rows: list[dict[str, Any]]) -> str | None:
    for name in REGIME_COLUMN_CANDIDATES:
        if any(str(row.get(name, "")).strip() for row in rows):
            return name
    return None


def _safe_quantile(values: np.ndarray, quantile: float, minimum_n: int) -> dict[str, Any]:
    if len(values) < minimum_n:
        return {"value": None, "reliable": False, "n": int(len(values)), "minimum_n": minimum_n}
    return {
        "value": float(np.quantile(values, quantile)),
        "reliable": True,
        "n": int(len(values)),
        "minimum_n": minimum_n,
    }


def _target_diagnostics(
    true: np.ndarray,
    pred: np.ndarray,
    *,
    target_scale: float,
) -> dict[str, Any]:
    errors = pred - true
    abs_errors = np.abs(errors)
    n = int(len(true))
    true_std = float(np.std(true)) if n else 0.0
    pred_std = float(np.std(pred)) if n else 0.0
    sufficient_variance = true_std / max(float(target_scale), 1e-8) > 0.01
    regression_available = n >= 30 and sufficient_variance

    slope = None
    intercept = None
    r2 = None
    if regression_available:
        slope_value, intercept_value = np.polyfit(true, pred, 1)
        residual = pred - (slope_value * true + intercept_value)
        total = true - np.mean(true)
        ss_res = float(np.sum(residual**2))
        ss_tot = float(np.sum(total**2))
        slope = float(slope_value)
        intercept = float(intercept_value)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

    compression_ratio = pred_std / true_std if true_std > 0 else None
    regression_to_mean_warning = bool(
        regression_available
        and compression_ratio is not None
        and (slope is not None and slope < 0.85 or compression_ratio < 0.85)
    )

    return {
        "n": n,
        "target_scale": float(target_scale),
        "bias": float(np.mean(errors)) if n else 0.0,
        "mae": float(np.mean(abs_errors)) if n else 0.0,
        "rmse": float(np.sqrt(np.mean(errors**2))) if n else 0.0,
        "normalized_mae": float(np.mean(abs_errors) / max(float(target_scale), 1e-8)) if n else 0.0,
        "p95_abs_error": _safe_quantile(abs_errors, 0.95, 20),
        "p99_abs_error": _safe_quantile(abs_errors, 0.99, 100),
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "true_std": true_std,
        "pred_std": pred_std,
        "compression_ratio": compression_ratio,
        "regression_available": regression_available,
        "sufficient_true_variance": sufficient_variance,
        "regression_to_mean_warning": regression_to_mean_warning,
    }


def per_target_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_columns: list[str],
    target_scales: dict[str, float],
    *,
    split_name: str,
) -> dict[str, Any]:
    return {
        "split": split_name,
        "targets": {
            target: _target_diagnostics(
                y_true[:, index],
                y_pred[:, index],
                target_scale=float(target_scales.get(target, 1.0)),
            )
            for index, target in enumerate(target_columns)
        },
    }


def regime_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rows: list[dict[str, Any]],
    indices: np.ndarray,
    target_columns: list[str],
    target_scales: dict[str, float],
) -> dict[str, Any]:
    regime_column = discover_regime_column(rows)
    if regime_column is None:
        return {"regime_column": None, "regimes": {}}

    grouped: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        grouped[str(rows[int(index)].get(regime_column, ""))].append(int(index))

    regimes: dict[str, Any] = {}
    for label, label_indices in sorted(grouped.items()):
        local = np.asarray(label_indices, dtype=np.int64)
        regimes[label] = per_target_diagnostics(
            y_true[local],
            y_pred[local],
            target_columns,
            target_scales,
            split_name=f"{regime_column}:{label}",
        )
        regimes[label]["n"] = int(len(local))
    return {"regime_column": regime_column, "regimes": regimes}


def _wrapped_angle_error_deg(true_angle: np.ndarray, pred_angle: np.ndarray) -> np.ndarray:
    error = np.rad2deg(pred_angle - true_angle)
    return (error + 180.0) % 360.0 - 180.0


def _linear_fit_diagnostics(true_values: np.ndarray, pred_values: np.ndarray) -> dict[str, Any]:
    if len(true_values) < 30 or float(np.std(true_values)) <= 1e-8:
        return {"slope": None, "r2": None, "available": False}
    slope, intercept = np.polyfit(true_values, pred_values, 1)
    residual = pred_values - (slope * true_values + intercept)
    centered = true_values - np.mean(true_values)
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum(centered**2))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else None,
        "available": True,
    }


def vector_scale_from_training(
    y_true: np.ndarray,
    train_index: np.ndarray,
    target_columns: list[str],
    vector_pairs: dict[str, tuple[str, str]] | None = None,
) -> dict[str, dict[str, Any]]:
    vector_pairs = vector_pairs or VECTOR_TARGET_PAIRS
    scales: dict[str, dict[str, Any]] = {}
    for pair_name, (x_name, y_name) in vector_pairs.items():
        x_index = target_columns.index(x_name)
        y_index = target_columns.index(y_name)
        true_x = y_true[train_index, x_index]
        true_y = y_true[train_index, y_index]
        magnitude = np.sqrt(true_x**2 + true_y**2)
        scale = float(np.quantile(magnitude, 0.95)) if len(magnitude) else 0.0
        scales[pair_name] = {
            "vector_scale": max(scale, 1e-8),
            "scale_source": "training_split_true_magnitude_p95",
            "training_samples": int(len(magnitude)),
        }
    return scales


def _vector_pair_diagnostics(
    true_x: np.ndarray,
    true_y: np.ndarray,
    pred_x: np.ndarray,
    pred_y: np.ndarray,
    *,
    vector_scale: float,
    scale_source: str,
) -> dict[str, Any]:
    error_x = pred_x - true_x
    error_y = pred_y - true_y
    true_magnitude = np.sqrt(true_x**2 + true_y**2)
    pred_magnitude = np.sqrt(pred_x**2 + pred_y**2)
    magnitude_error = pred_magnitude - true_magnitude
    fit = _linear_fit_diagnostics(true_magnitude, pred_magnitude)
    compression_ratio = float(np.std(pred_magnitude) / (np.std(true_magnitude) + 1e-8))

    true_angle = np.arctan2(true_y, true_x)
    pred_angle = np.arctan2(pred_y, pred_x)
    angle_threshold = 0.1 * max(float(vector_scale), 1e-8)
    angle_mask = true_magnitude > angle_threshold
    angle_error = _wrapped_angle_error_deg(true_angle[angle_mask], pred_angle[angle_mask])
    abs_angle_error = np.abs(angle_error)

    cosine_mask = (true_magnitude > 1e-8) & (pred_magnitude > 1e-8)
    cosine = (
        (true_x[cosine_mask] * pred_x[cosine_mask] + true_y[cosine_mask] * pred_y[cosine_mask])
        / (true_magnitude[cosine_mask] * pred_magnitude[cosine_mask] + 1e-8)
    )

    return {
        "n": int(len(true_x)),
        "component_errors": {
            "MAE_x": float(np.mean(np.abs(error_x))) if len(error_x) else 0.0,
            "MAE_y": float(np.mean(np.abs(error_y))) if len(error_y) else 0.0,
            "RMSE_x": float(np.sqrt(np.mean(error_x**2))) if len(error_x) else 0.0,
            "RMSE_y": float(np.sqrt(np.mean(error_y**2))) if len(error_y) else 0.0,
        },
        "magnitude": {
            "vector_scale": float(vector_scale),
            "scale_source": scale_source,
            "magnitude_mae": float(np.mean(np.abs(magnitude_error))) if len(magnitude_error) else 0.0,
            "magnitude_rmse": float(np.sqrt(np.mean(magnitude_error**2))) if len(magnitude_error) else 0.0,
            "magnitude_bias": float(np.mean(magnitude_error)) if len(magnitude_error) else 0.0,
            "magnitude_slope": fit["slope"],
            "magnitude_r2": fit["r2"],
            "magnitude_fit_available": fit["available"],
            "magnitude_compression_ratio": compression_ratio,
        },
        "angle": {
            "angle_threshold": float(angle_threshold),
            "angle_defined_count": int(np.sum(angle_mask)),
            "angle_undefined_near_zero_count": int(np.sum(~angle_mask)),
            "mean_abs_angle_error_deg": float(np.mean(abs_angle_error)) if len(abs_angle_error) else None,
            "median_abs_angle_error_deg": float(np.median(abs_angle_error)) if len(abs_angle_error) else None,
            "p95_abs_angle_error_deg": float(np.quantile(abs_angle_error, 0.95)) if len(abs_angle_error) >= 20 else None,
            "p95_reliable": bool(len(abs_angle_error) >= 20),
        },
        "directional_cosine": {
            "cosine_defined_count": int(len(cosine)),
            "cosine_undefined_zero_vector_count": int(len(true_x) - len(cosine)),
            "mean_cosine_similarity": float(np.mean(cosine)) if len(cosine) else None,
            "median_cosine_similarity": float(np.median(cosine)) if len(cosine) else None,
            "p05_cosine_similarity": float(np.quantile(cosine, 0.05)) if len(cosine) >= 20 else None,
            "p05_reliable": bool(len(cosine) >= 20),
        },
    }


def vector_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rows: list[dict[str, Any]],
    target_columns: list[str],
    split_indices: dict[str, np.ndarray],
    *,
    vector_pairs: dict[str, tuple[str, str]] | None = None,
) -> dict[str, Any]:
    vector_pairs = vector_pairs or VECTOR_TARGET_PAIRS
    train_index = split_indices["train"]
    validation_index = split_indices["validation"]
    scales = vector_scale_from_training(y_true, train_index, target_columns, vector_pairs)

    def pair_for_indices(pair_name: str, indices: np.ndarray) -> dict[str, Any]:
        x_name, y_name = vector_pairs[pair_name]
        x_index = target_columns.index(x_name)
        y_index = target_columns.index(y_name)
        scale_data = scales[pair_name]
        return _vector_pair_diagnostics(
            y_true[indices, x_index],
            y_true[indices, y_index],
            y_pred[indices, x_index],
            y_pred[indices, y_index],
            vector_scale=float(scale_data["vector_scale"]),
            scale_source=str(scale_data["scale_source"]),
        )

    payload: dict[str, Any] = {
        "split": "validation",
        "vector_pairs": {pair_name: pair_for_indices(pair_name, validation_index) for pair_name in vector_pairs},
        "scale_policy": {
            "angle_threshold": "true_magnitude > 0.1 * vector_scale",
            "vector_scale": "95th percentile of true vector magnitude on training split",
        },
        "by_regime": {},
    }

    regime_column = "sweep_label" if any(str(row.get("sweep_label", "")).strip() for row in rows) else discover_regime_column(rows)
    payload["by_regime"]["regime_column"] = regime_column
    payload["by_regime"]["pairs"] = {}
    if regime_column is not None:
        validation_labels = np.asarray([str(rows[int(index)].get(regime_column, "")) for index in validation_index])
        for pair_name in ("B2", "S3", "A3"):
            regimes: dict[str, Any] = {}
            for label in sorted(set(validation_labels)):
                local_indices = validation_index[validation_labels == label]
                if len(local_indices) == 0:
                    continue
                regimes[label] = pair_for_indices(pair_name, local_indices)
            payload["by_regime"]["pairs"][pair_name] = regimes
    return payload
