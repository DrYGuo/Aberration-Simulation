"""Compact diagnostics for aberration coefficient regression results."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


REGIME_COLUMN_CANDIDATES = ("regime_name", "regime", "sweep_label")


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
