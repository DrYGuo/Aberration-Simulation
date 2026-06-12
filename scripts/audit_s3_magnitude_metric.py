"""Audit S3 magnitude-slope diagnostics from model-selection artifacts.

The compact GitHub artifact policy does not push raw predictions or model
checkpoints. This script therefore has two modes:

1. If a run folder contains ``validation_predictions_s3_audit.npz`` with
   arrays ``y_true`` and ``y_pred`` plus optional ``target_columns``, it computes
   the full requested slope/component/residual audit and plots.
2. Otherwise it performs a limited audit from ``vector_diagnostics.json`` and
   related compact artifacts, and explicitly reports that the slope cannot be
   recomputed from raw saved predictions.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any


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


DEFAULT_RUNS = {
    "old_v3_bin_diag": "training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_bin_diag_20260609_073514_utc",
    "v3_smoothl1_champion": "training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_plateau_clip_smoothl1_20260610_071108_utc",
    "v5_s3tail60k_seed23": "training_results/model_selection_loop/D66_grouped_width320_lr6e-4_dropout0.075_s3tail60k_plateau_clip_smoothl1_seed23_20260611_090007_utc",
}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def wrapped_angle_error_deg(true_angle: np.ndarray, pred_angle: np.ndarray) -> np.ndarray:
    import numpy as np

    error = np.rad2deg(pred_angle - true_angle)
    return (error + 180.0) % 360.0 - 180.0


def fit_stats(true_values: np.ndarray, pred_values: np.ndarray) -> dict[str, Any]:
    import numpy as np

    true_values = np.asarray(true_values, dtype=float)
    pred_values = np.asarray(pred_values, dtype=float)
    mask = np.isfinite(true_values) & np.isfinite(pred_values)
    true_values = true_values[mask]
    pred_values = pred_values[mask]
    n = int(len(true_values))
    if n == 0:
        return {
            "n": 0,
            "ols_slope": None,
            "ols_intercept": None,
            "through_origin_slope": None,
            "correlation": None,
            "r2": None,
            "mae": None,
            "rmse": None,
            "bias": None,
            "median_error": None,
            "p95_abs_error": None,
            "true_range": None,
            "pred_range": None,
        }
    error = pred_values - true_values
    ols_slope = None
    ols_intercept = None
    r2 = None
    correlation = None
    if n >= 2 and float(np.std(true_values)) > 1e-12:
        ols_slope, ols_intercept = np.polyfit(true_values, pred_values, 1)
        fitted = ols_slope * true_values + ols_intercept
        ss_res = float(np.sum((pred_values - fitted) ** 2))
        ss_tot = float(np.sum((pred_values - np.mean(pred_values)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else None
        correlation = float(np.corrcoef(true_values, pred_values)[0, 1]) if float(np.std(pred_values)) > 1e-12 else None
    denom = float(np.sum(true_values**2))
    through_origin_slope = float(np.sum(true_values * pred_values) / denom) if denom > 1e-12 else None
    return {
        "n": n,
        "ols_slope": None if ols_slope is None else float(ols_slope),
        "ols_intercept": None if ols_intercept is None else float(ols_intercept),
        "through_origin_slope": through_origin_slope,
        "correlation": correlation,
        "r2": r2,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "median_error": float(np.median(error)),
        "p95_abs_error": float(np.quantile(np.abs(error), 0.95)) if n >= 2 else float(np.abs(error[0])),
        "true_range": [float(np.min(true_values)), float(np.max(true_values))],
        "pred_range": [float(np.min(pred_values)), float(np.max(pred_values))],
    }


def load_prediction_npz(run_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    path = run_dir / "validation_predictions_s3_audit.npz"
    if not path.exists():
        return None
    import numpy as np

    data = np.load(path, allow_pickle=True)
    y_true = np.asarray(data["y_true"], dtype=float)
    y_pred = np.asarray(data["y_pred"], dtype=float)
    target_columns = [str(value) for value in data["target_columns"]] if "target_columns" in data else TARGET_COLUMNS
    return y_true, y_pred, target_columns


def load_audit_csv(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "s3_magnitude_metric_audit_validation.csv"
    if not path.exists():
        return None
    import numpy as np

    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None

    def column(name: str) -> np.ndarray:
        return np.asarray([float(row[name]) for row in rows], dtype=float)

    return {
        "path": str(path),
        "n": len(rows),
        "true_x": column("true_s3_x"),
        "true_y": column("true_s3_y"),
        "pred_x": column("pred_s3_x"),
        "pred_y": column("pred_s3_y"),
        "true_mag": column("true_s3_magnitude"),
        "pred_mag": column("pred_s3_magnitude"),
        "magnitude_error": column("magnitude_error"),
        "vector_residual": column("vector_residual_magnitude"),
        "angle_error_deg": column("angle_error_deg"),
        "high_mask": np.asarray([str(row["high_s3_bin"]).lower() == "true" for row in rows], dtype=bool),
        "vector_scale": float(rows[0]["vector_scale"]),
    }


def plot_audit(
    output_dir: Path,
    run_key: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_columns: list[str],
    high_mask: np.ndarray,
) -> list[str]:
    import matplotlib.pyplot as plt
    import numpy as np

    s3x = target_columns.index("S3_x")
    s3y = target_columns.index("S3_y")
    true_x = y_true[:, s3x]
    true_y = y_true[:, s3y]
    pred_x = y_pred[:, s3x]
    pred_y = y_pred[:, s3y]
    true_mag = np.hypot(true_x, true_y)
    pred_mag = np.hypot(pred_x, pred_y)
    mag_error = pred_mag - true_mag
    vector_residual = np.hypot(pred_x - true_x, pred_y - true_y)
    angle_error = wrapped_angle_error_deg(np.arctan2(true_y, true_x), np.arctan2(pred_y, pred_x))
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []

    def scatter(path: Path, x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str, identity: bool = False) -> None:
        fig, ax = plt.subplots(figsize=(4.2, 3.5))
        ax.scatter(x, y, s=8, alpha=0.5)
        if identity and len(x) and len(y):
            low = float(min(np.min(x), np.min(y)))
            high = float(max(np.max(x), np.max(y)))
            ax.plot([low, high], [low, high], "k--", linewidth=0.8)
        else:
            ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        outputs.append(str(path))

    scatter(plot_dir / f"{run_key}_S3_x_pred_vs_true.png", true_x, pred_x, "true S3_x", "pred S3_x", f"{run_key}: S3_x", True)
    scatter(plot_dir / f"{run_key}_S3_y_pred_vs_true.png", true_y, pred_y, "true S3_y", "pred S3_y", f"{run_key}: S3_y", True)
    scatter(plot_dir / f"{run_key}_S3_mag_pred_vs_true_all.png", true_mag, pred_mag, "true |S3|", "pred |S3|", f"{run_key}: |S3| all", True)
    scatter(
        plot_dir / f"{run_key}_S3_mag_pred_vs_true_high.png",
        true_mag[high_mask],
        pred_mag[high_mask],
        "true |S3|",
        "pred |S3|",
        f"{run_key}: |S3| high bin",
        True,
    )
    scatter(plot_dir / f"{run_key}_S3_mag_residual_vs_true_mag.png", true_mag, mag_error, "true |S3|", "pred |S3| - true |S3|", f"{run_key}: magnitude residual")
    scatter(plot_dir / f"{run_key}_S3_vector_residual_vs_true_mag.png", true_mag, vector_residual, "true |S3|", "vector residual magnitude", f"{run_key}: vector residual")
    scatter(plot_dir / f"{run_key}_S3_angle_error_vs_true_mag.png", true_mag, angle_error, "true |S3|", "angle error deg", f"{run_key}: angle error")
    return outputs


def plot_audit_from_csv(output_dir: Path, run_key: str, audit: dict[str, Any]) -> list[str]:
    import matplotlib.pyplot as plt
    import numpy as np

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []

    def scatter(path: Path, x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str, identity: bool = False) -> None:
        fig, ax = plt.subplots(figsize=(4.2, 3.5))
        ax.scatter(x, y, s=8, alpha=0.5)
        if identity and len(x) and len(y):
            low = float(min(np.min(x), np.min(y)))
            high = float(max(np.max(x), np.max(y)))
            ax.plot([low, high], [low, high], "k--", linewidth=0.8)
        else:
            ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        outputs.append(str(path))

    high_mask = audit["high_mask"]
    scatter(plot_dir / f"{run_key}_S3_x_pred_vs_true.png", audit["true_x"], audit["pred_x"], "true S3_x", "pred S3_x", f"{run_key}: S3_x", True)
    scatter(plot_dir / f"{run_key}_S3_y_pred_vs_true.png", audit["true_y"], audit["pred_y"], "true S3_y", "pred S3_y", f"{run_key}: S3_y", True)
    scatter(plot_dir / f"{run_key}_S3_mag_pred_vs_true_all.png", audit["true_mag"], audit["pred_mag"], "true |S3|", "pred |S3|", f"{run_key}: |S3| all", True)
    scatter(plot_dir / f"{run_key}_S3_mag_pred_vs_true_high.png", audit["true_mag"][high_mask], audit["pred_mag"][high_mask], "true |S3|", "pred |S3|", f"{run_key}: |S3| high bin", True)
    scatter(plot_dir / f"{run_key}_S3_mag_residual_vs_true_mag.png", audit["true_mag"], audit["magnitude_error"], "true |S3|", "pred |S3| - true |S3|", f"{run_key}: magnitude residual")
    scatter(plot_dir / f"{run_key}_S3_vector_residual_vs_true_mag.png", audit["true_mag"], audit["vector_residual"], "true |S3|", "vector residual magnitude", f"{run_key}: vector residual")
    scatter(plot_dir / f"{run_key}_S3_angle_error_vs_true_mag.png", audit["true_mag"], audit["angle_error_deg"], "true |S3|", "angle error deg", f"{run_key}: angle error")
    return outputs


def copy_existing_s3_plots(run_dir: Path, output_dir: Path, run_key: str) -> list[str]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    source_dir = run_dir / "plots" / "vector_diagnostics"
    for name in [
        "S3_pred_vs_true_magnitude.png",
        "S3_angle_error_vs_true_magnitude.png",
        "S3_pred_vs_true_angle.png",
    ]:
        source = source_dir / name
        if not source.exists():
            continue
        destination = plot_dir / f"{run_key}_{name}"
        shutil.copy2(source, destination)
        copied.append(str(destination))
    return copied


def audit_run(run_key: str, run_dir: Path, output_dir: Path) -> dict[str, Any]:
    vector = load_json(run_dir / "vector_diagnostics.json")
    manifest = load_json(run_dir / "run_manifest_model_loop.json")
    selection = load_json(run_dir / "selection_score.json")
    audit_csv = load_audit_csv(run_dir)
    prediction_data = load_prediction_npz(run_dir)
    result: dict[str, Any] = {
        "run_key": run_key,
        "run_dir": str(run_dir),
        "files_read": [
            str(path)
            for path in [
                run_dir / "vector_diagnostics.json",
                run_dir / "run_manifest_model_loop.json",
                run_dir / "selection_score.json",
            ]
            if path.exists()
        ],
        "has_raw_validation_predictions": prediction_data is not None,
        "has_s3_audit_csv": audit_csv is not None,
        "weighted_score": selection.get("weighted_score"),
        "dataset": manifest.get("dataset", {}),
        "existing_plots_copied": copy_existing_s3_plots(run_dir, output_dir, run_key),
    }

    s3 = vector.get("vector_pairs", {}).get("S3", {})
    s3_high = s3.get("magnitude_bins", {}).get("bins", {}).get("high", {})
    result["stored_vector_diagnostics"] = {
        "vector_scale": s3.get("magnitude", {}).get("vector_scale"),
        "scale_source": s3.get("magnitude", {}).get("scale_source"),
        "overall_magnitude": s3.get("magnitude", {}),
        "overall_angle": s3.get("angle", {}),
        "high_bin": s3_high,
    }

    if prediction_data is None:
        if audit_csv is not None:
            import numpy as np

            high_mask = audit_csv["high_mask"]
            angle_error_high = np.abs(audit_csv["angle_error_deg"][high_mask])
            result["audit_mode"] = "s3_audit_csv_recompute"
            result["files_read"].append(str(audit_csv["path"]))
            result["computed"] = {
                "formula": "true_mag = sqrt(S3_x^2 + S3_y^2); pred_mag = sqrt(pred_S3_x^2 + pred_S3_y^2)",
                "high_bin": "true_mag > 0.7 * vector_scale",
                "vector_scale": audit_csv["vector_scale"],
                "all_magnitude": fit_stats(audit_csv["true_mag"], audit_csv["pred_mag"]),
                "high_magnitude": fit_stats(audit_csv["true_mag"][high_mask], audit_csv["pred_mag"][high_mask]),
                "component_slopes": {
                    "S3_x_all": fit_stats(audit_csv["true_x"], audit_csv["pred_x"]),
                    "S3_y_all": fit_stats(audit_csv["true_y"], audit_csv["pred_y"]),
                    "S3_x_high": fit_stats(audit_csv["true_x"][high_mask], audit_csv["pred_x"][high_mask]),
                    "S3_y_high": fit_stats(audit_csv["true_y"][high_mask], audit_csv["pred_y"][high_mask]),
                },
                "high_angle": {
                    "mean_abs_angle_error_deg": float(np.mean(angle_error_high)) if len(angle_error_high) else None,
                    "p95_abs_angle_error_deg": float(np.quantile(angle_error_high, 0.95)) if len(angle_error_high) >= 2 else None,
                    "n": int(len(angle_error_high)),
                },
            }
            result["generated_plots"] = plot_audit_from_csv(output_dir, run_key, audit_csv)
            return result
        result["audit_mode"] = "compact_artifact_limited"
        result["limitation"] = (
            "Raw y_true/y_pred validation predictions are not present in the compact GitHub artifacts, "
            "so through-origin slope, component slopes, residual plots, and exact recomputation from saved "
            "predictions cannot be performed for this local audit."
        )
        return result

    y_true, y_pred, target_columns = prediction_data
    s3x = target_columns.index("S3_x")
    s3y = target_columns.index("S3_y")
    true_x = y_true[:, s3x]
    true_y = y_true[:, s3y]
    pred_x = y_pred[:, s3x]
    pred_y = y_pred[:, s3y]
    true_mag = np.hypot(true_x, true_y)
    pred_mag = np.hypot(pred_x, pred_y)
    vector_scale = float(s3.get("magnitude", {}).get("vector_scale") or np.quantile(true_mag, 0.95))
    high_mask = true_mag > 0.7 * vector_scale
    angle_error = np.abs(wrapped_angle_error_deg(np.arctan2(true_y[high_mask], true_x[high_mask]), np.arctan2(pred_y[high_mask], pred_x[high_mask])))

    result["audit_mode"] = "full_prediction_recompute"
    result["computed"] = {
        "formula": "true_mag = sqrt(S3_x^2 + S3_y^2); pred_mag = sqrt(pred_S3_x^2 + pred_S3_y^2)",
        "high_bin": "true_mag > 0.7 * vector_scale",
        "vector_scale": vector_scale,
        "all_magnitude": fit_stats(true_mag, pred_mag),
        "high_magnitude": fit_stats(true_mag[high_mask], pred_mag[high_mask]),
        "component_slopes": {
            "S3_x_all": fit_stats(true_x, pred_x),
            "S3_y_all": fit_stats(true_y, pred_y),
            "S3_x_high": fit_stats(true_x[high_mask], pred_x[high_mask]),
            "S3_y_high": fit_stats(true_y[high_mask], pred_y[high_mask]),
        },
        "high_angle": {
            "mean_abs_angle_error_deg": float(np.mean(angle_error)) if len(angle_error) else None,
            "p95_abs_angle_error_deg": float(np.quantile(angle_error, 0.95)) if len(angle_error) >= 2 else None,
            "n": int(len(angle_error)),
        },
    }
    result["generated_plots"] = plot_audit(output_dir, run_key, y_true, y_pred, target_columns, high_mask)
    return result


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return str(value)
    if digits == 0:
        try:
            return str(int(round(float(value))))
        except (TypeError, ValueError):
            return str(value)
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    full_runs = [run for run in payload["runs"] if run.get("computed")]
    compact_only_runs = [run for run in payload["runs"] if not run.get("computed")]
    limitation_lines = []
    if compact_only_runs:
        limitation_lines.append(
            "Compact-only historical runs do not include raw predictions or checkpoints, so intercepts, "
            "through-origin slopes, component slopes, and residual plots cannot be recomputed for those runs."
        )
    if full_runs:
        limitation_lines.append(
            "Runs with `s3_magnitude_metric_audit_validation.csv` are fully recomputed below, including "
            "OLS intercepts, through-origin slopes, component slopes, and residual plots."
        )
    if not limitation_lines:
        limitation_lines.append(payload["artifact_limitation"])

    lines = [
        "# S3 Magnitude Metric Audit",
        "",
        f"Created UTC: {payload['created_utc']}",
        "",
        "## Formula And Code-Path Check",
        "",
        "- `true_mag = sqrt(true_S3_x^2 + true_S3_y^2)`.",
        "- `pred_mag = sqrt(pred_S3_x^2 + pred_S3_y^2)`.",
        "- High-S3 bin is selected by true magnitude: `true_mag > 0.7 * vector_scale`.",
        "- `vector_scale` is the 95th percentile of true vector magnitude on the training split.",
        "- The model runner inverse-transforms normalized predictions once with `y_scaler.inverse_transform(pred_scaled)` before diagnostics.",
        "- `S3_x` and `S3_y` are adjacent target columns from the same physical target vector and use the same coefficient units.",
        "",
        "## Artifact Limitation",
        "",
        " ".join(limitation_lines),
        "",
        "## Stored Metric Comparison",
        "",
        "| run | has raw predictions | has S3 audit CSV | weighted score | high n | stored high OLS slope | stored high MAE | stored high bias | stored high RMSE | high angle mean deg | high angle p95 deg |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in payload["runs"]:
        high = run["stored_vector_diagnostics"].get("high_bin", {})
        mag = high.get("magnitude", {})
        angle = high.get("angle", {})
        lines.append(
            "| {run} | {raw} | {csv} | {score} | {n} | {slope} | {mae} | {bias} | {rmse} | {angle_mean} | {angle_p95} |".format(
                run=run["run_key"],
                raw=run["has_raw_validation_predictions"],
                csv=run.get("has_s3_audit_csv", False),
                score=fmt(run.get("weighted_score")),
                n=fmt(high.get("n"), 0),
                slope=fmt(mag.get("magnitude_slope")),
                mae=fmt(mag.get("magnitude_mae")),
                bias=fmt(mag.get("magnitude_bias")),
                rmse=fmt(mag.get("magnitude_rmse")),
                angle_mean=fmt(angle.get("mean_abs_angle_error_deg")),
                angle_p95=fmt(angle.get("p95_abs_angle_error_deg")),
            )
        )
    lines.extend(["", "## Full Recompute Tables", ""])
    if not full_runs:
        lines.append("No run folder contained `validation_predictions_s3_audit.npz` or `s3_magnitude_metric_audit_validation.csv`, so through-origin slopes and component slopes could not be recomputed from saved predictions in this local audit.")
    else:
        lines.append("### Magnitude")
        lines.append("")
        lines.append("| run | subset | OLS slope | intercept | through-origin slope | corr | R2 | MAE | bias | RMSE | n |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for run in full_runs:
            computed = run["computed"]
            for subset in ["all_magnitude", "high_magnitude"]:
                stats = computed[subset]
                lines.append(
                    f"| {run['run_key']} | {subset} | {fmt(stats['ols_slope'])} | {fmt(stats['ols_intercept'])} | {fmt(stats['through_origin_slope'])} | {fmt(stats['correlation'])} | {fmt(stats['r2'])} | {fmt(stats['mae'])} | {fmt(stats['bias'])} | {fmt(stats['rmse'])} | {fmt(stats['n'], 0)} |"
                )
        lines.extend(["", "### Components", ""])
        lines.append("| run | component subset | OLS slope | intercept | through-origin slope | corr | R2 | MAE | bias | RMSE | n |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for run in full_runs:
            for subset, stats in run["computed"].get("component_slopes", {}).items():
                lines.append(
                    f"| {run['run_key']} | {subset} | {fmt(stats['ols_slope'])} | {fmt(stats['ols_intercept'])} | {fmt(stats['through_origin_slope'])} | {fmt(stats['correlation'])} | {fmt(stats['r2'])} | {fmt(stats['mae'])} | {fmt(stats['bias'])} | {fmt(stats['rmse'])} | {fmt(stats['n'], 0)} |"
                )
    lines.extend(["", "## Plots", ""])
    for run in payload["runs"]:
        copied = run.get("existing_plots_copied", [])
        generated = run.get("generated_plots", [])
        lines.append(f"### {run['run_key']}")
        if copied:
            lines.append("Copied existing compact plots:")
            for plot in copied:
                lines.append(f"- `{plot}`")
        if generated:
            lines.append("Generated full audit plots:")
            for plot in generated:
                lines.append(f"- `{plot}`")
        if not copied and not generated:
            lines.append("- No plots available.")
        lines.append("")
    lines.extend(
        [
            "## Answers To Main Questions",
            "",
            "A. The stored high-S3 magnitude slope near `0.71` is present in `vector_diagnostics.json` and the batch summary for the v5 seed23 run.",
            "",
            "B. Code inspection shows it is an OLS-with-intercept slope from `np.polyfit(true_magnitude, pred_magnitude, 1)`. The intercept is recovered for runs with `s3_magnitude_metric_audit_validation.csv`; compact-only historical runs cannot provide it.",
            "",
            "C. The through-origin slope is computed when `s3_magnitude_metric_audit_validation.csv` or raw validation predictions are available; otherwise it cannot be recovered from historical compact artifacts.",
            "",
            "D. Component slopes for `S3_x` and `S3_y` are computed from the S3 audit CSV when present. For older compact-only runs, existing component scatter plots are images only.",
            "",
            "E. A low OLS slope can be affected by high-bin range restriction and nonzero intercept. Runs with the S3 audit CSV provide enough data to distinguish OLS-with-intercept behavior from through-origin/component slopes.",
            "",
            "F. Whether magnitude bias becomes more negative at larger true `|S3|` requires residual-vs-true-magnitude data. This is available for runs with the S3 audit CSV and unavailable for compact-only historical runs.",
            "",
            "G. Feature-bottleneck conclusions should be based on the full recompute rows when available, not on the high-bin OLS slope alone.",
            "",
            "## Recommendation",
            "",
            "Use the recomputed S3 audit CSV metrics when deciding whether S3 magnitude-loss candidates improve high-S3 magnitude MAE and bias without damaging component slopes or blind/stress performance.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("training_results/model_selection_reports"))
    parser.add_argument("--include-default-runs", action="store_true", default=True)
    parser.add_argument("--run", action="append", default=[], help="Run entry as key=path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stamp = utc_stamp()
    output_dir = args.output_root / f"s3_magnitude_metric_audit_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs: dict[str, str] = dict(DEFAULT_RUNS if args.include_default_runs else {})
    for item in args.run:
        key, sep, path = item.partition("=")
        if not sep:
            raise ValueError(f"--run must be key=path, got {item!r}")
        runs[key] = path
    run_payloads = [audit_run(key, Path(path), output_dir) for key, path in runs.items()]
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "runs": run_payloads,
        "artifact_limitation": (
            "The current GitHub artifact policy excludes raw predictions and model checkpoints. "
            "This report therefore verifies the stored metric and code path, but cannot recompute "
            "the 0.71 slope, through-origin slope, component slopes, or intercept from saved raw predictions."
        ),
    }
    json_path = output_dir / "s3_magnitude_metric_audit.json"
    md_path = args.output_root / f"s3_magnitude_metric_audit_{stamp}.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    write_markdown(md_path, payload)
    print("wrote", md_path)
    print("wrote", json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
