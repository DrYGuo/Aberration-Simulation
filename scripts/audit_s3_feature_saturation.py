"""Audit whether current enhanced features saturate for high-S3 targets."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from feature_regression_model import TARGET_COLUMNS, file_sha256, target_from_row


REQUESTED_BUT_UNAVAILABLE_FEATURES = [
    "inner radial-band m=2 harmonic",
    "middle radial-band m=2 harmonic",
    "outer radial-band m=2 harmonic",
    "r^4-weighted m=2 moment",
    "r^6-weighted m=2 moment",
    "contour/radius r50 angular anisotropy",
    "contour/radius r80 angular anisotropy",
    "contour/radius r95 angular anisotropy",
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open() as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def find_latest_csv(search_root: Path) -> Path:
    matches = sorted(
        search_root.glob("**/training_features_enhanced.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    if not matches:
        raise FileNotFoundError(f"No training_features_enhanced.csv found under {search_root}")
    return matches[-1]


def f(row: dict[str, str], name: str) -> float:
    value = row.get(name, "")
    if value in ("", None):
        return float("nan")
    return float(value)


def complex_magnitude(rows: list[dict[str, str]], real_name: str, imag_name: str) -> np.ndarray | None:
    if not rows or real_name not in rows[0] or imag_name not in rows[0]:
        return None
    real = np.asarray([f(row, real_name) for row in rows], dtype=float)
    imag = np.asarray([f(row, imag_name) for row in rows], dtype=float)
    return np.hypot(real, imag)


def complex_difference_magnitude(
    rows: list[dict[str, str]],
    left_real: str,
    left_imag: str,
    right_real: str,
    right_imag: str,
) -> np.ndarray | None:
    required = {left_real, left_imag, right_real, right_imag}
    if not rows or not required.issubset(rows[0]):
        return None
    left_r = np.asarray([f(row, left_real) for row in rows], dtype=float)
    left_i = np.asarray([f(row, left_imag) for row in rows], dtype=float)
    right_r = np.asarray([f(row, right_real) for row in rows], dtype=float)
    right_i = np.asarray([f(row, right_imag) for row in rows], dtype=float)
    return np.hypot(left_r - right_r, left_i - right_i)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def linear_stats(x: np.ndarray, y: np.ndarray) -> dict[str, float | int | None]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or float(np.std(x)) <= 1e-12:
        return {"n": int(len(x)), "slope": None, "intercept": None, "r2": None}
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else None
    return {"n": int(len(x)), "slope": float(slope), "intercept": float(intercept), "r2": None if r2 is None else float(r2)}


def summarize_feature(true_s3_mag: np.ndarray, feature: np.ndarray, high_threshold: float) -> dict[str, Any]:
    mask = np.isfinite(feature)
    high_mask = mask & (true_s3_mag > high_threshold)
    low_mask = mask & (true_s3_mag <= high_threshold)
    ranks_x = rankdata(true_s3_mag[mask]) if int(np.sum(mask)) else np.asarray([], dtype=float)
    ranks_y = rankdata(feature[mask]) if int(np.sum(mask)) else np.asarray([], dtype=float)
    high_stats = linear_stats(true_s3_mag[high_mask], feature[high_mask])
    overall_stats = linear_stats(true_s3_mag[mask], feature[mask])
    return {
        "n": int(np.sum(mask)),
        "pearson_corr": pearson(true_s3_mag[mask], feature[mask]) if int(np.sum(mask)) else None,
        "spearman_corr": pearson(ranks_x, ranks_y) if len(ranks_x) else None,
        "linear_feature_vs_true_s3_overall": overall_stats,
        "linear_feature_vs_true_s3_high_s3": high_stats,
        "high_s3_threshold": float(high_threshold),
        "mean_feature_low_or_medium_s3": float(np.nanmean(feature[low_mask])) if int(np.sum(low_mask)) else None,
        "mean_feature_high_s3": float(np.nanmean(feature[high_mask])) if int(np.sum(high_mask)) else None,
        "std_feature_high_s3": float(np.nanstd(feature[high_mask])) if int(np.sum(high_mask)) else None,
        "high_to_low_mean_ratio": (
            float(np.nanmean(feature[high_mask]) / (np.nanmean(feature[low_mask]) + 1e-8))
            if int(np.sum(high_mask)) and int(np.sum(low_mask))
            else None
        ),
    }


def build_features(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    candidates: dict[str, np.ndarray] = {}
    for name, value in [
        ("Eq43_S3_value_magnitude", complex_magnitude(rows, "S3_value_real", "S3_value_imag")),
        ("under_Rho_h2_magnitude", complex_magnitude(rows, "under_Rho_h2_real", "under_Rho_h2_imag")),
        ("over_Rho_h2_magnitude", complex_magnitude(rows, "over_Rho_h2_real", "over_Rho_h2_imag")),
        (
            "under_minus_over_Rho_h2_magnitude",
            complex_difference_magnitude(
                rows,
                "under_Rho_h2_real",
                "under_Rho_h2_imag",
                "over_Rho_h2_real",
                "over_Rho_h2_imag",
            ),
        ),
        ("under_Xigma_h2_magnitude", complex_magnitude(rows, "under_Xigma_h2_real", "under_Xigma_h2_imag")),
        ("over_Xigma_h2_magnitude", complex_magnitude(rows, "over_Xigma_h2_real", "over_Xigma_h2_imag")),
        (
            "under_minus_over_Xigma_h2_magnitude",
            complex_difference_magnitude(
                rows,
                "under_Xigma_h2_real",
                "under_Xigma_h2_imag",
                "over_Xigma_h2_real",
                "over_Xigma_h2_imag",
            ),
        ),
        ("under_Mu_h2_magnitude", complex_magnitude(rows, "under_Mu_h2_real", "under_Mu_h2_imag")),
        ("over_Mu_h2_magnitude", complex_magnitude(rows, "over_Mu_h2_real", "over_Mu_h2_imag")),
        (
            "under_minus_over_Mu_h2_magnitude",
            complex_difference_magnitude(
                rows,
                "under_Mu_h2_real",
                "under_Mu_h2_imag",
                "over_Mu_h2_real",
                "over_Mu_h2_imag",
            ),
        ),
    ]:
        if value is not None:
            candidates[name] = value
    return candidates


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# S3 Feature Saturation Audit",
        "",
        f"- CSV: `{report['csv_path']}`",
        f"- Rows: {report['n_rows']}",
        f"- SHA256: `{report['csv_sha256']}`",
        f"- S3 vector scale: {report['s3_vector_scale']:.6g}",
        f"- High-S3 threshold: {report['high_s3_threshold']:.6g}",
        f"- High-S3 rows: {report['n_high_s3']}",
        "",
        "## Available Current Features",
        "",
        "| feature | n | Pearson | Spearman | high-S3 slope | high-S3 R2 | high/low mean ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, stats in report["feature_stats"].items():
        high = stats["linear_feature_vs_true_s3_high_s3"]
        lines.append(
            "| {name} | {n} | {pearson} | {spearman} | {slope} | {r2} | {ratio} |".format(
                name=name,
                n=stats["n"],
                pearson=format_optional(stats["pearson_corr"]),
                spearman=format_optional(stats["spearman_corr"]),
                slope=format_optional(high["slope"]),
                r2=format_optional(high["r2"]),
                ratio=format_optional(stats["high_to_low_mean_ratio"]),
            )
        )
    lines.extend(
        [
            "",
            "## Requested Features Not Present In This CSV",
            "",
        ]
    )
    for name in report["requested_but_unavailable_features"]:
        lines.append(f"- {name}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            report["interpretation"],
            "",
        ]
    )
    path.write_text("\n".join(lines))


def format_optional(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4g}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", type=Path)
    parser.add_argument("--search-root", type=Path, default=Path("training_results/feature_regression_enhanced"))
    parser.add_argument("--output-root", type=Path, default=Path("training_results/model_selection_reports"))
    parser.add_argument("--vector-scale", type=float, default=90.73979949951172)
    parser.add_argument("--high-bin-fraction", type=float, default=0.7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = args.csv_path or find_latest_csv(args.search_root)
    csv_path = csv_path.resolve()
    rows, fieldnames = read_rows(csv_path)
    if not rows:
        raise RuntimeError(f"CSV is empty: {csv_path}")
    targets = np.asarray([target_from_row(row) for row in rows], dtype=float)
    s3_x_index = TARGET_COLUMNS.index("S3_x")
    s3_y_index = TARGET_COLUMNS.index("S3_y")
    true_s3_mag = np.hypot(targets[:, s3_x_index], targets[:, s3_y_index])
    high_threshold = float(args.high_bin_fraction * args.vector_scale)
    features = build_features(rows)
    feature_stats = {
        name: summarize_feature(true_s3_mag, values, high_threshold) for name, values in sorted(features.items())
    }
    high_slopes = [
        stats["linear_feature_vs_true_s3_high_s3"]["slope"]
        for stats in feature_stats.values()
        if stats["linear_feature_vs_true_s3_high_s3"]["slope"] is not None
    ]
    if not feature_stats:
        interpretation = (
            "No S3-sensitive current-feature candidates were found in the CSV. The next step should be feature "
            "engineering before interpreting data expansion as sufficient."
        )
    elif high_slopes and max(abs(float(value)) for value in high_slopes) < 1e-3:
        interpretation = (
            "Available current features show nearly flat high-S3 linear response. This supports feature saturation "
            "as a likely bottleneck."
        )
    else:
        interpretation = (
            "This audit is diagnostic only. If high-S3 model slope remains compressed after v5 data expansion, compare "
            "the high-S3 feature slopes here against new radial-band or radially weighted m=2 descriptors before "
            "expanding directly to 100k rows."
        )

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "csv_path": str(csv_path),
        "csv_sha256": file_sha256(csv_path),
        "n_rows": int(len(rows)),
        "field_count": int(len(fieldnames)),
        "s3_vector_scale": float(args.vector_scale),
        "high_bin_fraction": float(args.high_bin_fraction),
        "high_s3_threshold": high_threshold,
        "n_high_s3": int(np.sum(true_s3_mag > high_threshold)),
        "feature_stats": feature_stats,
        "requested_but_unavailable_features": REQUESTED_BUT_UNAVAILABLE_FEATURES,
        "interpretation": interpretation,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    json_path = args.output_root / f"s3_feature_saturation_audit_{stamp}.json"
    md_path = args.output_root / f"s3_feature_saturation_audit_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(md_path, report)
    print("wrote", md_path, flush=True)
    print("wrote", json_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
