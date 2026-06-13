"""Audit C1 sensitivity of selected feature columns in an enhanced CSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


EPS = 1e-8
C1_SCALE = 100.0
C1_BINS = {
    "near_zero": (0.0, 10.0),
    "low": (10.0, 40.0),
    "medium": (40.0, 75.0),
    "high": (75.0, math.inf),
}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def load_feature_columns(source_csv: Path) -> list[str]:
    for name in ("feature_columns_enhanced.json", "feature_columns.json"):
        path = source_csv.parent / name
        if path.exists():
            data = json.loads(path.read_text())
            if isinstance(data, dict) and "features" in data:
                return list(data["features"])
            return list(data)
    raise FileNotFoundError(f"No feature_columns*.json found beside {source_csv}")


def row_float(row: dict[str, str], name: str, default: float = 0.0) -> float:
    value = row.get(name, default)
    if value in (None, ""):
        return float(default)
    return float(value)


def linear_stats(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 3 or float(np.std(x)) < EPS or float(np.std(y)) < EPS:
        return {
            "n": int(len(x)),
            "pearson_r": None,
            "slope_feature_vs_c1": None,
            "intercept": None,
            "r2": None,
        }
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    corr = float(np.corrcoef(x, y)[0, 1])
    return {
        "n": int(len(x)),
        "pearson_r": corr,
        "slope_feature_vs_c1": float(slope),
        "intercept": float(intercept),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > EPS else None,
    }


def selected_features(feature_columns: list[str], rows: list[dict[str, str]], max_features: int) -> list[str]:
    preferred_terms = (
        "C1_value_real",
        "Cdf",
        "Xigma",
        "defocus_Xigma",
        "defocus_Mu",
        "defocus_Rho",
        "under_Xigma",
        "over_Xigma",
    )
    candidates = [
        feature
        for feature in feature_columns
        if feature in rows[0] and any(term in feature for term in preferred_terms)
    ]
    if len(candidates) <= max_features:
        return candidates

    c1 = np.asarray([row_float(row, "C1") for row in rows], dtype=float)
    scored = []
    for feature in candidates:
        values = np.asarray([row_float(row, feature) for row in rows], dtype=float)
        stats = linear_stats(c1, values)
        score = abs(stats["pearson_r"]) if stats["pearson_r"] is not None else -1.0
        scored.append((score, feature))
    return [feature for _, feature in sorted(scored, reverse=True)[:max_features]]


def bin_name(c1: float) -> str:
    value = abs(c1)
    for name, (low, high) in C1_BINS.items():
        if low <= value < high:
            return name
    return "high"


def feature_table(
    rows: list[dict[str, str]],
    features: list[str],
    *,
    label_filter: str | None = None,
    bin_filter: str | None = None,
) -> list[dict[str, Any]]:
    selected_rows = rows
    if label_filter is not None:
        selected_rows = [row for row in selected_rows if str(row.get("sweep_label", "")) == label_filter]
    if bin_filter is not None:
        selected_rows = [row for row in selected_rows if bin_name(row_float(row, "C1")) == bin_filter]
    c1 = np.asarray([row_float(row, "C1") for row in selected_rows], dtype=float)
    records = []
    for feature in features:
        values = np.asarray([row_float(row, feature) for row in selected_rows], dtype=float)
        stats = linear_stats(c1, values)
        records.append({"feature": feature, **stats})
    return sorted(
        records,
        key=lambda item: -1.0 if item["pearson_r"] is None else -abs(float(item["pearson_r"])),
        reverse=True,
    )


def plot_top_features(output_dir: Path, rows: list[dict[str, str]], records: list[dict[str, Any]], top_n: int) -> list[str]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    c1 = np.asarray([row_float(row, "C1") for row in rows], dtype=float)
    paths = []
    for record in records[:top_n]:
        feature = str(record["feature"])
        values = np.asarray([row_float(row, feature) for row in rows], dtype=float)
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        ax.scatter(c1, values, s=6, alpha=0.35)
        ax.set_xlabel("C1")
        ax.set_ylabel(feature)
        r = record.get("pearson_r")
        r_text = "n/a" if r is None else f"{float(r):.3f}"
        ax.set_title(f"{feature}\nr={r_text}", fontsize=8)
        fig.tight_layout()
        path = plot_dir / f"{feature.replace('/', '_')}_vs_C1.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(str(path))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("training_results/model_selection_reports"))
    parser.add_argument("--run-label", default="")
    parser.add_argument("--max-features", type=int, default=80)
    parser.add_argument("--top-plots", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_csv(args.csv_path)
    if not rows:
        raise RuntimeError(f"CSV is empty: {args.csv_path}")
    feature_columns = load_feature_columns(args.csv_path)
    features = selected_features(feature_columns, rows, args.max_features)
    stamp = utc_stamp()
    label = args.run_label or args.csv_path.parent.name
    output_dir = args.output_root / f"c1_feature_sensitivity_{label}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    overall = feature_table(rows, features)
    by_bin = {name: feature_table(rows, features, bin_filter=name) for name in C1_BINS}
    labels = sorted({str(row.get("sweep_label", "")) for row in rows})
    c1_labels = [label for label in labels if "C1" in label or label in {"coupled_full_random", "coupled_sparse_random"}]
    by_label = {label: feature_table(rows, features, label_filter=label) for label in c1_labels}
    plot_paths = plot_top_features(output_dir, rows, overall, args.top_plots)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "csv_path": str(args.csv_path),
        "run_label": label,
        "n_rows": len(rows),
        "feature_count_total": len(feature_columns),
        "feature_count_audited": len(features),
        "c1_scale": C1_SCALE,
        "c1_bins": {
            name: {"low_abs_c1": low, "high_abs_c1": None if math.isinf(high) else high}
            for name, (low, high) in C1_BINS.items()
        },
        "overall_top_features": overall[:20],
        "by_c1_magnitude_bin": {name: records[:20] for name, records in by_bin.items()},
        "by_coupling_label": {name: records[:12] for name, records in by_label.items()},
        "plot_paths": plot_paths,
    }
    json_path = output_dir / "c1_feature_sensitivity.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")

    csv_path = output_dir / "c1_feature_sensitivity_overall.csv"
    with csv_path.open("w", newline="") as handle:
        fieldnames = ["feature", "n", "pearson_r", "slope_feature_vs_c1", "intercept", "r2"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(overall)

    md_path = args.output_root / f"c1_feature_sensitivity_{label}_{stamp}.md"
    lines = [
        f"# C1 Feature Sensitivity - {label}",
        "",
        f"- CSV: `{args.csv_path}`",
        f"- Rows: `{len(rows)}`",
        f"- Audited features: `{len(features)}`",
        "",
        "## Top Overall Features",
        "",
        "| feature | n | pearson r | slope | r2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in overall[:15]:
        r = item["pearson_r"]
        slope = item["slope_feature_vs_c1"]
        r2 = item["r2"]
        lines.append(
            f"| `{item['feature']}` | {item['n']} | "
            f"{'n/a' if r is None else f'{r:.4f}'} | "
            f"{'n/a' if slope is None else f'{slope:.6g}'} | "
            f"{'n/a' if r2 is None else f'{r2:.4f}'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Large absolute correlations or slopes indicate C1-sensitive features. "
            "If explicit defocus-difference features do not rank above the existing "
            "collapsed C1/defocus features, C1 error is less likely to be fixed by "
            "adding more rows alone.",
            "",
            f"Detailed JSON: `{json_path}`",
            f"Overall CSV: `{csv_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n")
    print("C1 sensitivity report:", md_path, flush=True)
    print("C1 sensitivity JSON:", json_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
