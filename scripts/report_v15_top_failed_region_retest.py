#!/usr/bin/env python3
"""Report v15 errors on the exact v13 active top-failure rows.

The active failed-region report summarized v13 top active-hole failures
(`active_hole_search_top_failures.csv` from each active-search run). The v15
active-hole retest writes full per-run probe comparison tables to Drive. This
script joins those tables by `(source_run, active_probe_id)` and writes compact
v13-vs-v15 metrics for the exact top-failure subset.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any


ERROR_COLUMNS = (
    "weighted_abs_error",
    "overall_abs_error",
    "C1_abs_error",
    "C3_abs_error",
    "A1_vector_error",
    "B2_vector_error",
    "A2_vector_error",
    "S3_vector_error",
    "A3_vector_error",
)

DISPLAY = {
    "weighted_abs_error": ("weighted normalized abs", ""),
    "overall_abs_error": ("overall mixed-unit abs", ""),
    "C1_abs_error": ("C1", "nm"),
    "C3_abs_error": ("C3", "mm"),
    "A1_vector_error": ("A1 vector", "nm"),
    "B2_vector_error": ("B2 vector", "um"),
    "A2_vector_error": ("A2 vector", "um"),
    "S3_vector_error": ("S3 vector", "um"),
    "A3_vector_error": ("A3 vector", "um"),
}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def as_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize(values: list[float]) -> dict[str, Any]:
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return {"n": 0, "mae": None, "rmse": None, "median": None, "p95": None, "max": None}
    values_sorted = sorted(values)
    n = len(values)

    def quantile(frac: float) -> float:
        return values_sorted[min(n - 1, max(0, int(round(frac * (n - 1)))))]

    return {
        "n": n,
        "mae": sum(values) / n,
        "rmse": math.sqrt(sum(value * value for value in values) / n),
        "median": quantile(0.50),
        "p95": quantile(0.95),
        "max": values_sorted[-1],
    }


def discover_active_run_dirs(active_root: Path) -> list[Path]:
    return sorted(
        path
        for path in active_root.glob("v13_active_12d_hole_search*")
        if (path / "active_hole_search_top_failures.csv").exists()
    )


def load_top_failure_keys(active_root: Path) -> tuple[list[dict[str, str]], set[tuple[str, str]]]:
    rows: list[dict[str, str]] = []
    keys: set[tuple[str, str]] = set()
    for run_dir in discover_active_run_dirs(active_root):
        for row in read_csv_rows(run_dir / "active_hole_search_top_failures.csv"):
            row = dict(row)
            row["source_run"] = run_dir.name
            probe_id = str(row.get("active_probe_id", ""))
            if probe_id:
                keys.add((run_dir.name, probe_id))
            rows.append(row)
    return rows, keys


def load_retest_rows(retest_dir: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    for path in sorted(retest_dir.glob("v13_active_*/active_hole_retest_probe_comparison.csv")):
        source_run = path.parent.name
        for row in read_csv_rows(path):
            probe_id = str(row.get("active_probe_id", ""))
            if probe_id:
                rows[(source_run, probe_id)] = row
    return rows


def metric_rows(joined: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for column in ERROR_COLUMNS:
        label, unit = DISPLAY[column]
        v13_values = [value for row in joined if (value := as_float(row.get(f"v13_{column}"))) is not None]
        v15_values = [value for row in joined if (value := as_float(row.get(f"v15_{column}"))) is not None]
        deltas = [value for row in joined if (value := as_float(row.get(f"delta_{column}"))) is not None]
        v13 = summarize(v13_values)
        v15 = summarize(v15_values)
        delta = summarize(deltas)
        output.append(
            {
                "metric": label,
                "unit": unit,
                "n": min(v13["n"], v15["n"]),
                "v13_mae": v13["mae"],
                "v13_rmse": v13["rmse"],
                "v13_p95": v13["p95"],
                "v15_mae": v15["mae"],
                "v15_rmse": v15["rmse"],
                "v15_p95": v15["p95"],
                "delta_mae": None if v13["mae"] is None or v15["mae"] is None else v15["mae"] - v13["mae"],
                "delta_rmse": None if v13["rmse"] is None or v15["rmse"] is None else v15["rmse"] - v13["rmse"],
                "relative_rmse_change": None
                if v13["rmse"] in (None, 0)
                else (float(v15["rmse"]) - float(v13["rmse"])) / float(v13["rmse"]),
                "worse_fraction": None if not deltas else sum(1 for value in deltas if value > 0.0) / len(deltas),
                "delta_median": delta["median"],
                "delta_p95": delta["p95"],
            }
        )
    return output


def group_summary(joined: list[dict[str, str]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in joined:
        groups.setdefault(str(row.get(key, "")), []).append(row)
    rows: list[dict[str, Any]] = []
    for label, group_rows in sorted(groups.items()):
        v13 = [value for row in group_rows if (value := as_float(row.get("v13_weighted_abs_error"))) is not None]
        v15 = [value for row in group_rows if (value := as_float(row.get("v15_weighted_abs_error"))) is not None]
        if not v13 or not v15:
            continue
        v13_s = summarize(v13)
        v15_s = summarize(v15)
        rows.append(
            {
                key: label,
                "n": len(group_rows),
                "v13_weighted_rmse": v13_s["rmse"],
                "v15_weighted_rmse": v15_s["rmse"],
                "delta_weighted_rmse": float(v15_s["rmse"]) - float(v13_s["rmse"]),
                "relative_weighted_rmse_change": (float(v15_s["rmse"]) - float(v13_s["rmse"])) / float(v13_s["rmse"]),
            }
        )
    rows.sort(key=lambda row: float(row["v15_weighted_rmse"]), reverse=True)
    return rows


def write_report(path: Path, summary: dict[str, Any], metrics: list[dict[str, Any]]) -> None:
    lines = [
        "# V15 Retest On V13 Top Active-Hole Failures",
        "",
        f"Created UTC: `{summary['created_utc']}`",
        "",
        "Purpose: compare v13 and v15 on the exact rows selected as v13 active-hole top failures.",
        "",
        f"- active top-failure rows requested: `{summary['n_top_failure_rows']}`",
        f"- matched retest rows: `{summary['n_matched_rows']}`",
        f"- missing retest rows: `{summary['n_missing_rows']}`",
        f"- source retest dir: `{summary['retest_dir']}`",
        "",
        "| metric | unit | v13 RMSE | v15 RMSE | relative change | worse fraction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        rel = row["relative_rmse_change"]
        worse = row["worse_fraction"]
        lines.append(
            f"| {row['metric']} | {row['unit']} | {row['v13_rmse']} | {row['v15_rmse']} | "
            f"{'' if rel is None else rel} | {'' if worse is None else worse} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- Negative relative change means v15 improved the v13 top-failure subset.",
            "- This table is a harder population than the full active-hole probe set.",
            "- Promotion still requires broad benchmark and anchor gates, not only this repair score.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-root", type=Path, default=Path("training_results/model_selection_reports"))
    parser.add_argument("--retest-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("training_results/model_selection_reports"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    top_rows, keys = load_top_failure_keys(args.active_root)
    if not keys:
        raise RuntimeError(f"No top-failure rows found under {args.active_root}")
    retest_by_key = load_retest_rows(args.retest_dir)
    if not retest_by_key:
        raise RuntimeError(
            f"No full retest comparison CSVs found under {args.retest_dir}. "
            "Restore v15_active_hole_retest_latest from Drive first."
        )
    joined: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    top_by_key = {(row["source_run"], row.get("active_probe_id", "")): row for row in top_rows}
    for key in sorted(keys):
        retest = retest_by_key.get(key)
        if retest is None:
            source = top_by_key.get(key, {})
            missing.append({"source_run": key[0], "active_probe_id": key[1], "sweep_label": source.get("sweep_label", "")})
            continue
        source = top_by_key.get(key, {})
        merged = dict(source)
        merged.update(retest)
        merged["source_run"] = key[0]
        joined.append(merged)

    output_dir = args.output_root / f"v15_top_failed_region_retest_{utc_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = metric_rows(joined)
    regime_rows = group_summary(joined, "sweep_label")
    class_rows = group_summary(joined, "v15_failure_class")
    summary = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "active_root": str(args.active_root),
        "retest_dir": str(args.retest_dir),
        "n_top_failure_rows": len(keys),
        "n_matched_rows": len(joined),
        "n_missing_rows": len(missing),
        "source_run_counts": dict(Counter(row["source_run"] for row in joined)),
        "sweep_label_counts": dict(Counter(row.get("sweep_label", "") for row in joined)),
        "files": {
            "report": str(output_dir / "v15_top_failed_region_retest_report.md"),
            "metrics": str(output_dir / "v15_top_failed_region_retest_metrics.csv"),
            "by_regime": str(output_dir / "v15_top_failed_region_retest_by_regime.csv"),
            "by_failure_class": str(output_dir / "v15_top_failed_region_retest_by_failure_class.csv"),
            "missing": str(output_dir / "v15_top_failed_region_retest_missing_rows.csv"),
        },
    }
    (output_dir / "v15_top_failed_region_retest_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(
        output_dir / "v15_top_failed_region_retest_metrics.csv",
        metrics,
        [
            "metric",
            "unit",
            "n",
            "v13_mae",
            "v13_rmse",
            "v13_p95",
            "v15_mae",
            "v15_rmse",
            "v15_p95",
            "delta_mae",
            "delta_rmse",
            "relative_rmse_change",
            "worse_fraction",
            "delta_median",
            "delta_p95",
        ],
    )
    write_csv(
        output_dir / "v15_top_failed_region_retest_by_regime.csv",
        regime_rows,
        ["sweep_label", "n", "v13_weighted_rmse", "v15_weighted_rmse", "delta_weighted_rmse", "relative_weighted_rmse_change"],
    )
    write_csv(
        output_dir / "v15_top_failed_region_retest_by_failure_class.csv",
        class_rows,
        ["v15_failure_class", "n", "v13_weighted_rmse", "v15_weighted_rmse", "delta_weighted_rmse", "relative_weighted_rmse_change"],
    )
    write_csv(output_dir / "v15_top_failed_region_retest_missing_rows.csv", missing, ["source_run", "active_probe_id", "sweep_label"])
    write_report(output_dir / "v15_top_failed_region_retest_report.md", summary, metrics)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
