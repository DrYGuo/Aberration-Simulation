"""Generate the orthogonal_hard_benchmark_v1 diagnostic design.

This creates a compact coefficient design table only. It does not simulate
features, does not create training data, and does not affect primary
validation/model-selection scoring.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


VECTOR_ORDERS = {"A1": 2, "B2": 1, "A2": 3, "S3": 2, "A3": 4}
VECTOR_NAMES = ("A1", "B2", "A2", "S3", "A3")
TRUE_HARD_TARGETS = ("C1", "S3_x", "S3_y", "A3_x", "A3_y")
DEFAULT_CONFIG = Path("configs/orthogonal_hard_benchmark_v1.yaml")


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


def load_config(path: Path) -> dict[str, Any]:
    # The .yaml file is intentionally JSON-compatible YAML, so no PyYAML
    # dependency is needed in Colab.
    return json.loads(path.read_text())


def vector_xy(magnitude: float, angle_deg: float) -> tuple[float, float]:
    angle = math.radians(angle_deg)
    return magnitude * math.cos(angle), magnitude * math.sin(angle)


def phase_from_vector_angle(vector_name: str, angle_deg: float) -> float:
    order = VECTOR_ORDERS[vector_name]
    return (angle_deg / order) % 360.0


def empty_row(benchmark_id: str, design_type: str, family: str, row_index: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "benchmark_id": benchmark_id,
        "design_type": design_type,
        "diagnostic_only": True,
        "train_on_this": False,
        "selection_primary_metric": False,
        "split": benchmark_id,
        "sweep_label": benchmark_id,
        "coupling_family": family,
        "case_index": row_index,
        "C1": 0.0,
        "C3": 0.0,
        "relative_angle_category": "not_applicable",
        "relative_A1_S3_deg": "",
        "relative_B2_S3_deg": "",
        "relative_A3_S3_deg": "",
    }
    for name in VECTOR_NAMES:
        row[f"{name}_magnitude"] = 0.0
        row[f"{name}_vector_angle_deg"] = ""
        row[f"{name}_x"] = 0.0
        row[f"{name}_y"] = 0.0
        row[f"{name}_amp"] = 0.0
        row[f"{name}_phase"] = 0.0
    return row


def set_vector(row: dict[str, Any], name: str, magnitude: float, angle_deg: float) -> None:
    x, y = vector_xy(magnitude, angle_deg)
    row[f"{name}_magnitude"] = float(magnitude)
    row[f"{name}_vector_angle_deg"] = float(angle_deg % 360.0)
    row[f"{name}_x"] = float(x)
    row[f"{name}_y"] = float(y)
    row[f"{name}_amp"] = float(magnitude)
    row[f"{name}_phase"] = float(phase_from_vector_angle(name, angle_deg))


def cycle_value(values: list[Any], index: int, stride: int = 1) -> Any:
    return values[(index // stride) % len(values)]


def row_for_family(
    *,
    benchmark_id: str,
    design_type: str,
    family: str,
    local_index: int,
    global_index: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    c1_levels = list(config["c1_levels"])
    base_angles = list(config["base_angle_degrees"])
    rel_items = list(config["relative_angle_categories"].items())
    mags = config["vector_magnitude_levels"]

    row = empty_row(benchmark_id, design_type, family, global_index)
    c1_active = family.startswith("C1_")
    if c1_active:
        row["C1"] = float(cycle_value(c1_levels, local_index))

    base_angle = float(cycle_value(base_angles, local_index, stride=max(1, len(c1_levels) if c1_active else 1)))
    rel_name, rel_deg = rel_items[local_index % len(rel_items)]
    row["relative_angle_category"] = rel_name

    def mag(name: str, offset: int = 0) -> float:
        levels = list(mags[name])
        return float(levels[((local_index // max(1, len(rel_items))) + offset) % len(levels)])

    if family == "C1_S3":
        set_vector(row, "S3", mag("S3"), base_angle)
    elif family == "C1_A3":
        set_vector(row, "A3", mag("A3"), base_angle)
    elif family == "C1_S3_A3":
        set_vector(row, "S3", mag("S3"), base_angle)
        set_vector(row, "A3", mag("A3", 1), base_angle + float(rel_deg))
        row["relative_A3_S3_deg"] = float(rel_deg)
    elif family == "A1_S3":
        set_vector(row, "S3", mag("S3"), base_angle)
        set_vector(row, "A1", mag("A1"), base_angle + float(rel_deg))
        row["relative_A1_S3_deg"] = float(rel_deg)
    elif family == "B2_S3":
        set_vector(row, "S3", mag("S3"), base_angle)
        set_vector(row, "B2", mag("B2"), base_angle + float(rel_deg))
        row["relative_B2_S3_deg"] = float(rel_deg)
    elif family == "A3_S3":
        set_vector(row, "S3", mag("S3"), base_angle)
        set_vector(row, "A3", mag("A3"), base_angle + float(rel_deg))
        row["relative_A3_S3_deg"] = float(rel_deg)
    elif family == "A1_B2_S3":
        rel2_name, rel2_deg = rel_items[(local_index // len(rel_items)) % len(rel_items)]
        row["relative_angle_category"] = f"A1_{rel_name}__B2_{rel2_name}"
        set_vector(row, "S3", mag("S3"), base_angle)
        set_vector(row, "A1", mag("A1"), base_angle + float(rel_deg))
        set_vector(row, "B2", mag("B2"), base_angle + float(rel2_deg))
        row["relative_A1_S3_deg"] = float(rel_deg)
        row["relative_B2_S3_deg"] = float(rel2_deg)
    else:
        raise KeyError(f"unknown coupling family: {family}")
    return row


def generate_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    benchmark_id = str(config["benchmark_id"])
    design_type = str(config["design_type"])
    rows: list[dict[str, Any]] = []
    global_index = 0
    for family, count in config["coupling_families"].items():
        for local_index in range(int(count)):
            rows.append(
                row_for_family(
                    benchmark_id=benchmark_id,
                    design_type=design_type,
                    family=family,
                    local_index=local_index,
                    global_index=global_index,
                    config=config,
                )
            )
            global_index += 1
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denom_x = math.sqrt(sum(x * x for x in dx))
    denom_y = math.sqrt(sum(y * y for y in dy))
    if denom_x == 0.0 or denom_y == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(dx, dy)) / (denom_x * denom_y)


def numeric_factor_table(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    table: dict[str, list[float]] = {
        "C1": [],
        "A1_magnitude": [],
        "B2_magnitude": [],
        "S3_magnitude": [],
        "A3_magnitude": [],
        "A1_active": [],
        "B2_active": [],
        "S3_active": [],
        "A3_active": [],
        "relative_A1_S3_deg": [],
        "relative_B2_S3_deg": [],
        "relative_A3_S3_deg": [],
    }
    for row in rows:
        for key in ("C1", "A1_magnitude", "B2_magnitude", "S3_magnitude", "A3_magnitude"):
            table[key].append(float(row[key]))
        for name in ("A1", "B2", "S3", "A3"):
            table[f"{name}_active"].append(1.0 if float(row[f"{name}_magnitude"]) > 0 else 0.0)
        for key in ("relative_A1_S3_deg", "relative_B2_S3_deg", "relative_A3_S3_deg"):
            table[key].append(float(row[key]) if row[key] != "" else 0.0)
    return table


def write_correlation_matrix(path: Path, rows: list[dict[str, Any]]) -> float:
    table = numeric_factor_table(rows)
    names = list(table)
    max_abs = 0.0
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["factor", *names])
        for left in names:
            values = []
            for right in names:
                corr = pearson(table[left], table[right])
                if left != right:
                    max_abs = max(max_abs, abs(corr))
                values.append(corr)
            writer.writerow([left, *values])
    return max_abs


def factor_balance_summary(rows: list[dict[str, Any]], max_abs_corr: float) -> dict[str, Any]:
    level_counts: dict[str, Any] = {
        "coupling_family": dict(Counter(str(row["coupling_family"]) for row in rows)),
        "C1": dict(Counter(str(row["C1"]) for row in rows)),
        "relative_angle_category": dict(Counter(str(row["relative_angle_category"]) for row in rows)),
    }
    relative_angle_coverage = {
        key: dict(Counter(str(row[key]) for row in rows if row[key] != ""))
        for key in ("relative_A1_S3_deg", "relative_B2_S3_deg", "relative_A3_S3_deg")
    }
    zero_magnitude_counts = {
        name: sum(1 for row in rows if float(row[f"{name}_magnitude"]) == 0.0)
        for name in VECTOR_NAMES
    }
    nonzero_orientation_counts = {
        name: sum(1 for row in rows if float(row[f"{name}_magnitude"]) > 0.0)
        for name in VECTOR_NAMES
    }
    return {
        "design_type": str(rows[0]["design_type"]) if rows else "unknown",
        "row_count": len(rows),
        "level_counts": level_counts,
        "relative_angle_coverage": relative_angle_coverage,
        "coupling_family_counts": level_counts["coupling_family"],
        "zero_magnitude_vector_cases": zero_magnitude_counts,
        "nonzero_vector_orientation_cases": nonzero_orientation_counts,
        "maximum_absolute_pairwise_correlation": max_abs_corr,
        "design_quality_warning": bool(max_abs_corr > 0.70),
        "design_quality_warning_reason": (
            "High correlations can occur in this fractional diagnostic design because coupling-family activation is intentionally structured."
            if max_abs_corr > 0.70
            else ""
        ),
    }


def plot_summaries(output_dir: Path, rows: list[dict[str, Any]]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"plotting skipped: {exc}", flush=True)
        return []

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    family_counts = Counter(str(row["coupling_family"]) for row in rows)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(list(family_counts), list(family_counts.values()))
    ax.set_ylabel("rows")
    ax.set_title("Coupling family counts")
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    path = plot_dir / "coupling_family_counts.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for name in ("A1", "B2", "S3", "A3"):
        values = [float(row[f"{name}_magnitude"]) for row in rows if float(row[f"{name}_magnitude"]) > 0.0]
        ax.hist(values, bins=8, alpha=0.45, label=name)
    ax.set_xlabel("nonzero magnitude")
    ax.set_ylabel("count")
    ax.legend()
    ax.set_title("Vector magnitude coverage")
    fig.tight_layout()
    path = plot_dir / "vector_magnitude_coverage.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    paths.append(str(path))

    rel_counts = Counter(
        str(row[key])
        for row in rows
        for key in ("relative_A1_S3_deg", "relative_B2_S3_deg", "relative_A3_S3_deg")
        if row[key] != ""
    )
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(list(rel_counts), list(rel_counts.values()))
    ax.set_xlabel("relative angle deg")
    ax.set_ylabel("count")
    ax.set_title("Relative angle coverage")
    fig.tight_layout()
    path = plot_dir / "relative_angle_coverage.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    paths.append(str(path))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=Path("training_results/orthogonal_benchmarks"))
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd()
    config = load_config(args.config)
    rows = generate_rows(config)
    benchmark_id = str(config["benchmark_id"])
    run_name = f"{benchmark_id}_{utc_stamp()}"
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    table_path = output_dir / "orthogonal_design_table.csv"
    fieldnames = list(rows[0])
    write_csv(table_path, rows, fieldnames)
    max_abs_corr = write_correlation_matrix(output_dir / "factor_correlation_matrix.csv", rows)
    summary = factor_balance_summary(rows, max_abs_corr)
    summary_path = output_dir / "factor_balance_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    plot_paths = [] if args.skip_plots else plot_summaries(output_dir, rows)

    manifest = {
        "benchmark_id": benchmark_id,
        "run_name": run_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_commit(repo_root),
        "python": sys.version,
        "platform": platform.platform(),
        "config_path": str(args.config),
        "design_type": config["design_type"],
        "exact_orthogonal": False,
        "train_on_this": False,
        "selection_primary_metric": False,
        "diagnostic_only": True,
        "row_count": len(rows),
        "output_dir": str(output_dir),
        "design_table_path": str(table_path),
        "factor_balance_summary_path": str(summary_path),
        "factor_correlation_matrix_path": str(output_dir / "factor_correlation_matrix.csv"),
        "plot_paths": plot_paths,
        "artifact_policy": "compact diagnostic design only; no simulated feature CSVs, raw predictions, checkpoints, or archives",
        "future_evaluation_hook": "Use regression_diagnostics.orthogonal_benchmark_metrics with model predictions on this benchmark split.",
    }
    (output_dir / "orthogonal_design_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("orthogonal benchmark:", output_dir, flush=True)
    print("rows:", len(rows), flush=True)
    print("max_abs_pairwise_correlation:", max_abs_corr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
