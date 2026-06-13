"""Create enhanced-feature CSV variants with explicit defocus-difference features.

This is a no-simulation transform. It reads an existing
``training_features_enhanced.csv`` and writes a new cached feature table with
additional columns derived from the under/over focus Xigma, Mu, and Rho summary
features. The large CSV remains a Colab-local artifact; only manifests and
small summaries should be pushed.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

from feature_regression_model import file_sha256


EPS = 1e-8
CHARS = ("Xigma", "Mu", "Rho")
HARMONIC_ORDERS = (1, 2, 3, 4)


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


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open() as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def load_feature_columns(source_csv: Path) -> list[str]:
    for name in ("feature_columns_enhanced.json", "feature_columns.json"):
        path = source_csv.parent / name
        if path.exists():
            data = json.loads(path.read_text())
            if isinstance(data, dict) and "features" in data:
                return list(data["features"])
            return list(data)
    raise FileNotFoundError(f"No feature_columns*.json found beside {source_csv}")


def row_float(row: dict[str, Any], name: str, default: float = 0.0) -> float:
    value = row.get(name, default)
    if value in (None, ""):
        return float(default)
    return float(value)


def add_feature(row: dict[str, Any], name: str, value: float, feature_columns: list[str]) -> None:
    row[name] = float(value)
    if name not in feature_columns:
        feature_columns.append(name)


def add_mean_features(row: dict[str, Any], char_name: str, feature_columns: list[str]) -> None:
    under = row_float(row, f"under_{char_name}_mean")
    over = row_float(row, f"over_{char_name}_mean")
    denom = abs(under) + abs(over) + EPS
    add_feature(row, f"defocus_{char_name}_mean_under_minus_over", under - over, feature_columns)
    add_feature(row, f"defocus_{char_name}_mean_over_minus_under", over - under, feature_columns)
    add_feature(row, f"defocus_{char_name}_mean_sum", under + over, feature_columns)
    add_feature(row, f"defocus_{char_name}_mean_norm_under_minus_over", (under - over) / denom, feature_columns)
    add_feature(row, f"defocus_{char_name}_mean_norm_over_minus_under", (over - under) / denom, feature_columns)


def add_harmonic_features(row: dict[str, Any], char_name: str, order: int, feature_columns: list[str]) -> None:
    prefix = f"defocus_{char_name}_h{order}"
    ur = row_float(row, f"under_{char_name}_h{order}_real")
    ui = row_float(row, f"under_{char_name}_h{order}_imag")
    or_ = row_float(row, f"over_{char_name}_h{order}_real")
    oi = row_float(row, f"over_{char_name}_h{order}_imag")
    diff_r = ur - or_
    diff_i = ui - oi
    sum_r = ur + or_
    sum_i = ui + oi
    under_mag = (ur * ur + ui * ui) ** 0.5
    over_mag = (or_ * or_ + oi * oi) ** 0.5
    diff_mag = (diff_r * diff_r + diff_i * diff_i) ** 0.5
    sum_mag = (sum_r * sum_r + sum_i * sum_i) ** 0.5
    denom = under_mag + over_mag + EPS

    add_feature(row, f"{prefix}_under_minus_over_real", diff_r, feature_columns)
    add_feature(row, f"{prefix}_under_minus_over_imag", diff_i, feature_columns)
    add_feature(row, f"{prefix}_under_plus_over_real", sum_r, feature_columns)
    add_feature(row, f"{prefix}_under_plus_over_imag", sum_i, feature_columns)
    add_feature(row, f"{prefix}_under_minus_over_magnitude", diff_mag, feature_columns)
    add_feature(row, f"{prefix}_under_plus_over_magnitude", sum_mag, feature_columns)
    add_feature(row, f"{prefix}_norm_under_minus_over_real", diff_r / denom, feature_columns)
    add_feature(row, f"{prefix}_norm_under_minus_over_imag", diff_i / denom, feature_columns)
    add_feature(row, f"{prefix}_norm_under_minus_over_magnitude", diff_mag / denom, feature_columns)


def augment_rows(
    rows: list[dict[str, str]],
    source_fieldnames: list[str],
    source_features: list[str],
    *,
    mode: str,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    output_rows: list[dict[str, Any]] = [dict(row) for row in rows]
    output_fieldnames = list(source_fieldnames)
    feature_columns = list(source_features)
    new_features_before = set(feature_columns)

    include_harmonics = mode in {"c1_defocus_full", "defocus_full"}
    for row in output_rows:
        for char_name in CHARS:
            add_mean_features(row, char_name, feature_columns)
            if include_harmonics:
                for order in HARMONIC_ORDERS:
                    add_harmonic_features(row, char_name, order, feature_columns)

    for feature in feature_columns:
        if feature not in output_fieldnames:
            output_fieldnames.append(feature)
    new_features = [feature for feature in feature_columns if feature not in new_features_before]
    return output_rows, output_fieldnames, new_features


def write_label_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    counts = Counter(str(row.get("sweep_label", "")) for row in rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sweep_label", "n_rows"])
        writer.writeheader()
        for label, count in sorted(counts.items()):
            writer.writerow({"sweep_label": label, "n_rows": count})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("training_results/feature_regression_enhanced"))
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument(
        "--mode",
        choices=["c1_defocus_basic", "c1_defocus_full", "defocus_full"],
        default="c1_defocus_basic",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd()
    source_csv = args.source_csv.resolve()
    rows, fieldnames = read_csv(source_csv)
    if not rows:
        raise RuntimeError(f"source CSV is empty: {source_csv}")
    source_features = load_feature_columns(source_csv)
    output_rows, output_fieldnames, new_features = augment_rows(
        rows,
        fieldnames,
        source_features,
        mode=args.mode,
    )

    run_name = f"{args.run_prefix}_{utc_stamp()}"
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "training_features_enhanced.csv"
    write_csv(output_csv, output_rows, output_fieldnames)
    feature_columns_path = output_dir / "feature_columns_enhanced.json"
    feature_columns = [*source_features, *new_features]
    feature_columns_path.write_text(json.dumps(feature_columns, indent=2) + "\n")
    write_label_summary(output_dir / "label_summary.csv", output_rows)

    summary_path = output_dir / "feature_augmentation_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature_name", "feature_group"])
        writer.writeheader()
        for feature in new_features:
            group = "mean" if "_mean_" in feature else "harmonic"
            writer.writerow({"feature_name": feature, "feature_group": group})

    manifest = {
        "run_name": run_name,
        "dataset_version": args.dataset_version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_commit(repo_root),
        "generation_script": "scripts/augment_defocus_difference_features.py",
        "mode": args.mode,
        "python": sys.version,
        "platform": platform.platform(),
        "source_csv": str(source_csv),
        "source_csv_sha256": file_sha256(source_csv),
        "output_csv": str(output_csv),
        "output_csv_sha256": file_sha256(output_csv),
        "row_count": len(output_rows),
        "source_feature_count": len(source_features),
        "new_feature_count": len(new_features),
        "feature_count": len(feature_columns),
        "new_features": new_features,
        "feature_columns_path": str(feature_columns_path),
        "label_summary_path": str(output_dir / "label_summary.csv"),
        "feature_augmentation_summary_path": str(summary_path),
        "large_artifact_policy": "training_features_enhanced.csv is intentionally not pushed to GitHub",
    }
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("source CSV:", source_csv, flush=True)
    print("wrote augmented CSV:", output_csv, flush=True)
    print("rows:", len(output_rows), flush=True)
    print("features:", len(feature_columns), flush=True)
    print("new features:", len(new_features), flush=True)
    print("manifest:", output_dir / "dataset_manifest.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
