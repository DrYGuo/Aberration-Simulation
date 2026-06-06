"""Generate a targeted enhanced-feature dataset expansion.

This appends newly simulated hard-regime rows to the latest cached
``training_features_enhanced.csv``. The large combined CSV is intentionally
kept out of GitHub by the Colab worker artifact policy; only compact manifests
and summaries should be pushed.
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

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from feature_regression_model import FEATURE_COLUMNS as UNO_FEATURE_COLUMNS, file_sha256


UNDER_FOCUS_C1_OFFSET = -909
OVER_FOCUS_C1_OFFSET = 909
C1_OFFSETS = [UNDER_FOCUS_C1_OFFSET, OVER_FOCUS_C1_OFFSET]
PROFILE_RADIUS_PIXELS = 80
PROFILE_STEP_DEGREES = 10
NUM_PROFILE_LINES = int(180 / PROFILE_STEP_DEGREES) + 1

COMBINATION_FIELDS = (
    "sweep_label",
    "C1",
    "A1_amp",
    "A1_phase",
    "A2_amp",
    "A2_phase",
    "B2_amp",
    "B2_phase",
    "A3_amp",
    "A3_phase",
    "S3_amp",
    "S3_phase",
    "C3",
)

BASELINE_PARAMETERS = {
    "C1_offset": 0,
    "A3_amp": 0,
    "S3_amp": 0,
    "A2_amp": 0,
    "B2_amp": 0,
    "C1": 0,
    "C3": 0,
    "A1_amp": 0,
    "A1_phase": 0,
    "A2_phase": 0,
    "A3_phase": 0,
    "S3_phase": 0,
    "B2_phase": 0,
}

TARGETED_CASE_COUNTS = {
    "coupled_full_random": 4000,
    "coupled_sparse_random": 3500,
    "coupled_C1_C3_random": 3500,
    "coupled_A1_S3_random": 3500,
    "coupled_C3_A3_S3_random": 3000,
    "coupled_A1_B2_random": 2500,
    "coupled_A2_B2_random": 2500,
    "coupled_C3_B2_random": 2000,
    "coupled_A3_S3_random": 500,
}

ALL_FIELDS = ("C1", "C3", "A1", "A2", "B2", "A3", "S3")


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


def find_latest_csv(search_root: Path) -> Path:
    matches = sorted(
        search_root.glob("**/training_features_enhanced.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    if not matches:
        raise FileNotFoundError(f"No training_features_enhanced.csv found under {search_root}")
    return matches[-1]


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
    for name in ["feature_columns_enhanced.json", "feature_columns.json"]:
        path = source_csv.parent / name
        if path.exists():
            data = json.loads(path.read_text())
            if isinstance(data, dict) and "features" in data:
                return list(data["features"])
            return list(data)
    return list(UNO_FEATURE_COLUMNS) + extra_feature_columns()


def extra_feature_columns() -> list[str]:
    columns: list[str] = []
    for focus_name in ("under", "over"):
        for char_name in ("Xigma", "Mu", "Rho"):
            columns.append(f"{focus_name}_{char_name}_mean")
            for order in (1, 2, 3, 4):
                columns.append(f"{focus_name}_{char_name}_h{order}_real")
                columns.append(f"{focus_name}_{char_name}_h{order}_imag")
    return columns


def randomize(params: dict[str, Any], fields: tuple[str, ...], rng: np.random.Generator, *, sparse: bool = False) -> dict[str, Any]:
    def uniform(low: float, high: float) -> float:
        return float(rng.uniform(low, high))

    def amp(max_value: float, active_probability: float = 1.0) -> float:
        if rng.random() > active_probability:
            return 0.0
        low = 0.05 * max_value
        return uniform(low, max_value)

    active_probability = 0.85 if sparse else 1.0
    if "C1" in fields:
        params["C1"] = uniform(-100.0, 100.0)
    if "C3" in fields:
        params["C3"] = uniform(0.05, 2.0)
    if "A1" in fields:
        params["A1_amp"] = amp(60.0, active_probability)
        params["A1_phase"] = uniform(0.0, 180.0)
    if "A2" in fields:
        params["A2_amp"] = amp(16.0, active_probability)
        params["A2_phase"] = uniform(0.0, 120.0)
    if "B2" in fields:
        params["B2_amp"] = amp(3.0, active_probability)
        params["B2_phase"] = uniform(0.0, 360.0)
    if "A3" in fields:
        params["A3_amp"] = amp(100.0, active_probability)
        params["A3_phase"] = uniform(0.0, 90.0)
    if "S3" in fields:
        params["S3_amp"] = amp(100.0, active_probability)
        params["S3_phase"] = uniform(0.0, 180.0)
    return params


def generate_target_cases(seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    field_sets = {
        "coupled_full_random": ALL_FIELDS,
        "coupled_C1_C3_random": ("C1", "C3"),
        "coupled_A1_S3_random": ("A1", "S3"),
        "coupled_C3_A3_S3_random": ("C3", "A3", "S3"),
        "coupled_A1_B2_random": ("A1", "B2"),
        "coupled_A2_B2_random": ("A2", "B2"),
        "coupled_C3_B2_random": ("C3", "B2"),
        "coupled_A3_S3_random": ("A3", "S3"),
    }
    cases: list[dict[str, Any]] = []
    for label, count in TARGETED_CASE_COUNTS.items():
        for _ in range(count):
            params = dict(BASELINE_PARAMETERS)
            params["sweep_label"] = label
            if label == "coupled_sparse_random":
                active_count = int(rng.integers(2, min(5, len(ALL_FIELDS)) + 1))
                fields = tuple(rng.choice(ALL_FIELDS, size=active_count, replace=False))
                cases.append(randomize(params, fields, rng, sparse=True))
            else:
                cases.append(randomize(params, field_sets[label], rng))
    rng.shuffle(cases)
    return cases


def scalar_value(value):
    from aberration_simulation.backend import asnumpy

    value_np = asnumpy(value)
    if isinstance(value_np, np.ndarray):
        return value_np.item() if value_np.shape == () else value_np
    return value_np


def add_raw_harmonic_features(row: dict[str, Any], prefix: str, chars: dict[str, Any], angles_deg: np.ndarray) -> None:
    from aberration_simulation.backend import xp

    theta = xp.deg2rad(xp.asarray(angles_deg, dtype=float))
    for char_name in ("Xigma", "Mu", "Rho"):
        values = chars[char_name]
        row[f"{prefix}_{char_name}_mean"] = float(scalar_value(xp.mean(values)))
        for order in (1, 2, 3, 4):
            harmonic = 2 * xp.mean(values * xp.exp(1j * order * theta))
            harmonic = scalar_value(harmonic)
            row[f"{prefix}_{char_name}_h{order}_real"] = float(np.real(harmonic))
            row[f"{prefix}_{char_name}_h{order}_imag"] = float(np.imag(harmonic))


def simulate_rows(base_cases: list[dict[str, Any]], batch_base_cases: int) -> list[dict[str, Any]]:
    from aberration_simulation.backend import HAS_CUPY, xp
    from aberration_simulation.line_profiles import extract_line_profiles_from_stack
    from aberration_simulation.optics import SimulationConfig, run_simulation
    from aberration_simulation.uno_conventions import (
        add_complex_columns,
        compute_line_characteristics,
        compute_uno_values,
        select_under_over_pairs,
    )

    if not HAS_CUPY:
        raise RuntimeError("CuPy GPU path is required for targeted dataset expansion.")

    config = SimulationConfig(
        pix_dim=(256, 256),
        real_dim=(1280, 1280),
        app=30,
        sigma=2,
    )
    rows: list[dict[str, Any]] = []
    print("targeted base cases:", len(base_cases), flush=True)
    print("batch base cases:", batch_base_cases, flush=True)

    for batch_start in range(0, len(base_cases), batch_base_cases):
        batch_base_cases_list = base_cases[batch_start : batch_start + batch_base_cases]
        batch_parameters = []
        for base_case in batch_base_cases_list:
            for c1_offset in C1_OFFSETS:
                params = dict(base_case)
                params["C1_offset"] = c1_offset
                batch_parameters.append(params)

        probe_images, selected = run_simulation(config, batch_parameters)
        simulation_records = [dict(params) for params in batch_parameters]
        batch_pairs = select_under_over_pairs(
            simulation_records,
            COMBINATION_FIELDS,
            under_focus_c1_offset=UNDER_FOCUS_C1_OFFSET,
            over_focus_c1_offset=OVER_FOCUS_C1_OFFSET,
        )

        for local_case_index, (params, under_index, over_index) in enumerate(batch_pairs):
            stack = probe_images[:, :, [under_index, over_index]]
            profiles, coords = extract_line_profiles_from_stack(
                stack,
                num_lines=NUM_PROFILE_LINES,
                radius=PROFILE_RADIUS_PIXELS,
            )
            angles_deg = np.asarray(coords["angles_deg"], dtype=float)
            under_chars = compute_line_characteristics(profiles[:, :, 0], PROFILE_RADIUS_PIXELS)
            over_chars = compute_line_characteristics(profiles[:, :, 1], PROFILE_RADIUS_PIXELS)
            feature_values = compute_uno_values(under_chars, over_chars, angles_deg)

            row = {field: params.get(field, 0.0) for field in COMBINATION_FIELDS}
            row["case_index"] = batch_start + local_case_index
            row["under_index"] = batch_start * len(C1_OFFSETS) + under_index
            row["over_index"] = batch_start * len(C1_OFFSETS) + over_index
            row["under_C1_offset"] = UNDER_FOCUS_C1_OFFSET
            row["over_C1_offset"] = OVER_FOCUS_C1_OFFSET
            for name, value in feature_values.items():
                add_complex_columns(row, name, value)
            add_raw_harmonic_features(row, "under", under_chars, angles_deg)
            add_raw_harmonic_features(row, "over", over_chars, angles_deg)
            rows.append(row)

        del probe_images, selected, batch_parameters, simulation_records, batch_pairs
        xp.cuda.Stream.null.synchronize()
        xp.get_default_memory_pool().free_all_blocks()
        print(
            f"processed {min(batch_start + batch_base_cases, len(base_cases))}/{len(base_cases)} targeted cases",
            flush=True,
        )
    return rows


def write_label_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    counts = Counter(str(row.get("sweep_label", "")) for row in rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sweep_label", "n_rows"])
        writer.writeheader()
        for label, count in sorted(counts.items()):
            writer.writerow({"sweep_label": label, "n_rows": count})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path)
    parser.add_argument("--search-root", type=Path, default=Path("training_results"))
    parser.add_argument("--output-root", type=Path, default=Path("training_results/feature_regression_enhanced"))
    parser.add_argument("--run-prefix", default="enhanced_v3_targeted25k")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--batch-base-cases", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd()
    source_csv = args.source_csv or find_latest_csv(args.search_root)
    source_csv = source_csv.resolve()
    source_rows, source_fieldnames = read_csv(source_csv)
    if not source_rows:
        raise RuntimeError(f"source CSV is empty: {source_csv}")

    feature_columns = load_feature_columns(source_csv)
    target_cases = generate_target_cases(args.seed)
    assert len(target_cases) == 25000
    print("source CSV:", source_csv, flush=True)
    print("source rows:", len(source_rows), flush=True)
    print("new targeted rows planned:", len(target_cases), flush=True)
    print("targeted case counts:", dict(Counter(row["sweep_label"] for row in target_cases)), flush=True)

    run_name = f"{args.run_prefix}_{utc_stamp()}"
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    new_rows = simulate_rows(target_cases, args.batch_base_cases)
    combined_rows: list[dict[str, Any]] = [*source_rows, *new_rows]
    output_csv = output_dir / "training_features_enhanced.csv"
    write_csv(output_csv, combined_rows, source_fieldnames)
    (output_dir / "feature_columns_enhanced.json").write_text(json.dumps(feature_columns, indent=2) + "\n")
    write_label_summary(output_dir / "label_summary.csv", combined_rows)
    write_label_summary(output_dir / "new_targeted_label_summary.csv", new_rows)

    manifest = {
        "run_name": run_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_commit(repo_root),
        "python": sys.version,
        "platform": platform.platform(),
        "source_csv": str(source_csv),
        "source_csv_sha256": file_sha256(source_csv),
        "output_csv": str(output_csv),
        "output_csv_sha256": file_sha256(output_csv),
        "source_rows": len(source_rows),
        "new_targeted_rows": len(new_rows),
        "combined_rows": len(combined_rows),
        "random_seed": args.seed,
        "targeted_case_counts": TARGETED_CASE_COUNTS,
        "feature_columns_path": str(output_dir / "feature_columns_enhanced.json"),
        "label_summary_path": str(output_dir / "label_summary.csv"),
        "new_targeted_label_summary_path": str(output_dir / "new_targeted_label_summary.csv"),
        "large_artifact_policy": "training_features_enhanced.csv is intentionally not pushed to GitHub",
    }
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote combined enhanced dataset:", output_csv, flush=True)
    print("combined rows:", len(combined_rows), flush=True)
    print("manifest:", output_dir / "dataset_manifest.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
