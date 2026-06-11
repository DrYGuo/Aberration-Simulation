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
import itertools
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

from feature_regression_model import FEATURE_COLUMNS as UNO_FEATURE_COLUMNS, TARGET_COLUMNS, file_sha256, target_from_row


UNDER_FOCUS_C1_OFFSET = -909
OVER_FOCUS_C1_OFFSET = 909
C1_OFFSETS = [UNDER_FOCUS_C1_OFFSET, OVER_FOCUS_C1_OFFSET]
PROFILE_RADIUS_PIXELS = 80
PROFILE_STEP_DEGREES = 10
NUM_PROFILE_LINES = int(180 / PROFILE_STEP_DEGREES) + 1
DATASET_VERSION = "enhanced_v3_targeted25k"
SPLIT_HINT_FIELD = "dataset_split_hint"
SPLIT_HINT_TRAINING_ONLY = "training_only"
DATASET_SOURCE_FIELD = "dataset_source"
DATASET_VERSION_FIELD = "dataset_version"
TRUE_HARD_TARGETS = ("C1", "S3_x", "S3_y", "A3_x", "A3_y")
COUPLING_EPSILON = 1e-8

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


def find_or_bootstrap_parent_csv(args: argparse.Namespace) -> Path:
    if args.parent_csv:
        return args.parent_csv
    try:
        return find_latest_csv(args.search_root)
    except FileNotFoundError:
        if not args.bootstrap_if_missing:
            raise
    print(
        f"No training_features_enhanced.csv found under {args.search_root}; bootstrapping parent enhanced dataset.",
        flush=True,
    )
    run_dataset_bootstrap(args.bootstrap_notebook, args.bootstrap_timeout, args.bootstrap_output_dir)
    return find_latest_csv(args.search_root)


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open() as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def require_source_columns(fieldnames: list[str]) -> None:
    required = {
        "sweep_label",
        "C1",
        "C3",
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
    }
    missing = sorted(required.difference(fieldnames))
    if missing:
        raise RuntimeError(f"source CSV is missing required columns: {missing}")


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


def fieldnames_with_metadata(source_fieldnames: list[str]) -> list[str]:
    return list(
        dict.fromkeys(
            [
                *source_fieldnames,
                DATASET_VERSION_FIELD,
                DATASET_SOURCE_FIELD,
                SPLIT_HINT_FIELD,
            ]
        )
    )


def attach_parent_metadata(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row.setdefault(DATASET_VERSION_FIELD, "parent_cached_dataset")
        row.setdefault(DATASET_SOURCE_FIELD, "parent")
        row.setdefault(SPLIT_HINT_FIELD, "")


def attach_new_row_metadata(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row[DATASET_VERSION_FIELD] = DATASET_VERSION
        row[DATASET_SOURCE_FIELD] = DATASET_VERSION
        row[SPLIT_HINT_FIELD] = SPLIT_HINT_TRAINING_ONLY


def extra_feature_columns() -> list[str]:
    columns: list[str] = []
    for focus_name in ("under", "over"):
        for char_name in ("Xigma", "Mu", "Rho"):
            columns.append(f"{focus_name}_{char_name}_mean")
            for order in (1, 2, 3, 4):
                columns.append(f"{focus_name}_{char_name}_h{order}_real")
                columns.append(f"{focus_name}_{char_name}_h{order}_imag")
    return columns


def randomize(
    params: dict[str, Any],
    fields: tuple[str, ...],
    rng: np.random.Generator,
    *,
    sparse: bool = False,
    s3_amp_range: tuple[float, float] | None = None,
) -> dict[str, Any]:
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
        if s3_amp_range is not None:
            params["S3_amp"] = uniform(*s3_amp_range)
        else:
            params["S3_amp"] = amp(100.0, active_probability)
        params["S3_phase"] = uniform(0.0, 180.0)
    return params


def load_generation_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"case_counts": dict(TARGETED_CASE_COUNTS)}
    data = json.loads(path.read_text())
    if "case_counts" in data:
        case_counts = data["case_counts"]
    elif "new_couplings_to_add" in data:
        case_counts = data["new_couplings_to_add"]
    else:
        case_counts = data
        data = {"case_counts": case_counts}
    if not isinstance(case_counts, dict):
        raise TypeError(f"case-count config must contain a mapping, got {type(case_counts).__name__}")
    parsed = {str(label): int(count) for label, count in case_counts.items()}
    if any(count <= 0 for count in parsed.values()):
        raise ValueError(f"all case counts must be positive: {parsed}")
    config = dict(data)
    config["case_counts"] = parsed
    return config


def generate_target_cases(seed: int, generation_config: dict[str, Any]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    case_counts = generation_config["case_counts"]
    sampling = generation_config.get("sampling", {})
    s3_tail = sampling.get("s3_tail", {}) if isinstance(sampling, dict) else {}
    s3_tail_enabled = bool(s3_tail.get("enabled", False))
    s3_amp_range = None
    if s3_tail_enabled:
        s3_amp_range = (
            float(s3_tail.get("s3_amp_min", 63.5)),
            float(s3_tail.get("s3_amp_max", 100.0)),
        )
        if s3_amp_range[0] < 0 or s3_amp_range[1] <= s3_amp_range[0]:
            raise ValueError(f"invalid S3 tail amplitude range: {s3_amp_range}")
    s3_tail_labels = set(s3_tail.get("labels", [])) if isinstance(s3_tail, dict) else set()
    force_s3_in_sparse = bool(s3_tail.get("force_s3_in_sparse", False)) if isinstance(s3_tail, dict) else False
    field_sets = {
        "S3_high_random": ("S3",),
        "coupled_full_random": ALL_FIELDS,
        "coupled_C1_C3_random": ("C1", "C3"),
        "coupled_C1_C3_S3_random": ("C1", "C3", "S3"),
        "coupled_A1_S3_random": ("A1", "S3"),
        "coupled_B2_S3_random": ("B2", "S3"),
        "coupled_A1_B2_S3_random": ("A1", "B2", "S3"),
        "coupled_C3_A3_S3_random": ("C3", "A3", "S3"),
        "coupled_A1_B2_random": ("A1", "B2"),
        "coupled_A2_B2_random": ("A2", "B2"),
        "coupled_C3_B2_random": ("C3", "B2"),
        "coupled_A3_S3_random": ("A3", "S3"),
    }
    cases: list[dict[str, Any]] = []
    for label, count in case_counts.items():
        if label != "coupled_sparse_random" and label not in field_sets:
            raise KeyError(f"unknown targeted case label: {label}")
        for _ in range(count):
            params = dict(BASELINE_PARAMETERS)
            params["sweep_label"] = label
            label_s3_range = s3_amp_range if s3_tail_enabled and label in s3_tail_labels else None
            if label == "coupled_sparse_random":
                active_count = int(rng.integers(2, min(5, len(ALL_FIELDS)) + 1))
                if s3_tail_enabled and force_s3_in_sparse:
                    available = tuple(field for field in ALL_FIELDS if field != "S3")
                    other_count = max(1, active_count - 1)
                    fields = ("S3", *tuple(rng.choice(available, size=other_count, replace=False)))
                else:
                    fields = tuple(rng.choice(ALL_FIELDS, size=active_count, replace=False))
                cases.append(randomize(params, fields, rng, sparse=True, s3_amp_range=label_s3_range))
            else:
                cases.append(randomize(params, field_sets[label], rng, s3_amp_range=label_s3_range))
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


def row_target_dict(row: dict[str, Any]) -> dict[str, float]:
    values = target_from_row(row)
    return {name: float(values[index]) for index, name in enumerate(TARGET_COLUMNS)}


def target_matrix(rows: list[dict[str, Any]], target_names: tuple[str, ...]) -> np.ndarray:
    if not rows:
        return np.zeros((0, len(target_names)), dtype=float)
    converted = [row_target_dict(row) for row in rows]
    return np.asarray([[target_values[name] for name in target_names] for target_values in converted], dtype=float)


def hard_target_scales(rows: list[dict[str, Any]]) -> dict[str, float]:
    matrix = target_matrix(rows, TRUE_HARD_TARGETS)
    scales: dict[str, float] = {}
    for index, name in enumerate(TRUE_HARD_TARGETS):
        if matrix.size:
            value = float(np.max(np.abs(matrix[:, index])))
        else:
            value = 0.0
        scales[name] = max(value, 1e-8)
    return scales


def s3_magnitude_histogram(rows: list[dict[str, Any]], bins: int = 20) -> dict[str, Any]:
    matrix = target_matrix(rows, ("S3_x", "S3_y"))
    if not len(matrix):
        return {"n": 0, "counts": [], "bin_edges": []}
    magnitudes = np.hypot(matrix[:, 0], matrix[:, 1])
    max_edge = max(100.0, float(np.max(magnitudes)))
    counts, edges = np.histogram(magnitudes, bins=bins, range=(0.0, max_edge))
    return {
        "n": int(len(magnitudes)),
        "max_magnitude": float(np.max(magnitudes)),
        "p50_magnitude": float(np.percentile(magnitudes, 50)),
        "p90_magnitude": float(np.percentile(magnitudes, 90)),
        "p95_magnitude": float(np.percentile(magnitudes, 95)),
        "count_gt_63_5": int(np.sum(magnitudes > 63.5)),
        "fraction_gt_63_5": float(np.mean(magnitudes > 63.5)),
        "counts": counts.astype(int).tolist(),
        "bin_edges": edges.tolist(),
    }


def active_groups(row: dict[str, Any]) -> int:
    active = 0
    if abs(float(row.get("C1") or 0.0)) > COUPLING_EPSILON:
        active += 1
    if abs(float(row.get("C3") or 0.0)) > COUPLING_EPSILON:
        active += 1
    for name in ("A1", "A2", "B2", "A3", "S3"):
        if abs(float(row.get(f"{name}_amp") or 0.0)) > COUPLING_EPSILON:
            active += 1
    return active


def coupling_density(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = np.asarray([active_groups(row) for row in rows], dtype=float)
    if not len(counts):
        return {"n": 0}
    return {
        "n": int(len(counts)),
        "mean_active_groups": float(np.mean(counts)),
        "median_active_groups": float(np.median(counts)),
        "min_active_groups": int(np.min(counts)),
        "max_active_groups": int(np.max(counts)),
        "fraction_active_groups_ge_2": float(np.mean(counts >= 2)),
        "fraction_active_groups_ge_3": float(np.mean(counts >= 3)),
        "fraction_active_groups_ge_5": float(np.mean(counts >= 5)),
    }


def per_regime_hard_target_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_label: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_label.setdefault(str(row.get("sweep_label", "")), []).append(row)

    stats: dict[str, Any] = {}
    for label, label_rows in sorted(by_label.items()):
        matrix = np.abs(target_matrix(label_rows, TRUE_HARD_TARGETS))
        label_stats: dict[str, Any] = {"n": len(label_rows), "targets": {}}
        for index, target in enumerate(TRUE_HARD_TARGETS):
            values = matrix[:, index] if matrix.size else np.asarray([], dtype=float)
            label_stats["targets"][target] = {
                "mean_abs": float(np.mean(values)) if len(values) else 0.0,
                "std_abs": float(np.std(values)) if len(values) else 0.0,
                "max_abs": float(np.max(values)) if len(values) else 0.0,
            }
        stats[label] = label_stats
    return stats


def pairwise_hard_target_histograms(
    output_dir: Path,
    rows: list[dict[str, Any]],
    scales: dict[str, float],
    bins: int = 20,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    matrix = target_matrix(rows, TRUE_HARD_TARGETS)
    histogram_dir = output_dir / "hard_target_histograms"
    histogram_dir.mkdir(parents=True, exist_ok=True)
    histograms: dict[str, Any] = {}
    for left, right in itertools.combinations(TRUE_HARD_TARGETS, 2):
        left_index = TRUE_HARD_TARGETS.index(left)
        right_index = TRUE_HARD_TARGETS.index(right)
        left_scale = max(scales[left], 1e-8)
        right_scale = max(scales[right], 1e-8)
        x = matrix[:, left_index] / left_scale if len(matrix) else np.asarray([], dtype=float)
        y = matrix[:, right_index] / right_scale if len(matrix) else np.asarray([], dtype=float)
        counts, x_edges, y_edges = np.histogram2d(x, y, bins=bins, range=[[-1.0, 1.0], [-1.0, 1.0]])
        key = f"{left}__{right}"
        png_path = histogram_dir / f"{key}_hist2d.png"
        fig, ax = plt.subplots(figsize=(4, 3.5))
        image = ax.imshow(
            counts.T,
            origin="lower",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            aspect="auto",
            cmap="viridis",
        )
        ax.set_xlabel(f"{left} / scale")
        ax.set_ylabel(f"{right} / scale")
        ax.set_title(key, fontsize=9)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        histograms[key] = {
            "counts": counts.astype(int).tolist(),
            "x_edges": x_edges.tolist(),
            "y_edges": y_edges.tolist(),
            "png_path": str(png_path),
        }
    return histograms


def write_targeted_audit(
    path: Path,
    combined_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    new_scales = hard_target_scales(new_rows)
    new_matrix = target_matrix(new_rows, TRUE_HARD_TARGETS)
    normalized_abs = np.abs(new_matrix) / np.asarray([new_scales[name] for name in TRUE_HARD_TARGETS])[None, :]
    new_labels = np.asarray([str(row.get("sweep_label", "")) for row in new_rows])
    full_random_mask = new_labels == "coupled_full_random"
    full_random_norm = normalized_abs[full_random_mask] if len(normalized_abs) else np.zeros((0, len(TRUE_HARD_TARGETS)))
    a3_s3_count = Counter(str(row.get("sweep_label", "")) for row in new_rows).get("coupled_A3_S3_random", 0)
    audit = {
        "dataset_version": DATASET_VERSION,
        "true_hard_targets": list(TRUE_HARD_TARGETS),
        "hard_target_scale_definition": "max(abs(target_min), abs(target_max)) on newly appended rows, not full span",
        "new_hard_target_scales": new_scales,
        "new_rows_per_regime": dict(Counter(str(row.get("sweep_label", "")) for row in new_rows)),
        "total_rows_per_regime_after_merge": dict(Counter(str(row.get("sweep_label", "")) for row in combined_rows)),
        "new_s3_magnitude_histogram": s3_magnitude_histogram(new_rows),
        "combined_s3_magnitude_histogram": s3_magnitude_histogram(combined_rows),
        "combined_coupling_density": coupling_density(combined_rows),
        "new_coupling_density": coupling_density(new_rows),
        "fraction_new_rows_at_least_3_of_5_hard_targets_above_half_scale": float(
            np.mean(np.sum(normalized_abs > 0.5, axis=1) >= 3)
        )
        if len(normalized_abs)
        else 0.0,
        "fraction_coupled_full_random_rows_all_5_hard_targets_below_20_percent_scale": float(
            np.mean(np.all(full_random_norm < 0.2, axis=1))
        )
        if len(full_random_norm)
        else 0.0,
        "new_per_regime_hard_target_stats": per_regime_hard_target_stats(new_rows),
        "combined_per_regime_hard_target_stats": per_regime_hard_target_stats(combined_rows),
        "a3_s3_coverage_warning": a3_s3_count < 1000,
        "a3_s3_coverage_note": (
            "coupled_A3_S3_random has fewer than 1000 newly appended rows; review before treating "
            "A3/S3 pair coverage as saturated."
        )
        if a3_s3_count < 1000
        else "coupled_A3_S3_random has at least 1000 newly appended rows in this expansion.",
    }
    audit["pairwise_hard_target_histograms"] = pairwise_hard_target_histograms(path.parent, new_rows, new_scales)
    path.write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-csv", "--source-csv", dest="parent_csv", type=Path)
    parser.add_argument("--search-root", type=Path, default=Path("training_results"))
    parser.add_argument("--output-root", type=Path, default=Path("training_results/feature_regression_enhanced"))
    parser.add_argument("--run-prefix", default="enhanced_v3_targeted25k")
    parser.add_argument("--dataset-version", default=DATASET_VERSION)
    parser.add_argument("--case-counts-json", type=Path)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--batch-base-cases", type=int, default=64)
    parser.add_argument(
        "--bootstrap-if-missing",
        action="store_true",
        help="Run the enhanced dataset bootstrap notebook if no parent training_features_enhanced.csv exists.",
    )
    parser.add_argument(
        "--bootstrap-notebook",
        type=Path,
        default=Path("notebooks/uno_feature_regression_enhanced_dataset_bootstrap.ipynb"),
    )
    parser.add_argument("--bootstrap-timeout", type=int, default=3600)
    parser.add_argument("--bootstrap-output-dir", type=Path, default=Path("colab_worker_logs"))
    return parser.parse_args()


def main() -> int:
    global DATASET_VERSION
    args = parse_args()
    DATASET_VERSION = args.dataset_version
    repo_root = Path.cwd()
    source_csv = find_or_bootstrap_parent_csv(args)
    source_csv = source_csv.resolve()
    source_rows, source_fieldnames = read_csv(source_csv)
    if not source_rows:
        raise RuntimeError(f"source CSV is empty: {source_csv}")
    require_source_columns(source_fieldnames)

    feature_columns = load_feature_columns(source_csv)
    attach_parent_metadata(source_rows)
    generation_config = load_generation_config(args.case_counts_json)
    case_counts = generation_config["case_counts"]
    target_cases = generate_target_cases(args.seed, generation_config)
    print("source CSV:", source_csv, flush=True)
    print("source rows:", len(source_rows), flush=True)
    print("feature columns:", len(feature_columns), flush=True)
    print("target columns:", TARGET_COLUMNS, flush=True)
    print("new targeted rows planned:", len(target_cases), flush=True)
    print("targeted case counts:", dict(Counter(row["sweep_label"] for row in target_cases)), flush=True)

    run_name = f"{args.run_prefix}_{utc_stamp()}"
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    new_rows = simulate_rows(target_cases, args.batch_base_cases)
    attach_new_row_metadata(new_rows)
    combined_rows: list[dict[str, Any]] = [*source_rows, *new_rows]
    output_fieldnames = fieldnames_with_metadata(source_fieldnames)
    output_csv = output_dir / "training_features_enhanced.csv"
    write_csv(output_csv, combined_rows, output_fieldnames)
    (output_dir / "feature_columns_enhanced.json").write_text(json.dumps(feature_columns, indent=2) + "\n")
    write_label_summary(output_dir / "label_summary.csv", combined_rows)
    write_label_summary(output_dir / "new_targeted_label_summary.csv", new_rows)
    audit_path = output_dir / "targeted25k_audit.json"
    audit = write_targeted_audit(audit_path, combined_rows, new_rows)

    manifest = {
        "run_name": run_name,
        "dataset_version": DATASET_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_commit(repo_root),
        "generation_script": "scripts/generate_targeted_enhanced_dataset.py",
        "python": sys.version,
        "platform": platform.platform(),
        "parent_dataset_path": str(source_csv),
        "parent_dataset_sha256": file_sha256(source_csv),
        "output_csv": str(output_csv),
        "output_csv_sha256": file_sha256(output_csv),
        "row_count_before_expansion": len(source_rows),
        "appended_training_only_row_count": len(new_rows),
        "total_rows": len(combined_rows),
        "random_seed": args.seed,
        "targeted_case_counts": case_counts,
        "generation_config": generation_config,
        "target_columns": TARGET_COLUMNS,
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "regime_counts_after_merge": dict(Counter(str(row.get("sweep_label", "")) for row in combined_rows)),
        "new_rows_per_regime": audit["new_rows_per_regime"],
        "split_policy": "New rows are training-only unless a later, explicit versioned split experiment changes validation/blind/stress benchmark definitions.",
        "dataset_split_hint_field": SPLIT_HINT_FIELD,
        "new_row_split_hint": SPLIT_HINT_TRAINING_ONLY,
        "feature_columns_path": str(output_dir / "feature_columns_enhanced.json"),
        "label_summary_path": str(output_dir / "label_summary.csv"),
        "new_targeted_label_summary_path": str(output_dir / "new_targeted_label_summary.csv"),
        "targeted25k_audit_path": str(audit_path),
        "large_artifact_policy": "training_features_enhanced.csv is intentionally not pushed to GitHub",
    }
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote combined enhanced dataset:", output_csv, flush=True)
    print("combined rows:", len(combined_rows), flush=True)
    print("manifest:", output_dir / "dataset_manifest.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
