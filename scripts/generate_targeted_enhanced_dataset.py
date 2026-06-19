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
import math
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
    "sampling_method",
    "sampling_relative_angle_bin",
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
SAMPLING_METADATA_FIELDS = (
    "sampling_method",
    "sampling_relative_angle_bin",
    "sampling_candidate_role",
    "sampling_parent_nn_distance_12d",
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
SAMPLING_DIMENSIONS = (
    "C1_bin",
    "C1_value",
    "C1_sign",
    "C3_value",
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
)
VECTOR_ORDERS = {"A1": 2, "B2": 1, "A2": 3, "S3": 2, "A3": 4}
PHASE_PERIODS = {"A1": 180.0, "B2": 360.0, "A2": 120.0, "S3": 180.0, "A3": 90.0}
RELATIVE_ANGLE_DEGREES = {
    "aligned": 0.0,
    "orthogonal": 90.0,
    "anti_aligned": 180.0,
}


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
                *SAMPLING_METADATA_FIELDS,
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


def attach_new_row_metadata(rows: list[dict[str, Any]], split_hint: str = SPLIT_HINT_TRAINING_ONLY) -> None:
    for row in rows:
        row[DATASET_VERSION_FIELD] = DATASET_VERSION
        row[DATASET_SOURCE_FIELD] = DATASET_VERSION
        row[SPLIT_HINT_FIELD] = split_hint


def extra_feature_columns() -> list[str]:
    columns: list[str] = []
    for focus_name in ("under", "over"):
        for char_name in ("Xigma", "Mu", "Rho"):
            columns.append(f"{focus_name}_{char_name}_mean")
            for order in (1, 2, 3, 4):
                columns.append(f"{focus_name}_{char_name}_h{order}_real")
                columns.append(f"{focus_name}_{char_name}_h{order}_imag")
    return columns


def latin_hypercube(n_rows: int, dimensions: tuple[str, ...], rng: np.random.Generator) -> list[dict[str, float]]:
    if n_rows <= 0:
        return []
    columns: dict[str, np.ndarray] = {}
    for name in dimensions:
        columns[name] = (rng.permutation(n_rows).astype(float) + rng.random(n_rows)) / float(n_rows)
    return [
        {name: float(columns[name][row_index]) for name in dimensions}
        for row_index in range(n_rows)
    ]


def vector_angle_from_phase(group: str, phase_deg: float) -> float:
    return (VECTOR_ORDERS[group] * float(phase_deg)) % 360.0


def phase_from_vector_angle(group: str, vector_angle_deg: float) -> float:
    return (float(vector_angle_deg) / VECTOR_ORDERS[group]) % PHASE_PERIODS[group]


def apply_relative_angle_controls(
    params: dict[str, Any],
    pairs: list[list[str]],
    angle_bin: str,
    rng: np.random.Generator,
) -> None:
    if not pairs or angle_bin == "random":
        return
    delta = RELATIVE_ANGLE_DEGREES.get(angle_bin)
    if delta is None:
        raise ValueError(f"unknown relative-angle bin: {angle_bin!r}")
    jitter = float(rng.uniform(-7.5, 7.5))
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError(f"relative-angle pair must contain two vector groups: {pair!r}")
        left, right = pair
        if left not in VECTOR_ORDERS or right not in VECTOR_ORDERS:
            continue
        if float(params.get(f"{left}_amp", 0.0) or 0.0) <= COUPLING_EPSILON:
            continue
        if float(params.get(f"{right}_amp", 0.0) or 0.0) <= COUPLING_EPSILON:
            continue
        right_angle = vector_angle_from_phase(right, float(params.get(f"{right}_phase", 0.0) or 0.0))
        params[f"{left}_phase"] = phase_from_vector_angle(left, right_angle + delta + jitter)


def randomize(
    params: dict[str, Any],
    fields: tuple[str, ...],
    rng: np.random.Generator,
    *,
    sparse: bool = False,
    s3_amp_range: tuple[float, float] | None = None,
    c1_magnitude_bins: tuple[tuple[float, float], ...] | None = None,
    unit_values: dict[str, float] | None = None,
) -> dict[str, Any]:
    def unit(name: str) -> float:
        if unit_values is not None and name in unit_values:
            return min(max(float(unit_values[name]), 0.0), np.nextafter(1.0, 0.0))
        return float(rng.random())

    def uniform(low: float, high: float, name: str) -> float:
        return float(low + unit(name) * (high - low))

    def amp(max_value: float, name: str, active_probability: float = 1.0) -> float:
        raw = unit(name)
        if raw > active_probability:
            return 0.0
        scaled = raw / max(active_probability, 1e-8)
        low = 0.05 * max_value
        return float(low + scaled * (max_value - low))

    active_probability = 0.85 if sparse else 1.0
    if "C1" in fields:
        if c1_magnitude_bins:
            bin_index = min(int(unit("C1_bin") * len(c1_magnitude_bins)), len(c1_magnitude_bins) - 1)
            low, high = c1_magnitude_bins[bin_index]
            magnitude = uniform(low, high, "C1_value")
            params["C1"] = magnitude if unit("C1_sign") < 0.5 else -magnitude
        else:
            params["C1"] = uniform(-100.0, 100.0, "C1_value")
    if "C3" in fields:
        params["C3"] = uniform(0.05, 2.0, "C3_value")
    if "A1" in fields:
        params["A1_amp"] = amp(60.0, "A1_amp", active_probability)
        params["A1_phase"] = uniform(0.0, 180.0, "A1_phase")
    if "A2" in fields:
        params["A2_amp"] = amp(16.0, "A2_amp", active_probability)
        params["A2_phase"] = uniform(0.0, 120.0, "A2_phase")
    if "B2" in fields:
        params["B2_amp"] = amp(3.0, "B2_amp", active_probability)
        params["B2_phase"] = uniform(0.0, 360.0, "B2_phase")
    if "A3" in fields:
        params["A3_amp"] = amp(100.0, "A3_amp", active_probability)
        params["A3_phase"] = uniform(0.0, 90.0, "A3_phase")
    if "S3" in fields:
        if s3_amp_range is not None:
            params["S3_amp"] = uniform(s3_amp_range[0], s3_amp_range[1], "S3_amp")
        else:
            params["S3_amp"] = amp(100.0, "S3_amp", active_probability)
        params["S3_phase"] = uniform(0.0, 180.0, "S3_phase")
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


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def parent_reference_rows_for_candidate_selection(
    parent_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    split_hints = set(str(item) for item in config.get("reference_split_hints", ["", SPLIT_HINT_TRAINING_ONLY]))
    dataset_versions = set(str(item) for item in config.get("reference_dataset_versions", []))
    output: list[dict[str, Any]] = []
    for row in parent_rows:
        hint = str(row.get(SPLIT_HINT_FIELD, ""))
        version = str(row.get(DATASET_VERSION_FIELD, ""))
        if dataset_versions and version not in dataset_versions:
            continue
        if hint in split_hints:
            output.append(row)
    if output:
        return output
    return parent_rows


def normalized_12d_target_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    matrix = target_matrix(rows, tuple(TARGET_COLUMNS))
    if not len(matrix):
        return np.zeros((0, len(TARGET_COLUMNS)), dtype=np.float32)
    return (matrix / target_scale_array()[None, :]).astype(np.float32)


def nearest_distance_to_reference(
    reference: np.ndarray,
    query: np.ndarray,
    *,
    chunk_size: int = 25_000,
) -> tuple[np.ndarray, str]:
    if len(reference) == 0 or len(query) == 0:
        return np.zeros(len(query), dtype=np.float32), "empty"
    try:
        from sklearn.neighbors import NearestNeighbors

        model = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean")
        model.fit(reference)
        chunks = []
        for start in range(0, len(query), chunk_size):
            distances, _ = model.kneighbors(query[start : start + chunk_size])
            chunks.append(distances[:, 0])
        return np.concatenate(chunks).astype(np.float32), "sklearn_exact_or_sampled_reference"
    except Exception as exc:
        if len(reference) * len(query) > 50_000_000:
            raise RuntimeError(
                "candidate_selection nearest-neighbor lookup requires scikit-learn for this dataset size; "
                f"reference={len(reference)}, query={len(query)}"
            ) from exc
        chunks = []
        for start in range(0, len(query), min(chunk_size, 1000)):
            block = query[start : start + min(chunk_size, 1000)]
            distances = np.sqrt(np.sum((block[:, None, :] - reference[None, :, :]) ** 2, axis=2))
            chunks.append(np.min(distances, axis=1))
        return np.concatenate(chunks).astype(np.float32), "numpy_reference"


def select_bridge_indices(
    distances: np.ndarray,
    candidate_indices: np.ndarray,
    count: int,
    *,
    quantile: float,
) -> np.ndarray:
    if count <= 0 or len(candidate_indices) == 0:
        return np.asarray([], dtype=np.int64)
    selected_distances = distances[candidate_indices]
    target = float(np.quantile(distances, quantile))
    order = np.argsort(np.abs(selected_distances - target), kind="mergesort")
    return candidate_indices[order[: min(count, len(order))]]


def select_candidate_cases_by_parent_nn(
    cases: list[dict[str, Any]],
    *,
    desired_counts: dict[str, int],
    parent_rows: list[dict[str, Any]],
    config: dict[str, Any],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    reference_rows = parent_reference_rows_for_candidate_selection(parent_rows, config)
    reference_sample_size = int(config.get("reference_sample_size", 250_000))
    reference_matrix = normalized_12d_target_matrix(reference_rows)
    if len(reference_matrix) > reference_sample_size:
        sample_index = rng.choice(len(reference_matrix), size=reference_sample_size, replace=False)
        reference_matrix = reference_matrix[sample_index]

    query_matrix = normalized_12d_target_matrix(cases)
    distances, method = nearest_distance_to_reference(
        reference_matrix,
        query_matrix,
        chunk_size=int(config.get("query_chunk_size", 25_000)),
    )
    far_fraction = float(config.get("far_fraction", 0.8))
    bridge_fraction = float(config.get("bridge_fraction", 1.0 - far_fraction))
    bridge_quantile = float(config.get("bridge_quantile", 0.55))
    if far_fraction < 0 or bridge_fraction < 0 or far_fraction + bridge_fraction <= 0:
        raise ValueError("candidate_selection far_fraction/bridge_fraction must be nonnegative and nonzero")

    by_label: dict[str, list[int]] = {}
    for index, row in enumerate(cases):
        by_label.setdefault(str(row.get("sweep_label", "")), []).append(index)

    selected: list[dict[str, Any]] = []
    selection_summary: dict[str, Any] = {}
    for label, desired_count in desired_counts.items():
        label_indices = np.asarray(by_label.get(label, []), dtype=np.int64)
        if len(label_indices) < desired_count:
            raise RuntimeError(
                f"candidate selection for {label} has only {len(label_indices)} candidates for {desired_count} requested rows"
            )
        label_distances = distances[label_indices]
        far_count = int(round(desired_count * far_fraction / max(far_fraction + bridge_fraction, 1e-8)))
        far_count = min(max(far_count, 0), desired_count)
        bridge_count = desired_count - far_count

        far_order = np.argsort(label_distances, kind="mergesort")[::-1]
        far_indices = label_indices[far_order[:far_count]]
        remaining_mask = np.ones(len(label_indices), dtype=bool)
        remaining_mask[far_order[:far_count]] = False
        remaining = label_indices[remaining_mask]
        bridge_indices = select_bridge_indices(
            distances,
            remaining,
            bridge_count,
            quantile=bridge_quantile,
        )
        chosen = np.concatenate([far_indices, bridge_indices])
        if len(chosen) < desired_count:
            chosen_set = set(int(index) for index in chosen)
            fill = [int(index) for index in label_indices[far_order] if int(index) not in chosen_set]
            chosen = np.concatenate([chosen, np.asarray(fill[: desired_count - len(chosen)], dtype=np.int64)])
        for role, indices in [("far_nn", far_indices), ("bridge_anchor", bridge_indices)]:
            for case_index in indices:
                cases[int(case_index)]["sampling_candidate_role"] = role
                cases[int(case_index)]["sampling_parent_nn_distance_12d"] = float(distances[int(case_index)])
        chosen_set = set(int(index) for index in chosen)
        for case_index in chosen_set:
            cases[int(case_index)].setdefault("sampling_candidate_role", "fill")
            cases[int(case_index)].setdefault("sampling_parent_nn_distance_12d", float(distances[int(case_index)]))
        label_chosen_distances = distances[np.asarray(sorted(chosen_set), dtype=np.int64)]
        selection_summary[label] = {
            "candidate_count": int(len(label_indices)),
            "selected_count": int(len(chosen_set)),
            "far_count": int(len(far_indices)),
            "bridge_count": int(len(bridge_indices)),
            "selected_distance_median": float(np.median(label_chosen_distances)),
            "selected_distance_p05": float(np.percentile(label_chosen_distances, 5)),
            "selected_distance_p95": float(np.percentile(label_chosen_distances, 95)),
        }
        selected.extend(cases[index] for index in sorted(chosen_set))

    rng.shuffle(selected)
    selected_counts = Counter(str(row.get("sweep_label", "")) for row in selected)
    if {label: int(selected_counts[label]) for label in desired_counts} != desired_counts:
        raise RuntimeError(f"candidate selection count mismatch: selected={dict(selected_counts)}, desired={desired_counts}")
    print("candidate selection method:", method, flush=True)
    print("candidate selection reference rows:", len(reference_rows), "sampled:", len(reference_matrix), flush=True)
    print("candidate selection summary:", json.dumps(selection_summary, sort_keys=True)[:4000], flush=True)
    return selected


def generate_target_cases(
    seed: int,
    generation_config: dict[str, Any],
    *,
    parent_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    case_counts = generation_config["case_counts"]
    candidate_selection = generation_config.get("candidate_selection", {})
    candidate_selection_enabled = bool(candidate_selection.get("enabled", False)) if isinstance(candidate_selection, dict) else False
    oversample_multiplier = float(candidate_selection.get("oversample_multiplier", 1.0)) if isinstance(candidate_selection, dict) else 1.0
    if candidate_selection_enabled and oversample_multiplier <= 1.0:
        raise ValueError("candidate_selection.oversample_multiplier must be > 1 when candidate selection is enabled")
    generation_case_counts = (
        {
            label: int(np.ceil(count * oversample_multiplier))
            for label, count in case_counts.items()
        }
        if candidate_selection_enabled
        else case_counts
    )
    sampling = generation_config.get("sampling", {})
    space_filling = sampling.get("space_filling", {}) if isinstance(sampling, dict) else {}
    space_filling_enabled = bool(space_filling.get("enabled", False)) if isinstance(space_filling, dict) else False
    space_filling_method = str(space_filling.get("method", "latin_hypercube")) if isinstance(space_filling, dict) else "latin_hypercube"
    if space_filling_enabled and space_filling_method != "latin_hypercube":
        raise ValueError(f"unsupported space-filling method: {space_filling_method!r}")
    space_filling_labels = set(space_filling.get("labels", [])) if isinstance(space_filling, dict) else set()
    relative_angles = sampling.get("relative_angles", {}) if isinstance(sampling, dict) else {}
    relative_angles_enabled = bool(relative_angles.get("enabled", False)) if isinstance(relative_angles, dict) else False
    relative_angle_bins = (
        list(relative_angles.get("bins", ["aligned", "orthogonal", "anti_aligned", "random"]))
        if isinstance(relative_angles, dict)
        else ["aligned", "orthogonal", "anti_aligned", "random"]
    )
    relative_angle_label_pairs = (
        relative_angles.get("label_pairs", {}) if isinstance(relative_angles, dict) else {}
    )
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
    c1_balanced = sampling.get("c1_balanced", {}) if isinstance(sampling, dict) else {}
    c1_balanced_enabled = bool(c1_balanced.get("enabled", False)) if isinstance(c1_balanced, dict) else False
    c1_balanced_labels = set(c1_balanced.get("labels", [])) if isinstance(c1_balanced, dict) else set()
    c1_magnitude_bins: tuple[tuple[float, float], ...] | None = None
    if c1_balanced_enabled:
        raw_bins = c1_balanced.get("magnitude_bins", [[0.0, 10.0], [10.0, 40.0], [40.0, 75.0], [75.0, 100.0]])
        parsed_bins: list[tuple[float, float]] = []
        for item in raw_bins:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError(f"invalid C1 magnitude bin: {item!r}")
            low, high = float(item[0]), float(item[1])
            if low < 0 or high <= low or high > 100.0:
                raise ValueError(f"invalid C1 magnitude bin range: {(low, high)}")
            parsed_bins.append((low, high))
        if not parsed_bins:
            raise ValueError("C1 balanced sampling requires at least one magnitude bin")
        c1_magnitude_bins = tuple(parsed_bins)
    active_failed_subspace = sampling.get("active_failed_subspace_jitter", {}) if isinstance(sampling, dict) else {}
    active_failed_subspace_enabled = (
        bool(active_failed_subspace.get("enabled", False)) if isinstance(active_failed_subspace, dict) else False
    )
    active_failed_subspace_labels = (
        set(active_failed_subspace.get("labels", ["active_failed_subspace_jitter"]))
        if isinstance(active_failed_subspace, dict)
        else {"active_failed_subspace_jitter"}
    )
    field_sets = {
        "S3_high_random": ("S3",),
        "coupled_full_random": ALL_FIELDS,
        "coupled_C1_A1_random": ("C1", "A1"),
        "coupled_C1_A2_random": ("C1", "A2"),
        "coupled_C1_A3_random": ("C1", "A3"),
        "coupled_C1_B2_random": ("C1", "B2"),
        "coupled_C1_S3_random": ("C1", "S3"),
        "coupled_C1_A1_C3_random": ("C1", "A1", "C3"),
        "coupled_C1_A1_S3_random": ("C1", "A1", "S3"),
        "coupled_C1_C3_random": ("C1", "C3"),
        "coupled_C1_C3_A2_random": ("C1", "C3", "A2"),
        "coupled_C1_A1_C3_A2_random": ("C1", "A1", "C3", "A2"),
        "coupled_C1_C3_S3_random": ("C1", "C3", "S3"),
        "coupled_C1_A3_S3_random": ("C1", "A3", "S3"),
        "coupled_A1_S3_random": ("A1", "S3"),
        "coupled_B2_S3_random": ("B2", "S3"),
        "coupled_A1_B2_S3_random": ("A1", "B2", "S3"),
        "coupled_C3_A3_S3_random": ("C3", "A3", "S3"),
        "coupled_A1_B2_random": ("A1", "B2"),
        "coupled_A1_A2_B2_random": ("A1", "A2", "B2"),
        "coupled_A2_B2_random": ("A2", "B2"),
        "coupled_C3_B2_random": ("C3", "B2"),
        "coupled_A3_S3_random": ("A3", "S3"),
    }
    cases: list[dict[str, Any]] = []
    for label, count in generation_case_counts.items():
        if active_failed_subspace_enabled and label in active_failed_subspace_labels:
            cases.extend(generate_active_failed_subspace_cases(count, rng, active_failed_subspace))
            continue
        if label != "coupled_sparse_random" and label not in field_sets:
            raise KeyError(f"unknown targeted case label: {label}")
        use_space_filling = space_filling_enabled and label in space_filling_labels
        label_units = (
            latin_hypercube(count, SAMPLING_DIMENSIONS, rng)
            if use_space_filling
            else [None] * count
        )
        relative_pairs = relative_angle_label_pairs.get(label, []) if relative_angles_enabled else []
        for label_index in range(count):
            params = dict(BASELINE_PARAMETERS)
            params["sweep_label"] = label
            params["sampling_method"] = space_filling_method if use_space_filling else "random"
            angle_bin = (
                str(relative_angle_bins[label_index % len(relative_angle_bins)])
                if relative_pairs and relative_angle_bins
                else ""
            )
            params["sampling_relative_angle_bin"] = angle_bin
            label_s3_range = s3_amp_range if s3_tail_enabled and label in s3_tail_labels else None
            label_c1_bins = c1_magnitude_bins if c1_balanced_enabled and label in c1_balanced_labels else None
            if label == "coupled_sparse_random":
                active_count = int(rng.integers(2, min(5, len(ALL_FIELDS)) + 1))
                if s3_tail_enabled and force_s3_in_sparse:
                    available = tuple(field for field in ALL_FIELDS if field != "S3")
                    other_count = max(1, active_count - 1)
                    fields = ("S3", *tuple(rng.choice(available, size=other_count, replace=False)))
                else:
                    fields = tuple(rng.choice(ALL_FIELDS, size=active_count, replace=False))
                randomized = randomize(
                    params,
                    fields,
                    rng,
                    sparse=True,
                    s3_amp_range=label_s3_range,
                    c1_magnitude_bins=label_c1_bins,
                    unit_values=label_units[label_index],
                )
                apply_relative_angle_controls(randomized, relative_pairs, angle_bin, rng)
                cases.append(randomized)
            else:
                randomized = randomize(
                    params,
                    field_sets[label],
                    rng,
                    s3_amp_range=label_s3_range,
                    c1_magnitude_bins=label_c1_bins,
                    unit_values=label_units[label_index],
                )
                apply_relative_angle_controls(randomized, relative_pairs, angle_bin, rng)
                cases.append(randomized)
    rng.shuffle(cases)
    if candidate_selection_enabled:
        if parent_rows is None:
            raise ValueError("candidate selection requires parent_rows")
        cases = select_candidate_cases_by_parent_nn(
            cases,
            desired_counts=case_counts,
            parent_rows=parent_rows,
            config=candidate_selection,
            rng=rng,
        )
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
            for field in SAMPLING_METADATA_FIELDS:
                row[field] = params.get(field, "")
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


TARGET_PHYSICAL_SCALES = {
    "C1": 100.0,
    "C3": 2.0,
    "A1_x": 60.0,
    "A1_y": 60.0,
    "B2_x": 3.0,
    "B2_y": 3.0,
    "A2_x": 16.0,
    "A2_y": 16.0,
    "S3_x": 100.0,
    "S3_y": 100.0,
    "A3_x": 100.0,
    "A3_y": 100.0,
}


def load_active_failed_subspace_centers(path: Path) -> list[dict[str, Any]]:
    data = read_json(path)
    centers = data.get("targeted_subspace_centers", [])
    if not isinstance(centers, list) or not centers:
        raise ValueError(f"active failed-subspace spec has no targeted_subspace_centers: {path}")
    return [dict(center) for center in centers]


def center_weight(center: dict[str, Any]) -> float:
    n = float(center.get("n") or 1.0)
    error = float(center.get("median_weighted_error") or 0.01)
    nn_distance = float(center.get("median_nn_distance_12d") or 0.5)
    class_bonus = 1.25 if center.get("dominant_failure_class") == "coverage_limited_sparse_failure" else 1.0
    return max(1e-6, math.sqrt(max(n, 1.0)) * max(error, 1e-4) * max(nn_distance, 0.1) * class_bonus)


def active_center_to_normalized_vector(center: dict[str, Any]) -> np.ndarray:
    values = []
    for name in TARGET_COLUMNS:
        value = float(center.get(f"center_{name}") or 0.0)
        values.append(value / TARGET_PHYSICAL_SCALES[name])
    return np.asarray(values, dtype=float)


def normalized_vector_to_case(
    values: np.ndarray,
    *,
    sweep_label: str,
    sampling_method: str,
    source_center: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    values = np.clip(values, -1.0, 1.0)
    row = dict(BASELINE_PARAMETERS)
    row["sweep_label"] = sweep_label
    row["sampling_method"] = sampling_method
    row["sampling_candidate_role"] = "active_failed_subspace_jitter"
    row["sampling_parent_nn_distance_12d"] = source_center.get("median_nn_distance_12d", "")
    row["C1"] = float(np.clip(values[0] * TARGET_PHYSICAL_SCALES["C1"], -100.0, 100.0))
    row["C3"] = float(np.clip(values[1] * TARGET_PHYSICAL_SCALES["C3"], 0.0, 2.0))
    index_by_name = {name: index for index, name in enumerate(TARGET_COLUMNS)}
    for group, max_amp in (("A1", 60.0), ("B2", 3.0), ("A2", 16.0), ("S3", 100.0), ("A3", 100.0)):
        x = float(values[index_by_name[f"{group}_x"]] * TARGET_PHYSICAL_SCALES[f"{group}_x"])
        y = float(values[index_by_name[f"{group}_y"]] * TARGET_PHYSICAL_SCALES[f"{group}_y"])
        amp = min(max_amp, math.hypot(x, y))
        if amp <= 1e-10:
            angle = float(rng.uniform(0.0, 360.0))
        else:
            angle = math.degrees(math.atan2(y, x)) % 360.0
        row[f"{group}_amp"] = float(amp)
        row[f"{group}_phase"] = phase_from_vector_angle(group, angle)
    set_relative_angle_bin(row)
    return row


def generate_active_failed_subspace_cases(
    count: int,
    rng: np.random.Generator,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    spec_path = Path(config["spec_path"])
    centers = load_active_failed_subspace_centers(spec_path)
    weights = np.asarray([center_weight(center) for center in centers], dtype=float)
    weights = weights / np.sum(weights)
    sigma = float(config.get("normalized_jitter_sigma", 0.075))
    min_sigma = float(config.get("normalized_jitter_min_sigma", 0.025))
    max_sigma = float(config.get("normalized_jitter_max_sigma", 0.14))
    sigma = float(np.clip(sigma, min_sigma, max_sigma))
    sweep_label = str(config.get("sweep_label", "active_failed_subspace_jitter"))
    sampling_method = str(config.get("sampling_method", "active_failed_subspace_jitter"))
    rows = []
    for _ in range(count):
        center = centers[int(rng.choice(len(centers), p=weights))]
        base = active_center_to_normalized_vector(center)
        jittered = base + rng.normal(loc=0.0, scale=sigma, size=base.shape)
        jittered[1] = np.clip(jittered[1], 0.0, 1.0)
        rows.append(
            normalized_vector_to_case(
                jittered,
                sweep_label=sweep_label,
                sampling_method=sampling_method,
                source_center=center,
                rng=rng,
            )
        )
    return rows


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


def target_scale_array() -> np.ndarray:
    return np.asarray([100.0, 2.0, 60.0, 60.0, 3.0, 3.0, 16.0, 16.0, 100.0, 100.0, 100.0, 100.0], dtype=float)


def relative_angle_deg(row: dict[str, Any], left: str, right: str) -> float:
    left_angle = vector_angle_from_phase(left, float(row.get(f"{left}_phase", 0.0) or 0.0))
    right_angle = vector_angle_from_phase(right, float(row.get(f"{right}_phase", 0.0) or 0.0))
    delta = (left_angle - right_angle + 180.0) % 360.0 - 180.0
    return float(delta)


def angle_category(delta_deg: float) -> str:
    value = abs(float(delta_deg))
    if value <= 22.5 or value >= 157.5:
        return "aligned_or_anti_aligned" if value <= 22.5 else "anti_aligned"
    if 67.5 <= value <= 112.5:
        return "orthogonal"
    return "oblique"


def set_relative_angle_bin(row: dict[str, Any]) -> None:
    pairs = []
    for left, right in (("A1", "S3"), ("B2", "S3"), ("A3", "S3")):
        if float(row.get(f"{left}_amp") or 0.0) > COUPLING_EPSILON and float(row.get(f"{right}_amp") or 0.0) > COUPLING_EPSILON:
            pairs.append(f"{left}_{right}:{angle_category(relative_angle_deg(row, left, right))}")
    row["sampling_relative_angle_bin"] = ";".join(pairs)


def sampled_nearest_neighbor_distances(
    matrix: np.ndarray,
    *,
    rng: np.random.Generator,
    sample_size: int = 3000,
    chunk_size: int = 300,
) -> dict[str, Any]:
    if len(matrix) < 2:
        return {"n_sampled": int(len(matrix))}
    sample_size = min(int(sample_size), len(matrix))
    sample_indices = rng.choice(len(matrix), size=sample_size, replace=False)
    sample = matrix[sample_indices]
    nearest: list[np.ndarray] = []
    for start in range(0, sample_size, chunk_size):
        block = sample[start : start + chunk_size]
        distances = np.sqrt(np.sum((block[:, None, :] - sample[None, :, :]) ** 2, axis=2))
        row_indices = np.arange(start, min(start + chunk_size, sample_size))
        distances[np.arange(len(block)), row_indices] = np.inf
        nearest.append(np.min(distances, axis=1))
    values = np.concatenate(nearest)
    return {
        "n_sampled": int(sample_size),
        "dimension": int(matrix.shape[1]),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def coefficient_coverage_diagnostics(rows: list[dict[str, Any]], seed: int = 123) -> dict[str, Any]:
    matrix = target_matrix(rows, tuple(TARGET_COLUMNS))
    if not len(matrix):
        return {"n": 0}
    normalized = matrix / target_scale_array()[None, :]
    rng = np.random.default_rng(seed)
    marginal_bins: dict[str, Any] = {}
    for index, target in enumerate(TARGET_COLUMNS):
        if target == "C3":
            hist_range = (0.0, 1.0)
        else:
            hist_range = (-1.0, 1.0)
        counts, edges = np.histogram(normalized[:, index], bins=20, range=hist_range)
        marginal_bins[target] = {
            "counts": counts.astype(int).tolist(),
            "bin_edges": edges.tolist(),
            "empty_bins": int(np.sum(counts == 0)),
            "min_nonzero_bin_count": int(np.min(counts[counts > 0])) if np.any(counts > 0) else 0,
        }
    pair_bins: dict[str, Any] = {}
    for left, right in [
        ("S3_x", "S3_y"),
        ("A3_x", "A3_y"),
        ("B2_x", "B2_y"),
        ("A1_x", "S3_x"),
        ("B2_x", "S3_x"),
        ("A3_x", "S3_x"),
    ]:
        left_index = TARGET_COLUMNS.index(left)
        right_index = TARGET_COLUMNS.index(right)
        counts, _, _ = np.histogram2d(
            normalized[:, left_index],
            normalized[:, right_index],
            bins=16,
            range=[[-1.0, 1.0], [-1.0, 1.0]],
        )
        pair_bins[f"{left}__{right}"] = {
            "empty_bins": int(np.sum(counts == 0)),
            "occupied_bins": int(np.sum(counts > 0)),
            "total_bins": int(counts.size),
            "min_nonzero_bin_count": int(np.min(counts[counts > 0])) if np.any(counts > 0) else 0,
        }
    relative_pairs: dict[str, Any] = {}
    for left, right in [("A1", "S3"), ("B2", "S3"), ("A3", "S3")]:
        active_rows = [
            row
            for row in rows
            if float(row.get(f"{left}_amp", 0.0) or 0.0) > COUPLING_EPSILON
            and float(row.get(f"{right}_amp", 0.0) or 0.0) > COUPLING_EPSILON
        ]
        deltas = [relative_angle_deg(row, left, right) for row in active_rows]
        categories = Counter(angle_category(delta) for delta in deltas)
        relative_pairs[f"{left}_{right}"] = {
            "n": len(active_rows),
            "category_counts": dict(categories),
            "mean_abs_delta_deg": float(np.mean(np.abs(deltas))) if deltas else 0.0,
            "p95_abs_delta_deg": float(np.percentile(np.abs(deltas), 95)) if deltas else 0.0,
        }
    sampling_counts = {
        "method_counts": dict(Counter(str(row.get("sampling_method", "")) for row in rows)),
        "relative_angle_bin_counts": dict(Counter(str(row.get("sampling_relative_angle_bin", "")) for row in rows)),
    }
    return {
        "n": int(len(rows)),
        "target_columns": TARGET_COLUMNS,
        "normalization_scales": dict(zip(TARGET_COLUMNS, target_scale_array().tolist())),
        "marginal_bins": marginal_bins,
        "pair_bins": pair_bins,
        "relative_angle_coverage": relative_pairs,
        "sampling_metadata": sampling_counts,
        "sampled_nearest_neighbor_distance": sampled_nearest_neighbor_distances(normalized, rng=rng),
    }


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
        "new_coefficient_coverage": coefficient_coverage_diagnostics(new_rows),
        "combined_coefficient_coverage": coefficient_coverage_diagnostics(combined_rows),
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
        "--new-row-split-hint",
        default=SPLIT_HINT_TRAINING_ONLY,
        help=(
            "dataset_split_hint assigned to newly generated rows. The default "
            "keeps expansion rows training-only. Use a non-training hint, such "
            "as benchmark_v2, only with an explicit frozen split manifest."
        ),
    )
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
    target_cases = generate_target_cases(args.seed, generation_config, parent_rows=source_rows)
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
    attach_new_row_metadata(new_rows, args.new_row_split_hint)
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
        "appended_training_only_row_count": (
            len(new_rows) if args.new_row_split_hint == SPLIT_HINT_TRAINING_ONLY else 0
        ),
        "appended_non_training_benchmark_row_count": (
            0 if args.new_row_split_hint == SPLIT_HINT_TRAINING_ONLY else len(new_rows)
        ),
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
        "new_row_split_hint": args.new_row_split_hint,
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
