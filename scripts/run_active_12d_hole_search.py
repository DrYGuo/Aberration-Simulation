"""Active 12D hole-search diagnostics for the frozen v13 regression model.

This script is diagnostic-only. It proposes coefficient vectors in normalized
12D target space, optionally simulates/extracts features for the selected probe
set, runs frozen-model inference if a checkpoint is available, and writes a
compact recommendation for the next data-expansion strategy.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from feature_regression_model import TARGET_COLUMNS, Standardizer, file_sha256, target_from_row
from generate_targeted_enhanced_dataset import (
    ALL_FIELDS,
    COMBINATION_FIELDS,
    PHASE_PERIODS,
    SAMPLING_METADATA_FIELDS,
    angle_category,
    latin_hypercube,
    phase_from_vector_angle,
    relative_angle_deg,
    simulate_rows,
    target_scale_array,
    vector_angle_from_phase,
)
from run_model_selection_candidate import (
    DEFAULT_TARGET_PHYSICAL_SCALES,
    TARGET_WEIGHTS,
    build_model,
    frozen_benchmark_split,
    predict_numpy_chunked,
    row_float,
)


VECTOR_GROUPS = ("A1", "A2", "B2", "S3", "A3")
VECTOR_ORDERS = {"A1": 2, "B2": 1, "A2": 3, "S3": 2, "A3": 4}
AMP_MAX = {"A1": 60.0, "A2": 16.0, "B2": 3.0, "S3": 100.0, "A3": 100.0}
GENE_NAMES = (
    "C1",
    "C3",
    "A1_amp",
    "A1_angle",
    "A2_amp",
    "A2_angle",
    "B2_amp",
    "B2_angle",
    "S3_amp",
    "S3_angle",
    "A3_amp",
    "A3_angle",
)
REFERENCE_LABELS = (
    "coupled_full_random",
    "coupled_sparse_random",
    "coupled_A3_S3_random",
    "coupled_B2_S3_random",
    "coupled_A1_B2_S3_random",
    "coupled_A1_S3_random",
    "coupled_C3_A3_S3_random",
    "S3_high_random",
)


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
    return result.stdout.strip() if result.returncode == 0 else None


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open() as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def load_feature_columns(csv_path: Path) -> list[str]:
    for name in ("feature_columns_enhanced.json", "feature_columns.json"):
        path = csv_path.parent / name
        if path.exists():
            data = read_json(path)
            if isinstance(data, dict) and "features" in data:
                return list(data["features"])
            return list(data)
    raise FileNotFoundError(f"missing feature columns beside {csv_path}")


def target_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    if not rows:
        return np.zeros((0, len(TARGET_COLUMNS)), dtype=np.float32)
    return np.asarray([target_from_row(row) for row in rows], dtype=np.float32)


def normalized_target_matrix(rows: list[dict[str, Any]], scales: np.ndarray) -> np.ndarray:
    matrix = target_matrix(rows)
    return (matrix / scales[None, :]).astype(np.float32) if len(matrix) else matrix


def normalized_target_from_row(row: dict[str, Any], scales: np.ndarray) -> np.ndarray:
    return np.asarray(target_from_row(row), dtype=np.float32) / scales


def gene_to_row(gene: np.ndarray, label: str, mode: str) -> dict[str, Any]:
    gene = np.clip(np.asarray(gene, dtype=float), 0.0, np.nextafter(1.0, 0.0))
    row: dict[str, Any] = {
        "sweep_label": label,
        "sampling_method": f"active_12d_{mode}",
        "sampling_relative_angle_bin": "",
        "sampling_candidate_role": mode,
        "sampling_parent_nn_distance_12d": "",
        "C1": -100.0 + 200.0 * gene[0],
        "C3": 0.05 + 1.95 * gene[1],
        "A1_amp": AMP_MAX["A1"] * gene[2],
        "A1_phase": phase_from_vector_angle("A1", 360.0 * gene[3]),
        "A2_amp": AMP_MAX["A2"] * gene[4],
        "A2_phase": phase_from_vector_angle("A2", 360.0 * gene[5]),
        "B2_amp": AMP_MAX["B2"] * gene[6],
        "B2_phase": phase_from_vector_angle("B2", 360.0 * gene[7]),
        "S3_amp": AMP_MAX["S3"] * gene[8],
        "S3_phase": phase_from_vector_angle("S3", 360.0 * gene[9]),
        "A3_amp": AMP_MAX["A3"] * gene[10],
        "A3_phase": phase_from_vector_angle("A3", 360.0 * gene[11]),
    }
    if label == "S3_high_random":
        for group in ("A1", "A2", "B2", "A3"):
            row[f"{group}_amp"] = 0.0
        row["C1"] = 0.0
        row["C3"] = 0.0
        row["S3_amp"] = 65.0 + 35.0 * gene[8]
    elif label != "coupled_full_random":
        active = {
            "coupled_sparse_random": set(random_sparse_groups(gene)),
            "coupled_A3_S3_random": {"A3", "S3"},
            "coupled_B2_S3_random": {"B2", "S3"},
            "coupled_A1_B2_S3_random": {"A1", "B2", "S3"},
            "coupled_A1_S3_random": {"A1", "S3"},
            "coupled_C3_A3_S3_random": {"C3", "A3", "S3"},
        }.get(label, set(ALL_FIELDS))
        for scalar in ("C1", "C3"):
            if scalar not in active:
                row[scalar] = 0.0
        for group in VECTOR_GROUPS:
            if group not in active:
                row[f"{group}_amp"] = 0.0
    set_angle_bin(row)
    for field in COMBINATION_FIELDS:
        row.setdefault(field, 0.0)
    for field in SAMPLING_METADATA_FIELDS:
        row.setdefault(field, "")
    return row


def row_to_gene(row: dict[str, Any]) -> np.ndarray:
    values = [
        (float(row.get("C1", 0.0) or 0.0) + 100.0) / 200.0,
        (float(row.get("C3", 0.05) or 0.05) - 0.05) / 1.95,
    ]
    for group in ("A1", "A2", "B2", "S3", "A3"):
        amp = float(row.get(f"{group}_amp", 0.0) or 0.0) / AMP_MAX[group]
        angle = vector_angle_from_phase(group, float(row.get(f"{group}_phase", 0.0) or 0.0)) / 360.0
        values.extend([amp, angle])
    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def random_sparse_groups(gene: np.ndarray) -> list[str]:
    groups = list(ALL_FIELDS)
    scores = np.asarray(
        [gene[0], gene[1], gene[2], gene[4], gene[6], gene[10], gene[8]],
        dtype=float,
    )
    count = 2 + int(np.floor(4.0 * float(gene[3])))
    order = np.argsort(scores)[::-1]
    return [groups[int(index)] for index in order[: min(count, len(groups))]]


def set_angle_bin(row: dict[str, Any]) -> None:
    pairs = []
    for pair in (("A1", "S3"), ("B2", "S3"), ("A3", "S3")):
        left, right = pair
        if float(row.get(f"{left}_amp", 0.0) or 0.0) > 1e-8 and float(row.get(f"{right}_amp", 0.0) or 0.0) > 1e-8:
            pairs.append(f"{left}_{right}:{angle_category(relative_angle_deg(row, left, right))}")
    row["sampling_relative_angle_bin"] = ";".join(pairs)


def weighted_label_choice(rng: np.random.Generator, weights: dict[str, float]) -> str:
    labels = list(weights)
    raw = np.asarray([float(weights[label]) for label in labels], dtype=float)
    raw = raw / np.sum(raw)
    return str(rng.choice(labels, p=raw))


def lhs_genes(n_rows: int, rng: np.random.Generator) -> np.ndarray:
    try:
        from scipy.stats import qmc

        sampler = qmc.Sobol(d=len(GENE_NAMES), scramble=True, seed=int(rng.integers(0, 2**31 - 1)))
        m = int(math.ceil(math.log2(max(n_rows, 2))))
        return sampler.random_base2(m=m)[:n_rows]
    except Exception:
        rows = latin_hypercube(n_rows, GENE_NAMES, rng)
        return np.asarray([[row[name] for name in GENE_NAMES] for row in rows], dtype=float)


def build_nn_model(reference: np.ndarray, sample_size: int, seed: int):
    if len(reference) > sample_size:
        rng = np.random.default_rng(seed)
        reference = reference[rng.choice(len(reference), size=sample_size, replace=False)]
    try:
        from sklearn.neighbors import NearestNeighbors

        model = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean")
        model.fit(reference)
        return model, reference, "sklearn"
    except Exception:
        return None, reference, "numpy_sampled"


def nearest_distances(nn_model, reference: np.ndarray, query: np.ndarray, chunk_size: int) -> np.ndarray:
    if len(query) == 0:
        return np.asarray([], dtype=np.float32)
    if nn_model is not None:
        parts = []
        for start in range(0, len(query), chunk_size):
            distances, _ = nn_model.kneighbors(query[start : start + chunk_size])
            parts.append(distances[:, 0])
        return np.concatenate(parts).astype(np.float32)
    parts = []
    for start in range(0, len(query), min(chunk_size, 512)):
        block = query[start : start + min(chunk_size, 512)]
        distances = np.sqrt(np.sum((block[:, None, :] - reference[None, :, :]) ** 2, axis=2))
        parts.append(np.min(distances, axis=1))
    return np.concatenate(parts).astype(np.float32)


def residual_seed_rows(v13_rows: list[dict[str, str]], run_dir: Path, max_rows: int) -> list[dict[str, Any]]:
    path = run_dir / "residual_vs_nn_distance_top_residuals.csv"
    if not path.exists():
        return []
    seeds: list[dict[str, Any]] = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            try:
                index = int(row.get("row_index", ""))
            except ValueError:
                continue
            if 0 <= index < len(v13_rows):
                copied = dict(v13_rows[index])
                copied["source_residual_weighted_abs_error"] = row.get("weighted_abs_error", "")
                copied["source_residual_nn_distance_12d"] = row.get("nn_distance_12d", "")
                seeds.append(copied)
            if len(seeds) >= max_rows:
                break
    return seeds


def physical_relevance(rows: list[dict[str, Any]]) -> np.ndarray:
    scores = []
    for row in rows:
        active = 0
        for scalar in ("C1", "C3"):
            active += abs(float(row.get(scalar, 0.0) or 0.0)) > 1e-8
        for group in VECTOR_GROUPS:
            active += float(row.get(f"{group}_amp", 0.0) or 0.0) > 1e-8
        s3 = float(row.get("S3_amp", 0.0) or 0.0) / 100.0
        a3 = float(row.get("A3_amp", 0.0) or 0.0) / 100.0
        b2 = float(row.get("B2_amp", 0.0) or 0.0) / 3.0
        active_score = min(active / 5.0, 1.0)
        vector_score = 0.45 * s3 + 0.35 * a3 + 0.20 * b2
        scores.append(float(0.55 * active_score + 0.45 * vector_score))
    return np.asarray(scores, dtype=np.float32)


def residual_similarity(query: np.ndarray, residual_centers: np.ndarray) -> np.ndarray:
    if len(query) == 0 or len(residual_centers) == 0:
        return np.zeros(len(query), dtype=np.float32)
    try:
        from sklearn.neighbors import NearestNeighbors

        model = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean")
        model.fit(residual_centers)
        distances, _ = model.kneighbors(query)
        d = distances[:, 0]
    except Exception:
        d = []
        for start in range(0, len(query), 512):
            block = query[start : start + 512]
            distances = np.sqrt(np.sum((block[:, None, :] - residual_centers[None, :, :]) ** 2, axis=2))
            d.append(np.min(distances, axis=1))
        d = np.concatenate(d)
    sigma = max(float(np.quantile(d, 0.25)), 0.05)
    return np.exp(-d / sigma).astype(np.float32)


def boundary_penalty(genes: np.ndarray) -> np.ndarray:
    edge = np.minimum(genes, 1.0 - genes)
    near_edge = edge < 0.035
    return np.mean(near_edge, axis=1).astype(np.float32)


def score_candidates(
    rows: list[dict[str, Any]],
    reference_nn,
    reference_matrix: np.ndarray,
    residual_centers: np.ndarray,
    scales: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    query = normalized_target_matrix(rows, scales)
    distances = nearest_distances(
        reference_nn,
        reference_matrix,
        query,
        int(config["proposal"].get("nearest_neighbor_chunk_size", 25000)),
    )
    if len(distances):
        nn_score = (distances - np.min(distances)) / max(float(np.ptp(distances)), 1e-8)
    else:
        nn_score = np.zeros(len(rows), dtype=np.float32)
    sim_score = residual_similarity(query, residual_centers)
    phys_score = physical_relevance(rows)
    genes = np.asarray([row_to_gene(row) for row in rows], dtype=float)
    diversity_proxy = np.std(genes, axis=1).astype(np.float32)
    penalty = boundary_penalty(genes)
    weights = config["proposal"]["genetic_algorithm"]["fitness_weights"]
    score = (
        float(weights.get("nearest_neighbor_distance", 0.50)) * nn_score
        + float(weights.get("known_residual_similarity", 0.22)) * sim_score
        + float(weights.get("physical_relevance", 0.14)) * phys_score
        + float(weights.get("diversity_proxy", 0.08)) * diversity_proxy
        - float(weights.get("boundary_penalty", 0.06)) * penalty
    )
    return score.astype(np.float32), distances


def make_random_candidates(config: dict[str, Any], rng: np.random.Generator, count: int, mode: str) -> list[dict[str, Any]]:
    weights = config["proposal"]["regime_weights"]
    genes = lhs_genes(count, rng) if mode == "sobol_lhs" else rng.random((count, len(GENE_NAMES)))
    rows = []
    for gene in genes:
        label = weighted_label_choice(rng, weights)
        rows.append(gene_to_row(gene, label, mode))
    return rows


def make_residual_jitter_candidates(
    seed_rows: list[dict[str, Any]],
    config: dict[str, Any],
    rng: np.random.Generator,
    count: int,
) -> list[dict[str, Any]]:
    if not seed_rows:
        return make_random_candidates(config, rng, count, "residual_jitter")
    rows = []
    for _ in range(count * 4):
        base = seed_rows[int(rng.integers(0, len(seed_rows)))]
        gene = row_to_gene(base)
        scale = rng.uniform(0.025, 0.12, size=len(GENE_NAMES))
        jittered = np.clip(gene + rng.normal(0.0, scale), 0.0, 1.0)
        label = str(base.get("sweep_label", "")) or weighted_label_choice(rng, config["proposal"]["regime_weights"])
        if label not in config["proposal"]["regime_weights"]:
            label = weighted_label_choice(rng, config["proposal"]["regime_weights"])
        rows.append(gene_to_row(jittered, label, "residual_jitter"))
    return rows


def make_high_amp_alignment_candidates(
    config: dict[str, Any],
    rng: np.random.Generator,
    count: int,
) -> list[dict[str, Any]]:
    """Probe the high-amplitude A1/B2/S3/A3 alignment corner found in cycle 2."""
    rows: list[dict[str, Any]] = []
    a3_offsets = np.asarray([-12.0, -6.0, 0.0, 6.0, 12.0, 168.0, 180.0, 192.0])
    b2_offsets = np.asarray([35.0, 55.0, 75.0, 105.0, 125.0, 145.0])
    a1_offsets = np.asarray([35.0, 55.0, 75.0, 105.0, 125.0, 145.0])
    for _ in range(max(count * 3, count)):
        s3_angle = float(rng.uniform(0.0, 360.0))
        a3_angle = s3_angle + float(rng.choice(a3_offsets)) + float(rng.normal(0.0, 3.0))
        b2_angle = s3_angle + float(rng.choice(b2_offsets)) + float(rng.normal(0.0, 5.0))
        a1_angle = s3_angle + float(rng.choice(a1_offsets)) + float(rng.normal(0.0, 5.0))
        row: dict[str, Any] = {
            "sweep_label": "coupled_full_random",
            "sampling_method": "active_12d_high_amp_alignment",
            "sampling_candidate_role": "high_amp_alignment",
            "C1": float(rng.choice([-1.0, 1.0]) * rng.uniform(60.0, 100.0)) if rng.random() < 0.25 else float(-rng.uniform(60.0, 100.0)),
            "C3": float(rng.uniform(0.05, 0.45) if rng.random() < 0.75 else rng.uniform(0.45, 1.2)),
            "A1_amp": float(rng.uniform(42.0, 60.0)),
            "A1_phase": phase_from_vector_angle("A1", a1_angle),
            "A2_amp": float(rng.uniform(8.0, 16.0)),
            "A2_phase": phase_from_vector_angle("A2", float(rng.uniform(0.0, 360.0))),
            "B2_amp": float(rng.uniform(2.5, 3.0)),
            "B2_phase": phase_from_vector_angle("B2", b2_angle),
            "S3_amp": float(rng.uniform(90.0, 100.0)),
            "S3_phase": phase_from_vector_angle("S3", s3_angle),
            "A3_amp": float(rng.uniform(90.0, 100.0)),
            "A3_phase": phase_from_vector_angle("A3", a3_angle),
        }
        set_angle_bin(row)
        for field in COMBINATION_FIELDS:
            row.setdefault(field, 0.0)
        for field in SAMPLING_METADATA_FIELDS:
            row.setdefault(field, "")
        rows.append(row)
    return rows


def run_ga(
    config: dict[str, Any],
    rng: np.random.Generator,
    reference_nn,
    reference_matrix: np.ndarray,
    residual_centers: np.ndarray,
    scales: np.ndarray,
) -> list[dict[str, Any]]:
    ga = config["proposal"]["genetic_algorithm"]
    population_size = int(ga.get("population_size", 768))
    generations = int(ga.get("generations", 24))
    elite_n = max(2, int(round(population_size * float(ga.get("elite_fraction", 0.16)))))
    immigrant_n = int(round(population_size * float(ga.get("immigrant_fraction", 0.08))))
    mutation_probability = float(ga.get("mutation_probability", 0.18))
    mutation_scale = float(ga.get("mutation_scale", 0.10))
    weights = config["proposal"]["regime_weights"]
    labels = [weighted_label_choice(rng, weights) for _ in range(population_size)]
    population = rng.random((population_size, len(GENE_NAMES)))
    best_rows: list[dict[str, Any]] = []
    best_scores: list[float] = []
    for generation in range(generations):
        rows = [gene_to_row(gene, labels[index], "genetic_algorithm") for index, gene in enumerate(population)]
        scores, distances = score_candidates(rows, reference_nn, reference_matrix, residual_centers, scales, config)
        order = np.argsort(scores)[::-1]
        elites = population[order[:elite_n]]
        elite_labels = [labels[int(index)] for index in order[:elite_n]]
        for index in order[: min(80, len(order))]:
            row = rows[int(index)]
            row["ga_generation"] = generation
            row["active_search_score"] = float(scores[int(index)])
            row["sampling_parent_nn_distance_12d"] = float(distances[int(index)])
            best_rows.append(row)
            best_scores.append(float(scores[int(index)]))
        next_population = [gene.copy() for gene in elites]
        next_labels = list(elite_labels)
        while len(next_population) < population_size - immigrant_n:
            left = int(rng.integers(0, elite_n))
            right = int(rng.integers(0, elite_n))
            mask = rng.random(len(GENE_NAMES)) < 0.5
            child = np.where(mask, elites[left], elites[right])
            mutation_mask = rng.random(len(GENE_NAMES)) < mutation_probability
            child = child + mutation_mask * rng.normal(0.0, mutation_scale, size=len(GENE_NAMES))
            next_population.append(np.clip(child, 0.0, 1.0))
            next_labels.append(elite_labels[left] if rng.random() < 0.5 else elite_labels[right])
        while len(next_population) < population_size:
            next_population.append(rng.random(len(GENE_NAMES)))
            next_labels.append(weighted_label_choice(rng, weights))
        population = np.asarray(next_population, dtype=float)
        labels = next_labels
        print(f"GA generation {generation + 1}/{generations}: best={float(scores[order[0]]):.4f}", flush=True)
    ranked = [row for _, row in sorted(zip(best_scores, best_rows), key=lambda item: item[0], reverse=True)]
    return ranked


def greedy_select(rows: list[dict[str, Any]], scores: np.ndarray, distances: np.ndarray, count: int, scales: np.ndarray) -> list[dict[str, Any]]:
    if count <= 0 or not rows:
        return []
    order = np.argsort(scores)[::-1]
    selected: list[dict[str, Any]] = []
    selected_matrix: list[np.ndarray] = []
    min_sep = 0.035
    seen_keys: set[str] = set()
    for raw_index in order:
        row = dict(rows[int(raw_index)])
        vector = normalized_target_from_row(row, scales)
        key = hashlib.sha256(np.round(vector, 5).tobytes()).hexdigest()[:16]
        if key in seen_keys:
            continue
        if selected_matrix:
            d = np.sqrt(np.sum((np.asarray(selected_matrix) - vector[None, :]) ** 2, axis=1))
            if float(np.min(d)) < min_sep and len(selected) < int(0.8 * count):
                continue
        row["active_search_score"] = float(scores[int(raw_index)])
        row["sampling_parent_nn_distance_12d"] = float(distances[int(raw_index)])
        row["active_search_selected_rank"] = len(selected) + 1
        selected.append(row)
        selected_matrix.append(vector)
        seen_keys.add(key)
        if len(selected) >= count:
            break
    if len(selected) < count:
        for raw_index in order:
            row = dict(rows[int(raw_index)])
            vector = normalized_target_from_row(row, scales)
            key = hashlib.sha256(np.round(vector, 5).tobytes()).hexdigest()[:16]
            if key in seen_keys:
                continue
            row["active_search_score"] = float(scores[int(raw_index)])
            row["sampling_parent_nn_distance_12d"] = float(distances[int(raw_index)])
            row["active_search_selected_rank"] = len(selected) + 1
            selected.append(row)
            seen_keys.add(key)
            if len(selected) >= count:
                break
    return selected


def propose_candidates(
    v13_rows: list[dict[str, str]],
    split_indices: dict[str, np.ndarray],
    run_dir: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    proposal = config["proposal"]
    seed = int(proposal.get("seed", 1507))
    rng = np.random.default_rng(seed)
    scales = target_scale_array().astype(np.float32)
    train_rows = [v13_rows[int(index)] for index in split_indices["train"]]
    train_matrix = normalized_target_matrix(train_rows, scales)
    nn_model, reference_matrix, nn_method = build_nn_model(
        train_matrix,
        int(proposal.get("reference_sample_size", 300000)),
        seed,
    )
    seed_rows = residual_seed_rows(v13_rows, run_dir, int(proposal.get("top_residual_seed_count", 1200)))
    residual_centers = normalized_target_matrix(seed_rows, scales)
    mode_counts = {str(k): int(v) for k, v in proposal["mode_counts"].items()}

    selected: list[dict[str, Any]] = []
    mode_summaries: dict[str, Any] = {}
    for mode, count in mode_counts.items():
        if count <= 0:
            continue
        if mode == "genetic_algorithm":
            raw_rows = run_ga(config, rng, nn_model, reference_matrix, residual_centers, scales)
        elif mode == "residual_jitter":
            raw_rows = make_residual_jitter_candidates(seed_rows, config, rng, count)
        elif mode == "high_amp_alignment":
            raw_rows = make_high_amp_alignment_candidates(config, rng, count)
        elif mode in {"far_nn", "bridge_anchor", "sobol_lhs"}:
            pool_count = max(count * 12, int(proposal.get("candidate_pool_size", 45000)) // max(len(mode_counts), 1))
            raw_rows = make_random_candidates(config, rng, pool_count, "sobol_lhs" if mode == "sobol_lhs" else mode)
        else:
            raw_rows = make_random_candidates(config, rng, count * 8, mode)
        scores, distances = score_candidates(raw_rows, nn_model, reference_matrix, residual_centers, scales, config)
        if mode == "bridge_anchor":
            target = float(np.quantile(distances, 0.55)) if len(distances) else 0.0
            scores = -np.abs(distances - target).astype(np.float32) + 0.15 * physical_relevance(raw_rows)
        chosen = greedy_select(raw_rows, scores, distances, count, scales)
        for row in chosen:
            row["proposal_mode"] = mode
        selected.extend(chosen)
        mode_summaries[mode] = {
            "raw_candidates": int(len(raw_rows)),
            "selected": int(len(chosen)),
            "nn_distance_median": float(np.median([float(row["sampling_parent_nn_distance_12d"]) for row in chosen])) if chosen else None,
            "score_median": float(np.median([float(row["active_search_score"]) for row in chosen])) if chosen else None,
        }
        print(f"proposal mode {mode}: selected {len(chosen)}/{count}", flush=True)

    rng.shuffle(selected)
    for index, row in enumerate(selected):
        row["active_probe_id"] = f"active_probe_{index:05d}"
        row["dataset_version"] = config["workflow_id"]
        row["dataset_source"] = "active_12d_hole_search"
        row["dataset_split_hint"] = "diagnostic_only"

    selected_path = output_dir / "selected_probe_design.csv"
    write_csv(
        selected_path,
        selected,
        [
            "active_probe_id",
            "proposal_mode",
            "sweep_label",
            "sampling_method",
            "sampling_relative_angle_bin",
            "sampling_candidate_role",
            "sampling_parent_nn_distance_12d",
            "active_search_score",
            "active_search_selected_rank",
            *COMBINATION_FIELDS,
            "dataset_version",
            "dataset_source",
            "dataset_split_hint",
        ],
    )
    summary = {
        "seed": seed,
        "selected_probe_count": int(len(selected)),
        "reference_train_rows": int(len(train_rows)),
        "reference_matrix_rows": int(len(reference_matrix)),
        "nearest_neighbor_method": nn_method,
        "residual_seed_rows": int(len(seed_rows)),
        "mode_summaries": mode_summaries,
        "selected_probe_design_csv": str(selected_path),
        "regime_counts": dict(Counter(str(row.get("sweep_label", "")) for row in selected)),
        "proposal_mode_counts": dict(Counter(str(row.get("proposal_mode", "")) for row in selected)),
    }
    (output_dir / "proposal_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return selected, summary


def find_checkpoint(run_dir: Path) -> Path | None:
    for name in ("model_loop_candidate.pt", "model.pt"):
        path = run_dir / name
        if path.exists():
            return path
    matches = sorted(run_dir.glob("*.pt"))
    return matches[0] if matches else None


def feature_matrix(rows: list[dict[str, Any]], feature_columns: list[str]) -> np.ndarray:
    return np.asarray([[row_float(row, name) for name in feature_columns] for row in rows], dtype=np.float32)


def run_inference(
    v13_rows: list[dict[str, str]],
    v13_csv: Path,
    run_dir: Path,
    split_indices: dict[str, np.ndarray],
    probe_rows: list[dict[str, Any]],
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    checkpoint = find_checkpoint(run_dir)
    if checkpoint is None:
        return None, {
            "status": "skipped",
            "reason": f"no .pt checkpoint found under {run_dir}",
        }
    try:
        import torch
    except Exception as exc:
        return None, {"status": "skipped", "reason": f"torch unavailable: {type(exc).__name__}: {exc}"}

    metrics_path = run_dir / "metrics_model_loop.json"
    metrics = read_json(metrics_path)
    training_config = metrics.get("training_config", {})
    feature_columns = load_feature_columns(v13_csv)
    X_train_full = feature_matrix(v13_rows, feature_columns)
    y_train_full = target_matrix(v13_rows)
    train_index = split_indices["train"]
    x_scaler = Standardizer().fit(X_train_full[train_index])
    y_scaler = Standardizer().fit(y_train_full[train_index])
    X_probe = feature_matrix(probe_rows, feature_columns)
    y_true = target_matrix(probe_rows)
    Xn_probe = x_scaler.transform(X_probe).astype(np.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(
        Xn_probe.shape[1],
        len(TARGET_COLUMNS),
        int(training_config.get("hidden_dim", 320)),
        float(training_config.get("dropout_probability", training_config.get("dropout", 0.075))),
        str(training_config.get("architecture", "grouped_heads")),
    ).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        pred_scaled = predict_numpy_chunked(
            model,
            torch.tensor(Xn_probe, dtype=torch.float32, device=device),
            batch_size=int(config.get("inference", {}).get("predict_batch_size", 65536)),
        )
    y_pred = y_scaler.inverse_transform(pred_scaled)
    manifest = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "metrics_path": str(metrics_path),
        "feature_columns": str(v13_csv.parent / "feature_columns_enhanced.json"),
        "feature_count": int(len(feature_columns)),
        "device": device,
        "n_probe_rows": int(len(probe_rows)),
    }
    return y_pred.astype(np.float32), manifest


def vector_error(y_true: np.ndarray, y_pred: np.ndarray, row: int, group: str) -> float:
    xi = TARGET_COLUMNS.index(f"{group}_x")
    yi = TARGET_COLUMNS.index(f"{group}_y")
    return float(np.hypot(y_pred[row, xi] - y_true[row, xi], y_pred[row, yi] - y_true[row, yi]))


def true_amp(y_true: np.ndarray, row: int, group: str) -> float:
    xi = TARGET_COLUMNS.index(f"{group}_x")
    yi = TARGET_COLUMNS.index(f"{group}_y")
    return float(np.hypot(y_true[row, xi], y_true[row, yi]))


def relative_angle_from_y(y_true: np.ndarray, row: int, left: str, right: str) -> float:
    lx = y_true[row, TARGET_COLUMNS.index(f"{left}_x")]
    ly = y_true[row, TARGET_COLUMNS.index(f"{left}_y")]
    rx = y_true[row, TARGET_COLUMNS.index(f"{right}_x")]
    ry = y_true[row, TARGET_COLUMNS.index(f"{right}_y")]
    left_angle = math.degrees(math.atan2(float(ly), float(lx)))
    right_angle = math.degrees(math.atan2(float(ry), float(rx)))
    return float((left_angle - right_angle + 180.0) % 360.0 - 180.0)


def summarize_evaluation(
    probe_rows: list[dict[str, Any]],
    y_pred: np.ndarray,
    train_matrix: np.ndarray,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    scales = target_scale_array().astype(np.float32)
    y_true = target_matrix(probe_rows)
    normalized_abs = np.abs(y_pred - y_true) / scales[None, :]
    weights = np.asarray([TARGET_WEIGHTS[name] for name in TARGET_COLUMNS], dtype=np.float32)
    weights = weights / max(float(np.sum(weights)), 1e-8)
    weighted_error = np.sum(normalized_abs * weights[None, :], axis=1)
    query_matrix = y_true / scales[None, :]
    nn_model, reference_matrix, nn_method = build_nn_model(
        train_matrix,
        int(config["proposal"].get("reference_sample_size", 300000)),
        int(config["proposal"].get("seed", 1507)) + 1,
    )
    nn_dist = nearest_distances(
        nn_model,
        reference_matrix,
        query_matrix.astype(np.float32),
        int(config["proposal"].get("nearest_neighbor_chunk_size", 25000)),
    )
    failure_top_fraction = float(config["classification"].get("failure_top_fraction", 0.10))
    threshold = float(np.quantile(weighted_error, 1.0 - failure_top_fraction))
    sparse_threshold = float(np.quantile(nn_dist, float(config["classification"].get("coverage_limited_nn_quantile", 0.75))))
    dense_threshold = float(np.quantile(nn_dist, float(config["classification"].get("dense_failure_nn_quantile", 0.50))))

    rows_payload: list[dict[str, Any]] = []
    for i, row in enumerate(probe_rows):
        rel_a3_s3 = relative_angle_from_y(y_true, i, "A3", "S3")
        rel_b2_s3 = relative_angle_from_y(y_true, i, "B2", "S3")
        failure_class = "not_top_failure"
        if float(weighted_error[i]) >= threshold:
            if float(nn_dist[i]) >= sparse_threshold:
                failure_class = "coverage_limited_sparse_failure"
            elif float(nn_dist[i]) <= dense_threshold:
                failure_class = "dense_model_feature_loss_failure"
            else:
                failure_class = "mixed_failure"
        rows_payload.append(
            {
                "active_probe_id": row.get("active_probe_id", f"active_probe_{i:05d}"),
                "proposal_mode": row.get("proposal_mode", ""),
                "sweep_label": row.get("sweep_label", ""),
                "sampling_relative_angle_bin": row.get("sampling_relative_angle_bin", ""),
                "nn_distance_12d": float(nn_dist[i]),
                "overall_abs_error": float(np.mean(np.abs(y_pred[i] - y_true[i]))),
                "weighted_abs_error": float(weighted_error[i]),
                "failure_class": failure_class,
                "C1_abs_error": float(abs(y_pred[i, TARGET_COLUMNS.index("C1")] - y_true[i, TARGET_COLUMNS.index("C1")])),
                "C3_abs_error": float(abs(y_pred[i, TARGET_COLUMNS.index("C3")] - y_true[i, TARGET_COLUMNS.index("C3")])),
                "A1_vector_error": vector_error(y_true, y_pred, i, "A1"),
                "B2_vector_error": vector_error(y_true, y_pred, i, "B2"),
                "A2_vector_error": vector_error(y_true, y_pred, i, "A2"),
                "S3_vector_error": vector_error(y_true, y_pred, i, "S3"),
                "A3_vector_error": vector_error(y_true, y_pred, i, "A3"),
                "true_S3_amp": true_amp(y_true, i, "S3"),
                "true_A3_amp": true_amp(y_true, i, "A3"),
                "true_B2_amp": true_amp(y_true, i, "B2"),
                "relative_angle_A3_S3": rel_a3_s3,
                "relative_angle_B2_S3": rel_b2_s3,
                "relative_angle_A3_S3_category": angle_category(rel_a3_s3),
                "relative_angle_B2_S3_category": angle_category(rel_b2_s3),
            }
        )

    fieldnames = list(rows_payload[0]) if rows_payload else []
    top_rows = sorted(rows_payload, key=lambda item: float(item["weighted_abs_error"]), reverse=True)
    write_csv(output_dir / "active_hole_search_top_failures.csv", top_rows[: max(50, int(len(top_rows) * 0.10))], fieldnames)
    write_csv(output_dir / "active_hole_nn_error_summary.csv", rows_payload, fieldnames)

    regime_rows = summarize_groups(rows_payload, "sweep_label")
    angle_rows = summarize_angle_groups(rows_payload)
    write_csv(output_dir / "active_hole_regime_summary.csv", regime_rows, list(regime_rows[0]) if regime_rows else ["group"])
    write_csv(output_dir / "active_hole_relative_angle_summary.csv", angle_rows, list(angle_rows[0]) if angle_rows else ["group"])
    clusters = cluster_failures(top_rows[: max(50, int(len(top_rows) * 0.10))], y_true, probe_rows, output_dir)

    corr = safe_corr(weighted_error, nn_dist)
    spearman = safe_corr(rankdata(weighted_error), rankdata(nn_dist))
    failure_counts = Counter(row["failure_class"] for row in rows_payload)
    summary = {
        "status": "complete",
        "n_probes": int(len(probe_rows)),
        "nearest_neighbor_method": nn_method,
        "weighted_error": {
            "median": float(np.median(weighted_error)),
            "p90": float(np.quantile(weighted_error, 0.90)),
            "p95": float(np.quantile(weighted_error, 0.95)),
            "max": float(np.max(weighted_error)),
        },
        "nn_distance_12d": {
            "median": float(np.median(nn_dist)),
            "p75": float(np.quantile(nn_dist, 0.75)),
            "p95": float(np.quantile(nn_dist, 0.95)),
            "max": float(np.max(nn_dist)),
        },
        "correlation_weighted_error_vs_nn_distance": corr,
        "spearman_weighted_error_vs_nn_distance": spearman,
        "failure_threshold_weighted_error": threshold,
        "sparse_threshold_nn_distance": sparse_threshold,
        "dense_threshold_nn_distance": dense_threshold,
        "failure_class_counts": dict(failure_counts),
        "top_failure_median_nn_distance": float(np.median([float(row["nn_distance_12d"]) for row in top_rows[: max(1, int(len(top_rows) * failure_top_fraction))]])),
        "all_probe_median_nn_distance": float(np.median(nn_dist)),
        "top_failure_table": str(output_dir / "active_hole_search_top_failures.csv"),
        "nn_error_table": str(output_dir / "active_hole_nn_error_summary.csv"),
        "regime_summary": str(output_dir / "active_hole_regime_summary.csv"),
        "relative_angle_summary": str(output_dir / "active_hole_relative_angle_summary.csv"),
        "hole_clusters": clusters,
    }
    (output_dir / "active_hole_search_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_plots(rows_payload, output_dir)
    return summary


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


def safe_corr(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 3 or len(right) < 3 or float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def summarize_groups(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(row)
    output = []
    for label, group_rows in sorted(grouped.items()):
        errors = np.asarray([float(row["weighted_abs_error"]) for row in group_rows], dtype=float)
        distances = np.asarray([float(row["nn_distance_12d"]) for row in group_rows], dtype=float)
        output.append(
            {
                key: label,
                "n": len(group_rows),
                "median_weighted_error": float(np.median(errors)),
                "p95_weighted_error": float(np.quantile(errors, 0.95)),
                "median_nn_distance_12d": float(np.median(distances)),
                "p95_nn_distance_12d": float(np.quantile(distances, 0.95)),
                "coverage_limited_failures": sum(row["failure_class"] == "coverage_limited_sparse_failure" for row in group_rows),
                "dense_failures": sum(row["failure_class"] == "dense_model_feature_loss_failure" for row in group_rows),
            }
        )
    return output


def summarize_angle_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for key in ("relative_angle_A3_S3_category", "relative_angle_B2_S3_category"):
        for summary in summarize_groups(rows, key):
            summary["angle_pair"] = key.replace("relative_angle_", "").replace("_category", "")
            output.append(summary)
    return output


def cluster_failures(
    top_rows: list[dict[str, Any]],
    y_true: np.ndarray,
    probe_rows: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    if not top_rows:
        return []
    id_to_index = {str(row.get("active_probe_id")): index for index, row in enumerate(probe_rows)}
    indices = [id_to_index[str(row["active_probe_id"])] for row in top_rows if str(row["active_probe_id"]) in id_to_index]
    matrix = y_true[indices] / target_scale_array()[None, :]
    cluster_count = min(8, max(2, int(math.sqrt(len(indices) / 6))))
    labels = np.zeros(len(indices), dtype=int)
    method = "single_cluster"
    if len(indices) >= cluster_count * 3:
        try:
            from sklearn.cluster import KMeans

            model = KMeans(n_clusters=cluster_count, random_state=17, n_init=10)
            labels = model.fit_predict(matrix)
            method = "sklearn_kmeans"
        except Exception:
            labels = np.asarray([i % cluster_count for i in range(len(indices))], dtype=int)
            method = "round_robin_fallback"
    clusters = []
    for cluster_id in sorted(set(int(label) for label in labels)):
        member_positions = np.where(labels == cluster_id)[0]
        member_rows = [top_rows[int(position)] for position in member_positions]
        center = np.mean(matrix[member_positions], axis=0)
        errors = np.asarray([float(row["weighted_abs_error"]) for row in member_rows], dtype=float)
        distances = np.asarray([float(row["nn_distance_12d"]) for row in member_rows], dtype=float)
        regimes = Counter(str(row["sweep_label"]) for row in member_rows)
        classes = Counter(str(row["failure_class"]) for row in member_rows)
        clusters.append(
            {
                "cluster_id": cluster_id,
                "method": method,
                "n": int(len(member_rows)),
                "median_weighted_error": float(np.median(errors)),
                "median_nn_distance_12d": float(np.median(distances)),
                "dominant_regime": regimes.most_common(1)[0][0] if regimes else "",
                "dominant_failure_class": classes.most_common(1)[0][0] if classes else "",
                **{f"center_{name}_normalized": float(center[i]) for i, name in enumerate(TARGET_COLUMNS)},
            }
        )
    write_csv(output_dir / "active_hole_clusters.csv", clusters, list(clusters[0]) if clusters else ["cluster_id"])
    return clusters


def write_plots(rows: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    distances = np.asarray([float(row["nn_distance_12d"]) for row in rows], dtype=float)
    errors = np.asarray([float(row["weighted_abs_error"]) for row in rows], dtype=float)
    classes = [str(row["failure_class"]) for row in rows]
    colors = {
        "coverage_limited_sparse_failure": "tab:red",
        "dense_model_feature_loss_failure": "tab:purple",
        "mixed_failure": "tab:orange",
        "not_top_failure": "tab:blue",
    }
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, color in colors.items():
        mask = np.asarray([item == name for item in classes], dtype=bool)
        if np.any(mask):
            ax.scatter(distances[mask], errors[mask], s=12, alpha=0.6, label=name, color=color)
    ax.set_xlabel("12D nearest-neighbor distance to v13 train")
    ax.set_ylabel("weighted normalized absolute error")
    ax.set_title("Active hole search: error vs 12D coverage")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "error_vs_nn_distance.png", dpi=140)
    plt.close(fig)

    for key, label in [("true_S3_amp", "S3 amplitude"), ("true_A3_amp", "A3 amplitude"), ("true_B2_amp", "B2 amplitude")]:
        values = np.asarray([float(row[key]) for row in rows], dtype=float)
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.scatter(values, errors, s=10, alpha=0.55)
        ax.set_xlabel(label)
        ax.set_ylabel("weighted normalized absolute error")
        ax.set_title(f"Active hole search: error vs {label}")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / f"error_vs_{key}.png", dpi=140)
        plt.close(fig)


def write_recommendation(summary: dict[str, Any], output_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    counts = Counter()
    for cluster in summary.get("hole_clusters", []):
        counts[str(cluster.get("dominant_regime", ""))] += int(cluster.get("n", 0))
    total = sum(counts.values()) or 1
    sparse = int(summary.get("failure_class_counts", {}).get("coverage_limited_sparse_failure", 0))
    dense = int(summary.get("failure_class_counts", {}).get("dense_model_feature_loss_failure", 0))
    if sparse > dense * 1.5:
        primary = "targeted_data_expansion"
    elif dense > sparse:
        primary = "feature_model_loss_investigation"
    else:
        primary = "mixed_targeted_expansion_plus_model_diagnostics"
    regime_plan = {
        label: max(0.03, min(0.35, count / total))
        for label, count in counts.items()
        if label
    }
    if not regime_plan:
        regime_plan = {
            "coupled_full_random": 0.30,
            "coupled_sparse_random": 0.18,
            "coupled_A3_S3_random": 0.15,
            "coupled_B2_S3_random": 0.12,
            "coupled_A1_B2_S3_random": 0.10,
            "coupled_C3_A3_S3_random": 0.10,
            "S3_high_random": 0.05,
        }
    norm = sum(regime_plan.values())
    regime_plan = {label: value / norm for label, value in sorted(regime_plan.items())}
    payload = {
        "plan_name": "active_hole_search_recommended_sampling_plan",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "based_on": str(output_dir),
        "current_champion_to_freeze": config["champion"]["name"],
        "primary_next_move": primary,
        "do_not_train_from_this_workflow": True,
        "recommended_future_dataset": config["recommended_plan"]["future_dataset_name"],
        "recommended_new_rows": {
            "diagnostic_minimum": 100000,
            "moderate_expansion": 250000,
            "large_expansion_only_if_coverage_limited_persists": 500000,
        },
        "candidate_selection_mix": {
            "far_nn_from_active_failures": 0.45,
            "failure_cluster_jitter": 0.20,
            "bridge_anchor": 0.20,
            "sobol_lhs_global_balance": 0.10,
            "orthogonal_diagnostic_controls": 0.05,
        },
        "regime_fraction_suggestion": regime_plan,
        "decision_rules": {
            "expand_data": "Use if top failures are mostly coverage_limited_sparse_failure and error remains correlated with 12D NN distance.",
            "feature_or_model_change": "Use if dense_model_feature_loss_failure dominates or specific A3/S3/B2 angle/amplitude failures persist at low NN distance.",
            "measurement_geometry_change": "Consider if dense failures cluster around physically ambiguous C1/A3/S3 couplings under the current two-defocus geometry.",
        },
    }
    path = output_dir / "active_hole_search_recommended_sampling_plan.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def write_report(
    output_dir: Path,
    config: dict[str, Any],
    proposal_summary: dict[str, Any],
    simulation_manifest: dict[str, Any],
    inference_manifest: dict[str, Any],
    evaluation_summary: dict[str, Any] | None,
    recommendation: dict[str, Any] | None,
) -> None:
    lines = [
        "# Active 12D Hole Search v1",
        "",
        f"Created UTC: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "Purpose: freeze the v13 1M seed23 champion and actively probe holes in normalized 12D aberration space.",
        "",
        f"- selected probes: `{proposal_summary.get('selected_probe_count')}`",
        f"- reference train rows: `{proposal_summary.get('reference_train_rows')}`",
        f"- NN method: `{proposal_summary.get('nearest_neighbor_method')}`",
        f"- simulation status: `{simulation_manifest.get('status')}`",
        f"- inference status: `{inference_manifest.get('status')}`",
        "",
        "## Proposal Modes",
        "",
        "| mode | selected | median NN distance | median score |",
        "|---|---:|---:|---:|",
    ]
    for mode, row in proposal_summary.get("mode_summaries", {}).items():
        lines.append(
            f"| `{mode}` | {row.get('selected')} | {row.get('nn_distance_median')} | {row.get('score_median')} |"
        )
    if evaluation_summary:
        lines.extend(
            [
                "",
                "## Evaluation",
                "",
                f"- median weighted error: `{evaluation_summary['weighted_error']['median']}`",
                f"- p95 weighted error: `{evaluation_summary['weighted_error']['p95']}`",
                f"- median NN distance: `{evaluation_summary['nn_distance_12d']['median']}`",
                f"- p95 NN distance: `{evaluation_summary['nn_distance_12d']['p95']}`",
                f"- corr(error, NN distance): `{evaluation_summary['correlation_weighted_error_vs_nn_distance']}`",
                f"- Spearman corr(error, NN distance): `{evaluation_summary['spearman_weighted_error_vs_nn_distance']}`",
                f"- failure classes: `{evaluation_summary['failure_class_counts']}`",
                "",
                "Key compact artifacts:",
                "",
                "- `active_hole_search_top_failures.csv`",
                "- `active_hole_clusters.csv`",
                "- `active_hole_regime_summary.csv`",
                "- `active_hole_relative_angle_summary.csv`",
                "- `active_hole_search_recommended_sampling_plan.json`",
            ]
        )
    if recommendation:
        lines.extend(
            [
                "",
                "## Recommendation",
                "",
                f"- primary next move: **{recommendation['primary_next_move']}**",
                f"- recommended future dataset: `{recommendation['recommended_future_dataset']}`",
                "- this workflow remains diagnostic-only; it does not train a new model.",
            ]
        )
    lines.append("")
    (output_dir / "active_hole_search_report.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/active_12d_hole_search_v1.json"))
    parser.add_argument("--v13-csv", type=Path, required=True)
    parser.add_argument("--v13-run-dir", type=Path, required=True)
    parser.add_argument("--benchmark-split-manifest", type=Path, default=Path("configs/benchmark_split_v12_v2_row_keys.json"))
    parser.add_argument("--output-root", type=Path, default=Path("training_results/model_selection_reports"))
    parser.add_argument("--proposal-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    run_name = f"{config.get('run_prefix', 'v13_active_12d_hole_search')}_{utc_stamp()}"
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "active_hole_search_state.json"

    v13_rows, _ = read_csv(args.v13_csv)
    if not v13_rows:
        raise RuntimeError(f"v13 CSV is empty: {args.v13_csv}")
    split_indices = frozen_benchmark_split(v13_rows, manifest_path=args.benchmark_split_manifest)

    state = {
        "status": "started",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_commit(Path.cwd()),
        "python": sys.version,
        "platform": platform.platform(),
        "config": str(args.config),
        "v13_csv": str(args.v13_csv),
        "v13_csv_sha256": file_sha256(args.v13_csv),
        "v13_run_dir": str(args.v13_run_dir),
        "benchmark_split_manifest": str(args.benchmark_split_manifest),
        "output_dir": str(output_dir),
        "split_counts": {name: int(len(values)) for name, values in split_indices.items()},
    }
    state_path.write_text(json.dumps(state, indent=2) + "\n")

    selected, proposal_summary = propose_candidates(v13_rows, split_indices, args.v13_run_dir, config, output_dir)
    state["status"] = "proposals_complete"
    state["proposal_summary"] = proposal_summary
    state_path.write_text(json.dumps(state, indent=2) + "\n")

    simulation_manifest: dict[str, Any] = {"status": "skipped", "reason": "simulation disabled or proposal-only"}
    probe_rows: list[dict[str, Any]] = []
    if config.get("simulation", {}).get("enabled", True) and not args.proposal_only:
        try:
            probe_rows = simulate_rows(selected, int(config.get("simulation", {}).get("batch_base_cases", 192)))
            for source, simulated in zip(selected, probe_rows):
                simulated.update(
                    {
                        "active_probe_id": source.get("active_probe_id", ""),
                        "proposal_mode": source.get("proposal_mode", ""),
                        "sampling_candidate_role": source.get("sampling_candidate_role", ""),
                        "active_search_score": source.get("active_search_score", ""),
                        "sampling_parent_nn_distance_12d": source.get("sampling_parent_nn_distance_12d", ""),
                        "dataset_version": config["workflow_id"],
                        "dataset_source": "active_12d_hole_search",
                        "dataset_split_hint": "diagnostic_only",
                    }
                )
            feature_path = output_dir / "active_hole_search_probe_features.csv"
            feature_columns = load_feature_columns(args.v13_csv)
            fieldnames = list(dict.fromkeys([*probe_rows[0].keys(), *feature_columns])) if probe_rows else []
            if probe_rows:
                write_csv(feature_path, probe_rows, fieldnames)
            simulation_manifest = {
                "status": "complete",
                "probe_rows": int(len(probe_rows)),
                "probe_features_csv": str(feature_path),
                "large_artifact_policy": "Drive backup only; excluded from GitHub worker globs.",
            }
        except Exception as exc:
            simulation_manifest = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "note": "Proposal artifacts remain useful and resumable.",
            }
    (output_dir / "simulation_manifest.json").write_text(json.dumps(simulation_manifest, indent=2) + "\n")
    state["status"] = "simulation_" + str(simulation_manifest.get("status"))
    state["simulation_manifest"] = simulation_manifest
    state_path.write_text(json.dumps(state, indent=2) + "\n")

    inference_manifest: dict[str, Any] = {"status": "skipped", "reason": "no simulated probe rows"}
    evaluation_summary: dict[str, Any] | None = None
    recommendation: dict[str, Any] | None = None
    if probe_rows and config.get("inference", {}).get("enabled", True) and simulation_manifest.get("status") == "complete":
        y_pred, inference_manifest = run_inference(
            v13_rows,
            args.v13_csv,
            args.v13_run_dir,
            split_indices,
            probe_rows,
            config,
            output_dir,
        )
        if y_pred is not None:
            train_rows = [v13_rows[int(index)] for index in split_indices["train"]]
            train_matrix = normalized_target_matrix(train_rows, target_scale_array().astype(np.float32))
            evaluation_summary = summarize_evaluation(probe_rows, y_pred, train_matrix, config, output_dir)
            recommendation = write_recommendation(evaluation_summary, output_dir, config)
    (output_dir / "inference_manifest.json").write_text(json.dumps(inference_manifest, indent=2) + "\n")
    write_report(output_dir, config, proposal_summary, simulation_manifest, inference_manifest, evaluation_summary, recommendation)

    final_summary = {
        "status": "complete" if evaluation_summary else "partial",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workflow_id": config["workflow_id"],
        "output_dir": str(output_dir),
        "proposal_summary": proposal_summary,
        "simulation_manifest": simulation_manifest,
        "inference_manifest": inference_manifest,
        "evaluation_summary": evaluation_summary,
        "recommendation_path": str(output_dir / "active_hole_search_recommended_sampling_plan.json") if recommendation else None,
        "large_artifact_policy": config.get("artifact_policy", {}),
    }
    (output_dir / "active_hole_search_summary.json").write_text(json.dumps(final_summary, indent=2) + "\n")
    state["status"] = final_summary["status"]
    state["finished_utc"] = datetime.now(timezone.utc).isoformat()
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    print("active hole-search output:", output_dir, flush=True)
    print("active hole-search status:", final_summary["status"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
