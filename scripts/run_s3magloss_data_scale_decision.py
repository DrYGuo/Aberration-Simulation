"""State-machine runner for S3 magnitude-loss data scaling.

Each invocation performs one heavy step at most, so the Colab worker can run it
across several cycles without changing config after every cycle.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


BASELINE_CANDIDATE_PREFIX = "D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_bin_diag"
S3MAG_CANDIDATE_PREFIX = "D66_grouped_width320_lr6e-4_dropout0.075_targeted25k_s3magloss"
S3MAG_50K_PREFIX = "D66_grouped_width320_lr6e-4_dropout0.075_s3magloss_50k_total"
S3MAG_100K_PREFIX = "D66_grouped_width320_lr6e-4_dropout0.075_s3magloss_100k_total"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_command(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="" if line.endswith("\n") else "\n", flush=True)
    returncode = int(process.wait())
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def latest_run_dir(output_root: Path, candidate_prefix: str) -> Path | None:
    matches = sorted(output_root.glob(f"{candidate_prefix}_*/selection_score.json"))
    if not matches:
        return None
    return matches[-1].parent


def latest_dataset_csv(search_root: Path, prefix: str) -> Path | None:
    matches = sorted(
        search_root.glob(f"{prefix}_*/training_features_enhanced.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    return matches[-1] if matches else None


def ensure_v3_csv(args: argparse.Namespace) -> Path:
    v3_csv = latest_dataset_csv(args.dataset_root, "enhanced_v3_targeted25k")
    if v3_csv is not None:
        return v3_csv

    v2_csv = latest_dataset_csv(args.dataset_root, "enhanced_v2_coupled16k_stratified_dropout")
    if v2_csv is None:
        run_command(
            [
                "python3",
                "scripts/run_notebook_headless.py",
                str(args.bootstrap_notebook),
                "--output-dir",
                str(args.bootstrap_output_dir),
                "--timeout",
                str(args.bootstrap_timeout),
            ]
        )
        v2_csv = latest_dataset_csv(args.dataset_root, "enhanced_v2_coupled16k_stratified_dropout")
    if v2_csv is None:
        raise FileNotFoundError("could not find or bootstrap enhanced v2 feature CSV")

    run_command(
        [
            "python3",
            "scripts/generate_targeted_enhanced_dataset.py",
            "--parent-csv",
            str(v2_csv),
            "--run-prefix",
            "enhanced_v3_targeted25k",
            "--dataset-version",
            "enhanced_v3_targeted25k",
            "--seed",
            "31",
        ]
    )
    v3_csv = latest_dataset_csv(args.dataset_root, "enhanced_v3_targeted25k")
    if v3_csv is None:
        raise FileNotFoundError("v3 generation finished but no v3 feature CSV was found")
    return v3_csv


def train_candidate(
    *,
    csv_path: Path,
    candidate_id: str,
    baseline_metrics: Path,
    args: argparse.Namespace,
) -> None:
    run_command(
        [
            "python3",
            "scripts/run_model_selection_candidate.py",
            "--family",
            "enhanced",
            "--candidate-id",
            candidate_id,
            "--csv-path",
            str(csv_path),
            "--output-root",
            str(args.model_output_root),
            "--architecture",
            "grouped_heads",
            "--hidden-dim",
            "320",
            "--dropout",
            "0.075",
            "--learning-rate",
            "6e-4",
            "--weight-decay",
            "1e-4",
            "--residual-penalty",
            "3e-3",
            "--s3-magnitude-loss-weight",
            str(args.s3_magnitude_loss_weight),
            "--s3-magnitude-loss-kind",
            "smooth_l1",
            "--s3-magnitude-low-bin-weight",
            "1.0",
            "--s3-magnitude-medium-bin-weight",
            "2.0",
            "--s3-magnitude-high-bin-weight",
            "4.0",
            "--max-epochs",
            "6000",
            "--eval-every",
            "25",
            "--patience-epochs",
            "1000",
            "--selection-config",
            "experiments/model_selection_weights.json",
            "--baseline-metrics",
            str(baseline_metrics),
            "--easy-regression-limit",
            "0.10",
            "--split-seed",
            "7",
        ]
    )


def generate_dataset(
    *,
    parent_csv: Path,
    config: Path,
    run_prefix: str,
    dataset_version: str,
    seed: int,
) -> Path:
    run_command(
        [
            "python3",
            "scripts/generate_targeted_enhanced_dataset.py",
            "--parent-csv",
            str(parent_csv),
            "--case-counts-json",
            str(config),
            "--run-prefix",
            run_prefix,
            "--dataset-version",
            dataset_version,
            "--seed",
            str(seed),
        ]
    )
    csv_path = latest_dataset_csv(Path("training_results/feature_regression_enhanced"), run_prefix)
    if csv_path is None:
        raise FileNotFoundError(f"dataset generation finished but no CSV was found for {run_prefix}")
    return csv_path


def run_metrics(run_dir: Path) -> dict[str, Any]:
    selection = load_json(run_dir / "selection_score.json")
    metrics = load_json(run_dir / "metrics_model_loop.json")
    vector = load_json(run_dir / "vector_diagnostics.json")
    s3_high = vector["vector_pairs"]["S3"]["magnitude_bins"]["bins"]["high"]
    b2 = vector["vector_pairs"]["B2"]
    a3 = vector["vector_pairs"]["A3"]
    return {
        "run_dir": str(run_dir),
        "weighted_score": selection["weighted_score"],
        "true_hard_target_normalized_mae": selection["components"].get("true_hard_target_normalized_mae"),
        "overall_normalized_mae": selection["components"].get("overall_normalized_mae"),
        "blind_normalized_mae": metrics["splits"]["blind"]["overall_normalized_mae"],
        "stress_normalized_mae": metrics["splits"]["stress"]["overall_normalized_mae"],
        "S3_high_magnitude_mae": s3_high["magnitude"]["magnitude_mae"],
        "S3_high_magnitude_bias": s3_high["magnitude"]["magnitude_bias"],
        "S3_high_magnitude_slope": s3_high["magnitude"]["magnitude_slope"],
        "S3_high_mean_abs_angle_error_deg": s3_high["angle"]["mean_abs_angle_error_deg"],
        "S3_high_p95_abs_angle_error_deg": s3_high["angle"]["p95_abs_angle_error_deg"],
        "B2_magnitude_mae": b2["magnitude"]["magnitude_mae"],
        "B2_mean_abs_angle_error_deg": b2["angle"]["mean_abs_angle_error_deg"],
        "A3_magnitude_mae": a3["magnitude"]["magnitude_mae"],
        "A3_mean_abs_angle_error_deg": a3["angle"]["mean_abs_angle_error_deg"],
    }


def relative_change(current: float, baseline: float) -> float:
    return (float(current) - float(baseline)) / max(abs(float(baseline)), 1e-8)


def success_against_baseline(current: dict[str, Any], baseline: dict[str, Any], *, stage: str) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    slope_gain = float(current["S3_high_magnitude_slope"]) - float(baseline["S3_high_magnitude_slope"])
    if slope_gain < 0.05:
        reasons.append(f"S3 high slope gain too small: {slope_gain:.4f}")
    if abs(float(current["S3_high_magnitude_bias"])) > 0.95 * abs(float(baseline["S3_high_magnitude_bias"])):
        reasons.append("S3 high magnitude bias did not move clearly toward zero")
    if float(current["S3_high_magnitude_mae"]) > 0.98 * float(baseline["S3_high_magnitude_mae"]):
        reasons.append("S3 high magnitude MAE did not improve enough")
    if relative_change(current["weighted_score"], baseline["weighted_score"]) > 0.08:
        reasons.append("weighted score degraded by more than 8%")
    if relative_change(current["true_hard_target_normalized_mae"], baseline["true_hard_target_normalized_mae"]) > 0.08:
        reasons.append("true hard-target MAE degraded by more than 8%")
    if relative_change(current["blind_normalized_mae"], baseline["blind_normalized_mae"]) > 0.10:
        reasons.append("blind normalized MAE degraded by more than 10%")
    if relative_change(current["stress_normalized_mae"], baseline["stress_normalized_mae"]) > 0.10:
        reasons.append("stress normalized MAE degraded by more than 10%")
    for key in ["B2_magnitude_mae", "B2_mean_abs_angle_error_deg", "A3_magnitude_mae", "A3_mean_abs_angle_error_deg"]:
        if relative_change(current[key], baseline[key]) > 0.10:
            reasons.append(f"{key} degraded by more than 10%")
    return not reasons, reasons


def write_manifest(summary_dir: Path, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "batch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        write_csv(summary_dir / "batch_summary.csv", rows, fields)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("training_results/feature_regression_enhanced"))
    parser.add_argument("--model-output-root", type=Path, default=Path("training_results/model_selection_loop"))
    parser.add_argument("--summary-root", type=Path, default=Path("training_results/model_selection_batches"))
    parser.add_argument("--s3-magnitude-loss-weight", type=float, default=1.0)
    parser.add_argument("--bootstrap-notebook", type=Path, default=Path("notebooks/uno_feature_regression_enhanced_dataset_bootstrap.ipynb"))
    parser.add_argument("--bootstrap-timeout", type=int, default=3600)
    parser.add_argument("--bootstrap-output-dir", type=Path, default=Path("colab_worker_logs"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = f"s3magloss_data_scale_decision_{utc_stamp()}"
    summary_dir = args.summary_root / run_id
    rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "v3 targeted25k is baseline; v4b is rejected as model-improvement direction",
        "new_rows_split_policy": "new expansion rows are training-only; validation/blind/stress benchmarks remain fixed",
        "s3_couplings_included": [
            "coupled_A1_S3_random",
            "coupled_B2_S3_random",
            "coupled_A3_S3_random",
            "coupled_C3_A3_S3_random",
            "coupled_A1_B2_S3_random",
            "coupled_full_random",
            "coupled_sparse_random",
        ],
        "status": "unknown",
    }

    baseline_dir = latest_run_dir(args.model_output_root, BASELINE_CANDIDATE_PREFIX)
    if baseline_dir is None:
        manifest["status"] = "blocked_missing_v3_bin_baseline"
        write_manifest(summary_dir, manifest, rows)
        print("missing baseline v3 bin diagnostics")
        return 0
    baseline_metrics = run_metrics(baseline_dir)
    rows.append({"stage": "baseline", **baseline_metrics})

    s3mag_dir = latest_run_dir(args.model_output_root, S3MAG_CANDIDATE_PREFIX)
    if s3mag_dir is None:
        v3_csv = ensure_v3_csv(args)
        train_candidate(
            csv_path=v3_csv,
            candidate_id=S3MAG_CANDIDATE_PREFIX,
            baseline_metrics=baseline_dir / "metrics_model_loop.json",
            args=args,
        )
        manifest["status"] = "trained_s3magloss_on_v3"
        write_manifest(summary_dir, manifest, rows)
        return 0

    s3mag_metrics = run_metrics(s3mag_dir)
    s3mag_ok, s3mag_reasons = success_against_baseline(s3mag_metrics, baseline_metrics, stage="v3_s3magloss")
    rows.append({"stage": "v3_s3magloss", "passed": s3mag_ok, "failure_reasons": "; ".join(s3mag_reasons), **s3mag_metrics})
    if not s3mag_ok:
        manifest["status"] = "stop_reconsider_strategy"
        manifest["failure_reasons"] = s3mag_reasons
        write_manifest(summary_dir, manifest, rows)
        print("S3 magnitude loss did not pass expansion gate:", s3mag_reasons)
        return 0

    csv_50k = latest_dataset_csv(args.dataset_root, "enhanced_v5_s3mag_50k_total")
    if csv_50k is None:
        v3_csv = ensure_v3_csv(args)
        csv_50k = generate_dataset(
            parent_csv=v3_csv,
            config=Path("configs/targeted_expansion_s3_couplings_to_50k_total.json"),
            run_prefix="enhanced_v5_s3mag_50k_total",
            dataset_version="enhanced_v5_s3mag_50k_total",
            seed=51,
        )
        manifest["status"] = "generated_50k_total_dataset"
        manifest["dataset_50k_csv"] = str(csv_50k)
        write_manifest(summary_dir, manifest, rows)
        return 0

    run_50k = latest_run_dir(args.model_output_root, S3MAG_50K_PREFIX)
    if run_50k is None:
        train_candidate(
            csv_path=csv_50k,
            candidate_id=S3MAG_50K_PREFIX,
            baseline_metrics=baseline_dir / "metrics_model_loop.json",
            args=args,
        )
        manifest["status"] = "trained_50k_total_s3magloss"
        write_manifest(summary_dir, manifest, rows)
        return 0

    metrics_50k = run_metrics(run_50k)
    ok_50k, reasons_50k = success_against_baseline(metrics_50k, baseline_metrics, stage="50k_total")
    rows.append({"stage": "50k_total", "passed": ok_50k, "failure_reasons": "; ".join(reasons_50k), **metrics_50k})
    if not ok_50k:
        manifest["status"] = "stop_after_50k_reconsider_strategy"
        manifest["failure_reasons"] = reasons_50k
        write_manifest(summary_dir, manifest, rows)
        return 0

    csv_100k = latest_dataset_csv(args.dataset_root, "enhanced_v6_s3mag_100k_total")
    if csv_100k is None:
        csv_100k = generate_dataset(
            parent_csv=csv_50k,
            config=Path("configs/targeted_expansion_s3_couplings_to_100k_total.json"),
            run_prefix="enhanced_v6_s3mag_100k_total",
            dataset_version="enhanced_v6_s3mag_100k_total",
            seed=52,
        )
        manifest["status"] = "generated_100k_total_dataset"
        manifest["dataset_100k_csv"] = str(csv_100k)
        write_manifest(summary_dir, manifest, rows)
        return 0

    run_100k = latest_run_dir(args.model_output_root, S3MAG_100K_PREFIX)
    if run_100k is None:
        train_candidate(
            csv_path=csv_100k,
            candidate_id=S3MAG_100K_PREFIX,
            baseline_metrics=baseline_dir / "metrics_model_loop.json",
            args=args,
        )
        manifest["status"] = "trained_100k_total_s3magloss"
        write_manifest(summary_dir, manifest, rows)
        return 0

    metrics_100k = run_metrics(run_100k)
    ok_100k, reasons_100k = success_against_baseline(metrics_100k, baseline_metrics, stage="100k_total")
    rows.append({"stage": "100k_total", "passed": ok_100k, "failure_reasons": "; ".join(reasons_100k), **metrics_100k})
    manifest["status"] = "complete_100k_evaluated" if ok_100k else "complete_100k_failed_gate"
    manifest["failure_reasons"] = reasons_100k
    write_manifest(summary_dir, manifest, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
