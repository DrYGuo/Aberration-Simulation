"""Run a small batch of model-selection candidates with a walltime guard."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


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


def completed_job(output_root: Path, candidate_id: str) -> Path | None:
    matches = sorted(output_root.glob(f"{candidate_id}_*/selection_score.json"))
    return matches[-1] if matches else None


def selection_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        selection = load_json(path)
    except (OSError, json.JSONDecodeError):
        return {}
    summary: dict[str, Any] = {
        "selection_score_path": str(path),
        "weighted_score": selection.get("weighted_score"),
        "rejected": selection.get("rejected"),
    }
    components = selection.get("components", {})
    if isinstance(components, dict):
        for key in [
            "true_hard_target_normalized_mae",
            "hard_label_normalized_mae",
            "hard_label_normalized_p95",
            "overall_normalized_mae",
            "overall_normalized_p95",
            "easy_target_normalized_mae_relative_change",
        ]:
            if key in components:
                summary[key] = components[key]
    metrics_path = selection.get("metrics_path")
    if metrics_path:
        try:
            metrics = load_json(Path(metrics_path))
        except (OSError, json.JSONDecodeError):
            metrics = {}
        splits = metrics.get("splits", {}) if isinstance(metrics, dict) else {}
        for split_name in ["validation", "blind", "stress"]:
            split = splits.get(split_name, {}) if isinstance(splits, dict) else {}
            if "overall_normalized_mae" in split:
                summary[f"{split_name}_normalized_mae"] = split["overall_normalized_mae"]
    vector_path = path.parent / "vector_diagnostics.json"
    if vector_path.exists():
        try:
            vector = load_json(vector_path)
        except (OSError, json.JSONDecodeError):
            vector = {}
        pairs = vector.get("vector_pairs", {}) if isinstance(vector, dict) else {}
        s3 = pairs.get("S3", {}) if isinstance(pairs, dict) else {}
        s3_high = (
            s3.get("magnitude_bins", {})
            .get("bins", {})
            .get("high", {})
            if isinstance(s3, dict)
            else {}
        )
        s3_high_magnitude = s3_high.get("magnitude", {}) if isinstance(s3_high, dict) else {}
        s3_high_angle = s3_high.get("angle", {}) if isinstance(s3_high, dict) else {}
        if s3_high_magnitude:
            summary["S3_high_magnitude_mae"] = s3_high_magnitude.get("magnitude_mae")
            summary["S3_high_magnitude_bias"] = s3_high_magnitude.get("magnitude_bias")
            summary["S3_high_magnitude_slope"] = s3_high_magnitude.get("magnitude_slope")
        if s3_high_angle:
            summary["S3_high_mean_abs_angle_error_deg"] = s3_high_angle.get("mean_abs_angle_error_deg")
            summary["S3_high_p95_abs_angle_error_deg"] = s3_high_angle.get("p95_abs_angle_error_deg")
        for pair_name in ["B2", "A3"]:
            pair = pairs.get(pair_name, {}) if isinstance(pairs, dict) else {}
            if pair:
                summary[f"{pair_name}_magnitude_mae"] = pair.get("magnitude", {}).get("magnitude_mae")
                summary[f"{pair_name}_mean_abs_angle_error_deg"] = pair.get("angle", {}).get("mean_abs_angle_error_deg")
    return summary


def minutes_remaining(start_time: float, max_runtime_minutes: float) -> float:
    return max_runtime_minutes - (time.monotonic() - start_time) / 60.0


def command_for_job(job: dict[str, Any], defaults: dict[str, Any], output_root: Path) -> list[str]:
    model = dict(job.get("model", {}))
    candidate_id = str(job["candidate_id"])
    command = [
        sys.executable,
        "scripts/run_model_selection_candidate.py",
        "--family",
        str(model.get("family", defaults.get("family", "enhanced"))),
        "--candidate-id",
        candidate_id,
        "--output-root",
        str(output_root),
        "--architecture",
        str(model.get("architecture", "grouped_heads")),
        "--hidden-dim",
        str(model["hidden_dim"]),
        "--dropout",
        str(model["dropout"]),
        "--learning-rate",
        str(model["learning_rate"]),
        "--weight-decay",
        str(model.get("weight_decay", defaults.get("weight_decay", 1e-4))),
        "--residual-penalty",
        str(model.get("residual_penalty", defaults.get("residual_penalty", 3e-3))),
        "--max-epochs",
        str(model.get("max_epochs", defaults.get("max_epochs", 6000))),
        "--eval-every",
        str(model.get("eval_every", defaults.get("eval_every", 25))),
        "--patience-epochs",
        str(model.get("patience_epochs", defaults.get("patience_epochs", 1000))),
        "--easy-regression-limit",
        str(defaults.get("easy_regression_limit", 0.10)),
    ]
    torch_seed = model.get("torch_seed", job.get("torch_seed", defaults.get("torch_seed")))
    if torch_seed is not None:
        command.extend(["--torch-seed", str(torch_seed)])
    for option_name, flag_name in [
        ("batch_size", "--batch-size"),
        ("component_loss_kind", "--component-loss-kind"),
        ("component_smooth_l1_beta", "--component-smooth-l1-beta"),
        ("grad_clip_norm", "--grad-clip-norm"),
        ("lr_scheduler", "--lr-scheduler"),
        ("lr_plateau_factor", "--lr-plateau-factor"),
        ("lr_plateau_patience_evals", "--lr-plateau-patience-evals"),
        ("min_learning_rate", "--min-learning-rate"),
    ]:
        value = model.get(option_name, defaults.get(option_name))
        if value is not None:
            command.extend([flag_name, str(value)])
    if bool(model.get("shuffle_batches", defaults.get("shuffle_batches", False))):
        command.append("--shuffle-batches")
    csv_path = job.get("csv_path", model.get("csv_path", defaults.get("csv_path")))
    if csv_path:
        command.extend(["--csv-path", str(csv_path)])
    else:
        command.extend(["--search-root", str(defaults.get("search_root", "training_results"))])
    baseline_metrics = job.get("baseline_metrics", defaults.get("baseline_metrics"))
    if baseline_metrics:
        command.extend(["--baseline-metrics", str(baseline_metrics)])
    selection_config = defaults.get("selection_config")
    if selection_config:
        command.extend(["--selection-config", str(selection_config)])
    split_seed = job.get("split_seed", job.get("seed", defaults.get("split_seed")))
    if split_seed is not None:
        command.extend(["--split-seed", str(split_seed)])
    s3_loss_weight = model.get("s3_magnitude_loss_weight", defaults.get("s3_magnitude_loss_weight", 0.0))
    if float(s3_loss_weight) > 0:
        command.extend(["--s3-magnitude-loss-weight", str(s3_loss_weight)])
        command.extend(
            [
                "--s3-magnitude-loss-kind",
                str(model.get("s3_magnitude_loss_kind", defaults.get("s3_magnitude_loss_kind", "smooth_l1"))),
                "--s3-magnitude-low-bin-weight",
                str(model.get("s3_magnitude_low_bin_weight", defaults.get("s3_magnitude_low_bin_weight", 1.0))),
                "--s3-magnitude-medium-bin-weight",
                str(model.get("s3_magnitude_medium_bin_weight", defaults.get("s3_magnitude_medium_bin_weight", 2.0))),
                "--s3-magnitude-high-bin-weight",
                str(model.get("s3_magnitude_high_bin_weight", defaults.get("s3_magnitude_high_bin_weight", 4.0))),
            ]
        )
    if defaults.get("bootstrap_if_missing"):
        command.append("--bootstrap-if-missing")
        if defaults.get("bootstrap_notebook"):
            command.extend(["--bootstrap-notebook", str(defaults["bootstrap_notebook"])])
        if defaults.get("bootstrap_timeout"):
            command.extend(["--bootstrap-timeout", str(defaults["bootstrap_timeout"])])
    return command


def run_command(command: list[str]) -> int:
    print("$", " ".join(command), flush=True)
    process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="" if line.endswith("\n") else "\n", flush=True)
    return int(process.wait())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("training_results/model_selection_loop"))
    parser.add_argument("--summary-root", type=Path, default=Path("training_results/model_selection_batches"))
    parser.add_argument("--max-runtime-minutes", type=float)
    parser.add_argument("--force-all", action="store_true")
    parser.add_argument("--force-jobs", default="", help="Comma-separated job IDs to rerun even if complete.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_json(args.batch_config)
    if not config.get("enabled", False):
        raise RuntimeError(f"batch config is disabled: {args.batch_config}")

    batch_id = str(config["batch_id"])
    max_runtime = float(args.max_runtime_minutes or config.get("max_runtime_minutes", 55))
    estimated_minutes = float(config.get("estimated_minutes_per_job", 10))
    safety_margin = float(config.get("walltime_safety_margin_minutes", 6))
    defaults = dict(config.get("defaults", {}))
    force_jobs = {item.strip() for item in args.force_jobs.split(",") if item.strip()}
    started = time.monotonic()

    run_id = f"{batch_id}_{utc_stamp()}"
    summary_dir = args.summary_root / run_id
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    status = "complete"
    for job in config.get("jobs", []):
        job_id = str(job["job_id"])
        candidate_id = str(job["candidate_id"])
        if not job.get("enabled", True):
            rows.append({"job_id": job_id, "candidate_id": candidate_id, "status": "disabled"})
            continue
        if not args.force_all and job_id not in force_jobs:
            completed = completed_job(args.output_root, candidate_id)
            if completed is not None:
                row = {
                    "job_id": job_id,
                    "candidate_id": candidate_id,
                    "status": "skipped_completed",
                }
                row.update(selection_summary(completed))
                rows.append(row)
                continue
        remaining = minutes_remaining(started, max_runtime)
        if remaining < estimated_minutes + safety_margin:
            status = "walltime_guard_stopped"
            rows.append(
                {
                    "job_id": job_id,
                    "candidate_id": candidate_id,
                    "status": "not_started_walltime_guard",
                    "minutes_remaining": remaining,
                }
            )
            break

        command = command_for_job(job, defaults, args.output_root)
        job_started = time.monotonic()
        returncode = run_command(command)
        elapsed = (time.monotonic() - job_started) / 60.0
        row = {
            "job_id": job_id,
            "candidate_id": candidate_id,
            "status": "complete" if returncode == 0 else "failed",
            "returncode": returncode,
            "elapsed_minutes": elapsed,
        }
        completed = completed_job(args.output_root, candidate_id)
        row.update(selection_summary(completed))
        rows.append(row)
        if returncode:
            status = "failed"
            break

    manifest = {
        "batch_id": batch_id,
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "batch_config": str(args.batch_config),
        "max_runtime_minutes": max_runtime,
        "estimated_minutes_per_job": estimated_minutes,
        "walltime_safety_margin_minutes": safety_margin,
        "jobs": rows,
        "artifact_policy": config.get("artifact_policy", {}),
    }
    (summary_dir / "batch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_csv(
        summary_dir / "batch_summary.csv",
        rows,
        [
            "job_id",
            "candidate_id",
            "status",
            "returncode",
            "elapsed_minutes",
            "selection_score_path",
            "weighted_score",
            "rejected",
            "true_hard_target_normalized_mae",
            "hard_label_normalized_mae",
            "hard_label_normalized_p95",
            "overall_normalized_mae",
            "overall_normalized_p95",
            "easy_target_normalized_mae_relative_change",
            "validation_normalized_mae",
            "blind_normalized_mae",
            "stress_normalized_mae",
            "S3_high_magnitude_mae",
            "S3_high_magnitude_bias",
            "S3_high_magnitude_slope",
            "S3_high_mean_abs_angle_error_deg",
            "S3_high_p95_abs_angle_error_deg",
            "B2_magnitude_mae",
            "B2_mean_abs_angle_error_deg",
            "A3_magnitude_mae",
            "A3_mean_abs_angle_error_deg",
            "minutes_remaining",
        ],
    )
    print("batch manifest:", summary_dir / "batch_manifest.json", flush=True)
    print("batch status:", status, flush=True)
    return 0 if status in {"complete", "walltime_guard_stopped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
