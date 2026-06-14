"""Write a compact learning-curve report from model-selection batch summaries."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


DEFAULT_BATCHES = [
    ("v6_100k", Path("training_results/model_selection_batches/v6_benchmark_gap100k_smoothl1_20260612_051859_utc/batch_summary.csv")),
    ("v9_250k", Path("training_results/model_selection_batches/v9_gap250k_d66_20260614_062446_utc/batch_summary.csv")),
]


METRIC_COLUMNS = [
    "weighted_score",
    "true_hard_target_normalized_mae",
    "validation_normalized_mae",
    "blind_normalized_mae",
    "stress_normalized_mae",
    "S3_high_magnitude_mae",
    "S3_high_magnitude_bias",
    "S3_high_magnitude_slope",
    "B2_magnitude_mae",
    "A3_magnitude_mae",
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


def parse_batch_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.parent.name, path
    label, path = value.split("=", 1)
    return label, Path(path)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def best_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    complete = [row for row in rows if row.get("status") == "complete" and to_float(row.get("weighted_score")) is not None]
    if not complete:
        return None
    return min(complete, key=lambda row: float(row["weighted_score"]))


def summarize_batch(label: str, path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"label": label, "path": str(path), "status": "missing"}
    rows = read_rows(path)
    best = best_row(rows)
    if best is None:
        return {"label": label, "path": str(path), "status": "no_complete_rows", "n_rows": len(rows)}
    summary: dict[str, Any] = {
        "label": label,
        "path": str(path),
        "status": "complete",
        "n_candidate_rows": len(rows),
        "best_job_id": best.get("job_id"),
        "best_candidate_id": best.get("candidate_id"),
    }
    for key in METRIC_COLUMNS:
        summary[key] = to_float(best.get(key))
    return summary


def improvement_percent(previous: float | None, current: float | None, *, lower_is_better: bool = True) -> float | None:
    if previous is None or current is None or abs(previous) < 1e-12:
        return None
    if lower_is_better:
        return 100.0 * (previous - current) / abs(previous)
    return 100.0 * (current - previous) / abs(previous)


def write_markdown(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Data-Scale Learning Curve",
        "",
        f"Created UTC: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "| dataset | best candidate | weighted | hard MAE | blind MAE | stress MAE | high-S3 MAE | high-S3 slope | B2 mag MAE | A3 mag MAE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        if item.get("status") != "complete":
            lines.append(f"| {item['label']} | {item['status']} |  |  |  |  |  |  |  |  |")
            continue
        lines.append(
            "| {label} | `{candidate}` | {weighted:.5f} | {hard:.5f} | {blind:.5f} | {stress:.5f} | {s3:.3f} | {slope:.3f} | {b2:.4f} | {a3:.3f} |".format(
                label=item["label"],
                candidate=item.get("best_candidate_id", ""),
                weighted=item.get("weighted_score") or 0.0,
                hard=item.get("true_hard_target_normalized_mae") or 0.0,
                blind=item.get("blind_normalized_mae") or 0.0,
                stress=item.get("stress_normalized_mae") or 0.0,
                s3=item.get("S3_high_magnitude_mae") or 0.0,
                slope=item.get("S3_high_magnitude_slope") or 0.0,
                b2=item.get("B2_magnitude_mae") or 0.0,
                a3=item.get("A3_magnitude_mae") or 0.0,
            )
        )
    lines.extend(["", "## Incremental Improvements", ""])
    complete = [item for item in summaries if item.get("status") == "complete"]
    for previous, current in zip(complete, complete[1:]):
        weighted = improvement_percent(previous.get("weighted_score"), current.get("weighted_score"))
        hard = improvement_percent(previous.get("true_hard_target_normalized_mae"), current.get("true_hard_target_normalized_mae"))
        blind = improvement_percent(previous.get("blind_normalized_mae"), current.get("blind_normalized_mae"))
        stress = improvement_percent(previous.get("stress_normalized_mae"), current.get("stress_normalized_mae"))
        lines.append(
            "- `{prev}` -> `{cur}`: weighted `{weighted:.1f}%`, hard `{hard:.1f}%`, blind `{blind:.1f}%`, stress `{stress:.1f}%`.".format(
                prev=previous["label"],
                cur=current["label"],
                weighted=weighted if weighted is not None else 0.0,
                hard=hard if hard is not None else 0.0,
                blind=blind if blind is not None else 0.0,
                stress=stress if stress is not None else 0.0,
            )
        )
    lines.extend(
        [
            "",
            "Stopping heuristic: if doubling/near-doubling data improves weighted score or hard-regime metrics by less than about 3-5% across two seeds, inspect coverage/residual diagnostics before expanding again.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-summary", action="append", default=[], help="Label/path pair, e.g. v11_500k=.../batch_summary.csv")
    parser.add_argument("--include-default-batches", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("training_results/model_selection_reports"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    batches = list(DEFAULT_BATCHES) if args.include_default_batches else []
    batches.extend(parse_batch_arg(item) for item in args.batch_summary)
    if not batches:
        raise RuntimeError("no batch summaries specified")
    summaries = [summarize_batch(label, path) for label, path in batches]
    args.output_root.mkdir(parents=True, exist_ok=True)
    stem = f"data_scale_learning_curve_{utc_stamp()}"
    json_path = args.output_root / f"{stem}.json"
    md_path = args.output_root / f"{stem}.md"
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "batches": summaries,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    write_markdown(md_path, summaries)
    print("learning curve json:", json_path)
    print("learning curve report:", md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
