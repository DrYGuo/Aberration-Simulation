#!/usr/bin/env python3
"""Freeze the generalization benchmark v1 composition.

This script is report/config infrastructure only. It does not simulate probes,
run inference, or train a model.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def csv_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def latest(pattern: str) -> Path | None:
    matches = sorted(Path(".").glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def load_optional_json(path: Path | None) -> dict[str, Any] | None:
    return read_json(path) if path and path.exists() else None


def component_rows(
    config: dict[str, Any],
    new_hole_dir: Path,
    broad_dir: Path | None,
    anchor_dir: Path | None,
    score_summary: dict[str, Any] | None,
    top_summary: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    components = config["components"]
    new_hole_count = csv_count(new_hole_dir / "selected_probe_design.csv")
    rows: list[dict[str, Any]] = []
    for name, item in components.items():
        source = item["source"]
        n_rows: int | str = ""
        status = "defined"
        if name == "new_hole_challenge":
            source = str(new_hole_dir / "selected_probe_design.csv")
            n_rows = new_hole_count
            status = "frozen_design_pending_simulation"
        elif name == "broad_12d_representative_validation":
            source = str(broad_dir / "broad_12d_representative_validation_design.csv") if broad_dir else source
            n_rows = csv_count(Path(source)) if broad_dir else ""
            status = "frozen_design_pending_simulation"
        elif name == "anchor_regression_guard" and anchor_dir:
            source = str(anchor_dir / "anchor_easy_validation_design.csv")
            n_rows = csv_count(Path(source))
            status = "frozen_design_pending_simulation"
        elif name == "active_hole_repair":
            if top_summary:
                n_rows = top_summary.get("n_matched_rows", "")
            status = "scored_for_v13_v15"
        elif name == "broad_fixed_validation_blind_stress":
            status = "scored_for_v13_v15"
        elif name in {"hard_vector_diagnostics", "anchor_regression_guard"}:
            status = "partially_scored_for_fixed_splits"
        rows.append(
            {
                "component": name,
                "weight": item["weight"],
                "role": item["role"],
                "source": source,
                "n_rows": n_rows,
                "status": status,
                "train_on_this": item["train_on_this"],
            }
        )
    if score_summary:
        for model_row in score_summary.get("model_rows", []):
            rows.append(
                {
                    "component": f"current_available_score_{model_row.get('model')}",
                    "weight": "",
                    "role": "current diagnostic score before new-hole challenge",
                    "source": "benchmark_suite_scoring_v1",
                    "n_rows": "",
                    "status": f"available_score={model_row.get('available_suite_score')}; promotable={model_row.get('promotable_now')}; gates={model_row.get('gate_failures')}",
                    "train_on_this": False,
                }
            )
    return rows


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Generalization Benchmark v1",
        "",
        f"Created UTC: `{payload['created_utc']}`",
        "",
        "Purpose: freeze a model-selection benchmark that prioritizes 12D generalization over narrow fixed-split wins.",
        "",
        f"- benchmark id: `{payload['benchmark_id']}`",
        f"- train on this: `{payload['train_on_this']}`",
        f"- new-hole design dir: `{payload['new_hole_design_dir']}`",
        f"- new-hole probe count: `{payload['new_hole_probe_count']}`",
        f"- broad representative design dir: `{payload['broad_representative_design_dir']}`",
        f"- broad representative probe count: `{payload['broad_representative_probe_count']}`",
        f"- anchor/easy design dir: `{payload['anchor_easy_design_dir']}`",
        f"- anchor/easy probe count: `{payload['anchor_easy_probe_count']}`",
        "",
        "## Components",
        "",
        "| component | weight | status | rows | role |",
        "|---|---:|---|---:|---|",
    ]
    for row in payload["components"]:
        lines.append(
            f"| `{row['component']}` | {row['weight']} | {row['status']} | {row['n_rows']} | {row['role']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- v13 remains the production baseline until a candidate passes broad, active-hole, new-hole, vector, and anchor gates.",
            "- v15 proves the active-hole errors are learnable, but it is not promoted because broad fixed benchmarks regress.",
            "- The new-hole challenge, broad representative validation, and anchor/easy validation rows are intentionally held out from training.",
            "- These frozen designs should be simulated/evaluated next for v13 and v15 before v16 training.",
            "- v16 sampling should be decided from this benchmark, not from the old blind/stress split alone.",
            "",
            "## Next Action",
            "",
            "Simulate the frozen new-hole, broad representative, and anchor/easy probes and run v13/v15 inference. Then update benchmark-suite scoring so the full generalization suite is populated.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/generalization_benchmark_v1.json"))
    parser.add_argument("--new-hole-dir", type=Path, required=True)
    parser.add_argument("--broad-dir", type=Path)
    parser.add_argument("--anchor-dir", type=Path)
    parser.add_argument("--benchmark-suite-summary", type=Path)
    parser.add_argument("--top-failure-summary", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("training_results/model_selection_reports"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    score_summary = load_optional_json(args.benchmark_suite_summary)
    top_summary = load_optional_json(args.top_failure_summary)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")
    output_dir = args.output_root / f"generalization_benchmark_v1_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    components = component_rows(config, args.new_hole_dir, args.broad_dir, args.anchor_dir, score_summary, top_summary)
    new_hole_count = csv_count(args.new_hole_dir / "selected_probe_design.csv")
    broad_count = csv_count(args.broad_dir / "broad_12d_representative_validation_design.csv") if args.broad_dir else 0
    anchor_count = csv_count(args.anchor_dir / "anchor_easy_validation_design.csv") if args.anchor_dir else 0
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_id": config["benchmark_id"],
        "description": config["description"],
        "output_dir": str(output_dir),
        "config": str(args.config),
        "train_on_this": config["train_on_this"],
        "selection_primary_metric": config["selection_primary_metric"],
        "diagnostic_only_until_new_hole_scores_exist": config["diagnostic_only_until_new_hole_scores_exist"],
        "baseline_model": config["baseline_model"],
        "new_hole_design_dir": str(args.new_hole_dir),
        "new_hole_probe_count": new_hole_count,
        "broad_representative_design_dir": str(args.broad_dir) if args.broad_dir else None,
        "broad_representative_probe_count": broad_count,
        "anchor_easy_design_dir": str(args.anchor_dir) if args.anchor_dir else None,
        "anchor_easy_probe_count": anchor_count,
        "components": components,
        "promotion_gates": config["promotion_gates"],
        "v16_training_policy_after_benchmark": config["v16_training_policy_after_benchmark"],
        "input_summaries": {
            "benchmark_suite_summary": str(args.benchmark_suite_summary) if args.benchmark_suite_summary else None,
            "top_failure_summary": str(args.top_failure_summary) if args.top_failure_summary else None,
        },
        "files": {
            "manifest": str(output_dir / "generalization_benchmark_manifest.json"),
            "components": str(output_dir / "generalization_benchmark_components.csv"),
            "scoring_policy": str(output_dir / "generalization_benchmark_scoring_policy.json"),
            "report": str(output_dir / "generalization_benchmark_report.md"),
        },
    }
    (output_dir / "generalization_benchmark_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    (output_dir / "generalization_benchmark_scoring_policy.json").write_text(
        json.dumps(
            {
                "benchmark_id": config["benchmark_id"],
                "components": config["components"],
                "promotion_gates": config["promotion_gates"],
                "new_hole_design_policy": config["new_hole_design_policy"],
                "broad_12d_representative_validation_policy": config["broad_12d_representative_validation_policy"],
                "anchor_easy_validation_policy": config["anchor_easy_validation_policy"],
                "v16_training_policy_after_benchmark": config["v16_training_policy_after_benchmark"],
            },
            indent=2,
        )
        + "\n"
    )
    write_csv(
        output_dir / "generalization_benchmark_components.csv",
        components,
        ["component", "weight", "role", "source", "n_rows", "status", "train_on_this"],
    )
    write_report(output_dir / "generalization_benchmark_report.md", payload)
    print(json.dumps({"output_dir": str(output_dir), "new_hole_probe_count": new_hole_count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
