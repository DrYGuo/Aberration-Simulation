"""Regenerate the current architecture diagram and metric table."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = (
    REPO_ROOT
    / "training_results/model_selection_loop/"
    "D66_grouped_width320_lr6e-4_dropout0.075_v12benchmarkv2_500k_seed7_20260615_005333_utc"
)
METRICS_PATH = RUN_DIR / "metrics_model_loop.json"
SVG_PATH = REPO_ROOT / "docs/current_grouped_head_model_architecture.svg"
MD_PATH = REPO_ROOT / "docs/current_grouped_head_model_metrics.md"
CSV_PATH = REPO_ROOT / "docs/current_grouped_head_model_metrics.csv"

TARGETS = [
    "C1",
    "C3",
    "A1_x",
    "A1_y",
    "B2_x",
    "B2_y",
    "A2_x",
    "A2_y",
    "S3_x",
    "S3_y",
    "A3_x",
    "A3_y",
]


def fmt(value: float, digits: int = 4) -> str:
    if abs(value) >= 100:
        return f"{value:.2f}"
    if abs(value) >= 10:
        return f"{value:.3f}"
    return f"{value:.{digits}f}"


def text(x: int, y: int, value: str, klass: str = "body", anchor: str = "start") -> str:
    return f'<text x="{x}" y="{y}" class="{klass}" text-anchor="{anchor}">{html.escape(value)}</text>'


def rect(x: int, y: int, w: int, h: int, klass: str, rx: int = 10) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" class="{klass}"/>'


def line(x1: int, y1: int, x2: int, y2: int, klass: str = "arrow") -> str:
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="{klass}"/>'


def path(d: str, klass: str = "arrow") -> str:
    return f'<path d="{d}" class="{klass}"/>'


def metric_rows(metrics: dict) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    target_metrics = metrics["targets"]
    for target in TARGETS:
        item = target_metrics[target]
        mae = float(item["mae"])
        rmse = float(item["rmse"])
        mse = rmse * rmse
        rows.append(
            {
                "coefficient": target,
                "validation_mae": fmt(mae),
                "validation_mse": fmt(mse),
                "validation_rmse": fmt(rmse),
                "normalized_mae": fmt(float(item["normalized_mae"]), 5),
            }
        )
    return rows


def write_csv(rows: list[dict[str, str]]) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "coefficient",
                "validation_mae",
                "validation_mse",
                "validation_rmse",
                "normalized_mae",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(metrics: dict, rows: list[dict[str, str]]) -> None:
    config = metrics["training_config"]
    dataset = metrics["dataset"]
    lines = [
        "# Current Grouped-Head Regression Model",
        "",
        f"Run: `{metrics['run_name']}`",
        "",
        "Latest completed run as of this document is the v12 benchmark-v2 500K evaluation. "
        "The v13/1M workflow may be running, but no v13 metrics were available when this table was generated.",
        "",
        "## Model",
        "",
        f"- Architecture: `{config['architecture']}` grouped-head residual MLP",
        f"- Feature count: `{config['feature_count']}`",
        f"- Hidden width: `{config['hidden_dim']}`",
        f"- Dropout: `{config['dropout_probability']}`",
        f"- Optimizer/loss: AdamW + `{config['component_loss_kind']}` component loss",
        f"- Dataset rows: `{dataset['n_rows']}`",
        f"- Train / validation / blind / stress: `{metrics['n_train']}` / `{metrics['n_validation']}` / `{metrics['n_blind']}` / `{metrics['n_stress']}`",
        "",
        "## Validation Per-Coefficient Error",
        "",
        "MAE, MSE, and RMSE are in the same physical units as each target coefficient; "
        "MSE is the squared unit. Normalized MAE is divided by the configured physical target scale.",
        "",
        "| coefficient | validation MAE | validation MSE | validation RMSE | normalized MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['coefficient']}` | {row['validation_mae']} | {row['validation_mse']} | "
            f"{row['validation_rmse']} | {row['normalized_mae']} |"
        )
    lines.extend(
        [
            "",
            "Source files:",
            f"- `{METRICS_PATH.relative_to(REPO_ROOT)}`",
            f"- `{CSV_PATH.relative_to(REPO_ROOT)}`",
        ]
    )
    MD_PATH.write_text("\n".join(lines) + "\n")


def table_svg(rows: list[dict[str, str]]) -> list[str]:
    parts: list[str] = []
    x = 70
    y = 805
    col = [x, x + 230, x + 425, x + 620, x + 815]
    parts.append(rect(x, y - 45, 1040, 480, "panel", 12))
    parts.append(text(x + 24, y - 10, "Latest validation error by coefficient", "box-title"))
    parts.append(text(x + 24, y + 18, "MAE/RMSE use target units; MSE uses squared target units.", "small"))
    header_y = y + 55
    headers = ["coefficient", "MAE", "MSE", "RMSE", "norm. MAE"]
    for xx, header in zip(col, headers):
        parts.append(text(xx + 16, header_y, header, "table-head"))
    parts.append(f'<line x1="{x + 16}" y1="{header_y + 14}" x2="{x + 1010}" y2="{header_y + 14}" class="rule"/>')
    row_y = header_y + 42
    for index, row in enumerate(rows):
        if index % 2 == 0:
            parts.append(f'<rect x="{x + 16}" y="{row_y - 22}" width="995" height="28" class="stripe"/>')
        values = [
            row["coefficient"],
            row["validation_mae"],
            row["validation_mse"],
            row["validation_rmse"],
            row["normalized_mae"],
        ]
        for xx, value in zip(col, values):
            parts.append(text(xx + 16, row_y, value, "table-cell"))
        row_y += 28
    return parts


def write_svg(metrics: dict, rows: list[dict[str, str]]) -> None:
    config = metrics["training_config"]
    dataset = metrics["dataset"]
    width = 1800
    height = 1270
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Current grouped-head residual MLP architecture for aberration coefficient regression</title>',
        '<desc id="desc">Diagram of the current regression neural network and validation MAE/MSE table for the latest completed v12 benchmark-v2 run. No convolution layers are used.</desc>',
        "<defs>",
        '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">',
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#334155"/>',
        "</marker>",
        "<style>",
        ".title { font-family: Arial, Helvetica, sans-serif; font-size: 34px; font-weight: 700; fill: #111827; }",
        ".subtitle { font-family: Arial, Helvetica, sans-serif; font-size: 18px; fill: #374151; }",
        ".box-title { font-family: Arial, Helvetica, sans-serif; font-size: 20px; font-weight: 700; fill: #111827; }",
        ".body { font-family: Arial, Helvetica, sans-serif; font-size: 15px; fill: #1f2937; }",
        ".small { font-family: Arial, Helvetica, sans-serif; font-size: 13px; fill: #475569; }",
        ".table-head { font-family: Arial, Helvetica, sans-serif; font-size: 14px; font-weight: 700; fill: #111827; }",
        ".table-cell { font-family: Arial, Helvetica, sans-serif; font-size: 13px; fill: #1f2937; }",
        ".arrow { stroke: #334155; stroke-width: 2.3; fill: none; marker-end: url(#arrow); }",
        ".thin-arrow { stroke: #64748b; stroke-width: 1.8; fill: none; marker-end: url(#arrow); }",
        ".rule { stroke: #94a3b8; stroke-width: 1.2; }",
        ".input { fill: #e0f2fe; stroke: #0284c7; stroke-width: 2; }",
        ".process { fill: #f8fafc; stroke: #64748b; stroke-width: 2; }",
        ".trunk { fill: #ecfdf5; stroke: #059669; stroke-width: 2; }",
        ".head-scalar { fill: #fff7ed; stroke: #ea580c; stroke-width: 2; }",
        ".head-low { fill: #fefce8; stroke: #ca8a04; stroke-width: 2; }",
        ".head-high { fill: #fdf2f8; stroke: #db2777; stroke-width: 2; }",
        ".output { fill: #eef2ff; stroke: #4f46e5; stroke-width: 2; }",
        ".warning { fill: #fee2e2; stroke: #dc2626; stroke-width: 2; }",
        ".panel { fill: #f8fafc; stroke: #cbd5e1; stroke-width: 1.5; }",
        ".stripe { fill: #eef2f7; opacity: 0.75; }",
        "</style>",
        "</defs>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        text(70, 70, "Latest Regression Neural Network: D66 Grouped-Head Residual MLP", "title"),
        text(70, 104, "Feature-vector regression from probe-derived under/over-focus summaries to 12 aberration coefficients. No convolution layers.", "subtitle"),
    ]

    # Top architecture row.
    parts.extend(
        [
            rect(70, 160, 280, 150, "input"),
            text(96, 198, "Input Features", "box-title"),
            text(96, 230, f"D = {config['feature_count']} standardized features"),
            text(96, 256, "Uno Xigma/Mu/Rho summaries"),
            text(96, 282, "Under/over-focus harmonics"),
            rect(70, 355, 280, 100, "process"),
            text(96, 393, "Preprocessing", "box-title"),
            text(96, 425, "Scaler fit on train only"),
            line(210, 310, 210, 355),
            rect(450, 150, 320, 100, "process"),
            text(478, 188, "Linear Skip Branch", "box-title"),
            text(478, 220, "Linear(D, 12)"),
            rect(450, 330, 320, 220, "trunk"),
            text(478, 370, "Shared Residual Trunk", "box-title"),
            text(478, 404, f"Linear(D, {config['hidden_dim']})"),
            text(478, 432, "SiLU + LayerNorm + Dropout"),
            text(478, 460, f"Linear({config['hidden_dim']}, {config['hidden_dim']})"),
            text(478, 488, "SiLU + LayerNorm + Dropout"),
            text(478, 516, f"Linear({config['hidden_dim']}, {config['hidden_dim']})"),
            line(350, 405, 450, 405),
            path("M 350 405 C 390 405, 395 200, 450 200", "thin-arrow"),
            rect(870, 145, 330, 105, "head-scalar"),
            text(900, 184, "Scalar Head", "box-title"),
            text(900, 216, "C1, C3"),
            rect(870, 315, 330, 115, "head-low"),
            text(900, 354, "Low-Order Vector Head", "box-title"),
            text(900, 386, "A1_x/y, B2_x/y, A2_x/y"),
            rect(870, 490, 330, 115, "head-high"),
            text(900, 529, "High-Order Vector Head", "box-title"),
            text(900, 561, "S3_x/y, A3_x/y"),
            path("M 770 440 C 815 440, 820 198, 870 198"),
            line(770, 440, 870, 373),
            path("M 770 440 C 815 440, 820 548, 870 548"),
            rect(1295, 320, 140, 110, "process"),
            text(1323, 362, "Concat", "box-title"),
            text(1323, 394, "2 + 6 + 4 = 12"),
            path("M 1200 198 C 1250 198, 1260 355, 1295 355", "thin-arrow"),
            line(1200, 373, 1295, 373, "thin-arrow"),
            path("M 1200 548 C 1250 548, 1260 410, 1295 410", "thin-arrow"),
            rect(1490, 190, 210, 95, "process"),
            text(1518, 229, "Residual Add", "box-title"),
            text(1518, 260, "Skip + grouped heads"),
            path("M 770 200 C 1040 90, 1330 90, 1505 190"),
            line(1435, 375, 1490, 238),
            rect(1430, 440, 300, 165, "output"),
            text(1460, 480, "Final 12 Targets", "box-title"),
            text(1460, 512, "C1, C3, A1_x/y"),
            text(1460, 540, "B2_x/y, A2_x/y"),
            text(1460, 568, "S3_x/y, A3_x/y"),
            line(1595, 285, 1595, 440),
        ]
    )

    # Summary panels.
    parts.extend(
        [
            rect(70, 630, 505, 105, "panel"),
            text(98, 670, "Latest Completed Run", "box-title"),
            text(98, 700, "v12 benchmark-v2 500K, seed 7"),
            text(98, 726, f"Rows train/val/blind/stress: {metrics['n_train']} / {metrics['n_validation']} / {metrics['n_blind']} / {metrics['n_stress']}", "small"),
            rect(635, 630, 505, 105, "panel"),
            text(663, 670, "Training Setup", "box-title"),
            text(663, 700, f"Width {config['hidden_dim']}, dropout {config['dropout_probability']}, SmoothL1, AdamW"),
            text(663, 726, "Mini-batch + chunked evaluation; no image/CNN input", "small"),
            rect(1200, 630, 530, 105, "panel"),
            text(1228, 670, "Generalization Metrics", "box-title"),
            text(1228, 700, f"Overall normalized MAE: {fmt(float(metrics['overall_normalized_mae']), 5)}"),
            text(1228, 726, f"Overall normalized p95 error: {fmt(float(metrics['overall_normalized_p95_abs_error']), 5)}", "small"),
        ]
    )
    parts.extend(table_svg(rows))
    parts.append(text(70, 1248, "Prediction = LinearSkip(standardized features) + GroupedHeadResidual(standardized features). Latest table uses validation split.", "small"))
    parts.append("</svg>")
    SVG_PATH.write_text("\n".join(parts) + "\n")


def main() -> int:
    metrics = json.loads(METRICS_PATH.read_text())
    rows = metric_rows(metrics)
    write_csv(rows)
    write_markdown(metrics, rows)
    write_svg(metrics, rows)
    print("wrote:", SVG_PATH)
    print("wrote:", MD_PATH)
    print("wrote:", CSV_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
