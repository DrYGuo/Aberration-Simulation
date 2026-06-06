"""Run one model-only regression candidate from an existing feature CSV.

This script intentionally does not generate simulations. It is the Colab worker
entry point for architecture and hyperparameter selection using cached feature
tables.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from feature_regression_model import TARGET_COLUMNS, Standardizer, file_sha256, target_from_row
from select_regression_model import score_run


TARGET_WEIGHTS = {
    "C1": 1.5,
    "C3": 0.8,
    "A1_x": 1.1,
    "A1_y": 1.1,
    "B2_x": 0.8,
    "B2_y": 0.8,
    "A2_x": 0.9,
    "A2_y": 0.9,
    "S3_x": 1.5,
    "S3_y": 1.5,
    "A3_x": 1.7,
    "A3_y": 1.7,
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


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open() as handle:
        return list(csv.DictReader(handle))


def row_float(row: dict[str, str], name: str, default: float = 0.0) -> float:
    value = row.get(name, default)
    if value in (None, ""):
        return float(default)
    return float(value)


def find_latest_csv(search_root: Path, filename: str) -> Path:
    matches = sorted(search_root.glob(f"**/{filename}"), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} under {search_root}. "
            "This model-loop runner uses existing cached feature CSVs only."
        )
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


def ensure_csv_available(args: argparse.Namespace, filename: str) -> Path:
    if args.csv_path:
        return args.csv_path
    try:
        return find_latest_csv(args.search_root, filename)
    except FileNotFoundError:
        if not args.bootstrap_if_missing:
            raise
    run_dataset_bootstrap(
        args.bootstrap_notebook,
        args.bootstrap_timeout,
        args.bootstrap_output_dir,
    )
    return find_latest_csv(args.search_root, filename)


def find_feature_columns(csv_path: Path, family: str) -> list[str]:
    candidates = []
    if family == "enhanced":
        candidates = [
            csv_path.parent / "feature_columns_enhanced.json",
            csv_path.parent / "feature_columns.json",
        ]
    elif family == "raw_angles":
        candidates = [
            csv_path.parent / "feature_columns_raw_angles.json",
            csv_path.parent / "feature_columns_enhanced.json",
            csv_path.parent / "feature_columns.json",
        ]
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text())
            if isinstance(data, dict) and "features" in data:
                return list(data["features"])
            return list(data)
    raise FileNotFoundError(
        f"No feature_columns*.json found beside {csv_path}. "
        "Refusing to infer features from CSV headers because target columns are also present."
    )


def prepare_dataset(csv_path: Path, feature_columns: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, str]]]:
    rows = load_rows(csv_path)
    X = np.asarray(
        [[row_float(row, name) for name in feature_columns] for row in rows],
        dtype=np.float32,
    )
    y = np.asarray([target_from_row(row) for row in rows], dtype=np.float32)
    labels = np.asarray([row.get("sweep_label", "") for row in rows])
    return X, y, labels, rows


def stratified_train_test_split(labels: np.ndarray, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for label in sorted(set(labels)):
        indices = np.where(labels == label)[0]
        shuffled = rng.permutation(indices)
        n_test = max(1, int(round(test_fraction * len(shuffled)))) if len(shuffled) > 1 else 1
        test_parts.append(shuffled[:n_test])
        train_parts.append(shuffled[n_test:])
    train_index = np.concatenate([part for part in train_parts if len(part)])
    test_index = np.concatenate([part for part in test_parts if len(part)])
    return rng.permutation(train_index), rng.permutation(test_index)


def import_torch():
    import torch
    import torch.nn as nn
    return torch, nn


def build_residual_model(input_dim: int, output_dim: int, hidden_dim: int, dropout: float):
    torch, nn = import_torch()

    class ResidualRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(input_dim, output_dim)
            self.residual = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)

        def forward(self, x):
            return self.linear(x) + self.residual(x)

    return ResidualRegressor()


def build_grouped_head_model(input_dim: int, output_dim: int, hidden_dim: int, dropout: float):
    torch, nn = import_torch()
    if output_dim != len(TARGET_COLUMNS):
        raise ValueError(f"grouped_heads expects {len(TARGET_COLUMNS)} targets, got {output_dim}")

    class GroupedHeadRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(input_dim, output_dim)
            self.trunk = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            )
            self.scalar_head = self._head(2)
            self.low_order_head = self._head(6)
            self.high_order_head = self._head(4)

        def _head(self, out_dim):
            head_hidden = max(32, hidden_dim // 2)
            head = nn.Sequential(
                nn.Linear(hidden_dim, head_hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, out_dim),
            )
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            return head

        def residual(self, x):
            z = self.trunk(x)
            return torch.cat(
                [
                    self.scalar_head(z),
                    self.low_order_head(z),
                    self.high_order_head(z),
                ],
                dim=1,
            )

        def forward(self, x):
            return self.linear(x) + self.residual(x)

    return GroupedHeadRegressor()


def build_model(input_dim: int, output_dim: int, hidden_dim: int, dropout: float, architecture: str):
    if architecture == "residual_mlp":
        return build_residual_model(input_dim, output_dim, hidden_dim, dropout)
    if architecture == "grouped_heads":
        return build_grouped_head_model(input_dim, output_dim, hidden_dim, dropout)
    raise ValueError(f"unknown architecture: {architecture}")


def weighted_mse(pred, target, target_weights):
    torch, _ = import_torch()
    return torch.mean((pred - target) ** 2 * target_weights[None, :])


def summarize_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray,
    train_index: np.ndarray,
    test_index: np.ndarray,
    training_config: dict[str, Any],
) -> dict[str, Any]:
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    metrics: dict[str, Any] = {
        "n_samples": int(len(y_true)),
        "n_train": int(len(train_index)),
        "n_test": int(len(test_index)),
        "overall_mae": float(abs_errors.mean()),
        "overall_rmse": float(np.sqrt(np.mean(errors**2))),
        "targets": {},
        "test_targets": {},
        "labels": {},
        "training_config": training_config,
    }
    for i, name in enumerate(TARGET_COLUMNS):
        target_errors = errors[:, i]
        test_errors = errors[test_index, i]
        metrics["targets"][name] = {
            "mae": float(np.mean(np.abs(target_errors))),
            "rmse": float(np.sqrt(np.mean(target_errors**2))),
            "p95_abs_error": float(np.quantile(np.abs(target_errors), 0.95)),
        }
        metrics["test_targets"][name] = {
            "mae": float(np.mean(np.abs(test_errors))),
            "rmse": float(np.sqrt(np.mean(test_errors**2))),
            "p95_abs_error": float(np.quantile(np.abs(test_errors), 0.95)),
        }
    for label in sorted(set(labels)):
        for split_name, indices in [("all", np.arange(len(labels))), ("test", test_index)]:
            mask_indices = indices[labels[indices] == label]
            if len(mask_indices) == 0:
                continue
            label_errors = errors[mask_indices]
            metrics["labels"].setdefault(label, {})[split_name] = {
                "n": int(len(mask_indices)),
                "mae": float(np.mean(np.abs(label_errors))),
                "rmse": float(np.sqrt(np.mean(label_errors**2))),
            }
    return metrics


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_scale_summary(path: Path, names: list[str], data: np.ndarray) -> None:
    rows = []
    for i, name in enumerate(names):
        column = data[:, i]
        rows.append(
            {
                "name": name,
                "mean": float(np.mean(column)),
                "std": float(np.std(column)),
                "min": float(np.min(column)),
                "max": float(np.max(column)),
                "p01": float(np.quantile(column, 0.01)),
                "p99": float(np.quantile(column, 0.99)),
                "near_constant": bool(np.std(column) < 1e-8),
            }
        )
    write_csv(path, rows, ["name", "mean", "std", "min", "max", "p01", "p99", "near_constant"])


def plot_history(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([row["epoch"] for row in history], [row["train_loss"] for row in history], label="train")
    ax.plot([row["epoch"] for row in history], [row["test_loss"] for row in history], label="test")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("weighted scaled MSE")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_scatter(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray,
    test_index: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    ncols = 4
    nrows = int(math.ceil(len(TARGET_COLUMNS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 8))
    axes = axes.ravel()
    test_labels = labels[test_index]
    unique_labels = sorted(set(test_labels))
    color_map = plt.get_cmap("tab20")
    colors = {
        label: color_map(i % color_map.N)
        for i, label in enumerate(unique_labels)
    }
    for i, name in enumerate(TARGET_COLUMNS):
        ax = axes[i]
        true = y_true[test_index, i]
        pred = y_pred[test_index, i]
        for label in unique_labels:
            mask = test_labels == label
            ax.scatter(
                true[mask],
                pred[mask],
                s=6,
                alpha=0.55,
                color=colors[label],
                label=label if i == 0 else None,
            )
        low = float(min(true.min(), pred.min()))
        high = float(max(true.max(), pred.max()))
        ax.plot([low, high], [low, high], "k--", linewidth=0.8)
        ax.set_title(name, fontsize=9)
    for j in range(len(TARGET_COLUMNS), len(axes)):
        axes[j].axis("off")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            legend_labels,
            loc="center right",
            fontsize=6,
            markerscale=1.5,
            frameon=True,
        )
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["enhanced", "raw_angles"], required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--csv-path", type=Path)
    parser.add_argument("--search-root", type=Path, default=Path("training_results"))
    parser.add_argument("--output-root", type=Path, default=Path("training_results/model_selection_loop"))
    parser.add_argument("--architecture", choices=["residual_mlp", "grouped_heads"], default="residual_mlp")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--residual-penalty", type=float, default=3e-3)
    parser.add_argument("--max-epochs", type=int, default=6000)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--patience-epochs", type=int, default=1000)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--easy-regression-limit", type=float, default=0.10)
    parser.add_argument("--baseline-metrics", type=Path)
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument(
        "--bootstrap-if-missing",
        action="store_true",
        help="Generate the cached feature CSV once if it is absent.",
    )
    parser.add_argument(
        "--bootstrap-notebook",
        type=Path,
        default=Path("notebooks/uno_feature_regression_enhanced_dataset_bootstrap.ipynb"),
        help="Dataset-only notebook used when --bootstrap-if-missing is set.",
    )
    parser.add_argument(
        "--bootstrap-timeout",
        type=int,
        default=3600,
        help="Timeout in seconds for dataset bootstrap notebook execution.",
    )
    parser.add_argument(
        "--bootstrap-output-dir",
        type=Path,
        default=Path("colab_worker_logs"),
        help="Where to write the executed bootstrap notebook manifest.",
    )
    return parser.parse_args()


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    values = vars(args).copy()
    for key, value in list(values.items()):
        if isinstance(value, Path):
            values[key] = str(value)
    return values


def default_baseline_metrics(csv_path: Path, family: str) -> Path | None:
    names = ["metrics_enhanced.json"] if family == "enhanced" else ["metrics_raw_angles.json", "metrics_enhanced.json"]
    for name in names:
        path = csv_path.parent / name
        if path.exists():
            return path
    return None


def write_preflight_failure(output_root: Path, candidate_id: str, message: str, details: dict[str, Any]) -> Path:
    output_dir = output_root / f"{candidate_id}_preflight_failure_{utc_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "preflight_failure",
        "candidate_id": candidate_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "message": message,
        "details": details,
    }
    path = output_dir / "preflight_failure_model_loop.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print("preflight failure:", path)
    print(message)
    return path


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd()
    filename = "training_features_enhanced.csv" if args.family == "enhanced" else "training_features_raw_angles.csv"
    try:
        csv_path = ensure_csv_available(args, filename)
    except FileNotFoundError as exc:
        write_preflight_failure(
            args.output_root,
            args.candidate_id,
            str(exc),
            {
                "family": args.family,
                "search_root": str(args.search_root),
                "required_filename": filename,
                "uses_existing_cached_csv_only": True,
            },
        )
        return 2
    csv_path = csv_path.resolve()
    try:
        feature_columns = find_feature_columns(csv_path, args.family)
    except FileNotFoundError as exc:
        write_preflight_failure(
            args.output_root,
            args.candidate_id,
            str(exc),
            {
                "family": args.family,
                "csv_path": str(csv_path),
                "uses_existing_cached_csv_only": True,
            },
        )
        return 2

    run_name = f"{args.candidate_id}_{utc_stamp()}"
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("candidate:", args.candidate_id)
    print("family:", args.family)
    print("source CSV:", csv_path)
    print("features:", len(feature_columns))
    print("output:", output_dir)

    X, y, labels, rows = prepare_dataset(csv_path, feature_columns)
    train_index, test_index = stratified_train_test_split(labels, args.test_fraction, args.split_seed)

    x_scaler = Standardizer().fit(X[train_index])
    y_scaler = Standardizer().fit(y[train_index])
    Xn = x_scaler.transform(X).astype(np.float32)
    yn = y_scaler.transform(y).astype(np.float32)

    torch, _ = import_torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(Xn.shape[1], yn.shape[1], args.hidden_dim, args.dropout, args.architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    target_weights = torch.tensor(
        [TARGET_WEIGHTS[name] for name in TARGET_COLUMNS],
        dtype=torch.float32,
        device=device,
    )

    x_train = torch.tensor(Xn[train_index], device=device)
    y_train = torch.tensor(yn[train_index], device=device)
    x_test = torch.tensor(Xn[test_index], device=device)
    y_test = torch.tensor(yn[test_index], device=device)

    history: list[dict[str, float]] = []
    best_state = None
    best_epoch = None
    best_test_loss = float("inf")
    epochs_since_best = 0

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred_train = model(x_train)
        residual = model.residual(x_train)
        loss = weighted_mse(pred_train, y_train, target_weights) + args.residual_penalty * torch.mean(residual**2)
        loss.backward()
        optimizer.step()

        if epoch % args.eval_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                train_loss = float(weighted_mse(model(x_train), y_train, target_weights).detach().cpu())
                test_loss = float(weighted_mse(model(x_test), y_test, target_weights).detach().cpu())
            history.append({"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss})
            print(f"epoch {epoch:5d} train={train_loss:.6f} test={test_loss:.6f}")
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_epoch = epoch
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                epochs_since_best = 0
            else:
                epochs_since_best += args.eval_every
            if epochs_since_best >= args.patience_epochs:
                print("early stopping at epoch", epoch)
                break

    if best_state is not None:
        model.load_state_dict({name: value.to(device) for name, value in best_state.items()})

    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.tensor(Xn, device=device)).detach().cpu().numpy()
    y_pred = y_scaler.inverse_transform(pred_scaled)

    training_config = {
        "candidate_id": args.candidate_id,
        "family": args.family,
        "architecture": args.architecture,
        "max_epochs": args.max_epochs,
        "eval_every": args.eval_every,
        "patience_epochs": args.patience_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "residual_penalty": args.residual_penalty,
        "hidden_dim": args.hidden_dim,
        "dropout_probability": args.dropout,
        "split_strategy": "stratified_by_sweep_label",
        "split_seed": args.split_seed,
        "best_epoch": best_epoch,
        "best_test_weighted_mse_scaled": best_test_loss,
        "feature_count": len(feature_columns),
        "scaler": "standard_train_split",
    }
    metrics = summarize_predictions(y, y_pred, labels, train_index, test_index, training_config)
    metrics["run_name"] = run_name
    metrics_path = output_dir / "metrics_model_loop.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    baseline_path = args.baseline_metrics or default_baseline_metrics(csv_path, args.family)
    baseline = json.loads(baseline_path.read_text()) if baseline_path and baseline_path.exists() else None
    selection = score_run(
        metrics_path,
        metrics,
        baseline=baseline,
        easy_regression_limit=args.easy_regression_limit,
    )
    (output_dir / "selection_score.json").write_text(json.dumps(selection, indent=2) + "\n")

    write_scale_summary(output_dir / "feature_scale_summary.csv", feature_columns, X)
    write_scale_summary(output_dir / "target_scale_summary.csv", TARGET_COLUMNS, y)
    write_csv(
        output_dir / "training_history_summary.csv",
        history,
        ["epoch", "train_loss", "test_loss"],
    )
    plot_history(output_dir / "training_history_model_loop.png", history)
    plot_scatter(output_dir / "prediction_scatter_model_loop.png", y, y_pred, labels, test_index)

    if args.save_model:
        torch.save(model.state_dict(), output_dir / "model_loop_candidate.pt")

    manifest = {
        "run_name": run_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_commit(repo_root),
        "python": sys.version,
        "platform": platform.platform(),
        "device": device,
        "candidate": jsonable_args(args),
        "baseline_metrics_path": None if baseline_path is None else str(baseline_path),
        "dataset": {
            "csv_path": str(csv_path),
            "csv_sha256": file_sha256(csv_path),
            "n_rows": int(len(rows)),
            "n_train": int(len(train_index)),
            "n_test": int(len(test_index)),
        },
        "model": {
            "type": f"standardized_linear_plus_{args.architecture}",
            "input_dim": int(Xn.shape[1]),
            "output_dim": int(yn.shape[1]),
            "hidden_dim": args.hidden_dim,
            "dropout_probability": args.dropout,
        },
        "feature_columns": feature_columns,
        "target_columns": TARGET_COLUMNS,
        "output_dir": str(output_dir),
        "metrics_path": str(metrics_path),
        "selection_score_path": str(output_dir / "selection_score.json"),
    }
    (output_dir / "run_manifest_model_loop.json").write_text(json.dumps(manifest, indent=2) + "\n")

    registry_path = output_dir / "model_registry_model_loop.csv"
    registry_row = {
        "run_name": run_name,
        "candidate_id": args.candidate_id,
        "family": args.family,
        "architecture": args.architecture,
        "input_dim": Xn.shape[1],
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "best_epoch": best_epoch,
        "best_test_weighted_mse_scaled": best_test_loss,
        "overall_mae": metrics["overall_mae"],
        "overall_rmse": metrics["overall_rmse"],
        "selection_weighted_score": selection["weighted_score"],
        "selection_rejected": selection["rejected"],
        "source_csv_sha256": manifest["dataset"]["csv_sha256"],
    }
    write_csv(registry_path, [registry_row], list(registry_row))

    print("metrics:", metrics_path)
    print("selection:", output_dir / "selection_score.json")
    print("weighted_score:", selection["weighted_score"])
    print("rejected:", selection["rejected"], selection["rejection_reasons"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
