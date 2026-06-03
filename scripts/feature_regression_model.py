"""Hybrid regression from Uno feature values to aberration coefficients."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FEATURE_COLUMNS = [
    "C1_value_real",
    "C3_value_real",
    "A1_value_real",
    "A1_value_imag",
    "B2_value_real",
    "B2_value_imag",
    "A2_value_real",
    "A2_value_imag",
    "S3_value_real",
    "S3_value_imag",
    "A3_value_real",
    "A3_value_imag",
]


HARMONIC_TARGETS = [
    ("A1", "A1_amp", "A1_phase", 2),
    ("B2", "B2_amp", "B2_phase", 1),
    ("A2", "A2_amp", "A2_phase", 3),
    ("S3", "S3_amp", "S3_phase", 2),
    ("A3", "A3_amp", "A3_phase", 4),
]


TARGET_COLUMNS = [
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


def _float(row, name, default=0.0):
    value = row.get(name, default)
    if value in (None, ""):
        return float(default)
    return float(value)


def coefficient_vector(row, amp_field, phase_field, order):
    amp = _float(row, amp_field)
    theta = np.deg2rad(order * _float(row, phase_field))
    return amp * np.cos(theta), amp * np.sin(theta)


def target_from_row(row):
    target = [_float(row, "C1"), _float(row, "C3")]
    for _, amp_field, phase_field, order in HARMONIC_TARGETS:
        target.extend(coefficient_vector(row, amp_field, phase_field, order))
    return target


def load_rows(csv_path):
    with Path(csv_path).open() as handle:
        return list(csv.DictReader(handle))


def prepare_dataset(csv_path):
    rows = load_rows(csv_path)
    X = np.asarray([[ _float(row, name) for name in FEATURE_COLUMNS] for row in rows], dtype=np.float32)
    y = np.asarray([target_from_row(row) for row in rows], dtype=np.float32)
    labels = np.asarray([row.get("sweep_label", "") for row in rows])
    return X, y, labels, rows


def train_test_split(n_samples, test_fraction=0.2, seed=7):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    n_test = max(1, int(round(test_fraction * n_samples)))
    test_index = indices[:n_test]
    train_index = indices[n_test:]
    return train_index, test_index


class Standardizer:
    def fit(self, data):
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)
        self.std = np.where(self.std == 0, 1.0, self.std)
        return self

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def _import_torch():
    import torch
    import torch.nn as nn
    return torch, nn


def build_hybrid_model(input_dim, output_dim, hidden_dim=96):
    torch, nn = _import_torch()

    class HybridFeatureRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(input_dim, output_dim)
            self.residual = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, output_dim),
            )
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)

        def forward(self, x):
            return self.linear(x) + self.residual(x)

    return HybridFeatureRegressor()


def describe_hybrid_model(input_dim=None, output_dim=None, hidden_dim=96):
    """Return the current hybrid regressor structure as plain text."""
    input_dim = len(FEATURE_COLUMNS) if input_dim is None else input_dim
    output_dim = len(TARGET_COLUMNS) if output_dim is None else output_dim
    linear_params = input_dim * output_dim + output_dim
    residual_params = (
        input_dim * hidden_dim + hidden_dim
        + hidden_dim * hidden_dim + hidden_dim
        + hidden_dim * output_dim + output_dim
    )
    lines = [
        "HybridFeatureRegressor",
        f"  input_dim: {input_dim}",
        f"  output_dim: {output_dim}",
        "  forward(x): linear(x) + residual(x)",
        "  linear:",
        f"    Linear({input_dim}, {output_dim})",
        "  residual:",
        f"    Linear({input_dim}, {hidden_dim})",
        "    SiLU()",
        f"    Linear({hidden_dim}, {hidden_dim})",
        "    SiLU()",
        f"    Linear({hidden_dim}, {output_dim})",
        "    final residual layer initialized to zero",
        f"  linear_params: {linear_params}",
        f"  residual_params: {residual_params}",
        f"  total_params: {linear_params + residual_params}",
    ]
    return "\n".join(lines)


def train_hybrid_regressor(
    csv_path,
    output_dir,
    test_fraction=0.2,
    seed=7,
    epochs=2500,
    learning_rate=1e-3,
    residual_penalty=1e-2,
    hidden_dim=48,
    weight_decay=1e-4,
):
    torch, nn = _import_torch()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, labels, rows = prepare_dataset(csv_path)
    train_index, test_index = train_test_split(len(X), test_fraction=test_fraction, seed=seed)

    x_scaler = Standardizer().fit(X[train_index])
    y_scaler = Standardizer().fit(y[train_index])
    Xn = x_scaler.transform(X).astype(np.float32)
    yn = y_scaler.transform(y).astype(np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_hybrid_model(Xn.shape[1], yn.shape[1], hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    x_train = torch.tensor(Xn[train_index], device=device)
    y_train = torch.tensor(yn[train_index], device=device)
    x_test = torch.tensor(Xn[test_index], device=device)
    y_test = torch.tensor(yn[test_index], device=device)

    history = []
    best_test_loss = None
    best_epoch = None
    best_state = None
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_train)
        residual = model.residual(x_train)
        loss = loss_fn(pred, y_train) + residual_penalty * torch.mean(residual ** 2)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                train_loss = float(loss_fn(model(x_train), y_train).detach().cpu())
                test_loss = float(loss_fn(model(x_test), y_test).detach().cpu())
            if best_test_loss is None or test_loss < best_test_loss:
                best_test_loss = test_loss
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            history.append({
                "epoch": epoch,
                "train_mse_scaled": train_loss,
                "test_mse_scaled": test_loss,
                "best_test_mse_scaled": best_test_loss,
            })

    if best_state is not None:
        model.load_state_dict({name: value.to(device) for name, value in best_state.items()})

    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.tensor(Xn, device=device)).detach().cpu().numpy()
    pred = y_scaler.inverse_transform(pred_scaled)

    metrics = summarize_predictions(y, pred, labels, train_index, test_index)
    metrics["training_config"] = {
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "residual_penalty": float(residual_penalty),
        "hidden_dim": int(hidden_dim),
        "weight_decay": float(weight_decay),
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "best_test_mse_scaled": None if best_test_loss is None else float(best_test_loss),
    }
    save_training_outputs(
        output_dir=output_dir,
        model=model,
        history=history,
        metrics=metrics,
        rows=rows,
        y_true=y,
        y_pred=pred,
        train_index=train_index,
        test_index=test_index,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        device=device,
        hidden_dim=hidden_dim,
    )
    return metrics


def summarize_predictions(y_true, y_pred, labels, train_index, test_index):
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    metrics = {
        "n_samples": int(len(y_true)),
        "n_train": int(len(train_index)),
        "n_test": int(len(test_index)),
        "overall_mae": float(abs_errors.mean()),
        "overall_rmse": float(np.sqrt(np.mean(errors ** 2))),
        "targets": {},
        "test_targets": {},
        "labels": {},
    }
    for i, name in enumerate(TARGET_COLUMNS):
        metrics["targets"][name] = {
            "mae": float(abs_errors[:, i].mean()),
            "rmse": float(np.sqrt(np.mean(errors[:, i] ** 2))),
        }
        test_errors = errors[test_index, i]
        metrics["test_targets"][name] = {
            "mae": float(np.abs(test_errors).mean()),
            "rmse": float(np.sqrt(np.mean(test_errors ** 2))),
        }
    for label in sorted(set(labels)):
        mask = labels == label
        if np.any(mask):
            label_errors = errors[mask]
            metrics["labels"][label] = {
                "n": int(mask.sum()),
                "mae": float(np.abs(label_errors).mean()),
                "rmse": float(np.sqrt(np.mean(label_errors ** 2))),
            }
    return metrics


def save_training_outputs(
    output_dir,
    model,
    history,
    metrics,
    rows,
    y_true,
    y_pred,
    train_index,
    test_index,
    x_scaler,
    y_scaler,
    device,
    hidden_dim,
):
    torch, _ = _import_torch()
    output_dir = Path(output_dir)

    with (output_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
    with (output_dir / "normalization.json").open("w") as handle:
        json.dump({"features": FEATURE_COLUMNS, "targets": TARGET_COLUMNS, "x": x_scaler.to_dict(), "y": y_scaler.to_dict()}, handle, indent=2)
    (output_dir / "model_summary.txt").write_text(
        describe_hybrid_model(len(FEATURE_COLUMNS), len(TARGET_COLUMNS), hidden_dim=hidden_dim) + "\n"
    )
    with (output_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_mse_scaled", "test_mse_scaled", "best_test_mse_scaled"])
        writer.writeheader()
        writer.writerows(history)

    split = np.asarray(["train"] * len(rows), dtype=object)
    split[test_index] = "test"
    with (output_dir / "predictions.csv").open("w", newline="") as handle:
        fieldnames = ["row_index", "split", "sweep_label"] + [f"true_{name}" for name in TARGET_COLUMNS] + [f"pred_{name}" for name in TARGET_COLUMNS]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            out = {"row_index": index, "split": split[index], "sweep_label": row.get("sweep_label", "")}
            out.update({f"true_{name}": float(y_true[index, i]) for i, name in enumerate(TARGET_COLUMNS)})
            out.update({f"pred_{name}": float(y_pred[index, i]) for i, name in enumerate(TARGET_COLUMNS)})
            writer.writerow(out)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_columns": FEATURE_COLUMNS,
            "target_columns": TARGET_COLUMNS,
            "device_used": device,
        },
        output_dir / "hybrid_feature_regressor.pt",
    )
    plot_training_history(history, output_dir)
    plot_prediction_scatter(y_true, y_pred, output_dir)


def plot_training_history(history, output_dir):
    output_dir = Path(output_dir)
    epochs = [item["epoch"] for item in history]
    train = [item["train_mse_scaled"] for item in history]
    test = [item["test_mse_scaled"] for item in history]
    fig, axis = plt.subplots(figsize=(6.2, 4.2))
    axis.plot(epochs, train, label="train")
    axis.plot(epochs, test, label="test")
    axis.set_yscale("log")
    axis.set_xlabel("epoch")
    axis.set_ylabel("scaled MSE")
    axis.set_title("Hybrid feature-regression training")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    path = output_dir / "training_history.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.show()
    print("saved:", path)


def plot_prediction_scatter(y_true, y_pred, output_dir):
    output_dir = Path(output_dir)
    ncols = 4
    nrows = int(np.ceil(len(TARGET_COLUMNS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.6 * nrows), squeeze=False)
    for i, name in enumerate(TARGET_COLUMNS):
        axis = axes[i // ncols, i % ncols]
        axis.scatter(y_true[:, i], y_pred[:, i], s=18, alpha=0.65)
        lo = float(min(np.nanmin(y_true[:, i]), np.nanmin(y_pred[:, i])))
        hi = float(max(np.nanmax(y_true[:, i]), np.nanmax(y_pred[:, i])))
        if lo == hi:
            hi = lo + 1
        axis.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
        axis.set_title(name)
        axis.set_xlabel("true")
        axis.set_ylabel("pred")
        axis.grid(alpha=0.25)
    for j in range(len(TARGET_COLUMNS), nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    fig.suptitle("Predicted vs true aberration coefficient vectors", fontsize=13)
    fig.tight_layout()
    path = output_dir / "prediction_scatter.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.show()
    print("saved:", path)
