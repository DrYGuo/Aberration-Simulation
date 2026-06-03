"""Relationship plots for Uno coefficient diagnostics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aberration_simulation.uno_conventions import (
    PRIMARY_PHASE_CONVENTIONS,
    UNO_HARMONIC_ORDERS,
)


def circular_difference_deg(a, b, period):
    return np.abs((a - b + period / 2) % period - period / 2)


def fitted_slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x != 0)
    if not np.any(valid):
        return np.nan
    return float(np.sum(x[valid] * y[valid]) / np.sum(x[valid] ** 2))


def plot_scalar_relationship_for_spec(rows, spec, output_dir):
    output_dir = Path(output_dir)
    value_name = spec["value_name"]
    input_field = spec["input_field"]
    selected_rows = [row for row in rows if row["sweep_label"] == spec["label"]]
    if not selected_rows:
        raise ValueError(f"No rows found for {spec['label']}.")

    input_value = np.asarray([row[input_field] for row in selected_rows], dtype=float)
    output_value = np.asarray([row[value_name + "_real"] for row in selected_rows], dtype=float)
    slope = fitted_slope(input_value, output_value)
    residual = output_value - slope * input_value if np.isfinite(slope) else np.full_like(output_value, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].scatter(input_value, output_value, s=48, alpha=0.9)
    x_line = np.linspace(float(np.nanmin(input_value)), float(np.nanmax(input_value)), 100)
    if np.isfinite(slope):
        axes[0].plot(x_line, slope * x_line, color="black", linestyle="--", linewidth=1, label=f"fit slope={slope:.4g}")
        axes[0].legend(fontsize=8)
    axes[0].set_title(f"{value_name} vs {input_field}")
    axes[0].set_xlabel(input_field)
    axes[0].set_ylabel(value_name)
    axes[0].grid(alpha=0.3)

    axes[1].scatter(input_value, residual, s=48, alpha=0.9)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("linear-fit residual")
    axes[1].set_xlabel(input_field)
    axes[1].set_ylabel(f"{value_name} residual")
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"{spec['label']}: scalar one-coefficient sweep", fontsize=12)
    fig.tight_layout()
    plot_path = output_dir / f"relationship_{value_name}.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.show()
    print(spec["label"], "cases=", len(selected_rows), "slope=", slope, "plot=", plot_path)
    return plot_path


def plot_harmonic_relationship_for_spec(rows, spec, output_dir):
    output_dir = Path(output_dir)
    value_name = spec["value_name"]
    amp_field = spec["amp_field"]
    phase_field = spec["phase_field"]
    order = UNO_HARMONIC_ORDERS[value_name]
    period = 360.0 / order
    selected_rows = [row for row in rows if row["sweep_label"] == spec["label"]]
    if not selected_rows:
        raise ValueError(f"No rows found for {spec['label']}.")

    input_amp = np.asarray([row[amp_field] for row in selected_rows], dtype=float)
    input_phase = np.asarray([row[phase_field] % period for row in selected_rows], dtype=float)
    output_amp = np.asarray([row[value_name + "_abs"] for row in selected_rows], dtype=float)
    output_phase = np.asarray([row[value_name + "_phase_deg"] for row in selected_rows], dtype=float)
    phase_error = circular_difference_deg(output_phase, input_phase, period)
    slope = fitted_slope(input_amp, output_amp)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    amp_scatter = axes[0].scatter(input_amp, output_amp, c=input_phase, cmap="twilight", s=42, alpha=0.9)
    x_line = np.linspace(0, float(np.nanmax(input_amp)) * 1.05, 100)
    if np.isfinite(slope):
        axes[0].plot(x_line, slope * x_line, color="black", linestyle="--", linewidth=1, label=f"fit slope={slope:.4g}")
        axes[0].legend(fontsize=8)
    axes[0].set_title(f"{value_name} amplitude vs {amp_field}")
    axes[0].set_xlabel(amp_field)
    axes[0].set_ylabel(f"abs({value_name})")
    axes[0].grid(alpha=0.3)
    fig.colorbar(amp_scatter, ax=axes[0], label=f"{phase_field} mod {period:g} deg")

    phase_scatter = axes[1].scatter(input_phase, output_phase, c=input_amp, cmap="viridis", s=42, alpha=0.9)
    axes[1].plot([0, period], [0, period], color="black", linestyle="--", linewidth=1)
    axes[1].set_xlim(-0.04 * period, 1.04 * period)
    axes[1].set_ylim(-0.04 * period, 1.04 * period)
    axes[1].set_title(f"{value_name} phase vs {phase_field}")
    axes[1].set_xlabel(f"{phase_field} mod period (deg)")
    axes[1].set_ylabel(f"{value_name}_phase_deg")
    axes[1].grid(alpha=0.3)
    fig.colorbar(phase_scatter, ax=axes[1], label=amp_field)

    axes[2].scatter(input_amp, phase_error, c=input_phase, cmap="twilight", s=42, alpha=0.9)
    axes[2].set_title("phase error")
    axes[2].set_xlabel(amp_field)
    axes[2].set_ylabel("abs wrapped phase error (deg)")
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"{spec['label']}: one-coefficient sweep | convention {PRIMARY_PHASE_CONVENTIONS[value_name]}", fontsize=12)
    fig.tight_layout()
    plot_path = output_dir / f"relationship_{value_name}.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.show()
    print(spec["label"], "cases=", len(selected_rows), "slope=", slope, "plot=", plot_path)
    return plot_path


def plot_relationships(rows, scalar_specs, harmonic_specs, output_dir):
    plot_paths = []
    plot_paths.extend(plot_scalar_relationship_for_spec(rows, spec, output_dir) for spec in scalar_specs)
    plot_paths.extend(plot_harmonic_relationship_for_spec(rows, spec, output_dir) for spec in harmonic_specs)
    return plot_paths


def plot_relationship_summary(rows, sweep_specs, output_dir):
    output_dir = Path(output_dir)
    fig, axes = plt.subplots(2, len(sweep_specs), figsize=(4.0 * len(sweep_specs), 7.2), squeeze=False)

    for column, spec in enumerate(sweep_specs):
        selected_rows = [row for row in rows if row["sweep_label"] == spec["label"]]
        if not selected_rows:
            axes[0, column].set_title(spec["label"] + " (no rows)")
            axes[1, column].set_title(spec["label"] + " (no rows)")
            continue

        if "input_field" in spec:
            value_name = spec["value_name"]
            input_field = spec["input_field"]
            input_value = np.asarray([row[input_field] for row in selected_rows], dtype=float)
            output_value = np.asarray([row[value_name + "_real"] for row in selected_rows], dtype=float)
            slope = fitted_slope(input_value, output_value)
            residual = output_value - slope * input_value if np.isfinite(slope) else np.full_like(output_value, np.nan)

            axes[0, column].scatter(input_value, output_value, s=24)
            axes[0, column].set_title(spec["label"])
            axes[0, column].set_xlabel(input_field)
            axes[0, column].set_ylabel(value_name)
            axes[0, column].grid(alpha=0.25)

            axes[1, column].scatter(input_value, residual, s=24)
            axes[1, column].axhline(0, color="black", linestyle="--", linewidth=1)
            axes[1, column].set_xlabel(input_field)
            axes[1, column].set_ylabel("residual")
            axes[1, column].grid(alpha=0.25)
            continue

        value_name = spec["value_name"]
        amp_field = spec["amp_field"]
        phase_field = spec["phase_field"]
        order = UNO_HARMONIC_ORDERS[value_name]
        period = 360.0 / order
        input_amp = np.asarray([row[amp_field] for row in selected_rows], dtype=float)
        input_phase = np.asarray([row[phase_field] % period for row in selected_rows], dtype=float)
        output_amp = np.asarray([row[value_name + "_abs"] for row in selected_rows], dtype=float)
        output_phase = np.asarray([row[value_name + "_phase_deg"] for row in selected_rows], dtype=float)

        axes[0, column].scatter(input_amp, output_amp, c=input_phase, cmap="twilight", s=22)
        axes[0, column].set_title(spec["label"])
        axes[0, column].set_xlabel(amp_field)
        axes[0, column].set_ylabel(f"abs({value_name})")
        axes[0, column].grid(alpha=0.25)

        axes[1, column].scatter(input_phase, output_phase, c=input_amp, cmap="viridis", s=22)
        axes[1, column].plot([0, period], [0, period], color="black", linestyle="--", linewidth=1)
        axes[1, column].set_xlim(-0.04 * period, 1.04 * period)
        axes[1, column].set_ylim(-0.04 * period, 1.04 * period)
        axes[1, column].set_xlabel(f"{phase_field} mod period")
        axes[1, column].set_ylabel(f"{value_name}_phase_deg")
        axes[1, column].grid(alpha=0.25)

    fig.suptitle("Uno coefficient relationships: scalar values, amplitude, and phase", fontsize=14)
    fig.tight_layout()
    plot_path = output_dir / "uno_coefficient_relationship_summary.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.show()
    print("saved:", plot_path)
    return plot_path


def plot_c1_c3_coupling_maps(rows, grid_label, output_dir):
    output_dir = Path(output_dir)
    selected_rows = [row for row in rows if row["sweep_label"] == grid_label]
    if not selected_rows:
        raise ValueError(f"No rows found for {grid_label}.")

    c1_values = np.asarray(sorted({float(row["C1"]) for row in selected_rows}), dtype=float)
    c3_values = np.asarray(sorted({float(row["C3"]) for row in selected_rows}), dtype=float)
    c1_index = {value: index for index, value in enumerate(c1_values)}
    c3_index = {value: index for index, value in enumerate(c3_values)}
    c1_map = np.full((len(c3_values), len(c1_values)), np.nan, dtype=float)
    c3_map = np.full_like(c1_map, np.nan)

    for row in selected_rows:
        iy = c3_index[float(row["C3"])]
        ix = c1_index[float(row["C1"])]
        c1_map[iy, ix] = float(row["C1_value_real"])
        c3_map[iy, ix] = float(row["C3_value_real"])

    extent = [float(c1_values.min()), float(c1_values.max()), float(c3_values.min()), float(c3_values.max())]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), squeeze=False)
    for axis, data, title, label in [
        (axes[0, 0], c1_map, "C1_value over coupled C1/C3 sweep", "C1_value"),
        (axes[0, 1], c3_map, "C3_value over coupled C1/C3 sweep", "C3_value"),
    ]:
        im = axis.imshow(data, origin="lower", aspect="auto", extent=extent, cmap="viridis")
        axis.set_title(title)
        axis.set_xlabel("input C1")
        axis.set_ylabel("input C3")
        axis.set_xticks(c1_values)
        axis.set_yticks(c3_values)
        axis.grid(color="white", alpha=0.25, linewidth=0.7)
        fig.colorbar(im, ax=axis, label=label)

    fig.suptitle("Coupled scalar response maps for C1 and C3", fontsize=12)
    fig.tight_layout()
    plot_path = output_dir / "relationship_c1_c3_coupling_maps.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.show()
    print("saved:", plot_path)
    return plot_path


def plot_a1_s3_coupling_map(rows, grid_label, value_name, value_column, output_name, output_dir):
    output_dir = Path(output_dir)
    selected_rows = [row for row in rows if row["sweep_label"] == grid_label]
    if not selected_rows:
        raise ValueError(f"No rows found for {grid_label}.")

    a1_amps = np.asarray(sorted({float(row["A1_amp"]) for row in selected_rows}), dtype=float)
    s3_amps = np.asarray(sorted({float(row["S3_amp"]) for row in selected_rows}), dtype=float)
    a1_phases = np.asarray(sorted({float(row["A1_phase"]) for row in selected_rows}), dtype=float)
    s3_phases = np.asarray(sorted({float(row["S3_phase"]) for row in selected_rows}), dtype=float)

    phase_maps = {}
    values_for_limits = []
    for a1_phase in a1_phases:
        for s3_phase in s3_phases:
            data = np.full((len(s3_amps), len(a1_amps)), np.nan, dtype=float)
            for row in selected_rows:
                if not np.isclose(float(row["A1_phase"]), a1_phase):
                    continue
                if not np.isclose(float(row["S3_phase"]), s3_phase):
                    continue
                iy = int(np.where(np.isclose(s3_amps, float(row["S3_amp"])))[0][0])
                ix = int(np.where(np.isclose(a1_amps, float(row["A1_amp"])))[0][0])
                data[iy, ix] = float(row[value_column])
            phase_maps[(a1_phase, s3_phase)] = data
            values_for_limits.append(data[np.isfinite(data)])

    finite_values = np.concatenate([values for values in values_for_limits if values.size])
    vmin = float(np.nanmin(finite_values))
    vmax = float(np.nanmax(finite_values))
    extent = [float(a1_amps.min()), float(a1_amps.max()), float(s3_amps.min()), float(s3_amps.max())]

    fig = plt.figure(figsize=(3.0 * len(a1_phases) + 0.7, 2.7 * len(s3_phases)))
    grid = fig.add_gridspec(
        len(s3_phases),
        len(a1_phases) + 1,
        width_ratios=[1.0] * len(a1_phases) + [0.08],
        wspace=0.22,
        hspace=0.36,
    )
    last_image = None
    for row_index, s3_phase in enumerate(s3_phases):
        for col_index, a1_phase in enumerate(a1_phases):
            axis = fig.add_subplot(grid[row_index, col_index])
            data = phase_maps[(a1_phase, s3_phase)]
            last_image = axis.imshow(
                data, origin="lower", aspect="auto", extent=extent,
                cmap="viridis", vmin=vmin, vmax=vmax,
            )
            axis.set_title(f"A1 phase={a1_phase:g} deg\nS3 phase={s3_phase:g} deg", fontsize=8)
            axis.set_xticks(a1_amps)
            axis.set_yticks(s3_amps)
            axis.grid(color="white", alpha=0.22, linewidth=0.6)
            if row_index == len(s3_phases) - 1:
                axis.set_xlabel("A1_amp")
            else:
                axis.set_xticklabels([])
            if col_index == 0:
                axis.set_ylabel("S3_amp")
            else:
                axis.set_yticklabels([])

    colorbar_axis = fig.add_subplot(grid[:, -1])
    fig.colorbar(last_image, cax=colorbar_axis, label=value_column)
    fig.suptitle(f"{value_name} over coupled A1/S3 sweep | color = {value_column}", fontsize=12, y=0.98)
    plot_path = output_dir / output_name
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.show()
    print("saved:", plot_path)
    return plot_path


def plot_a1_s3_coupling_maps(rows, grid_label, output_dir):
    return [
        plot_a1_s3_coupling_map(
            rows, grid_label, "A1_value", "A1_value_abs",
            "relationship_a1_s3_coupling_a1_value_abs.png", output_dir,
        ),
        plot_a1_s3_coupling_map(
            rows, grid_label, "S3_value", "S3_value_abs",
            "relationship_a1_s3_coupling_s3_value_abs.png", output_dir,
        ),
    ]


def plot_wide_harmonic_amplitude_relationship_for_spec(rows, spec, output_dir):
    output_dir = Path(output_dir)
    value_name = spec["value_name"]
    amp_field = spec["amp_field"]
    phase_field = spec["phase_field"]
    selected_rows = [row for row in rows if row["sweep_label"] == spec["label"]]
    if not selected_rows:
        raise ValueError(f"No rows found for {spec['label']}.")

    input_amp = np.asarray([row[amp_field] for row in selected_rows], dtype=float)
    input_phase = np.asarray([row[phase_field] for row in selected_rows], dtype=float)
    output_amp = np.asarray([row[value_name + "_abs"] for row in selected_rows], dtype=float)
    slope = fitted_slope(input_amp, output_amp)
    residual = output_amp - slope * input_amp if np.isfinite(slope) else np.full_like(output_amp, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    amp_scatter = axes[0].scatter(input_amp, output_amp, c=input_phase, cmap="twilight", s=46, alpha=0.9)
    x_line = np.linspace(0, float(np.nanmax(input_amp)) * 1.05, 100)
    if np.isfinite(slope):
        axes[0].plot(x_line, slope * x_line, color="black", linestyle="--", linewidth=1, label=f"fit slope={slope:.4g}")
        axes[0].legend(fontsize=8)
    axes[0].set_title(f"{value_name} wide range")
    axes[0].set_xlabel(amp_field)
    axes[0].set_ylabel(f"abs({value_name})")
    axes[0].grid(alpha=0.3)
    fig.colorbar(amp_scatter, ax=axes[0], label=phase_field)

    axes[1].scatter(input_amp, residual, c=input_phase, cmap="twilight", s=46, alpha=0.9)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("linear-fit residual")
    axes[1].set_xlabel(amp_field)
    axes[1].set_ylabel(f"abs({value_name}) residual")
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"{spec['label']}: fixed-phase wide-amplitude sweep", fontsize=12)
    fig.tight_layout()
    plot_path = output_dir / spec["output_name"]
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.show()
    print(
        spec["label"],
        "cases=", len(selected_rows),
        "slope=", slope,
        "mean abs residual=", float(np.nanmean(np.abs(residual))),
        "plot=", plot_path,
    )
    return plot_path


def plot_wide_harmonic_amplitude_relationships(rows, wide_specs, output_dir):
    return [
        plot_wide_harmonic_amplitude_relationship_for_spec(rows, spec, output_dir)
        for spec in wide_specs
    ]
