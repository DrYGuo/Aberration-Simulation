"""Probe-shape gallery plots for Uno relationship diagnostics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def safe_label(label):
    return (
        label.lower()
        .replace("/", "_")
        .replace(" ", "_")
        .replace("+", "p")
        .replace("-", "m")
    )


def representative_probe_rows(rows, spec, max_columns=6):
    selected_rows = [row for row in rows if row["sweep_label"] == spec["label"]]
    if not selected_rows:
        return []

    if "input_field" in spec:
        selected_rows = sorted(selected_rows, key=lambda row: row[spec["input_field"]])
    else:
        selected_rows = sorted(
            selected_rows,
            key=lambda row: (row[spec["amp_field"]], row[spec["phase_field"]]),
        )

    if len(selected_rows) <= max_columns:
        return selected_rows
    indices = np.linspace(0, len(selected_rows) - 1, max_columns).round().astype(int)
    return [selected_rows[int(index)] for index in indices]


def probe_case_title(spec, row):
    if "input_field" in spec:
        return f"{spec['input_field']}={row[spec['input_field']]:g}"
    return f"{spec['amp_field']}={row[spec['amp_field']]:g}\n{spec['phase_field']}={row[spec['phase_field']]:g} deg"


def probe_edge_fraction(image, border=8):
    edge_mask = np.zeros(image.shape, dtype=bool)
    edge_mask[:border, :] = True
    edge_mask[-border:, :] = True
    edge_mask[:, :border] = True
    edge_mask[:, -border:] = True
    total = float(np.nansum(image))
    if total <= 0:
        return np.nan
    return float(np.nansum(image[edge_mask]) / total)


def probe_display_limits(image):
    vmax = float(np.nanpercentile(image, 99.8))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(image))
    return 0.0, vmax


def plot_probe_shape_gallery_for_spec(
    rows,
    probe_np,
    spec,
    output_dir,
    under_focus_c1_offset=-909,
    over_focus_c1_offset=909,
    max_columns=6,
):
    output_dir = Path(output_dir)
    selected_rows = representative_probe_rows(rows, spec, max_columns=max_columns)
    if not selected_rows:
        raise ValueError(f"No probe rows found for {spec['label']}.")

    fig, axes = plt.subplots(
        2,
        len(selected_rows),
        figsize=(2.35 * len(selected_rows), 4.7),
        squeeze=False,
    )
    diagnostics = []
    for col, row in enumerate(selected_rows):
        for axis_row, index_key, offset_label in [
            (0, "under_index", f"C1_offset={under_focus_c1_offset} nm"),
            (1, "over_index", f"C1_offset={over_focus_c1_offset} nm"),
        ]:
            image = probe_np[:, :, int(row[index_key])]
            vmin, vmax = probe_display_limits(image)
            edge_fraction = probe_edge_fraction(image)
            peak = float(np.nanmax(image))
            diagnostics.append((
                probe_case_title(spec, row).replace("\n", ", "),
                offset_label,
                peak,
                edge_fraction,
            ))

            axis = axes[axis_row, col]
            im = axis.imshow(image, cmap="magma", vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=axis, fraction=0.046, pad=0.02)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.text(
                0.03,
                0.05,
                f"edge={100 * edge_fraction:.1f}%",
                transform=axis.transAxes,
                color="white",
                fontsize=7,
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 1.5},
            )
            if col == 0:
                axis.set_ylabel(offset_label, fontsize=9)
            if axis_row == 0:
                axis.set_title(probe_case_title(spec, row), fontsize=9)

    fig.suptitle(f"Probe shapes: {spec['label']} | each panel scaled independently", fontsize=12)
    fig.tight_layout()
    plot_path = output_dir / f"probe_shapes_{safe_label(spec['label'])}.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.show()
    print("saved:", plot_path)
    print(f"{spec['label']} probe diagnostics: peak intensity and 8-pixel edge intensity fraction")
    for title, offset_label, peak, edge_fraction in diagnostics:
        print(f"  {title} | {offset_label}: peak={peak:.4g}, edge={100 * edge_fraction:.2f}%")
    return plot_path


def plot_probe_shape_galleries(
    rows,
    probe_np,
    sweep_specs,
    output_dir,
    under_focus_c1_offset=-909,
    over_focus_c1_offset=909,
):
    return [
        plot_probe_shape_gallery_for_spec(
            rows,
            probe_np,
            spec,
            output_dir,
            under_focus_c1_offset=under_focus_c1_offset,
            over_focus_c1_offset=over_focus_c1_offset,
        )
        for spec in sweep_specs
    ]
