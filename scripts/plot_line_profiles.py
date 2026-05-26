#!/usr/bin/env python
"""Generate paired C1-offset line-profile plots from the smoke test."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aberration_simulation.backend import asnumpy
from aberration_simulation.line_profiles import extract_line_profiles_from_stack


C1_OFFSETS = (-909, 909)
COMBINATION_FIELDS = (
    "C1",
    "A1_amp",
    "A1_phase",
    "A2_amp",
    "A2_phase",
    "A3_amp",
    "A3_phase",
    "C3",
)


def load_smoke_outputs(path):
    data = np.load(path, allow_pickle=True)
    names = [str(name) for name in data["parameter_names"]]
    rows = data["parameters"]
    parameters = [
        {name: float(value) for name, value in zip(names, row)}
        for row in rows
    ]
    return data["probe_images"], parameters


def _fmt(value):
    return "{:g}".format(value)


def combination_key(params):
    return tuple(params[field] for field in COMBINATION_FIELDS)


def combination_title(params):
    parts = [
        "C1={}".format(_fmt(params["C1"])),
        "A1={}@{}".format(_fmt(params["A1_amp"]), _fmt(params["A1_phase"])),
        "A2={}@{}".format(_fmt(params["A2_amp"]), _fmt(params["A2_phase"])),
        "A3={}@{}".format(_fmt(params["A3_amp"]), _fmt(params["A3_phase"])),
        "C3={}".format(_fmt(params["C3"])),
    ]
    return ", ".join(parts)


def select_c1_offset_pairs(parameters):
    pairs = {}
    representatives = {}
    for index, params in enumerate(parameters):
        key = combination_key(params)
        representatives.setdefault(key, params)
        offset_map = pairs.setdefault(key, {})
        for c1_offset in C1_OFFSETS:
            if np.isclose(params["C1_offset"], c1_offset):
                offset_map[c1_offset] = index

    selected = []
    for key, offset_map in pairs.items():
        if all(c1_offset in offset_map for c1_offset in C1_OFFSETS):
            selected.append((representatives[key], [offset_map[c1_offset] for c1_offset in C1_OFFSETS]))
    return selected


def main():
    output_dir = ROOT / "outputs"
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / "smoke_probe_images.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            "Missing smoke-test output. Run scripts/run_smoke_test.py first."
        )

    for old_plot in plot_dir.glob("line_profiles_*.png"):
        old_plot.unlink()

    probe_images, parameters = load_smoke_outputs(npz_path)
    pairs = select_c1_offset_pairs(parameters)

    print("selected C1_offset paired profile cases:", len(pairs))

    for plot_index, (representative_params, source_indices) in enumerate(pairs):
        stack = probe_images[:, :, source_indices]
        profiles, coords = extract_line_profiles_from_stack(stack, num_lines=37, radius=80)
        profiles_np = asnumpy(profiles)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for local_index, c1_offset in enumerate(C1_OFFSETS):
            axes[local_index].imshow(stack[:, :, local_index], cmap="magma")
            axes[local_index].set_title("C1_offset={} nm".format(c1_offset))
            axes[local_index].set_axis_off()

        # The original notebook samples 36 angles; plot cardinal/diagonal
        # directions to keep each paired comparison readable.
        angle_indices = [
            index for index, angle in enumerate(coords["angles_deg"])
            if np.isclose(angle % 45, 0)
        ]
        for angle_index in angle_indices:
            angle = coords["angles_deg"][angle_index]
            axes[2].plot(
                profiles_np[angle_index, :, 0],
                linestyle="-",
                label="-909 nm, {:.0f} deg".format(angle),
            )
            axes[2].plot(
                profiles_np[angle_index, :, 1],
                linestyle="--",
                label="+909 nm, {:.0f} deg".format(angle),
            )
        axes[2].set_title("Line profiles")
        axes[2].set_xlabel("pixel along line")
        axes[2].set_ylabel("intensity")
        axes[2].legend(ncol=2, fontsize=7)

        fig.suptitle(combination_title(representative_params), fontsize=10)
        fig.tight_layout()

        plot_path = plot_dir / "line_profiles_{:03d}.png".format(plot_index)
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)

        print("saved: {} | source indices {} | {}".format(
            plot_path,
            source_indices,
            combination_title(representative_params),
        ))

    print("generated paired plots:", len(pairs))


if __name__ == "__main__":
    main()
