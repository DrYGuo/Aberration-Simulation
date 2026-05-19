#!/usr/bin/env python
"""Generate line-profile plots from the smoke-test simulation."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aberration_simulation.backend import asnumpy
from aberration_simulation.line_profiles import (
    choose_nonzero_parameter_indices,
    extract_line_profiles_from_stack,
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


def title_for(params):
    parts = [
        "A1={A1_amp:g}@{A1_phase:g}".format(**params),
        "A2={A2_amp:g}@{A2_phase:g}".format(**params),
        "A3={A3_amp:g}@{A3_phase:g}".format(**params),
        "C3={C3:g}",
    ]
    return ", ".join(parts).format(**params)


def main():
    output_dir = ROOT / "outputs"
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / "smoke_probe_images.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            "Missing smoke-test output. Run scripts/run_smoke_test.py first."
        )

    probe_images, parameters = load_smoke_outputs(npz_path)
    indices = choose_nonzero_parameter_indices(parameters, limit=4)
    stack = probe_images[:, :, indices]
    selected_params = [parameters[index] for index in indices]
    profiles, coords = extract_line_profiles_from_stack(stack, num_lines=9, radius=24)
    profiles_np = asnumpy(profiles)

    for local_index, params in enumerate(selected_params):
        image = stack[:, :, local_index]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        axes[0].imshow(image, cmap="magma")
        axes[0].set_title("Probe image")
        axes[0].set_axis_off()

        for angle_index, angle in enumerate(coords["angles_deg"]):
            axes[1].plot(
                profiles_np[angle_index, :, local_index],
                label="{:.0f} deg".format(angle),
            )
        axes[1].set_title("Line profiles")
        axes[1].set_xlabel("pixel along line")
        axes[1].set_ylabel("intensity")
        axes[1].legend(ncol=2, fontsize=8)
        fig.suptitle(title_for(params), fontsize=10)
        fig.tight_layout()

        plot_path = plot_dir / "line_profiles_{:02d}.png".format(local_index)
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        print("saved:", plot_path)


if __name__ == "__main__":
    main()
