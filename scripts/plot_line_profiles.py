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
from aberration_simulation.line_profiles import extract_line_profiles_from_stack


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
        "C1={C1:g}".format(**params),
        "C1_offset={C1_offset:g}".format(**params),
        "A1={A1_amp:g}@{A1_phase:g}".format(**params),
        "A2={A2_amp:g}@{A2_phase:g}".format(**params),
        "A3={A3_amp:g}@{A3_phase:g}".format(**params),
        "C3={C3:g}",
    ]
    return ", ".join(parts).format(**params)


def _find_case(parameters, predicate):
    for index, params in enumerate(parameters):
        if predicate(params):
            return index
    raise ValueError("Could not find requested line-profile case.")


def select_profile_indices(parameters):
    """Select deterministic baseline, C3-only, and A1-only profile cases."""
    selected = [
        _find_case(
            parameters,
            lambda params: all(
                np.isclose(params[key], 0)
                for key in ("A1_amp", "A2_amp", "A3_amp", "C1", "C1_offset", "C3")
            ),
        )
    ]

    for c3 in (0.3, 1.2, 2.0):
        selected.append(
            _find_case(
                parameters,
                lambda params, c3=c3: (
                    np.isclose(params["C3"], c3)
                    and np.isclose(params["A1_amp"], 0)
                    and np.isclose(params["A2_amp"], 0)
                    and np.isclose(params["A3_amp"], 0)
                    and np.isclose(params["C1"], 0)
                    and np.isclose(params["C1_offset"], 0)
                ),
            )
        )

    for a1_amp, a1_phase in ((10, 0), (20, 45), (60, 90)):
        selected.append(
            _find_case(
                parameters,
                lambda params, a1_amp=a1_amp, a1_phase=a1_phase: (
                    np.isclose(params["A1_amp"], a1_amp)
                    and np.isclose(params["A1_phase"], a1_phase)
                    and np.isclose(params["C3"], 0)
                    and np.isclose(params["A2_amp"], 0)
                    and np.isclose(params["A3_amp"], 0)
                    and np.isclose(params["C1"], 0)
                    and np.isclose(params["C1_offset"], 0)
                ),
            )
        )

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
    indices = select_profile_indices(parameters)
    stack = probe_images[:, :, indices]
    selected_params = [parameters[index] for index in indices]
    profiles, coords = extract_line_profiles_from_stack(stack, num_lines=9, radius=24)
    profiles_np = asnumpy(profiles)

    print("selected profile cases:")
    for output_index, source_index in enumerate(indices):
        print("  plot {:02d}: source index {}, {}".format(
            output_index,
            source_index,
            title_for(parameters[source_index]),
        ))

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

    print("generated plots:", len(indices))


if __name__ == "__main__":
    main()
