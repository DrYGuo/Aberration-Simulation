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
FAMILY_ORDER = {
    "baseline": 0,
    "a2": 1,
    "c3": 2,
    "a1": 3,
    "a3": 4,
    "c1": 5,
}
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


def aberration_family(params):
    """Return the isolated nonzero aberration family for naming and grouping."""
    if not np.isclose(params["A2_amp"], 0):
        return "a2"
    if not np.isclose(params["A1_amp"], 0):
        return "a1"
    if not np.isclose(params["C3"], 0):
        return "c3"
    if not np.isclose(params["A3_amp"], 0):
        return "a3"
    if not np.isclose(params["C1"], 0):
        return "c1"
    return "baseline"


def _slug(value):
    text = _fmt(value).replace("-", "m").replace(".", "p")
    return text


def plot_filename(plot_index, params):
    family = aberration_family(params)
    if family in ("a1", "a2", "a3"):
        amp_key = "{}_amp".format(family.upper())
        phase_key = "{}_phase".format(family.upper())
        suffix = "{}_amp{}_phase{}".format(
            family,
            _slug(params[amp_key]),
            _slug(params[phase_key]),
        )
    elif family == "c3":
        suffix = "c3_{}".format(_slug(params["C3"]))
    elif family == "c1":
        suffix = "c1_{}".format(_slug(params["C1"]))
    else:
        suffix = "baseline"
    return "line_profiles_{:03d}_{}.png".format(plot_index, suffix)


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
    selected.sort(
        key=lambda item: (
            FAMILY_ORDER.get(aberration_family(item[0]), 99),
            item[0]["A2_amp"],
            item[0]["A2_phase"],
            item[0]["C3"],
            item[0]["A1_amp"],
            item[0]["A1_phase"],
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
    pairs = select_c1_offset_pairs(parameters)

    print("selected C1_offset paired profile cases:", len(pairs))
    family_counts = {}
    for representative_params, _ in pairs:
        family = aberration_family(representative_params)
        family_counts[family] = family_counts.get(family, 0) + 1
    print("profile case families:", family_counts)

    for plot_index, (representative_params, source_indices) in enumerate(pairs):
        stack = probe_images[:, :, source_indices]
        profiles, coords = extract_line_profiles_from_stack(stack, num_lines=37, radius=80)
        profiles_np = asnumpy(profiles)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for local_index, c1_offset in enumerate(C1_OFFSETS):
            image_axis = axes[local_index, 0]

            image_axis.imshow(stack[:, :, local_index], cmap="magma")
            image_axis.set_title("Probe image: C1_offset={} nm".format(c1_offset))
            image_axis.set_axis_off()

        # The original notebook samples 36 angles; plot cardinal/diagonal
        # directions to keep each offset row readable.
        angle_indices = [
            index for index, angle in enumerate(coords["angles_deg"])
            if np.isclose(angle % 45, 0)
        ]
        for local_index, c1_offset in enumerate(C1_OFFSETS):
            profile_axis = axes[local_index, 1]
            for angle_index in angle_indices:
                angle = coords["angles_deg"][angle_index]
                profile_axis.plot(
                    profiles_np[angle_index, :, local_index],
                    label="{:.0f} deg".format(angle),
                )
            profile_axis.set_title("Line profiles: C1_offset={} nm".format(c1_offset))
            profile_axis.set_xlabel("pixel along line")
            profile_axis.set_ylabel("intensity")
            profile_axis.legend(ncol=4, fontsize=7)

        fig.suptitle(combination_title(representative_params), fontsize=10)
        fig.tight_layout()

        plot_path = plot_dir / plot_filename(plot_index, representative_params)
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
