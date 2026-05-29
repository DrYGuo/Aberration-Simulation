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
    "b2": 2,
    "c3": 3,
    "a1": 4,
    "a3": 5,
    "s3": 6,
    "c1": 7,
}
COMBINATION_FIELDS = (
    "C1",
    "A1_amp",
    "A1_phase",
    "A2_amp",
    "A2_phase",
    "B2_amp",
    "B2_phase",
    "A3_amp",
    "A3_phase",
    "S3_amp",
    "S3_phase",
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
    for params in parameters:
        params.setdefault("B2_amp", 0.0)
        params.setdefault("B2_phase", 0.0)
        params.setdefault("S3_amp", 0.0)
        params.setdefault("S3_phase", 0.0)
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
        "B2/C21={}@{}".format(_fmt(params["B2_amp"]), _fmt(params["B2_phase"])),
        "A3={}@{}".format(_fmt(params["A3_amp"]), _fmt(params["A3_phase"])),
        "S3/C32={}@{}".format(_fmt(params["S3_amp"]), _fmt(params["S3_phase"])),
        "C3={}".format(_fmt(params["C3"])),
    ]
    return ", ".join(parts)


def aberration_family(params):
    """Return the isolated nonzero aberration family for naming and grouping."""
    if not np.isclose(params["B2_amp"], 0) or not np.isclose(params["B2_phase"], 0):
        return "b2"
    if not np.isclose(params["A2_amp"], 0):
        return "a2"
    if not np.isclose(params["A1_amp"], 0):
        return "a1"
    if not np.isclose(params["C3"], 0):
        return "c3"
    if not np.isclose(params["S3_amp"], 0):
        return "s3"
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
    if family in ("a1", "a2", "b2", "a3", "s3"):
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


def cardinal_diagonal_angle_indices(coords):
    return [
        index for index, angle in enumerate(coords["angles_deg"])
        if np.isclose(angle % 45, 0)
    ]


def profile_line_colors(count):
    cmap = plt.get_cmap("tab10")
    return [cmap(index % cmap.N) for index in range(count)]


def overlay_profile_lines(axis, coords, angle_indices, colors):
    for color, angle_index in zip(colors, angle_indices):
        angle = coords["angles_deg"][angle_index]
        x = coords["x"][angle_index]
        y = coords["y"][angle_index]
        axis.plot(x, y, color=color, linewidth=0.9, alpha=0.85)
        axis.text(
            x[-1],
            y[-1],
            "{:.0f}".format(angle),
            color=color,
            fontsize=6,
            ha="center",
            va="center",
            bbox={"facecolor": "black", "alpha": 0.35, "edgecolor": "none", "pad": 1},
        )


def save_probe_summary(probe_images, pairs, plot_dir, family):
    family_pairs = [
        (params, source_indices)
        for params, source_indices in pairs
        if aberration_family(params) == family
    ]
    if not family_pairs:
        print("warning: no {} profile cases found in smoke-test output".format(family.upper()))
        return

    amp_key = "{}_amp".format(family.upper())
    phase_key = "{}_phase".format(family.upper())
    amps = sorted({params[amp_key] for params, _ in family_pairs})
    phases = sorted({params[phase_key] for params, _ in family_pairs})
    pair_lookup = {
        (params[amp_key], params[phase_key]): source_indices
        for params, source_indices in family_pairs
    }
    baseline_indices = next(
        (
            source_indices
            for params, source_indices in pairs
            if aberration_family(params) == "baseline"
        ),
        None,
    )

    for local_index, c1_offset in enumerate(C1_OFFSETS):
        fig, axes = plt.subplots(
            len(amps),
            len(phases),
            figsize=(1.7 * len(phases), 1.7 * len(amps)),
            squeeze=False,
        )
        for row, amp in enumerate(amps):
            for col, phase in enumerate(phases):
                axis = axes[row, col]
                source_indices = pair_lookup.get((amp, phase))
                if source_indices is None and np.isclose(amp, 0) and baseline_indices is not None:
                    source_indices = baseline_indices
                if source_indices is None:
                    axis.set_axis_off()
                    continue
                image_index = source_indices[local_index]
                axis.imshow(probe_images[:, :, image_index], cmap="magma")
                axis.set_xticks([])
                axis.set_yticks([])
                if row == 0:
                    axis.set_title("phase {}".format(_fmt(phase)), fontsize=7)
                if col == 0:
                    axis.set_ylabel("amp {}".format(_fmt(amp)), fontsize=7)

        title_name = "B2/C21" if family == "b2" else family.upper()
        fig.suptitle("{} probe summary, C1_offset={} nm".format(title_name, c1_offset), fontsize=12)
        fig.tight_layout()
        offset_slug = "m{}".format(abs(c1_offset)) if c1_offset < 0 else "p{}".format(c1_offset)
        plot_path = plot_dir / "line_profiles_000_{}_summary_c1_{}.png".format(family, offset_slug)
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
        print("saved {} summary:".format(title_name), plot_path)


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
            item[0]["B2_amp"],
            item[0]["B2_phase"],
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
    save_probe_summary(probe_images, pairs, plot_dir, "a2")
    save_probe_summary(probe_images, pairs, plot_dir, "b2")

    for plot_index, (representative_params, source_indices) in enumerate(pairs):
        stack = probe_images[:, :, source_indices]
        profiles, coords = extract_line_profiles_from_stack(stack, num_lines=37, radius=80)
        profiles_np = asnumpy(profiles)
        angle_indices = cardinal_diagonal_angle_indices(coords)
        colors = profile_line_colors(len(angle_indices))

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for local_index, c1_offset in enumerate(C1_OFFSETS):
            image_axis = axes[local_index, 0]

            image_axis.imshow(stack[:, :, local_index], cmap="magma")
            overlay_profile_lines(image_axis, coords, angle_indices, colors)
            image_axis.set_title("Probe image: C1_offset={} nm".format(c1_offset))
            image_axis.set_axis_off()

        # The original notebook samples 36 angles; plot cardinal/diagonal
        # directions to keep each offset row readable.
        for local_index, c1_offset in enumerate(C1_OFFSETS):
            profile_axis = axes[local_index, 1]
            for color, angle_index in zip(colors, angle_indices):
                angle = coords["angles_deg"][angle_index]
                profile_axis.plot(
                    profiles_np[angle_index, :, local_index],
                    color=color,
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
