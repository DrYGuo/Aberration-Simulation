#!/usr/bin/env python
"""Run a small aberration simulation as a smoke test."""

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aberration_simulation.backend import HAS_CUPY, asnumpy
from aberration_simulation.optics import (
    SimulationConfig,
    run_simulation,
    save_npz,
)


BASELINE_PARAMETERS = {
    "C1_offset": 0,
    "A3_amp": 0,
    "A2_amp": 0,
    "C1": 0,
    "C3": 0,
    "A1_amp": 0,
    "A1_phase": 0,
    "A2_phase": 0,
    "A3_phase": 0,
}
C1_OFFSETS = [-909, 909]


def smoke_parameter_grid():
    """Build targeted smoke cases paired at two C1 offset values."""
    base_cases = [dict(BASELINE_PARAMETERS)]

    for c3 in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 1.8, 2.0]:
        params = dict(BASELINE_PARAMETERS)
        params["C3"] = c3
        base_cases.append(params)

    for a1_amp in [2, 5, 10, 15, 20, 30, 40, 60]:
        for a1_phase in [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]:
            params = dict(BASELINE_PARAMETERS)
            params["A1_amp"] = a1_amp
            params["A1_phase"] = a1_phase
            base_cases.append(params)

    parameters = []
    for base_case in base_cases:
        for c1_offset in C1_OFFSETS:
            params = dict(base_case)
            params["C1_offset"] = c1_offset
            parameters.append(params)
    return parameters


def write_parameter_csv(path, parameters):
    keys = sorted(parameters[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(parameters)


def main():
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    config = SimulationConfig(
        pix_dim=(256, 256),
        real_dim=(1280, 1280),
        app=30,
        sigma=2,
    )
    parameters = smoke_parameter_grid()
    probe_images, selected = run_simulation(config, parameters)

    npz_path = output_dir / "smoke_probe_images.npz"
    csv_path = output_dir / "smoke_parameters.csv"
    save_npz(npz_path, probe_images, selected)
    write_parameter_csv(csv_path, selected)

    probe_np = asnumpy(probe_images)
    print("backend:", "cupy" if HAS_CUPY else "numpy/scipy")
    print("parameter combinations:", len(selected))
    print("probe image shape:", probe_np.shape)
    print("intensity range:", float(probe_np.min()), float(probe_np.max()))
    print("saved:", npz_path)
    print("saved:", csv_path)


if __name__ == "__main__":
    main()
