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
    run_simulation_from_sequences,
    save_npz,
)


SMOKE_SEQUENCES = {
    "C1_offset_sequence": [0],
    "A3_amp_sequence": [0, 20],
    "A2_amp_sequence": [0, 2],
    "C1_sequence": [0],
    "C3_sequence": [0, 0.3],
    "A1_amp_sequence": [0, 20],
    "A1_phase_sequence": [0],
    "A2_phase_sequence": [0, 60],
    "A3_phase_sequence": [0, 90],
}


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
        pix_dim=(64, 64),
        real_dim=(320, 320),
        app=30,
        sigma=1,
    )
    probe_images, selected = run_simulation_from_sequences(config, **SMOKE_SEQUENCES)

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
