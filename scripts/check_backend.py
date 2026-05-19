#!/usr/bin/env python
"""Report which array backend the simulation package is using."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aberration_simulation.backend import HAS_CUPY, xp


def main():
    print("HAS_CUPY:", HAS_CUPY)
    print("backend module:", xp.__name__)
    print("backend version:", getattr(xp, "__version__", "n/a"))
    if HAS_CUPY:
        print("cuda runtime:", xp.cuda.runtime.runtimeGetVersion())
        device = xp.cuda.Device()
        print("gpu device:", xp.cuda.runtime.getDeviceProperties(device.id)["name"].decode())


if __name__ == "__main__":
    main()
