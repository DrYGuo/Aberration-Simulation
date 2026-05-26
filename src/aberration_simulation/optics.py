"""Compatibility import for optical simulation backends.

When CuPy is installed, this module exposes the fully vectorized GPU
implementation from :mod:`aberration_simulation.gpu_optics`. Otherwise it
exposes the NumPy/SciPy CPU implementation from
:mod:`aberration_simulation.cpu_optics`.
"""

try:
    from .gpu_optics import *  # noqa: F401,F403

    USING_GPU_BACKEND = True
except ImportError:
    from .cpu_optics import *  # noqa: F401,F403

    USING_GPU_BACKEND = False
