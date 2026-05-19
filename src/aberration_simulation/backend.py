"""Array backend selection.

The original notebook targets CuPy. For small local tests and GitHub-friendly
execution, this module falls back to NumPy/SciPy when CuPy is not installed.
"""

try:
    import cupy as xp
    from cupyx.scipy.ndimage import gaussian_filter, map_coordinates

    HAS_CUPY = True
except ImportError:
    import numpy as xp
    from scipy.ndimage import gaussian_filter, map_coordinates

    HAS_CUPY = False


def asnumpy(array):
    """Return a NumPy array regardless of the active backend."""
    if HAS_CUPY:
        return xp.asnumpy(array)
    return array


def free_memory():
    """Release backend memory pools when the active backend supports it."""
    if HAS_CUPY:
        xp.get_default_memory_pool().free_all_blocks()
