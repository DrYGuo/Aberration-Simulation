"""CuPy GPU optical aberration and probe image simulation.

This module is intentionally CuPy-only for readability and for keeping the GPU
path separate from the CPU implementation. In contrast to the original
notebook, CTF construction is vectorized across the aberration coefficient
combinations in each batch.
"""

from dataclasses import dataclass
import gc

import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import gaussian_filter


PARAMETER_NAMES = [
    "C1_offset",
    "A3_amp",
    "A2_amp",
    "B2_amp",
    "C1",
    "C3",
    "A1_amp",
    "A1_phase",
    "A2_phase",
    "A3_phase",
    "B2_phase",
]


@dataclass
class Aberration:
    krivanek: str
    haider: str
    description: str
    amplitude: float
    angle: float
    n: int
    m: int

    def __post_init__(self):
        if self.m <= 0:
            self.angle = 0.0


@dataclass
class SimulationConfig:
    pix_dim: tuple = (256, 256)
    real_dim: tuple = (1280, 1280)
    eV: float = 0.8e3
    app: float = 30.0
    optic_axis: tuple = (0.0, 0.0)
    aperture_shift: tuple = (0.0, 0.0)
    tilt_units: str = "mrad"
    df: float = 0.0
    app_units: str = "mrad"
    sigma: float = 2.0


def wavev(E):
    """Relativistically corrected electron wavenumber for energy in eV."""
    hc = 1.23984193e4
    m0c2 = 5.109989461e5
    return cp.sqrt(E * (E + 2 * m0c2)) / hc


def convert_tilt_angles(tilt, tilt_units, rsize, eV, invA_out=False):
    """Convert mrad, pixel, or inverse-Angstrom tilt values."""
    if tilt_units == "mrad":
        k = wavev(eV)
        tilt_ = cp.asarray(tilt, dtype=float) * 1e-3 * k
    else:
        tilt_ = cp.asarray(tilt)

    if invA_out:
        return tilt_

    if tilt_units != "pixels":
        tilt_ = cp.round(tilt_ * cp.asarray(rsize[:2])).astype(int)
    return tilt_


def q_space_array(pixels, gridsize):
    """Return scaled 2D reciprocal-space coordinate arrays."""
    qspace = [
        cp.fft.fftfreq(int(pixels[i]), d=float(gridsize[i]) / int(pixels[i]))
        for i in (0, 1)
    ]
    shape = tuple(int(p) for p in pixels)
    return [
        cp.broadcast_to(q, shape)
        for q in (qspace[0][:, None], qspace[1][None, :])
    ]


def _as_array(values):
    return cp.asarray(list(values), dtype=float)


def build_parameter_table(
    C1_offset_sequence,
    A3_amp_sequence,
    A2_amp_sequence,
    C1_sequence,
    C3_sequence,
    A1_amp_sequence,
    A1_phase_sequence,
    A2_phase_sequence,
    A3_phase_sequence,
    B2_amp_sequence=(0,),
    B2_phase_sequence=(0,),
):
    """Build a flat CuPy parameter table in notebook meshgrid order."""
    arrays = [
        _as_array(C1_offset_sequence),
        _as_array(A3_amp_sequence),
        _as_array(A2_amp_sequence),
        _as_array(B2_amp_sequence),
        _as_array(C1_sequence),
        _as_array(C3_sequence),
        _as_array(A1_amp_sequence),
        _as_array(A1_phase_sequence),
        _as_array(A2_phase_sequence),
        _as_array(A3_phase_sequence),
        _as_array(B2_phase_sequence),
    ]
    mesh = cp.meshgrid(*arrays, indexing="ij")
    return {
        name: values.ravel()
        for name, values in zip(PARAMETER_NAMES, mesh)
    }


def parameter_table_to_records(parameter_table):
    """Convert a CuPy parameter table to a list of Python dictionaries."""
    table = {
        name: cp.asnumpy(values)
        for name, values in parameter_table.items()
    }
    count = len(next(iter(table.values()))) if table else 0
    return [
        {name: float(table[name][index]) for name in PARAMETER_NAMES}
        for index in range(count)
    ]


def build_parameter_grid(
    C1_offset_sequence,
    A3_amp_sequence,
    A2_amp_sequence,
    C1_sequence,
    C3_sequence,
    A1_amp_sequence,
    A1_phase_sequence,
    A2_phase_sequence,
    A3_phase_sequence,
    B2_amp_sequence=(0,),
    B2_phase_sequence=(0,),
):
    """Build a list of parameter dictionaries for API compatibility."""
    table = build_parameter_table(
        C1_offset_sequence,
        A3_amp_sequence,
        A2_amp_sequence,
        C1_sequence,
        C3_sequence,
        A1_amp_sequence,
        A1_phase_sequence,
        A2_phase_sequence,
        A3_phase_sequence,
        B2_amp_sequence,
        B2_phase_sequence,
    )
    return parameter_table_to_records(table)


def parameter_records_to_table(parameters):
    """Convert list-of-dict parameters into a CuPy parameter table."""
    return {
        name: cp.asarray([params.get(name, 0) for params in parameters], dtype=float)
        for name in PARAMETER_NAMES
    }


def slice_parameter_table(parameter_table, batch_index=0, batch_size=None):
    """Return a batch slice from a CuPy parameter table."""
    count = len(next(iter(parameter_table.values())))
    if batch_size is None:
        batch_size = count
    start_index = int(batch_index) * int(batch_size)
    end_index = min(start_index + int(batch_size), count)
    return {
        name: values[start_index:end_index]
        for name, values in parameter_table.items()
    }


def chi(q, qphi, lam, df=0.0, aberrations=None):
    """Compatibility phase function for one list of Aberration objects."""
    aberrations = aberrations or []
    qlam = q * lam
    phase = qlam ** 2 / 2 * df
    for ab in aberrations:
        phase += (
            qlam ** (ab.n + 1)
            * ab.amplitude
            / (ab.n + 1)
            * cp.cos(ab.m * (qphi - ab.angle))
        )
    return 2 * cp.pi * phase / lam


def aberrations_from_parameters(params):
    """Convert notebook-style coefficient values into Aberration objects."""
    return [
        Aberration("C1", "C1", "Defocus", 10 * params["C1"] + 10 * params["C1_offset"], 0, 1, 0),
        Aberration("A1", "A1", "2-Fold Astigmatism", 10 * params["A1_amp"], -cp.radians(params["A1_phase"]), 1, 2),
        Aberration("A2", "A2", "3-Fold Astigmatism", 1e4 * params["A2_amp"], -cp.radians(params["A2_phase"]), 2, 3),
        Aberration("B2", "C21", "Axial Coma", 1e4 * params.get("B2_amp", 0), -cp.radians(params.get("B2_phase", 0)), 2, 1),
        Aberration("A3", "A3", "4-Fold Astigmatism", 1e4 * params["A3_amp"], -cp.radians(params["A3_phase"]), 3, 4),
        Aberration("C3", "C3", "Spherical Aberration", 1e7 * params["C3"], 0, 3, 0),
    ]


def _phase_for_parameter_table(q_mask, qphi_mask, lam, df, parameter_table):
    """Vectorized aberration phase for all selected coefficient combinations."""
    qlam = q_mask[:, None] * lam

    C1_amp = 10 * parameter_table["C1"][None, :] + 10 * parameter_table["C1_offset"][None, :]
    A1_amp = 10 * parameter_table["A1_amp"][None, :]
    A2_amp = 1e4 * parameter_table["A2_amp"][None, :]
    B2_amp = 1e4 * parameter_table["B2_amp"][None, :]
    A3_amp = 1e4 * parameter_table["A3_amp"][None, :]
    C3_amp = 1e7 * parameter_table["C3"][None, :]

    A1_angle = -cp.radians(parameter_table["A1_phase"])[None, :]
    A2_angle = -cp.radians(parameter_table["A2_phase"])[None, :]
    B2_angle = -cp.radians(parameter_table["B2_phase"])[None, :]
    A3_angle = -cp.radians(parameter_table["A3_phase"])[None, :]
    qphi = qphi_mask[:, None]

    phase = qlam ** 2 / 2 * df
    phase = phase + qlam ** 2 * C1_amp / 2
    phase = phase + qlam ** 2 * A1_amp / 2 * cp.cos(2 * (qphi - A1_angle))
    phase = phase + qlam ** 3 * A2_amp / 3 * cp.cos(3 * (qphi - A2_angle))
    phase = phase + qlam ** 3 * B2_amp / 3 * cp.cos(qphi - B2_angle)
    phase = phase + qlam ** 4 * A3_amp / 4 * cp.cos(4 * (qphi - A3_angle))
    phase = phase + qlam ** 4 * C3_amp / 4
    return 2 * cp.pi * phase / lam


def make_contrast_transfer_function(
    config,
    parameters,
    q=None,
    batch_index=0,
    batch_size=None,
):
    """Generate a CTF tensor vectorized across coefficient combinations."""
    pix_dim = tuple(config.pix_dim)
    real_dim = tuple(config.real_dim)

    if q is None:
        q = q_space_array(pix_dim, real_dim[:2])

    parameter_table = (
        parameters
        if isinstance(parameters, dict)
        else parameter_records_to_table(parameters)
    )
    selected_table = slice_parameter_table(parameter_table, batch_index, batch_size)
    selected = parameter_table_to_records(selected_table)

    k = wavev(config.eV)
    optic_axis = convert_tilt_angles(
        config.optic_axis, config.tilt_units, real_dim, config.eV, invA_out=True
    )
    aperture_shift = convert_tilt_angles(
        config.aperture_shift, config.tilt_units, real_dim, config.eV, invA_out=True
    )

    if config.app is None:
        app = cp.amax(cp.abs(q))
    else:
        app = convert_tilt_angles(config.app, config.app_units, real_dim, config.eV, invA_out=True)

    qarray1 = cp.sqrt((q[0] - optic_axis[0]) ** 2 + (q[1] - optic_axis[1]) ** 2)
    qarray2 = (
        (q[0] - optic_axis[0] - aperture_shift[0]) ** 2
        + (q[1] - optic_axis[1] - aperture_shift[1]) ** 2
    )
    qphi = cp.arctan2(q[0] - optic_axis[0], q[1] - optic_axis[1])
    mask = qarray2 <= app ** 2

    ctf_tensor = cp.zeros((pix_dim[0], pix_dim[1], len(selected)), dtype=complex)
    if selected:
        chi_mask = _phase_for_parameter_table(
            qarray1[mask],
            qphi[mask],
            1.0 / k,
            config.df,
            selected_table,
        )
        ctf_flat = ctf_tensor.reshape((-1, len(selected)))
        ctf_flat[mask.ravel(), :] = cp.exp(-1j * chi_mask)

    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()
    return ctf_tensor, selected


def compute_probe_image(ctf_tensor, sigma=2.0):
    """Compute smoothed probe intensities from a CTF tensor."""
    ctf_real = cp.fft.ifft2(ctf_tensor, axes=(0, 1))
    shifted = cp.fft.fftshift(ctf_real, axes=(0, 1))
    probe_image = cp.abs(shifted * cp.conj(shifted))
    smooth_sigma = (sigma, sigma) + (0,) * (probe_image.ndim - 2)
    probe_image_final = gaussian_filter(probe_image, sigma=smooth_sigma, mode="constant")
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()
    return probe_image_final


def run_simulation(config, parameter_grid, batch_size=None):
    """Run vectorized CTF and probe-image computation."""
    ctf_tensor, selected = make_contrast_transfer_function(
        config, parameter_grid, batch_size=batch_size
    )
    probe_images = compute_probe_image(ctf_tensor, sigma=config.sigma)
    return probe_images, selected


def run_simulation_from_sequences(config, batch_size=None, **sequences):
    """Build a CuPy parameter table from sequences and run the GPU simulation."""
    parameter_table = build_parameter_table(**sequences)
    return run_simulation(config, parameter_table, batch_size=batch_size)


def save_npz(path, probe_images, parameters):
    """Save simulation outputs in a portable NumPy archive."""
    keys = sorted(parameters[0].keys()) if parameters else []
    parameter_table = np.array(
        [[params[key] for key in keys] for params in parameters],
        dtype=float,
    )
    np.savez_compressed(
        path,
        probe_images=cp.asnumpy(probe_images),
        parameter_names=np.array(keys),
        parameters=parameter_table,
    )
