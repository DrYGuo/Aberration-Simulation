"""Optical aberration and probe image simulation."""

from dataclasses import dataclass
import gc

import numpy as np

from .backend import asnumpy, free_memory, gaussian_filter, xp


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
    pix_dim: tuple = (64, 64)
    real_dim: tuple = (320, 320)
    eV: float = 0.8e3
    app: float = 30.0
    optic_axis: tuple = (0.0, 0.0)
    aperture_shift: tuple = (0.0, 0.0)
    tilt_units: str = "mrad"
    df: float = 0.0
    app_units: str = "mrad"
    sigma: float = 1.0


def wavev(E):
    """Relativistically corrected electron wavenumber for energy in eV."""
    hc = 1.23984193e4
    m0c2 = 5.109989461e5
    return xp.sqrt(E * (E + 2 * m0c2)) / hc


def convert_tilt_angles(tilt, tilt_units, rsize, eV, invA_out=False):
    """Convert mrad, pixel, or inverse-Angstrom tilt values."""
    if tilt_units == "mrad":
        k = wavev(eV)
        tilt_ = xp.asarray(tilt, dtype=float) * 1e-3 * k
    else:
        tilt_ = xp.asarray(tilt)

    if invA_out:
        return tilt_

    if tilt_units != "pixels":
        tilt_ = xp.round(tilt_ * xp.asarray(rsize[:2])).astype(int)
    return tilt_


def q_space_array(pixels, gridsize):
    """Return scaled 2D reciprocal-space coordinate arrays."""
    qspace = [
        xp.fft.fftfreq(int(pixels[i]), d=float(gridsize[i]) / int(pixels[i]))
        for i in (0, 1)
    ]
    shape = tuple(int(p) for p in pixels)
    return [
        xp.broadcast_to(q, shape)
        for q in (qspace[0][:, None], qspace[1][None, :])
    ]


def chi(q, qphi, lam, df=0.0, aberrations=None):
    """Aberration phase function."""
    aberrations = aberrations or []
    qlam = q * lam
    phase = qlam ** 2 / 2 * df
    for ab in aberrations:
        phase += (
            qlam ** (ab.n + 1)
            * ab.amplitude
            / (ab.n + 1)
            * xp.cos(ab.m * (qphi - ab.angle))
        )
    return 2 * xp.pi * phase / lam


def _as_backend_array(values):
    return xp.asarray(list(values), dtype=float)


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
):
    """Build a flat list of parameter dictionaries in notebook meshgrid order."""
    names = [
        "C1_offset",
        "A3_amp",
        "A2_amp",
        "C1",
        "C3",
        "A1_amp",
        "A1_phase",
        "A2_phase",
        "A3_phase",
    ]
    arrays = [
        _as_backend_array(C1_offset_sequence),
        _as_backend_array(A3_amp_sequence),
        _as_backend_array(A2_amp_sequence),
        _as_backend_array(C1_sequence),
        _as_backend_array(C3_sequence),
        _as_backend_array(A1_amp_sequence),
        _as_backend_array(A1_phase_sequence),
        _as_backend_array(A2_phase_sequence),
        _as_backend_array(A3_phase_sequence),
    ]
    mesh = xp.meshgrid(*arrays, indexing="ij")
    flat = [asnumpy(m.ravel()) for m in mesh]
    return [
        {name: float(values[i]) for name, values in zip(names, flat)}
        for i in range(len(flat[0]))
    ]


def aberrations_from_parameters(params):
    """Convert notebook-style coefficient values into Aberration objects."""
    return [
        Aberration("C1", "C1", "Defocus", 10 * params["C1"] + 10 * params["C1_offset"], 0, 1, 0),
        Aberration("A1", "A1", "2-Fold Astigmatism", 10 * params["A1_amp"], -xp.radians(params["A1_phase"]), 1, 2),
        Aberration("A2", "A2", "3-Fold Astigmatism", 1e4 * params["A2_amp"], -xp.radians(params["A2_phase"]), 2, 3),
        Aberration("A3", "A3", "4-Fold Astigmatism", 1e4 * params["A3_amp"], -xp.radians(params["A3_phase"]), 3, 4),
        Aberration("C3", "C3", "Spherical Aberration", 1e7 * params["C3"], 0, 3, 0),
    ]


def make_contrast_transfer_function(
    config,
    parameters,
    q=None,
    batch_index=0,
    batch_size=None,
):
    """Generate a CTF tensor with shape `(height, width, combinations)`."""
    pix_dim = tuple(config.pix_dim)
    real_dim = tuple(config.real_dim)

    if q is None:
        q = q_space_array(pix_dim, real_dim[:2])

    k = wavev(config.eV)
    optic_axis = convert_tilt_angles(
        config.optic_axis, config.tilt_units, real_dim, config.eV, invA_out=True
    )
    aperture_shift = convert_tilt_angles(
        config.aperture_shift, config.tilt_units, real_dim, config.eV, invA_out=True
    )

    if config.app is None:
        app = xp.amax(xp.abs(q))
    else:
        app = convert_tilt_angles(config.app, config.app_units, real_dim, config.eV, invA_out=True)

    if batch_size is None:
        batch_size = len(parameters)
    start_index = int(batch_index) * int(batch_size)
    end_index = min(start_index + int(batch_size), len(parameters))
    selected = parameters[start_index:end_index]

    ctf_tensor = xp.zeros((pix_dim[0], pix_dim[1], len(selected)), dtype=complex)
    qarray1 = xp.sqrt((q[0] - optic_axis[0]) ** 2 + (q[1] - optic_axis[1]) ** 2)
    qarray2 = (
        (q[0] - optic_axis[0] - aperture_shift[0]) ** 2
        + (q[1] - optic_axis[1] - aperture_shift[1]) ** 2
    )
    qphi = xp.arctan2(q[0] - optic_axis[0], q[1] - optic_axis[1])
    mask = qarray2 <= app ** 2
    lam = 1.0 / k

    for output_index, params in enumerate(selected):
        phase = xp.zeros(pix_dim, dtype=float)
        aberrations = aberrations_from_parameters(params)
        phase[mask] = chi(qarray1[mask], qphi[mask], lam, config.df, aberrations)
        ctf_tensor[:, :, output_index][mask] = xp.exp(-1j * phase[mask])

    free_memory()
    gc.collect()
    return ctf_tensor, selected


def compute_probe_image(ctf_tensor, sigma=1.0):
    """Compute smoothed probe intensities from a CTF tensor."""
    ctf_real = xp.fft.ifft2(ctf_tensor, axes=(0, 1))
    shifted = xp.fft.fftshift(ctf_real, axes=(0, 1))
    probe_image = xp.abs(shifted * xp.conj(shifted))
    smooth_sigma = (sigma, sigma) + (0,) * (probe_image.ndim - 2)
    probe_image_final = gaussian_filter(probe_image, sigma=smooth_sigma, mode="constant")
    free_memory()
    gc.collect()
    return probe_image_final


def run_simulation(config, parameter_grid, batch_size=None):
    """Run CTF and probe-image computation for the supplied parameter grid."""
    ctf_tensor, selected = make_contrast_transfer_function(
        config, parameter_grid, batch_size=batch_size
    )
    probe_images = compute_probe_image(ctf_tensor, sigma=config.sigma)
    return probe_images, selected


def save_npz(path, probe_images, parameters):
    """Save simulation outputs in a portable NumPy archive."""
    keys = sorted(parameters[0].keys()) if parameters else []
    parameter_table = np.array(
        [[params[key] for key in keys] for params in parameters],
        dtype=float,
    )
    np.savez_compressed(
        path,
        probe_images=asnumpy(probe_images),
        parameter_names=np.array(keys),
        parameters=parameter_table,
    )
