"""Uno et al. 2005 value and phase-convention helpers.

The raw complex values from formulas (38)-(44) are harmonic coefficients.
Their displayed physical phases are fitted to the simulation convention used by
the probe generator:

    phase = sign * raw_complex_phase / harmonic_order + offset

The result is wrapped to the coefficient period, 360 / harmonic_order degrees.
"""

import numpy as np


try:
    import cupy as cp
except Exception:  # pragma: no cover - optional GPU dependency
    cp = None


UNO_HARMONIC_ORDERS = {
    "A1_value": 2,
    "B2_value": 1,
    "A2_value": 3,
    "S3_value": 2,
    "A3_value": 4,
}


PRIMARY_PHASE_CONVENTIONS = {
    "A1_value": {"sign": -1.0, "offset_deg": 90.0},
    "B2_value": {"sign": -1.0, "offset_deg": 0.0},
    "A2_value": {"sign": -1.0, "offset_deg": 0.0},
    "S3_value": {"sign": -1.0, "offset_deg": 0.0},
    "A3_value": {"sign": -1.0, "offset_deg": 45.0},
}


def _array_module(array):
    if cp is not None and isinstance(array, cp.ndarray):
        return cp
    return np


def _to_numpy_value(value):
    if cp is not None and isinstance(value, cp.ndarray):
        value = cp.asnumpy(value)
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def compute_line_characteristics(profiles, radius):
    """Compute Uno profile quantities Xigma, Mu, and Rho.

    `profiles` has shape `(num_angles, 2 * radius + 1)`. The implementation
    uses the input array module, so CuPy input stays on the GPU and NumPy input
    stays on the CPU.
    """
    xp = _array_module(profiles)
    profiles = xp.asarray(profiles)
    j = xp.arange(-radius, radius + 1, dtype=float)
    center_index = int(xp.argmin(xp.abs(j)).item())
    p0 = profiles[:, center_index]

    W = xp.sum(profiles, axis=1)
    T = xp.sum(profiles ** 2, axis=1)
    W = xp.where(W == 0, xp.nan, W)
    T = xp.where(T == 0, xp.nan, T)

    Xigma = xp.sqrt(xp.sum((j[None, :] ** 2) * profiles, axis=1) / W)
    Mu = xp.sum(j[None, :] * profiles, axis=1) / W

    nonzero = j != 0
    curvature_sum = xp.sum(
        ((profiles[:, nonzero] - p0[:, None]) * profiles[:, nonzero])
        / xp.abs(j[nonzero])[None, :],
        axis=1,
    )
    Rho = (Xigma ** 2 / T) * curvature_sum

    return {"Xigma": Xigma, "Mu": Mu, "Rho": Rho}


def compute_uno_values(under_chars, over_chars, angles_deg):
    """Compute current Uno diagnostic values from paired focus profiles.

    Definitions:

    - `Cdf_value = mean(Xigma_under - Xigma_over)`
    - `C1_value = -Cdf_value`
    - `A1_value = 2 mean((Xigma_under - Xigma_over) exp(2 i theta))`
    - `B2_value = 2 mean((Mu_under + Mu_over) exp(i theta))`
    - `A2_value = 2 mean((Mu_under + Mu_over) exp(3 i theta))`
    - `Cs_value = mean(Rho_under - Rho_over)`
    - `C3_value = -mean(Rho_over)`
    - `S3_value = 2 mean((Rho_under - Rho_over) exp(2 i theta))`
    - `A3_value = 2 mean((Xigma_under - Xigma_over) exp(4 i theta))`

    Inputs may be NumPy or CuPy arrays. Outputs use the same array module as
    the profile-characteristic arrays.
    """
    xp = _array_module(under_chars["Xigma"])
    theta = xp.deg2rad(xp.asarray(angles_deg, dtype=float))
    N = len(theta)

    Xigma_diff = under_chars["Xigma"] - over_chars["Xigma"]
    Mu_sum = under_chars["Mu"] + over_chars["Mu"]
    Rho_diff = under_chars["Rho"] - over_chars["Rho"]

    Cdf_value = xp.sum(Xigma_diff) / N
    C1_value = -Cdf_value
    A1_value = 2 * xp.sum(Xigma_diff * xp.exp(2j * theta)) / N
    B2_value = 2 * xp.sum(Mu_sum * xp.exp(1j * theta)) / N
    A2_value = 2 * xp.sum(Mu_sum * xp.exp(3j * theta)) / N
    Cs_value = xp.sum(Rho_diff) / N
    C3_value = -xp.sum(over_chars["Rho"]) / N
    S3_value = 2 * xp.sum(Rho_diff * xp.exp(2j * theta)) / N
    A3_value = 2 * xp.sum(Xigma_diff * xp.exp(4j * theta)) / N

    return {
        "Cdf_value": Cdf_value,
        "C1_value": C1_value,
        "A1_value": A1_value,
        "B2_value": B2_value,
        "A2_value": A2_value,
        "Cs_value": Cs_value,
        "C3_value": C3_value,
        "S3_value": S3_value,
        "A3_value": A3_value,
    }


def combination_key(params, fields):
    """Build a metadata key for grouping under/over-focus simulations."""
    return tuple(params.get(field, 0.0) for field in fields)


def select_under_over_pairs(
    parameters,
    fields,
    under_focus_c1_offset=-909,
    over_focus_c1_offset=909,
):
    """Return `(representative_params, under_index, over_index)` tuples."""
    pairs = {}
    representatives = {}
    for index, params in enumerate(parameters):
        key = combination_key(params, fields)
        representatives.setdefault(key, params)
        pair = pairs.setdefault(key, {})
        if np.isclose(params["C1_offset"], under_focus_c1_offset):
            pair["under"] = index
        if np.isclose(params["C1_offset"], over_focus_c1_offset):
            pair["over"] = index

    selected_pairs = []
    for key, pair in pairs.items():
        if "under" in pair and "over" in pair:
            selected_pairs.append((representatives[key], pair["under"], pair["over"]))
    return selected_pairs


def wrap_period_deg(angle_deg, period_deg):
    """Wrap an angle to `[0, period_deg)`."""
    return float(np.mod(angle_deg, period_deg))


def harmonic_orientation_deg(phase_deg, order):
    """Convert a raw complex harmonic phase into one coefficient period."""
    return wrap_period_deg(phase_deg / order, 360.0 / order)


def interpreted_harmonic_phase_deg(raw_complex_phase_deg, order, convention):
    """Apply the fitted Uno-to-simulation phase convention."""
    period_deg = 360.0 / order
    return wrap_period_deg(
        convention["sign"] * raw_complex_phase_deg / order + convention["offset_deg"],
        period_deg,
    )


def add_complex_columns(row, name, value):
    """Add raw and convention-corrected complex coefficient columns to a row."""
    value = _to_numpy_value(value)
    raw_phase_deg = float(np.angle(value, deg=True))
    row[name] = str(value)
    row[name + "_real"] = float(np.real(value))
    row[name + "_imag"] = float(np.imag(value))
    row[name + "_abs"] = float(np.abs(value))
    row[name + "_complex_phase_deg"] = raw_phase_deg
    row[name + "_raw_phase_deg"] = raw_phase_deg

    order = UNO_HARMONIC_ORDERS.get(name)
    if order is None:
        row[name + "_phase_deg"] = raw_phase_deg
        return

    period_deg = 360.0 / order
    convention = PRIMARY_PHASE_CONVENTIONS[name]
    interpreted_phase_deg = interpreted_harmonic_phase_deg(
        raw_phase_deg,
        order,
        convention,
    )

    row[name + "_orientation_deg"] = harmonic_orientation_deg(raw_phase_deg, order)
    row[name + "_orientation_negated_deg"] = harmonic_orientation_deg(
        -raw_phase_deg,
        order,
    )
    row[name + "_orientation_sign_flipped_deg"] = harmonic_orientation_deg(
        raw_phase_deg + 180.0,
        order,
    )
    row[name + "_orientation_negated_sign_flipped_deg"] = harmonic_orientation_deg(
        -(raw_phase_deg + 180.0),
        order,
    )
    row[name + "_orientation_period_deg"] = period_deg
    row[name + "_phase_convention"] = "sign={sign:g}, offset={offset:g} deg".format(
        sign=convention["sign"],
        offset=convention["offset_deg"],
    )
    row[name + "_phase_sign"] = float(convention["sign"])
    row[name + "_phase_offset_deg"] = float(convention["offset_deg"])
    row[name + "_phase_period_deg"] = period_deg
    row[name + "_phase_deg"] = interpreted_phase_deg
    row[name + "_interpreted_phase_deg"] = interpreted_phase_deg
    row[name + "_interpreted_phase_period_deg"] = period_deg
