"""Uno et al. 2005 phase-convention helpers.

The raw complex values from formulas (38)-(44) are harmonic coefficients.
Their displayed physical phases are fitted to the simulation convention used by
the probe generator:

    phase = sign * raw_complex_phase / harmonic_order + offset

The result is wrapped to the coefficient period, 360 / harmonic_order degrees.
"""

import numpy as np


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
