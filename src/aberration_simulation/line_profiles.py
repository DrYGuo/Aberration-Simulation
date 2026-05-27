"""Line-profile extraction for probe image stacks."""

import numpy as np

from .backend import asnumpy, map_coordinates, xp


def extract_line_profiles_from_stack(image_stack, num_lines=37, radius=None, center=None):
    """Extract radial line profiles from a stack of images.

    Parameters
    ----------
    image_stack:
        Array with shape `(height, width, num_images)`.
    num_lines:
        Number of angles including the duplicated 180-degree endpoint. The
        returned profile count is `num_lines - 1`.
    radius:
        Half-length of each line in pixels. Defaults to the largest centered
        radius that fits inside the image.
    center:
        Optional `(x, y)` center. Defaults to the image center.

    Returns
    -------
    line_profiles, line_coordinates:
        Profiles have shape `(num_lines - 1, 2 * radius + 1, num_images)`.
    """
    stack = xp.asarray(image_stack)
    if stack.ndim != 3:
        raise ValueError("image_stack must have shape (height, width, num_images)")

    height, width, num_images = stack.shape
    if radius is None:
        radius = int(min(height, width) // 2 - 1)
    if center is None:
        center = (width / 2.0, height / 2.0)

    x_center, y_center = center
    angles = xp.linspace(0, 180, num_lines)[:-1]
    theta = xp.radians(angles).reshape(-1, 1)
    offsets = xp.linspace(-radius, radius, 2 * radius + 1).reshape(1, -1)

    x_base = x_center + xp.cos(theta) * offsets
    y_base = y_center + xp.sin(theta) * offsets

    profiles = []
    for image_index in range(num_images):
        coords = xp.vstack((y_base.reshape(1, -1), x_base.reshape(1, -1)))
        sampled = map_coordinates(
            stack[:, :, image_index],
            coords,
            order=1,
            mode="nearest",
        )
        profiles.append(sampled.reshape(num_lines - 1, 2 * radius + 1))

    line_profiles = xp.stack(profiles, axis=2)
    coordinates = {
        "angles_deg": asnumpy(angles),
        "x": asnumpy(x_base),
        "y": asnumpy(y_base),
        "radius": radius,
        "center": np.array(center, dtype=float),
    }
    return line_profiles, coordinates


def choose_nonzero_parameter_indices(parameters, limit=4):
    """Pick representative simulations with diverse nonzero aberrations."""
    groups = [
        [
            index for index, params in enumerate(parameters)
            if abs(params["C3"]) > 0 and abs(params["A1_amp"]) == 0
        ],
        [
            index for index, params in enumerate(parameters)
            if abs(params["A1_amp"]) > 0 and abs(params["C3"]) == 0
        ],
        [
            index for index, params in enumerate(parameters)
            if any(abs(params.get(key, 0)) > 0 for key in ("A2_amp", "B2_amp", "A3_amp", "C1", "C1_offset"))
        ],
        [
            index for index, params in enumerate(parameters)
            if sum(
                abs(params.get(key, 0)) > 0
                for key in ("A1_amp", "A2_amp", "B2_amp", "A3_amp", "C1", "C3", "C1_offset")
            ) > 1
        ],
    ]

    selected = []
    offset = 0
    while len(selected) < limit and any(offset < len(group) for group in groups):
        for group in groups:
            if offset < len(group) and group[offset] not in selected:
                selected.append(group[offset])
            if len(selected) >= limit:
                return selected
        offset += 1

    for index, params in enumerate(parameters):
        if index in selected:
            continue
        if any(abs(params.get(key, 0)) > 0 for key in ("A1_amp", "A2_amp", "B2_amp", "A3_amp", "C1", "C3", "C1_offset")):
            selected.append(index)
        if len(selected) >= limit:
            break

    return selected
