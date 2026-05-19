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
        center = ((width - 1) / 2.0, (height - 1) / 2.0)

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
    """Pick representative simulations with at least one nonzero aberration."""
    scored = []
    for index, params in enumerate(parameters):
        nonzero = sum(
            abs(params[key]) > 0
            for key in ("A1_amp", "A2_amp", "A3_amp", "C1", "C3", "C1_offset")
        )
        if nonzero:
            scored.append((nonzero, index))
    scored.sort(reverse=True)
    return [index for _, index in scored[:limit]]
