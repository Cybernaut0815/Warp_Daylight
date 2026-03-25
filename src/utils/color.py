"""
Vectorised colour gradient mapping.

Maps an array of scalar values to RGB colours by interpolating between
a user-defined (or default) colour ramp.
"""
import numpy as np


def create_gradient(
    values: np.ndarray,
    colors: list[tuple[int, int, int]],
    positions: list[float] | None = None,
) -> np.ndarray:
    """Map scalar *values* to RGB colours via piecewise-linear interpolation.

    Parameters
    ----------
    values:
        (N,) array of scalars to map.
    colors:
        At least two ``(R, G, B)`` tuples (0-255) defining the gradient
        stops.
    positions:
        Normalised (0-1) positions for each colour stop.  When ``None``,
        colours are distributed evenly.  Must have the same length as
        *colors*.

    Returns
    -------
    np.ndarray
        (N, 3) uint8 RGB array.

    Example
    -------
    >>> values = np.array([0, 0.25, 0.5, 0.75, 1.0])
    >>> colors = [(0, 255, 0), (255, 255, 0), (255, 0, 0)]
    >>> result = create_gradient(values, colors, [0.0, 0.5, 1.0])
    """
    if len(colors) < 2:
        raise ValueError("Must provide at least 2 colors for gradient")

    if positions is None:
        positions = np.linspace(0.0, 1.0, len(colors))
    else:
        if len(positions) != len(colors):
            raise ValueError("positions must have same length as colors")
        positions = np.array(positions)

    colors_arr = np.asarray(colors, dtype=np.float64)

    min_val = np.min(values)
    max_val = np.max(values)

    if max_val > min_val:
        normalized = np.clip(
            (values - min_val) / (max_val - min_val), 0.0, 1.0,
        )
    else:
        normalized = np.zeros(len(values), dtype=np.float64)

    segment_indices = np.searchsorted(positions, normalized, side="right") - 1
    segment_indices = np.clip(segment_indices, 0, len(positions) - 2)

    seg_starts = positions[segment_indices]
    seg_ends = positions[segment_indices + 1]
    seg_widths = seg_ends - seg_starts
    t = np.where(seg_widths > 0, (normalized - seg_starts) / seg_widths, 0.0)

    color_starts = colors_arr[segment_indices]
    color_ends = colors_arr[segment_indices + 1]
    t_expanded = t[:, np.newaxis]
    result = color_starts * (1.0 - t_expanded) + color_ends * t_expanded

    return np.clip(result, 0, 255).astype(np.uint8)
