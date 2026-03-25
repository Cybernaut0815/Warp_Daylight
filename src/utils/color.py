import numpy as np
from typing import List, Tuple


def create_gradient(
    values: np.ndarray,
    colors: List[Tuple[int, int, int]],
    positions: List[float] = None
) -> np.ndarray:
    """
    Create a color gradient based on input values.
    
    Args:
        values: Array of values to map to colors (N,)
        colors: List of RGB color tuples, e.g. [(0, 255, 0), (255, 255, 0), (255, 0, 0)]
        positions: List of positions (0-1) for each color. If None, colors are evenly spaced.
                   Must have same length as colors.
    
    Returns:
        colors_array: Array of RGB colors (N, 3) as uint8
    
    Example:
        # Green -> Yellow -> Red gradient
        values = np.array([0, 0.25, 0.5, 0.75, 1.0])
        colors = [(0, 255, 0), (255, 255, 0), (255, 0, 0)]
        positions = [0.0, 0.5, 1.0]
        result = create_gradient(values, colors, positions)
    """
    # Validate inputs
    if len(colors) < 2:
        raise ValueError("Must provide at least 2 colors for gradient")
    
    # If positions not provided, distribute colors evenly
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
            (values - min_val) / (max_val - min_val), 0.0, 1.0
        )
    else:
        normalized = np.zeros(len(values), dtype=np.float64)

    segment_indices = np.searchsorted(positions, normalized, side='right') - 1
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

