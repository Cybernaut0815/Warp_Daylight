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
    
    # Normalize values to 0-1 range
    min_val = np.min(values)
    max_val = np.max(values)
    
    if max_val > min_val:
        normalized_values = (values - min_val) / (max_val - min_val)
    else:
        # All values are the same
        normalized_values = np.zeros_like(values, dtype=np.float32)
    
    # Create output array
    output_colors = np.zeros((len(values), 3), dtype=np.uint8)
    
    # For each value, find the appropriate color segment and interpolate
    for i, val in enumerate(normalized_values):
        # Clamp value to [0, 1]
        val = np.clip(val, 0.0, 1.0)
        
        # Find which segment this value falls into
        segment_idx = 0
        for j in range(len(positions) - 1):
            if val >= positions[j] and val <= positions[j + 1]:
                segment_idx = j
                break
        
        # Get colors for this segment
        color_start = np.array(colors[segment_idx])
        color_end = np.array(colors[segment_idx + 1])
        
        # Calculate interpolation factor within this segment
        segment_start = positions[segment_idx]
        segment_end = positions[segment_idx + 1]
        
        if segment_end > segment_start:
            t = (val - segment_start) / (segment_end - segment_start)
        else:
            t = 0.0
        
        # Linear interpolation between colors
        color = color_start * (1 - t) + color_end * t
        output_colors[i] = color.astype(np.uint8)
    
    return output_colors

