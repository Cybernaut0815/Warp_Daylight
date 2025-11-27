# Sunlight Analysis Comparison

This document explains the differences between vertex-based and face-based sunlight analysis.

## Overview

The WarpRays library now supports two approaches for sunlight analysis:

1. **Vertex-Based Analysis** (example_annual_sunlight.py)
2. **Face-Based Analysis** (example_annual_sunlight_faces.py)

## Key Differences

### Vertex-Based Analysis

**Script:** `example_annual_sunlight.py`

**Features:**
- Analyzes sunlight at **vertex positions**
- Uses **vertex normals** (averaged from adjacent face normals)
- **Backface culling: DISABLED** by default
- Casts rays from all vertices regardless of orientation
- Results in smooth color gradients across the mesh
- Better for smooth surfaces and organic shapes

**Use Cases:**
- Visualizing smooth sunlight transitions
- Analyzing curved surfaces like terrain
- When you want continuous color gradients
- Interior surfaces that might receive indirect light

**Technical Details:**
```python
hitcounts = direct_sunlight.raycast_directional_batch(
    vertices, faces, 
    vertices,        # Ray origins at vertices
    vertex_normals,  # Averaged normals
    light_directions, 
    use_backface_culling=False  # Disabled
)
```

### Face-Based Analysis

**Script:** `example_annual_sunlight_faces.py`

**Features:**
- Analyzes sunlight at **face centroids** (triangle centers)
- Uses **face normals** (perpendicular to each triangle)
- **Backface culling: ENABLED** by default
- Automatically treats back-facing surfaces as occluded
- Results in per-face coloring (flat shading)
- More physically accurate for architectural analysis

**Use Cases:**
- Architectural sunlight studies
- Solar panel placement optimization
- Building facade analysis
- When surface orientation matters
- Accurate shadow analysis for flat surfaces

**Technical Details:**
```python
face_centroids = mesh_utils.compute_face_centroids(vertices, faces)

hitcounts = direct_sunlight.raycast_directional_batch(
    vertices, faces,
    face_centroids,  # Ray origins at face centers
    face_normals,    # Perpendicular to faces
    light_directions,
    use_backface_culling=True  # Enabled
)
```

## Backface Culling

### What is Backface Culling?

Backface culling optimizes raycasting by automatically treating surfaces facing away from the sun as occluded.

**How it works:**
- Computes dot product between surface normal and ray direction
- If dot product ≤ 0, surface is back-facing (won't receive direct sunlight)
- Skips expensive ray-mesh intersection test
- Treats back-facing surfaces as occluded

**Benefits:**
- More physically accurate (surfaces facing away don't receive direct light)
- Better performance (fewer ray-mesh intersection tests)
- Essential for architectural analysis

### When to Use Backface Culling

**Enable for:**
- ✅ Face-based analysis (architectural studies)
- ✅ Building facades and solar panels
- ✅ When surface orientation is important
- ✅ Outdoor architectural visualization

**Disable for:**
- ❌ Vertex-based smooth visualization
- ❌ Organic shapes and terrain
- ❌ When you want all surfaces analyzed
- ❌ Interior surfaces with indirect lighting

## Visualization Differences

### Vertex Colors (Smooth Shading)
- Colors interpolated between vertices
- Smooth gradients across faces
- Used by vertex-based analysis
- Better for organic shapes

### Face Colors (Flat Shading)
- Each face has uniform color
- Sharp transitions at edges
- Used by face-based analysis
- Better for architectural visualization

## Performance Comparison

Both approaches have similar performance for raycasting, but differ in:

- **Memory:** Face-based may use less memory (typically fewer faces than vertices)
- **Accuracy:** Face-based is more physically accurate with backface culling
- **Visualization:** Vertex-based produces smoother visualizations

## Quick Reference

| Feature | Vertex-Based | Face-Based |
|---------|-------------|------------|
| Analysis Points | Vertices | Face Centroids |
| Normals | Vertex Normals (averaged) | Face Normals (perpendicular) |
| Backface Culling | Disabled (default) | Enabled (recommended) |
| Visualization | Smooth gradients | Flat per-face |
| Best For | Organic shapes, terrain | Buildings, solar panels |
| Color Output | Vertex colors | Face colors |

## Example Usage

### Running Vertex-Based Analysis
```bash
python example_annual_sunlight.py
```

### Running Face-Based Analysis
```bash
python example_annual_sunlight_faces.py
```

## API Reference

### Optional Backface Culling Parameter

The `raycast_directional_batch()` function now accepts an optional parameter:

```python
def raycast_directional_batch(
    vertices: np.ndarray,
    face_indices: np.ndarray,
    start_positions: np.ndarray,
    start_normals: np.ndarray,
    light_directions: np.ndarray,
    offset_distance: float = 0.001,
    chunk_size: int = 1000,
    use_backface_culling: bool = False  # NEW PARAMETER
) -> np.ndarray:
```

**Default:** `False` (disabled) for backward compatibility

**Recommendation:**
- Set to `False` for vertex-based analysis
- Set to `True` for face-based analysis

## Mesh Utilities

New helper function for face-based analysis:

```python
from src.utils import mesh_utils

# Compute face centroids
face_centroids = mesh_utils.compute_face_centroids(vertices, faces)
```

## Visualization Utilities

Updated function now supports both vertex and face colors:

```python
from src.utils import visualization

# Vertex-based (smooth)
scene = visualization.create_scene_with_mesh_and_wireframe(
    vertices=vertices,
    faces=faces,
    vertex_colors=colors,  # Per-vertex colors
)

# Face-based (flat)
scene = visualization.create_scene_with_mesh_and_wireframe(
    vertices=vertices,
    faces=faces,
    face_colors=colors,  # Per-face colors
)
```

**Note:** If both `vertex_colors` and `face_colors` are provided, `face_colors` takes precedence.

