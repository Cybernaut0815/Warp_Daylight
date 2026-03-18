import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import time

from src.raycast import direct_sunlight
from src.utils import color, mesh_utils, visualization

if __name__ == "__main__":
    print("WarpRays - GPU Raycast Performance Test")
    print("="*60)
    
    # Load mesh with normals
    vertices, face_counts, face_indices, face_normals, vertex_normals = mesh_utils.load_mesh_with_normals(
        str(_PROJECT_ROOT / "data/meshes/bunny_closed.obj")
    )
    faces = face_indices.reshape(-1, 3)
    print(f"Loaded mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    # Create sun direction vectors in an arc above the model
    # NOTE: The batch kernel scales efficiently - try 500 or 1000+ directions!
    num_sun_samples = 50
    arc_rotation_deg = 45
    
    light_directions_array = visualization.create_manual_arc_sun_vectors(
        num_samples=num_sun_samples,
        arc_angle_range=(-60, 60),
        arc_rotation_deg=arc_rotation_deg
    )
    
    print("\nSun configuration:")
    print(f"  {len(light_directions_array)} light directions from -60° to 60°")
    print(f"  Arc rotation: {arc_rotation_deg}° around X-axis")
    
    # Prepare for raycasting
    start_positions = vertices.copy()
    start_normals = vertex_normals.copy()
    
    ### Performance Comparison ###
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    # Method 1: Individual kernel launches (old way)
    print("\nMethod 1: Individual launches (one per direction)")
    start_time = time.time()
    hitcounts_individual = np.zeros(len(vertices), dtype=np.int32)
    for light_dir in light_directions_array:
        light_dirs = np.tile(light_dir, (len(vertices), 1)).astype(np.float32)
        hitcounts = direct_sunlight.raycast_directional(
            vertices, faces, start_positions, start_normals, 
            light_dirs, offset_distance=0.001
        )
        hitcounts_individual += hitcounts
    time_individual = time.time() - start_time
    
    print(f"  Time: {time_individual*1000:.2f} ms")
    print(f"  Rays: {len(vertices) * len(light_directions_array):,}")
    
    # Method 2: Batch kernel (optimized)
    print("\nMethod 2: Batch kernel (optimized)")
    start_time = time.time()
    total_hitcounts = direct_sunlight.raycast_directional_batch(
        vertices, faces, start_positions, start_normals, 
        light_directions_array, offset_distance=0.001,
        chunk_size=1000
    )
    time_batch = time.time() - start_time
    
    print(f"  Time: {time_batch*1000:.2f} ms")
    print(f"  Rays: {len(vertices) * len(light_directions_array):,}")
    
    # Verify results match
    results_match = np.array_equal(hitcounts_individual, total_hitcounts)
    print(f"\n  Results match: {results_match}")
    print(f"  Speedup: {time_individual/time_batch:.1f}×")
    print("="*60)
    
    print("\nHit count statistics:")
    print(f"  Range: {np.min(total_hitcounts)} to {np.max(total_hitcounts)}")
    print(f"  Mean: {np.mean(total_hitcounts):.2f}")
    
    # Create gradient colors based on sunlight hours
    gradient_colors = [
        (0, 0, 128),      # Dark blue (no sunlight)
        (0, 128, 255),    # Light blue (little sunlight)
        (0, 255, 128),    # Cyan-green (some sunlight)
        (255, 255, 0),    # Yellow (good sunlight)
        (255, 128, 0),    # Orange (lots of sunlight)
        (255, 0, 0)       # Red (maximum sunlight)
    ]
    gradient_positions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    # Invert hit counts: fewer hits = more sunlight = higher value = warmer colors
    max_hits = np.max(total_hitcounts)
    inverted_hitcounts = max_hits - total_hitcounts
    
    vertex_colors = color.create_gradient(
        values=inverted_hitcounts,
        colors=gradient_colors,
        positions=gradient_positions
    )
    
    ###### Visualization ######
    print("\n" + "="*60)
    print("VISUALIZATION (using batch result)")
    print("="*60)
    
    # Create scene with mesh and wireframe
    print("Creating 3D scene with mesh and sun vectors...")
    scene = visualization.create_scene_with_mesh_and_wireframe(
        vertices=vertices,
        faces=faces,
        vertex_colors=vertex_colors,
        wireframe_color=(80, 80, 80, 255),
        line_width=1.0
    )
    
    # Add sun direction vectors
    visualization.add_sun_vectors_to_scene(
        scene=scene,
        light_directions=light_directions_array,
        distance=0.5,
        length=0.1,
        color=(255, 200, 0, 255)
    )
    
    # Show the scene
    print(f"\nDisplaying mesh with {len(light_directions_array)} sun direction vectors...")
    print("Color gradient: Blue (occluded) → Cyan → Yellow → Orange → Red (max sunlight)")
    scene.show(line_settings={'line_width': 1.0})
