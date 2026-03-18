"""
Example: Using Ladybug to calculate realistic sun positions for sunlight analysis
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import time

from src.raycast import direct_sunlight
from src.utils import sun_vectors, color, mesh_utils, visualization


if __name__ == "__main__":
    print("="*70)
    print("SINGLE DAY SUNLIGHT ANALYSIS using Ladybug Sun Positions")
    print("="*70)
    
    # Load mesh with normals
    print("\nLoading mesh...")
    vertices, face_counts, face_indices, face_normals, vertex_normals = mesh_utils.load_mesh_with_normals(
        str(_PROJECT_ROOT / "data/meshes/bunny_closed.obj")
    )
    faces = face_indices.reshape(-1, 3)
    print(f"  Vertices: {len(vertices)}, Faces: {len(faces)}")
    
    # ==========================================================================
    # Summer Solstice Day (June 21) - New York City
    # ==========================================================================
    print("\n" + "="*70)
    print("Summer Solstice (June 21) - New York City")
    print("="*70)
    
    try:
        light_directions_summer, timestamps_summer = sun_vectors.get_sun_vectors_for_typical_day(
            latitude=40.7128,   # NYC latitude
            longitude=-74.0060, # NYC longitude
            month=6,            # June
            day=21,             # Summer solstice
            time_step_minutes=30,  # Every 30 minutes
            timezone='America/New_York'
        )
        
        print(f"\nSun positions calculated:")
        print(f"  Total sun vectors: {len(light_directions_summer)}")
        print(f"  Time range: {timestamps_summer[0]} to {timestamps_summer[-1]}")
        print(f"  First sun time: {timestamps_summer[0]}")
        print(f"  Last sun time: {timestamps_summer[-1]}")
        
        # Run raycasting analysis
        print("\nRunning sunlight analysis...")
        start_time = time.time()
        
        hitcounts_summer = direct_sunlight.raycast_directional_batch(
            vertices, faces, vertices, vertex_normals,
            light_directions_summer, offset_distance=0.001
        )
        
        elapsed = time.time() - start_time
        print(f"  Completed in {elapsed*1000:.1f} ms")
        print(f"  Processed {len(vertices) * len(light_directions_summer):,} rays")
        
        # Calculate sunlight hours (assuming each timestamp represents equal time)
        time_step_hours = 0.5  # 30 minutes
        max_possible_hours = len(light_directions_summer) * time_step_hours
        
        # Convert hit counts to sunlight hours
        # hitcount = times occluded, so clear = total - hitcount
        clear_counts = len(light_directions_summer) - hitcounts_summer
        sunlight_hours = clear_counts * time_step_hours
        
        print(f"\nSunlight hours per vertex:")
        print(f"  Max possible: {max_possible_hours:.1f} hours")
        print(f"  Mean: {np.mean(sunlight_hours):.2f} hours")
        print(f"  Min: {np.min(sunlight_hours):.2f} hours")
        print(f"  Max: {np.max(sunlight_hours):.2f} hours")
        
    except ImportError as e:
        print(f"\nERROR: {e}")
        print("\nTo use Ladybug, install it:")
        print("  pip install ladybug-core")
        print("  pip install tzdata")
        print("\nFalling back to manual sun positions...")
        
        # Fallback: Use manual arc
        light_directions_summer = visualization.create_manual_arc_sun_vectors(
            num_samples=50,
            arc_angle_range=(-60, 60),
            arc_rotation_deg=45
        )
        
        hitcounts_summer = direct_sunlight.raycast_directional_batch(
            vertices, faces, vertices, vertex_normals,
            light_directions_summer, offset_distance=0.001
        )
    
    # ==========================================================================
    # VISUALIZATION
    # ==========================================================================
    print("\n" + "="*70)
    print("VISUALIZATION")
    print("="*70)
    
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
    max_hits = np.max(hitcounts_summer)
    inverted_hitcounts = max_hits - hitcounts_summer
    
    vertex_colors = color.create_gradient(
        values=inverted_hitcounts,
        colors=gradient_colors,
        positions=gradient_positions
    )
    
    # Create scene with mesh and wireframe
    print("\nCreating 3D scene...")
    scene = visualization.create_scene_with_mesh_and_wireframe(
        vertices=vertices,
        faces=faces,
        vertex_colors=vertex_colors,
        wireframe_color=(80, 80, 80, 255),
        line_width=1.0
    )
    
    # Add sun direction vectors visualization
    print("Adding sun direction vectors...")
    visualization.add_sun_vectors_to_scene(
        scene=scene,
        light_directions=light_directions_summer,
        distance=0.5,
        length=0.1,
        color=(255, 200, 0, 255)
    )
    
    print(f"  Added {len(light_directions_summer)} sun direction vectors")
    
    print("\nShowing 3D visualization...")
    print("Color legend:")
    print("  Red/Orange/Yellow: Maximum sunlight exposure")
    print("  Blue/Dark Blue: Minimum sunlight exposure (occluded)")
    print("  Yellow arrows: Sun direction vectors")
    
    scene.show(line_settings={'line_width': 1.0})
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
