"""
Example: Annual sunlight analysis for FACE NORMALS using Ladybug
Analyzes sunlight at face centroids with backface culling enabled
"""
import numpy as np
import time

from src.raycast import direct_sunlight
from src.utils import sun_vectors, color, mesh_utils, visualization


if __name__ == "__main__":
    print("="*70)
    print("ANNUAL SUNLIGHT ANALYSIS - Face-Based with Backface Culling")
    print("="*70)
    
    # Load mesh with normals
    print("\nLoading mesh...")
    vertices, face_counts, face_indices, face_normals, vertex_normals = mesh_utils.load_mesh_with_normals(
        "data/meshes/bunny_closed.obj"
    )
    faces = face_indices.reshape(-1, 3)
    print(f"  Vertices: {len(vertices)}, Faces: {len(faces)}")
    
    # Compute face centroids
    print("\nComputing face centroids...")
    face_centroids = mesh_utils.compute_face_centroids(vertices, faces)
    print(f"  Face centroids: {len(face_centroids)}")
    
    # ==========================================================================
    # ANNUAL ANALYSIS: Full Year - New York City
    # ==========================================================================
    print("\n" + "="*70)
    print("Annual Sunlight Analysis (Full Year 2024) - New York City")
    print("Using FACE CENTROIDS and FACE NORMALS with BACKFACE CULLING")
    print("="*70)
    
    try:
        light_directions_annual, timestamps_annual = sun_vectors.get_yearly_sun_vectors(
            latitude=40.7128,   # NYC latitude
            longitude=-74.0060, # NYC longitude
            year=2024,          # Year to analyze
            time_step_hours=1,  # Hourly samples
            timezone='America/New_York'
        )
        
        print(f"\nSun positions calculated for entire year:")
        print(f"  Total sun vectors: {len(light_directions_annual)}")
        print(f"  Time range: {timestamps_annual[0]} to {timestamps_annual[-1]}")
        print(f"  First sun time: {timestamps_annual[0]}")
        print(f"  Last sun time: {timestamps_annual[-1]}")
        
        # Run raycasting analysis on FACES with BACKFACE CULLING
        print("\nRunning annual sunlight analysis on FACES...")
        print("  (This may take a while due to large number of sun positions)")
        print("  BACKFACE CULLING: Enabled (faces not facing the sun are automatically occluded)")
        start_time = time.time()
        
        hitcounts_annual = direct_sunlight.raycast_directional_batch(
            vertices, faces, 
            face_centroids,  # Use face centroids as ray origins
            face_normals,    # Use face normals for backface culling
            light_directions_annual, 
            offset_distance=0.001,
            use_backface_culling=True  # Enable backface culling for faces
        )
        
        elapsed = time.time() - start_time
        print(f"  Completed in {elapsed:.2f} seconds ({elapsed*1000:.1f} ms)")
        print(f"  Processed {len(face_centroids) * len(light_directions_annual):,} rays")
        print(f"  Performance: {(len(face_centroids) * len(light_directions_annual)) / elapsed:,.0f} rays/second")
        
        # Calculate annual sunlight hours
        time_step_hours_annual = 1.0  # 1 hour per sample
        max_possible_hours_annual = len(light_directions_annual) * time_step_hours_annual
        
        # Convert hit counts to sunlight hours
        clear_counts_annual = len(light_directions_annual) - hitcounts_annual
        sunlight_hours_annual = clear_counts_annual * time_step_hours_annual
        
        print(f"\nAnnual sunlight hours per FACE:")
        print(f"  Max possible: {max_possible_hours_annual:.0f} hours")
        print(f"  Mean: {np.mean(sunlight_hours_annual):.1f} hours ({np.mean(sunlight_hours_annual)/24:.1f} days)")
        print(f"  Min: {np.min(sunlight_hours_annual):.1f} hours ({np.min(sunlight_hours_annual)/24:.1f} days)")
        print(f"  Max: {np.max(sunlight_hours_annual):.1f} hours ({np.max(sunlight_hours_annual)/24:.1f} days)")
        print(f"  Std Dev: {np.std(sunlight_hours_annual):.1f} hours")
        
        # ==========================================================================
        # VISUALIZATION
        # ==========================================================================
        print("\n" + "="*70)
        print("VISUALIZATION")
        print("="*70)
        
        # Create gradient colors for annual data
        gradient_colors_annual = [
            (0, 0, 128),      # Dark blue (no sunlight)
            (0, 128, 255),    # Light blue (little sunlight)
            (0, 255, 128),    # Cyan-green (some sunlight)
            (255, 255, 0),    # Yellow (good sunlight)
            (255, 128, 0),    # Orange (lots of sunlight)
            (255, 0, 0)       # Red (maximum sunlight)
        ]
        gradient_positions_annual = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        # Invert hit counts for coloring (fewer hits = more sunlight)
        max_hits_annual = np.max(hitcounts_annual)
        inverted_hitcounts_annual = max_hits_annual - hitcounts_annual
        
        face_colors = color.create_gradient(
            values=inverted_hitcounts_annual,
            colors=gradient_colors_annual,
            positions=gradient_positions_annual
        )
        
        # Create scene with mesh and wireframe using FACE COLORS
        print("\nCreating 3D scene for annual data (face-based)...")
        scene_annual = visualization.create_scene_with_mesh_and_wireframe(
            vertices=vertices,
            faces=faces,
            face_colors=face_colors,  # Use face colors instead of vertex colors
            wireframe_color=(80, 80, 80, 255),
            line_width=1.0
        )
        
        # Add sun direction vectors (subsample for visualization clarity)
        print("Adding sun direction vectors (subsampled for clarity)...")
        # Subsample sun vectors to avoid overcrowding visualization
        subsample_step = max(1, len(light_directions_annual) // 50)  # Show ~50 vectors
        light_directions_subsampled = light_directions_annual[::subsample_step]
        
        visualization.add_sun_vectors_to_scene(
            scene=scene_annual,
            light_directions=light_directions_subsampled,
            distance=0.5,
            length=0.1,
            color=(255, 200, 0, 255)
        )
        
        print(f"  Added {len(light_directions_subsampled)} sun direction vectors (from {len(light_directions_annual)} total)")
        
        print("\nShowing 3D visualization for ANNUAL FACE analysis...")
        print("Color legend:")
        print("  Red/Orange/Yellow: Maximum annual sunlight exposure")
        print("  Blue/Dark Blue: Minimum annual sunlight exposure (occluded or back-facing)")
        print("  Yellow arrows: Subsampled sun direction vectors across the year")
        print("\nNote: Backface culling means faces not facing the sun are automatically occluded")
        
        scene_annual.show(line_settings={'line_width': 1.0})
        
    except ImportError as e:
        print(f"\nERROR: {e}")
        print("\nTo use Ladybug for annual analysis, install it:")
        print("  pip install ladybug-core")
        print("  pip install tzdata")
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)

