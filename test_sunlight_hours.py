import numpy as np
import trimesh
import time

from src.IO import load_obj
from src.raycast import direct_sunlight
from src.helper import color

if __name__ == "__main__":
    print("WarpRays - GPU Raycast Performance Test")
    print("="*60)
    
    # Load mesh
    vertices, face_counts, face_indices = load_obj.load_vertices_faces_indices("data/meshes/bunny_closed.obj")
    print(f"Loaded mesh: {len(vertices)} vertices, {len(face_indices)//3} faces")
    
    # Reshape face_indices into faces array (n, 3) for triangular meshes
    faces = face_indices.reshape(-1, 3)
    
    # Compute face normals using cross product
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    # Edge vectors
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    # Face normals via cross product
    face_normals = np.cross(edge1, edge2)
    # Normalize
    face_normals = face_normals / (np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-10)
    
    # Compute vertex normals using trimesh.geometry.mean_vertex_normals
    vertex_normals = trimesh.geometry.mean_vertex_normals(
        vertex_count=len(vertices),
        faces=faces,
        face_normals=face_normals
    )
    
    # Create sun direction vectors in an arc above the model
    # First create vectors TOWARD sun positions, then negate for light direction FROM sun
    # NOTE: The batch kernel scales efficiently - try 500 or 1000+ directions!
    sun_angles = np.linspace(-60, 60, 50)  # -60 to 60 degrees arc with 50 samples
    arc_rotation_deg = 45  # Rotate the arc by 45 degrees along its diameter (around X-axis)
    arc_rotation_rad = np.radians(arc_rotation_deg)
    
    sun_positions = []  # Unit vectors pointing TOWARD sun positions (for visualization)
    light_directions = []  # Light direction vectors FROM sun (for raycasting)
    
    for angle_deg in sun_angles:
        angle_rad = np.radians(angle_deg)
        # Sun position: x-component varies with angle, y is up, z is forward
        sun_pos = np.array([
            np.sin(angle_rad),  # x: varies from -1 to 1
            np.cos(angle_rad),  # y: always positive (above)
            0.0                  # z: no forward/backward component
        ])
        
        # Rotate around X-axis to tilt the arc along its diameter
        # Rotation matrix around X-axis: [[1, 0, 0], [0, cos, -sin], [0, sin, cos]]
        cos_rot = np.cos(arc_rotation_rad)
        sin_rot = np.sin(arc_rotation_rad)
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, cos_rot, -sin_rot],
            [0, sin_rot, cos_rot]
        ])
        
        sun_pos = rotation_matrix @ sun_pos
        sun_pos = sun_pos / np.linalg.norm(sun_pos)  # Normalize
        
        sun_positions.append(sun_pos)
        light_directions.append(-sun_pos)  # Light comes FROM sun (negative of position)
    
    print("\nSun configuration:")
    print(f"  {len(light_directions)} light directions from {sun_angles[0]:.0f}° to {sun_angles[-1]:.0f}°")
    print(f"  Arc rotation: {arc_rotation_deg}° around X-axis")
    
    # Prepare for raycasting
    # Pass light directions FROM sun, kernel will reverse them to cast rays TOWARD sun
    # (to check if anything blocks the path to the sun)
    start_positions = vertices.copy()
    start_normals = vertex_normals.copy()
    
    # Convert list to numpy array
    light_directions_array = np.array(light_directions, dtype=np.float32)
    
    ### Performance Comparison ###
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    # Method 1: Individual kernel launches (old way)
    print("\nMethod 1: Individual launches (one per direction)")
    start_time = time.time()
    hitcounts_individual = np.zeros(len(vertices), dtype=np.int32)
    for i, light_dir in enumerate(light_directions):
        light_dirs = np.tile(light_dir, (len(vertices), 1)).astype(np.float32)
        hitcounts = direct_sunlight.raycast_directional(
            vertices, faces, start_positions, start_normals, 
            light_dirs, offset_distance=0.001
        )
        hitcounts_individual += hitcounts
    time_individual = time.time() - start_time
    
    print(f"  Time: {time_individual*1000:.2f} ms")
    print(f"  Rays: {len(vertices) * len(light_directions):,}")
    
    # Method 2: Batch kernel (optimized)
    print("\nMethod 2: Batch kernel (optimized)")
    start_time = time.time()
    total_hitcounts = direct_sunlight.raycast_directional_batch(
        vertices, faces, start_positions, start_normals, 
        light_directions_array, offset_distance=0.001
    )
    time_batch = time.time() - start_time
    
    print(f"  Time: {time_batch*1000:.2f} ms")
    print(f"  Rays: {len(vertices) * len(light_directions):,}")
    
    # Verify results match
    results_match = np.array_equal(hitcounts_individual, total_hitcounts)
    print(f"\n  Results match: {results_match}")
    print(f"  Speedup: {time_individual/time_batch:.1f}×")
    print("="*60)
    
    print("\nHit count statistics:")
    print(f"  Range: {np.min(total_hitcounts)} to {np.max(total_hitcounts)}")
    print(f"  Mean: {np.mean(total_hitcounts):.2f}")
    
    # Create vertex colors with gradient based on accumulated hit counts
    # Now backfacing vertices are counted as occluded (hit=1), so:
    # Low hits = mostly clear view to sun = bright (green)
    # High hits = mostly occluded from sun = dark (red)
    
    # Define gradient: Green -> Yellow -> Orange -> Red (auto-adjusts to min/max)
    gradient_colors = [
        (50, 255, 50),    # Bright green (low occlusion)
        (255, 255, 0),    # Yellow (slight occlusion)
        (255, 128, 0),    # Orange (medium occlusion)
        (255, 0, 0)       # Red (high occlusion)
    ]
    gradient_positions = [0.0, 0.33, 0.66, 1.0]
    
    vertex_colors = color.create_gradient(
        values=total_hitcounts,
        colors=gradient_colors,
        positions=gradient_positions
    )
    
    ###### Visualization ######
    print("\n" + "="*60)
    print("VISUALIZATION (using batch result)")
    print("="*60)
    
    # Create trimesh mesh object
    mesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh_obj.visual.vertex_colors = vertex_colors
    
    # Create scene
    scene = trimesh.Scene(mesh_obj)
    
    # Add wireframe with thin lines
    edges = mesh_obj.edges_unique
    edge_vertices = vertices[edges]
    wireframe_path = trimesh.load_path(edge_vertices)
    # Set color on entities instead of directly on the path
    for entity in wireframe_path.entities:
        entity.color = [80, 80, 80, 255]  # Gray color
    scene.add_geometry(wireframe_path)
    
    # Add sun direction vectors visualization
    print("Creating 3D scene with mesh and sun vectors...")
    
    sun_vector_distance = 0.5
    sun_vector_length = 0.1
    
    for sun_pos in sun_positions:
        # Position at distance 0.5 from origin (where the sun is)
        start_pos = sun_pos * sun_vector_distance
        # Arrow points toward the scene to show light direction
        end_pos = start_pos - sun_pos * sun_vector_length
        
        # Create individual path for this vector
        segment = np.array([[start_pos, end_pos]])
        sun_vector_path = trimesh.load_path(segment)
        
        # Color in bright yellow/orange
        for entity in sun_vector_path.entities:
            entity.color = [255, 200, 0, 255]
        
        scene.add_geometry(sun_vector_path)
    
    # Show the scene
    print(f"\nDisplaying mesh with {len(sun_positions)} sun direction vectors...")
    print("Color gradient: Green (clear) → Yellow → Orange → Red (occluded)")
    scene.show(line_settings={'line_width': 1.0})