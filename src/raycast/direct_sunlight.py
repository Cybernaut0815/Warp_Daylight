import warp as wp
import numpy as np

### -------------------------------------------------------------- ###
### All AI generated code below ###
### Needs testing and optimization ###
### -------------------------------------------------------------- ###

@wp.kernel
def raycast_kernel(
    mesh_id: wp.uint64,
    start_positions: wp.array(dtype=wp.vec3),
    start_normals: wp.array(dtype=wp.vec3),
    sun_directions: wp.array(dtype=wp.vec3),
    offset_distance: float,
    hit_counts: wp.array(dtype=int)
):
    """
    Raycast kernel that performs ray-mesh intersection with backface culling.
    
    Args:
        mesh_id: Warp mesh ID
        start_positions: Starting positions for rays (N, 3)
        start_normals: Normal vectors at starting positions (N, 3)
        sun_directions: Light direction vectors FROM the sun (N, 3)
        offset_distance: Distance to offset ray origin along surface normal
        hit_counts: Output array to store hit counts for each ray (N,)
    """
    tid = wp.tid()
    
    # Get ray origin, normal, and sun direction for this thread
    ray_origin = start_positions[tid]
    surface_normal = wp.normalize(start_normals[tid])
    light_dir = wp.normalize(sun_directions[tid])
    
    # Reverse light direction to cast rays FROM surface TOWARD sun
    ray_dir = -light_dir
    
    # Backface culling BEFORE raycasting for performance
    # Only cast ray if surface is facing towards the ray direction
    # If dot product is positive, surface normal and ray direction point in same hemisphere
    dot_product = wp.dot(surface_normal, ray_dir)
    
    if dot_product > 0.0:
        # Surface is front-facing, proceed with raycast
        # Offset ray origin along normal to avoid self-intersection
        ray_origin_offset = ray_origin + surface_normal * offset_distance
        
        # Perform ray-mesh intersection
        query = wp.mesh_query_ray(mesh_id, ray_origin_offset, ray_dir, 1.0e10)
        
        if query.result:
            hit_counts[tid] = 1  # Occluded
        else:
            hit_counts[tid] = 0  # Clear view to sun
    else:
        # Back-facing surface, don't cast ray (treat as occluded)
        hit_counts[tid] = 1


@wp.kernel
def raycast_batch_kernel(
    mesh_id: wp.uint64,
    start_positions: wp.array(dtype=wp.vec3),
    start_normals: wp.array(dtype=wp.vec3),
    light_directions: wp.array(dtype=wp.vec3),
    num_directions: int,
    offset_distance: float,
    use_backface_culling: int,
    hit_counts: wp.array(dtype=int)
):
    """
    Optimized batch raycast kernel that processes all sun directions in one launch.
    Each thread processes one vertex/face for all sun directions.
    
    Args:
        mesh_id: Warp mesh ID
        start_positions: Starting positions for rays (N, 3)
        start_normals: Normal vectors at starting positions (N, 3)
        light_directions: Light direction vectors FROM the sun (M, 3) for M directions
        num_directions: Number of sun directions
        offset_distance: Distance to offset ray origin along surface normal
        use_backface_culling: 1 to enable backface culling, 0 to disable
        hit_counts: Output array to store accumulated hit counts per vertex (N,)
    """
    vertex_id = wp.tid()
    
    # Get vertex position and normal
    ray_origin = start_positions[vertex_id]
    surface_normal = wp.normalize(start_normals[vertex_id])
    ray_origin_offset = ray_origin + surface_normal * offset_distance
    
    # Accumulate hits across all sun directions
    total_hits = int(0)
    
    # Process all sun directions for this vertex
    for dir_id in range(num_directions):
        light_dir = wp.normalize(light_directions[dir_id])
        ray_dir = -light_dir  # Reverse to cast toward sun
        
        # Optional backface culling
        if use_backface_culling == 1:
            dot_product = wp.dot(surface_normal, ray_dir)
            
            if dot_product > 0.0:
                # Surface is front-facing, cast ray
                query = wp.mesh_query_ray(mesh_id, ray_origin_offset, ray_dir, 1.0e10)
                
                if query.result:
                    total_hits += 1  # Occluded
                # else: clear (contributes 0)
            else:
                # Back-facing surface (treat as occluded)
                total_hits += 1
        else:
            # No backface culling - always cast ray
            query = wp.mesh_query_ray(mesh_id, ray_origin_offset, ray_dir, 1.0e10)
            
            if query.result:
                total_hits += 1  # Occluded
    
    hit_counts[vertex_id] = total_hits


def raycast_directional(
    vertices: np.ndarray,
    face_indices: np.ndarray,
    start_positions: np.ndarray,
    start_normals: np.ndarray,
    sun_directions: np.ndarray,
    offset_distance: float = 0.001
) -> np.ndarray:
    """
    Perform directional raycasting from given positions with backface culling.
    Accepts light direction FROM the sun, internally reverses to cast rays TOWARD the sun.
    
    Args:
        vertices: Mesh vertices array (V, 3)
        face_indices: Mesh face indices array (F, 3) for triangular meshes
        start_positions: Starting positions for rays (N, 3)
        start_normals: Normal vectors at starting positions (N, 3)
        sun_directions: Light direction vectors FROM the sun (N, 3)
        offset_distance: Distance to offset ray origin along surface normal (default: 0.001)
    
    Returns:
        hit_counts: Array of hit counts for each ray (N,)
    """
    # Ensure inputs are contiguous float32 arrays
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    face_indices = np.ascontiguousarray(face_indices, dtype=np.int32)
    start_positions = np.ascontiguousarray(start_positions, dtype=np.float32)
    start_normals = np.ascontiguousarray(start_normals, dtype=np.float32)
    sun_directions = np.ascontiguousarray(sun_directions, dtype=np.float32)
    
    # Create warp arrays
    vertices_wp = wp.array(vertices, dtype=wp.vec3)
    indices_wp = wp.array(face_indices.flatten(), dtype=int)
    start_positions_wp = wp.array(start_positions, dtype=wp.vec3)
    start_normals_wp = wp.array(start_normals, dtype=wp.vec3)
    sun_directions_wp = wp.array(sun_directions, dtype=wp.vec3)
    
    # Create warp mesh
    mesh = wp.Mesh(
        points=vertices_wp,
        indices=indices_wp
    )
    
    # Prepare output array
    num_rays = start_positions.shape[0]
    hit_counts = wp.zeros(num_rays, dtype=int)
    
    # Launch kernel
    wp.launch(
        kernel=raycast_kernel,
        dim=num_rays,
        inputs=[mesh.id, start_positions_wp, start_normals_wp, sun_directions_wp, offset_distance, hit_counts]
    )
    
    # Synchronize and return results
    wp.synchronize()
    
    return hit_counts.numpy()


def raycast_directional_batch(
    vertices: np.ndarray,
    face_indices: np.ndarray,
    start_positions: np.ndarray,
    start_normals: np.ndarray,
    light_directions: np.ndarray,
    offset_distance: float = 0.001,
    chunk_size: int = 1000,
    use_backface_culling: bool = False
) -> np.ndarray:
    """
    OPTIMIZED: Perform directional raycasting for MULTIPLE light directions in a single kernel launch.
    This is much faster than calling raycast_directional() multiple times.
    
    For very large numbers of directions (>1000), automatically chunks into batches.
    
    Args:
        vertices: Mesh vertices array (V, 3)
        face_indices: Mesh face indices array (F, 3) for triangular meshes
        start_positions: Starting positions for rays (N, 3)
        start_normals: Normal vectors at starting positions (N, 3)
        light_directions: Light direction vectors FROM the sun (M, 3) for M directions
        offset_distance: Distance to offset ray origin along surface normal (default: 0.001)
        chunk_size: Max directions per kernel launch (default: 1000, 0 = no chunking)
                   
                   Chunk size considerations:
                   - Small (250-500): Lower memory, better for limited VRAM
                   - Medium (500-1500): Balanced, good default
                   - Large (2000+): Better for high-end GPUs, more memory
                   - 0 (no chunking): Process all directions at once (best for <2000 directions)
                   
                   To find optimal size for your GPU, run: benchmark_chunk_size.py
        use_backface_culling: Enable backface culling (default: False)
                             Set to True for face-based analysis
    
    Returns:
        hit_counts: Array of accumulated hit counts per vertex/face (N,) - range [0, M]
    """
    # Ensure inputs are contiguous float32 arrays
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    face_indices = np.ascontiguousarray(face_indices, dtype=np.int32)
    start_positions = np.ascontiguousarray(start_positions, dtype=np.float32)
    start_normals = np.ascontiguousarray(start_normals, dtype=np.float32)
    light_directions = np.ascontiguousarray(light_directions, dtype=np.float32)
    
    num_vertices = start_positions.shape[0]
    num_directions = light_directions.shape[0]
    
    # Create warp arrays (reused across chunks)
    vertices_wp = wp.array(vertices, dtype=wp.vec3)
    indices_wp = wp.array(face_indices.flatten(), dtype=int)
    start_positions_wp = wp.array(start_positions, dtype=wp.vec3)
    start_normals_wp = wp.array(start_normals, dtype=wp.vec3)
    
    # Create warp mesh (once)
    mesh = wp.Mesh(
        points=vertices_wp,
        indices=indices_wp
    )
    
    # Prepare output array
    total_hit_counts = np.zeros(num_vertices, dtype=np.int32)
    
    # Convert backface culling boolean to int for kernel
    backface_culling_flag = 1 if use_backface_culling else 0
    
    # Determine if chunking is needed
    use_chunking = chunk_size > 0 and num_directions > chunk_size
    
    if use_chunking:
        # Process in chunks for very large numbers of directions
        num_chunks = (num_directions + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, num_directions)
            chunk_directions = light_directions[start_idx:end_idx]
            
            # Create warp array for this chunk
            light_directions_wp = wp.array(chunk_directions, dtype=wp.vec3)
            hit_counts = wp.zeros(num_vertices, dtype=int)
            
            # Launch kernel for this chunk
            wp.launch(
                kernel=raycast_batch_kernel,
                dim=num_vertices,
                inputs=[mesh.id, start_positions_wp, start_normals_wp, light_directions_wp, 
                       len(chunk_directions), offset_distance, backface_culling_flag, hit_counts]
            )
            
            wp.synchronize()
            total_hit_counts += hit_counts.numpy()
    else:
        # Process all directions in one kernel launch (optimal for <= chunk_size directions)
        light_directions_wp = wp.array(light_directions, dtype=wp.vec3)
        hit_counts = wp.zeros(num_vertices, dtype=int)
        
        # Launch kernel - one thread per vertex, each processes all directions
        wp.launch(
            kernel=raycast_batch_kernel,
            dim=num_vertices,
            inputs=[mesh.id, start_positions_wp, start_normals_wp, light_directions_wp, 
                   num_directions, offset_distance, backface_culling_flag, hit_counts]
        )
        
        wp.synchronize()
        total_hit_counts = hit_counts.numpy()
    
    return total_hit_counts