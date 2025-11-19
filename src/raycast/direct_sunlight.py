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
    ray_directions: wp.array(dtype=wp.vec3),
    hit_counts: wp.array(dtype=int)
):
    """
    Raycast kernel that performs ray-mesh intersection with backface culling.
    
    Args:
        mesh_id: Warp mesh ID
        start_positions: Starting positions for rays (N, 3)
        start_normals: Normal vectors at starting positions (N, 3)
        ray_directions: Direction vectors for rays (N, 3)
        hit_counts: Output array to store hit counts for each ray (N,)
    """
    tid = wp.tid()
    
    # Get ray origin, normal, and direction for this thread
    ray_origin = start_positions[tid]
    surface_normal = wp.normalize(start_normals[tid])
    ray_dir = wp.normalize(ray_directions[tid])
    
    # Backface culling BEFORE raycasting for performance
    # Only cast ray if surface is facing towards the ray direction
    # If dot product is positive, surface normal and ray direction point in same hemisphere
    dot_product = wp.dot(surface_normal, ray_dir)
    
    if dot_product > 0.0:
        # Surface is front-facing, proceed with raycast
        # Offset ray origin slightly along normal to avoid self-intersection
        ray_origin_offset = ray_origin + surface_normal * 0.001
        
        # Perform ray-mesh intersection
        query = wp.mesh_query_ray(mesh_id, ray_origin_offset, ray_dir, 1.0e10)
        
        if query.result:
            hit_counts[tid] = 1
        else:
            hit_counts[tid] = 0
    else:
        # Back-facing surface, don't cast ray
        hit_counts[tid] = 0


def raycast_directional(
    vertices: np.ndarray,
    face_indices: np.ndarray,
    start_positions: np.ndarray,
    start_normals: np.ndarray,
    ray_directions: np.ndarray
) -> np.ndarray:
    """
    Perform directional raycasting from given positions with backface culling.
    
    Args:
        vertices: Mesh vertices array (V, 3)
        face_indices: Mesh face indices array (F, 3) for triangular meshes
        start_positions: Starting positions for rays (N, 3)
        start_normals: Normal vectors at starting positions (N, 3)
        ray_directions: Direction vectors for rays (N, 3)
    
    Returns:
        hit_counts: Array of hit counts for each ray (N,)
    """
    # Ensure inputs are contiguous float32 arrays
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    face_indices = np.ascontiguousarray(face_indices, dtype=np.int32)
    start_positions = np.ascontiguousarray(start_positions, dtype=np.float32)
    start_normals = np.ascontiguousarray(start_normals, dtype=np.float32)
    ray_directions = np.ascontiguousarray(ray_directions, dtype=np.float32)
    
    # Create warp arrays
    vertices_wp = wp.array(vertices, dtype=wp.vec3)
    indices_wp = wp.array(face_indices.flatten(), dtype=int)
    start_positions_wp = wp.array(start_positions, dtype=wp.vec3)
    start_normals_wp = wp.array(start_normals, dtype=wp.vec3)
    ray_directions_wp = wp.array(ray_directions, dtype=wp.vec3)
    
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
        inputs=[mesh.id, start_positions_wp, start_normals_wp, ray_directions_wp, hit_counts]
    )
    
    # Synchronize and return results
    wp.synchronize()
    
    return hit_counts.numpy()