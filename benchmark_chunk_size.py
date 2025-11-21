"""
Benchmark script to find optimal chunk size for raycast_directional_batch()
"""
import numpy as np
import time

from src.raycast import direct_sunlight
from src.utils import mesh_utils, visualization


def benchmark_chunk_sizes(vertices, faces, vertex_normals, light_directions, chunk_sizes):
    """
    Benchmark different chunk sizes to find optimal performance.
    
    Args:
        vertices: Mesh vertices
        faces: Mesh faces
        vertex_normals: Vertex normals
        light_directions: Array of light directions to test with
        chunk_sizes: List of chunk sizes to test (use 0 for no chunking)
    
    Returns:
        Dictionary with results for each chunk size
    """
    start_positions = vertices.copy()
    start_normals = vertex_normals.copy()
    
    results = {}
    
    for chunk_size in chunk_sizes:
        # Warm-up run
        _ = direct_sunlight.raycast_directional_batch(
            vertices, faces, start_positions, start_normals, 
            light_directions, offset_distance=0.001, chunk_size=chunk_size
        )
        
        # Timed runs (average of 3)
        times = []
        for _ in range(3):
            start_time = time.time()
            hitcounts = direct_sunlight.raycast_directional_batch(
                vertices, faces, start_positions, start_normals, 
                light_directions, offset_distance=0.001, chunk_size=chunk_size
            )
            elapsed = time.time() - start_time
            times.append(elapsed)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        results[chunk_size] = {
            'avg_time_ms': avg_time * 1000,
            'std_time_ms': std_time * 1000,
            'times': times,
            'hitcounts': hitcounts
        }
        
        chunk_desc = f"No chunking" if chunk_size == 0 else f"Chunk size {chunk_size}"
        print(f"{chunk_desc:20s}: {avg_time*1000:7.2f} ms ± {std_time*1000:5.2f} ms")
    
    return results


if __name__ == "__main__":
    print("Chunk Size Optimization Benchmark")
    print("="*60)
    
    # Load mesh with normals
    vertices, face_counts, face_indices, face_normals, vertex_normals = mesh_utils.load_mesh_with_normals(
        "data/meshes/bunny_closed.obj"
    )
    faces = face_indices.reshape(-1, 3)
    print(f"Mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    # Create test light directions (2000 directions for comprehensive test)
    num_test_directions = 2000
    light_directions = visualization.create_manual_arc_sun_vectors(
        num_samples=num_test_directions,
        arc_angle_range=(-60, 60),
        arc_rotation_deg=45
    )
    print(f"Test directions: {len(light_directions)}")
    print()
    
    # Test different chunk sizes
    print("Testing different chunk sizes:")
    print("-"*60)
    
    chunk_sizes_to_test = [
        0,      # No chunking (all at once)
        250,    # Very small chunks
        500,    # Small chunks
        1000,   # Default
        2000,   # Large chunks
        5000,   # Very large chunks
    ]
    
    results = benchmark_chunk_sizes(
        vertices, faces, vertex_normals, 
        light_directions, chunk_sizes_to_test
    )
    
    # Find best
    print()
    print("="*60)
    best_chunk = min(results.items(), key=lambda x: x[1]['avg_time_ms'])
    chunk_desc = "No chunking" if best_chunk[0] == 0 else f"Chunk size {best_chunk[0]}"
    print(f"BEST: {chunk_desc} - {best_chunk[1]['avg_time_ms']:.2f} ms")
    
    # Verify all results are identical
    print()
    print("Verification:")
    reference = results[chunk_sizes_to_test[0]]['hitcounts']
    all_match = all(np.array_equal(r['hitcounts'], reference) for r in results.values())
    print(f"  All chunk sizes produce identical results: {all_match}")
    
    print()
    print("="*60)
    print("RECOMMENDATION:")
    print(f"  For your GPU and mesh size, use chunk_size={best_chunk[0]}")
    print(f"  Expected performance: ~{best_chunk[1]['avg_time_ms']:.1f} ms for {len(light_directions)} directions")
    print("="*60)

