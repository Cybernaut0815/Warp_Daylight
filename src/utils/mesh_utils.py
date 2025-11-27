"""
Utility functions for mesh operations
"""
import numpy as np
import trimesh
from typing import Tuple

from src.IO import load_obj


def load_mesh_with_normals(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a mesh from an OBJ file and compute face and vertex normals.
    
    Args:
        filepath: Path to the OBJ file
    
    Returns:
        vertices: Vertex positions (N, 3)
        face_counts: Number of vertices per face (M,)
        face_indices: Flattened face indices
        face_normals: Face normal vectors (M, 3)
        vertex_normals: Vertex normal vectors (N, 3)
    """
    # Load mesh
    vertices, face_counts, face_indices = load_obj.load_vertices_faces_indices(filepath)
    
    # Reshape face_indices into faces array (n, 3) for triangular meshes
    faces = face_indices.reshape(-1, 3)
    
    # Compute face normals
    face_normals = compute_face_normals(vertices, faces)
    
    # Compute vertex normals
    vertex_normals = trimesh.geometry.mean_vertex_normals(
        vertex_count=len(vertices),
        faces=faces,
        face_normals=face_normals
    )
    
    return vertices, face_counts, face_indices, face_normals, vertex_normals


def compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute face normals using cross product.
    
    Args:
        vertices: Vertex positions (N, 3)
        faces: Face indices (M, 3)
    
    Returns:
        face_normals: Normalized face normal vectors (M, 3)
    """
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
    
    return face_normals


def compute_face_centroids(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute face centroids (center points of triangles).
    
    Args:
        vertices: Vertex positions (N, 3)
        faces: Face indices (M, 3)
    
    Returns:
        centroids: Face centroid positions (M, 3)
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    # Centroid is the average of the three vertices
    centroids = (v0 + v1 + v2) / 3.0
    
    return centroids
