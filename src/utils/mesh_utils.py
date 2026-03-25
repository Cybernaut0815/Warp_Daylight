"""
Utility functions for mesh operations (loading, normals, centroids).
"""
import numpy as np
import trimesh

from src.IO import load_obj


def load_mesh_with_normals(
    filepath: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a mesh from an OBJ file and compute face / vertex normals.

    Parameters
    ----------
    filepath:
        Path to the ``.obj`` file.

    Returns
    -------
    vertices:
        (N, 3) vertex positions.
    face_counts:
        (M,) number of vertices per face.
    face_indices:
        Flattened face indices.
    face_normals:
        (M, 3) face normal vectors.
    vertex_normals:
        (N, 3) vertex normal vectors.
    """
    vertices, face_counts, face_indices = load_obj.load_vertices_faces_indices(filepath)

    faces = face_indices.reshape(-1, 3)
    face_normals = compute_face_normals(vertices, faces)
    vertex_normals = trimesh.geometry.mean_vertex_normals(
        vertex_count=len(vertices),
        faces=faces,
        face_normals=face_normals,
    )

    return vertices, face_counts, face_indices, face_normals, vertex_normals


def compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute unit face normals via the cross product of edge vectors.

    Parameters
    ----------
    vertices:
        (N, 3) vertex positions.
    faces:
        (M, 3) triangle face indices.

    Returns
    -------
    np.ndarray
        (M, 3) normalised face normals.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    face_normals = np.cross(v1 - v0, v2 - v0)
    face_normals = face_normals / (
        np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-10
    )

    return face_normals


def compute_face_centroids(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute the centroid (average of three vertices) of each triangle.

    Parameters
    ----------
    vertices:
        (N, 3) vertex positions.
    faces:
        (M, 3) triangle face indices.

    Returns
    -------
    np.ndarray
        (M, 3) face centroids.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    return (v0 + v1 + v2) / 3.0
