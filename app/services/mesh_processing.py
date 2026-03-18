"""
Service for converting USD-like mesh input (Pydantic models) into the numpy
arrays expected by the Warp raycast layer.

Responsibilities:
  - Convert lists to contiguous float32 / int32 numpy arrays
  - Triangulate quads so all faces are (N, 3) for Warp
  - Compute vertex normals, face normals, and face centroids when not provided
"""
import numpy as np
import trimesh
from dataclasses import dataclass

from app.models.mesh import MeshData
from src.utils.mesh_utils import compute_face_normals, compute_face_centroids


@dataclass
class ProcessedMesh:
    """Intermediate representation after converting MeshData to numpy."""
    vertices: np.ndarray          # (V, 3) float32
    faces: np.ndarray             # (F, 3) int32 -- always triangulated
    vertex_normals: np.ndarray    # (V, 3) float32
    face_normals: np.ndarray      # (F, 3) float32
    face_centroids: np.ndarray    # (F, 3) float32


def _triangulate_faces(
    face_vertex_counts: np.ndarray,
    face_vertex_indices: np.ndarray,
) -> np.ndarray:
    """
    Fan-triangulate quads into triangles.

    Returns:
        faces: (F_tri, 3) int32 array of triangle vertex indices.
    """
    triangles: list[list[int]] = []
    offset = 0
    for count in face_vertex_counts:
        if count == 3:
            triangles.append([
                face_vertex_indices[offset],
                face_vertex_indices[offset + 1],
                face_vertex_indices[offset + 2],
            ])
        elif count == 4:
            i0 = face_vertex_indices[offset]
            i1 = face_vertex_indices[offset + 1]
            i2 = face_vertex_indices[offset + 2]
            i3 = face_vertex_indices[offset + 3]
            triangles.append([i0, i1, i2])
            triangles.append([i0, i2, i3])
        offset += count

    return np.array(triangles, dtype=np.int32)


def process_mesh(mesh_data: MeshData) -> ProcessedMesh:
    """
    Convert a ``MeshData`` Pydantic model into numpy arrays ready for Warp.

    Steps:
      1. Convert points → float32 array
      2. Triangulate quads (if any) via fan triangulation
      3. Compute face normals and centroids
      4. Compute vertex normals (or use provided ones)
    """
    vertices = np.ascontiguousarray(mesh_data.points, dtype=np.float32)
    face_vertex_counts = np.array(mesh_data.face_vertex_counts, dtype=np.int32)
    face_vertex_indices = np.array(mesh_data.face_vertex_indices, dtype=np.int32)

    faces = _triangulate_faces(face_vertex_counts, face_vertex_indices)

    face_norms = compute_face_normals(vertices, faces)
    face_cents = compute_face_centroids(vertices, faces)

    if mesh_data.normals is not None:
        vertex_norms = np.ascontiguousarray(mesh_data.normals, dtype=np.float32)
    else:
        vertex_norms = trimesh.geometry.mean_vertex_normals(
            vertex_count=len(vertices),
            faces=faces,
            face_normals=face_norms,
        )
        vertex_norms = np.ascontiguousarray(vertex_norms, dtype=np.float32)

    return ProcessedMesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=vertex_norms,
        face_normals=face_norms,
        face_centroids=face_cents,
    )


def points_and_normals_to_numpy(
    points: list[list[float]],
    normals: list[list[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw point/normal lists to contiguous float32 arrays."""
    pts = np.ascontiguousarray(points, dtype=np.float32)
    nrm = np.ascontiguousarray(normals, dtype=np.float32)
    return pts, nrm
