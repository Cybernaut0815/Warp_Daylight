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
    counts = np.asarray(face_vertex_counts, dtype=np.int32)

    # Fast path: all faces are triangles — just reshape
    if np.all(counts == 3):
        return face_vertex_indices.reshape(-1, 3).astype(np.int32)

    # Fast path: all quads
    if np.all(counts == 4):
        quads = face_vertex_indices.reshape(-1, 4)
        tri_a = quads[:, [0, 1, 2]]
        tri_b = quads[:, [0, 2, 3]]
        return np.concatenate([tri_a, tri_b], axis=0).astype(np.int32)

    # Mixed tri/quad path — vectorised gather
    offsets = np.empty(len(counts) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])

    is_tri = counts == 3
    is_quad = counts == 4

    tri_starts = offsets[:-1][is_tri]
    tri_indices = np.column_stack([
        face_vertex_indices[tri_starts],
        face_vertex_indices[tri_starts + 1],
        face_vertex_indices[tri_starts + 2],
    ])

    quad_starts = offsets[:-1][is_quad]
    i0 = face_vertex_indices[quad_starts]
    i1 = face_vertex_indices[quad_starts + 1]
    i2 = face_vertex_indices[quad_starts + 2]
    i3 = face_vertex_indices[quad_starts + 3]
    quad_tri_a = np.column_stack([i0, i1, i2])
    quad_tri_b = np.column_stack([i0, i2, i3])

    return np.concatenate(
        [tri_indices, quad_tri_a, quad_tri_b], axis=0
    ).astype(np.int32)


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


def process_mesh_from_numpy(arrays: dict[str, np.ndarray]) -> ProcessedMesh:
    """
    Build a ``ProcessedMesh`` directly from numpy arrays (e.g. from an NPZ
    upload) without going through Pydantic validation.

    Expected *arrays* keys:
      - ``points``               (V, 3)
      - ``face_vertex_counts``   (F,)
      - ``face_vertex_indices``  (sum(counts),)
      - ``normals``              (V, 3)  — optional
    """
    vertices = np.ascontiguousarray(arrays["points"], dtype=np.float32)
    face_vertex_counts = np.asarray(arrays["face_vertex_counts"], dtype=np.int32)
    face_vertex_indices = np.asarray(arrays["face_vertex_indices"], dtype=np.int32)

    faces = _triangulate_faces(face_vertex_counts, face_vertex_indices)

    face_norms = compute_face_normals(vertices, faces)
    face_cents = compute_face_centroids(vertices, faces)

    if "normals" in arrays:
        vertex_norms = np.ascontiguousarray(arrays["normals"], dtype=np.float32)
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
