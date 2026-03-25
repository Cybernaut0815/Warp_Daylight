"""
Service for converting USD-like mesh input into the numpy arrays expected by
the Warp raycast layer.

Responsibilities:
  - Convert lists / dicts to contiguous float32 / int32 numpy arrays
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
    """Intermediate representation after converting raw mesh data to numpy.

    All arrays are contiguous and in the dtypes expected by Warp:
      - ``vertices``       (V, 3) float32
      - ``faces``          (F, 3) int32  — always triangulated
      - ``vertex_normals`` (V, 3) float32
      - ``face_normals``   (F, 3) float32
      - ``face_centroids`` (F, 3) float32
    """
    vertices: np.ndarray
    faces: np.ndarray
    vertex_normals: np.ndarray
    face_normals: np.ndarray
    face_centroids: np.ndarray


def _triangulate_faces(
    face_vertex_counts: np.ndarray,
    face_vertex_indices: np.ndarray,
) -> np.ndarray:
    """Fan-triangulate a mix of triangles and quads.

    Parameters
    ----------
    face_vertex_counts:
        (F,) int32 — number of vertices per face (3 or 4).
    face_vertex_indices:
        (sum(counts),) int32 — flattened vertex indices.

    Returns
    -------
    faces:
        (F_tri, 3) int32 array of triangle vertex indices.
    """
    counts = np.asarray(face_vertex_counts, dtype=np.int32)

    if np.all(counts == 3):
        return face_vertex_indices.reshape(-1, 3).astype(np.int32)

    if np.all(counts == 4):
        quads = face_vertex_indices.reshape(-1, 4)
        tri_a = quads[:, [0, 1, 2]]
        tri_b = quads[:, [0, 2, 3]]
        return np.concatenate([tri_a, tri_b], axis=0).astype(np.int32)

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


def _build_processed_mesh(
    vertices: np.ndarray,
    face_vertex_counts: np.ndarray,
    face_vertex_indices: np.ndarray,
    vertex_normals: np.ndarray | None,
) -> ProcessedMesh:
    """Shared builder used by both ``process_mesh`` and ``process_mesh_from_numpy``.

    Parameters
    ----------
    vertices:
        (V, 3) float-like vertex positions.
    face_vertex_counts:
        (F,) int-like per-face vertex counts.
    face_vertex_indices:
        (sum(counts),) int-like flattened face indices.
    vertex_normals:
        (V, 3) float-like vertex normals, or ``None`` to auto-compute via
        trimesh area-weighted averaging.
    """
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    face_vertex_counts = np.asarray(face_vertex_counts, dtype=np.int32)
    face_vertex_indices = np.asarray(face_vertex_indices, dtype=np.int32)

    faces = _triangulate_faces(face_vertex_counts, face_vertex_indices)
    face_norms = compute_face_normals(vertices, faces)
    face_cents = compute_face_centroids(vertices, faces)

    if vertex_normals is not None:
        v_norms = np.ascontiguousarray(vertex_normals, dtype=np.float32)
    else:
        v_norms = np.ascontiguousarray(
            trimesh.geometry.mean_vertex_normals(
                vertex_count=len(vertices),
                faces=faces,
                face_normals=face_norms,
            ),
            dtype=np.float32,
        )

    return ProcessedMesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=v_norms,
        face_normals=face_norms,
        face_centroids=face_cents,
    )


def process_mesh(mesh_data: MeshData) -> ProcessedMesh:
    """Convert a ``MeshData`` Pydantic model into numpy arrays ready for Warp."""
    return _build_processed_mesh(
        vertices=np.asarray(mesh_data.points),
        face_vertex_counts=np.asarray(mesh_data.face_vertex_counts),
        face_vertex_indices=np.asarray(mesh_data.face_vertex_indices),
        vertex_normals=np.asarray(mesh_data.normals) if mesh_data.normals is not None else None,
    )


def process_mesh_from_numpy(arrays: dict[str, np.ndarray]) -> ProcessedMesh:
    """Build a ``ProcessedMesh`` from a dict of numpy arrays (e.g. NPZ upload).

    Expected keys: ``points``, ``face_vertex_counts``,
    ``face_vertex_indices``, and optionally ``normals``.
    """
    return _build_processed_mesh(
        vertices=arrays["points"],
        face_vertex_counts=arrays["face_vertex_counts"],
        face_vertex_indices=arrays["face_vertex_indices"],
        vertex_normals=arrays.get("normals"),
    )


def points_and_normals_to_numpy(
    points: list[list[float]],
    normals: list[list[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw point/normal lists to contiguous float32 arrays."""
    pts = np.ascontiguousarray(points, dtype=np.float32)
    nrm = np.ascontiguousarray(normals, dtype=np.float32)
    return pts, nrm
