"""
Orchestration service that ties together sun-vector generation, mesh
preparation, GPU raycasting, and optional colour mapping.
"""
import logging
import time

import numpy as np
from datetime import datetime

from app.models.mesh import MeshData
from app.models.requests import SunConfig, AnalysisOptions
from app.services.mesh_processing import (
    ProcessedMesh,
    process_mesh,
    points_and_normals_to_numpy,
)
from src.raycast.direct_sunlight import raycast_directional_batch
from src.utils.sun_vectors import (
    get_yearly_sun_vectors,
    get_sun_vectors_from_ladybug,
)
from src.utils.color import create_gradient

logger = logging.getLogger(__name__)

DEFAULT_GRADIENT_COLORS = [
    (0, 0, 128),
    (0, 128, 255),
    (0, 255, 128),
    (255, 255, 0),
    (255, 128, 0),
    (255, 0, 0),
]
DEFAULT_GRADIENT_POSITIONS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _resolve_sun_vectors(sun: SunConfig) -> tuple[np.ndarray, float]:
    """Return ``(light_directions, time_step_hours)`` from a SunConfig.

    .. note::
        # TODO(roadmap): Add daylight factor from EPW files — EPW-based
        #   irradiance weighting would hook into this function so that each
        #   direction carries a weight (W/m²) in addition to the binary
        #   hit/miss from raycasting.
    """
    if sun.start_datetime and sun.end_datetime:
        start_dt = datetime.fromisoformat(sun.start_datetime)
        end_dt = datetime.fromisoformat(sun.end_datetime)
        step_minutes = sun.time_step_minutes or (sun.time_step_hours * 60)
        light_dirs, _ = get_sun_vectors_from_ladybug(
            latitude=sun.latitude,
            longitude=sun.longitude,
            start_datetime=start_dt,
            end_datetime=end_dt,
            time_step_minutes=step_minutes,
            timezone=sun.timezone,
            coordinate_system=sun.coordinate_system,
        )
        return light_dirs, step_minutes / 60.0

    light_dirs, _ = get_yearly_sun_vectors(
        latitude=sun.latitude,
        longitude=sun.longitude,
        year=sun.year,
        time_step_hours=sun.time_step_hours,
        timezone=sun.timezone,
        coordinate_system=sun.coordinate_system,
    )
    return light_dirs, float(sun.time_step_hours)


def _build_colors(
    hit_counts: np.ndarray,
    options: AnalysisOptions,
) -> np.ndarray | None:
    """Map hit counts to RGB colours via a gradient (higher sunlight = warmer).

    Returns an (N, 3) uint8 numpy array, or ``None`` when colours are disabled.
    """
    if not options.return_colors:
        return None

    max_hits = np.max(hit_counts) if len(hit_counts) > 0 else 0
    inverted = max_hits - hit_counts

    if options.color_gradient and len(options.color_gradient) >= 2:
        gradient_colors = [tuple(c) for c in options.color_gradient]
        gradient_positions = None
    else:
        gradient_colors = DEFAULT_GRADIENT_COLORS
        gradient_positions = DEFAULT_GRADIENT_POSITIONS

    return create_gradient(
        values=inverted.astype(np.float32),
        colors=gradient_colors,
        positions=gradient_positions,
    )


def _build_response(
    hit_counts: np.ndarray,
    num_sun_positions: int,
    time_step_hours: float,
    computation_time: float,
    num_elements: int,
    mesh_vertex_count: int,
    mesh_face_count: int,
    options: AnalysisOptions,
) -> dict:
    """Build the analysis result as a plain dict with numpy array values.

    Callers serialise via ``orjson`` with ``OPT_SERIALIZE_NUMPY``.
    """
    max_possible_hours = num_sun_positions * time_step_hours
    clear_counts = num_sun_positions - hit_counts
    sunlight_hours = clear_counts * time_step_hours

    return {
        "sunlight_hours": sunlight_hours,
        "hit_counts": hit_counts,
        "total_sun_positions": num_sun_positions,
        "max_possible_hours": max_possible_hours,
        "colors": _build_colors(hit_counts, options),
        "metadata": {
            "computation_time_seconds": round(computation_time, 4),
            "num_elements": num_elements,
            "num_sun_positions": num_sun_positions,
            "mesh_vertex_count": mesh_vertex_count,
            "mesh_face_count": mesh_face_count,
        },
    }


def _run_analysis(
    blocker: ProcessedMesh,
    start_positions: np.ndarray,
    start_normals: np.ndarray,
    sun: SunConfig,
    options: AnalysisOptions,
    *,
    label: str = "analysis",
) -> dict:
    """Core analysis pipeline shared by every public entry-point.

    Parameters
    ----------
    blocker:
        Mesh used for occlusion (vertices + triangulated faces).
    start_positions:
        (N, 3) positions to evaluate sunlight at.
    start_normals:
        (N, 3) surface normals at those positions.
    sun:
        Sun/location configuration.
    options:
        Raycast tuning knobs and output options.
    label:
        Human-readable tag used in log messages.
    """
    t0 = time.perf_counter()
    light_dirs, step_h = _resolve_sun_vectors(sun)
    t1 = time.perf_counter()

    hit_counts = raycast_directional_batch(
        vertices=blocker.vertices,
        face_indices=blocker.faces,
        start_positions=start_positions,
        start_normals=start_normals,
        light_directions=light_dirs,
        offset_distance=options.offset_distance,
        chunk_size=options.chunk_size,
        use_backface_culling=options.use_backface_culling,
    )
    t2 = time.perf_counter()

    result = _build_response(
        hit_counts=hit_counts,
        num_sun_positions=len(light_dirs),
        time_step_hours=step_h,
        computation_time=t2 - t1,
        num_elements=len(start_positions),
        mesh_vertex_count=len(blocker.vertices),
        mesh_face_count=len(blocker.faces),
        options=options,
    )
    t3 = time.perf_counter()

    logger.info(
        "[TIMING] %s  sun_vectors=%.1f ms (%d dirs)  "
        "GPU_raycast=%.1f ms  build_response=%.1f ms  TOTAL=%.1f ms",
        label,
        (t1 - t0) * 1000, len(light_dirs),
        (t2 - t1) * 1000,
        (t3 - t2) * 1000,
        (t3 - t0) * 1000,
    )
    return result


# ------------------------------------------------------------------
# Public API — JSON endpoints (mesh arrives as Pydantic MeshData)
# ------------------------------------------------------------------

def analyze_vertices(
    mesh_data: MeshData, sun: SunConfig, options: AnalysisOptions,
) -> dict:
    """Compute direct sunlight at mesh vertices (mesh is its own blocker)."""
    processed = process_mesh(mesh_data)
    return _run_analysis(
        processed, processed.vertices, processed.vertex_normals,
        sun, options, label="analyze_vertices",
    )


def analyze_faces(
    mesh_data: MeshData, sun: SunConfig, options: AnalysisOptions,
) -> dict:
    """Compute direct sunlight at face centroids (mesh is its own blocker)."""
    processed = process_mesh(mesh_data)
    return _run_analysis(
        processed, processed.face_centroids, processed.face_normals,
        sun, options, label="analyze_faces",
    )


def analyze_vertices_separate(
    blocking_mesh_data: MeshData,
    target_points: list[list[float]],
    target_normals: list[list[float]],
    sun: SunConfig,
    options: AnalysisOptions,
) -> dict:
    """Sunlight at arbitrary target vertices with a separate blocker mesh."""
    blocker = process_mesh(blocking_mesh_data)
    pts, nrm = points_and_normals_to_numpy(target_points, target_normals)
    return _run_analysis(
        blocker, pts, nrm, sun, options,
        label="analyze_vertices_separate",
    )


def analyze_faces_separate(
    blocking_mesh_data: MeshData,
    target_mesh_data: MeshData,
    sun: SunConfig,
    options: AnalysisOptions,
) -> dict:
    """Sunlight at target-mesh face centroids with a separate blocker mesh."""
    blocker = process_mesh(blocking_mesh_data)
    target = process_mesh(target_mesh_data)
    return _run_analysis(
        blocker, target.face_centroids, target.face_normals,
        sun, options, label="analyze_faces_separate",
    )


# ------------------------------------------------------------------
# Public API — Numpy endpoints (mesh arrives as pre-built arrays)
# ------------------------------------------------------------------

def analyze_vertices_numpy(
    processed: ProcessedMesh, sun: SunConfig, options: AnalysisOptions,
) -> dict:
    """Like ``analyze_vertices`` but accepts a pre-built ProcessedMesh."""
    return _run_analysis(
        processed, processed.vertices, processed.vertex_normals,
        sun, options, label="analyze_vertices_numpy",
    )


def analyze_faces_numpy(
    processed: ProcessedMesh, sun: SunConfig, options: AnalysisOptions,
) -> dict:
    """Like ``analyze_faces`` but accepts a pre-built ProcessedMesh."""
    return _run_analysis(
        processed, processed.face_centroids, processed.face_normals,
        sun, options, label="analyze_faces_numpy",
    )


def analyze_vertices_separate_numpy(
    blocker: ProcessedMesh,
    target_points: np.ndarray,
    target_normals: np.ndarray,
    sun: SunConfig,
    options: AnalysisOptions,
) -> dict:
    """Like ``analyze_vertices_separate`` but with pre-built numpy arrays."""
    return _run_analysis(
        blocker, target_points, target_normals,
        sun, options, label="analyze_vertices_separate_numpy",
    )


def analyze_faces_separate_numpy(
    blocker: ProcessedMesh,
    target: ProcessedMesh,
    sun: SunConfig,
    options: AnalysisOptions,
) -> dict:
    """Like ``analyze_faces_separate`` but with pre-built ProcessedMesh."""
    return _run_analysis(
        blocker, target.face_centroids, target.face_normals,
        sun, options, label="analyze_faces_separate_numpy",
    )


# ------------------------------------------------------------------
# Standalone sun-vector query
# ------------------------------------------------------------------

def compute_sun_vectors(sun: SunConfig) -> tuple[np.ndarray, list[datetime]]:
    """Return raw sun direction vectors and timestamps for a SunConfig."""
    if sun.start_datetime and sun.end_datetime:
        start_dt = datetime.fromisoformat(sun.start_datetime)
        end_dt = datetime.fromisoformat(sun.end_datetime)
        step_minutes = sun.time_step_minutes or (sun.time_step_hours * 60)
        return get_sun_vectors_from_ladybug(
            latitude=sun.latitude,
            longitude=sun.longitude,
            start_datetime=start_dt,
            end_datetime=end_dt,
            time_step_minutes=step_minutes,
            timezone=sun.timezone,
            coordinate_system=sun.coordinate_system,
        )
    return get_yearly_sun_vectors(
        latitude=sun.latitude,
        longitude=sun.longitude,
        year=sun.year,
        time_step_hours=sun.time_step_hours,
        timezone=sun.timezone,
        coordinate_system=sun.coordinate_system,
    )
