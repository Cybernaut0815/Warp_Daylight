"""
Orchestration service that ties together sun-vector generation, mesh
preparation, GPU raycasting, and optional colour mapping.
"""
import logging
import time
import numpy as np
from datetime import datetime
from typing import Optional

logger = logging.getLogger("analyze")

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


DEFAULT_GRADIENT_COLORS = [
    (0, 0, 128),
    (0, 128, 255),
    (0, 255, 128),
    (255, 255, 0),
    (255, 128, 0),
    (255, 0, 0),
]
DEFAULT_GRADIENT_POSITIONS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _resolve_sun_vectors(
    sun: SunConfig,
) -> tuple[np.ndarray, list[datetime], float]:
    """
    Generate sun direction vectors from a SunConfig.

    Returns (light_directions, timestamps, time_step_hours).
    """
    if sun.start_datetime and sun.end_datetime:
        start_dt = datetime.fromisoformat(sun.start_datetime)
        end_dt = datetime.fromisoformat(sun.end_datetime)
        step_minutes = sun.time_step_minutes or (sun.time_step_hours * 60)
        light_dirs, timestamps = get_sun_vectors_from_ladybug(
            latitude=sun.latitude,
            longitude=sun.longitude,
            start_datetime=start_dt,
            end_datetime=end_dt,
            time_step_minutes=step_minutes,
            timezone=sun.timezone,
            coordinate_system=sun.coordinate_system,
        )
        time_step_hours = step_minutes / 60.0
    else:
        light_dirs, timestamps = get_yearly_sun_vectors(
            latitude=sun.latitude,
            longitude=sun.longitude,
            year=sun.year,
            time_step_hours=sun.time_step_hours,
            timezone=sun.timezone,
            coordinate_system=sun.coordinate_system,
        )
        time_step_hours = float(sun.time_step_hours)

    return light_dirs, timestamps, time_step_hours


def _build_colors(
    hit_counts: np.ndarray,
    options: AnalysisOptions,
) -> Optional[np.ndarray]:
    """Map hit counts to RGB colours via a gradient (higher sunlight = warmer).

    Returns an (N, 3) uint8 numpy array, or None when colours are disabled.
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

    Callers are responsible for serialising (e.g. via orjson with
    ``OPT_SERIALIZE_NUMPY``).
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


def analyze_vertices(
    mesh_data: MeshData,
    sun: SunConfig,
    options: AnalysisOptions,
) -> dict:
    """Compute direct sunlight on mesh vertices (mesh is its own blocker)."""
    t0 = time.perf_counter()
    processed = process_mesh(mesh_data)
    t1 = time.perf_counter()

    light_dirs, timestamps, step_h = _resolve_sun_vectors(sun)
    t2 = time.perf_counter()

    hit_counts = raycast_directional_batch(
        vertices=processed.vertices,
        face_indices=processed.faces,
        start_positions=processed.vertices,
        start_normals=processed.vertex_normals,
        light_directions=light_dirs,
        offset_distance=options.offset_distance,
        chunk_size=options.chunk_size,
        use_backface_culling=options.use_backface_culling,
    )
    t3 = time.perf_counter()

    result = _build_response(
        hit_counts=hit_counts,
        num_sun_positions=len(light_dirs),
        time_step_hours=step_h,
        computation_time=t3 - t2,
        num_elements=len(processed.vertices),
        mesh_vertex_count=len(processed.vertices),
        mesh_face_count=len(processed.faces),
        options=options,
    )
    t4 = time.perf_counter()

    print(
        f"[TIMING] analyze_vertices breakdown:\n"
        f"    process_mesh:     {(t1 - t0) * 1000:7.1f} ms\n"
        f"    sun_vectors:      {(t2 - t1) * 1000:7.1f} ms  ({len(light_dirs)} dirs)\n"
        f"    GPU raycast:      {(t3 - t2) * 1000:7.1f} ms\n"
        f"    build_response:   {(t4 - t3) * 1000:7.1f} ms  (colors={options.return_colors})\n"
        f"    TOTAL service:    {(t4 - t0) * 1000:7.1f} ms",
        flush=True,
    )
    return result


def analyze_faces(
    mesh_data: MeshData,
    sun: SunConfig,
    options: AnalysisOptions,
) -> dict:
    """Compute direct sunlight on face centroids (mesh is its own blocker)."""
    t0 = time.perf_counter()
    processed = process_mesh(mesh_data)
    t1 = time.perf_counter()

    light_dirs, timestamps, step_h = _resolve_sun_vectors(sun)
    t2 = time.perf_counter()

    hit_counts = raycast_directional_batch(
        vertices=processed.vertices,
        face_indices=processed.faces,
        start_positions=processed.face_centroids,
        start_normals=processed.face_normals,
        light_directions=light_dirs,
        offset_distance=options.offset_distance,
        chunk_size=options.chunk_size,
        use_backface_culling=options.use_backface_culling,
    )
    t3 = time.perf_counter()

    result = _build_response(
        hit_counts=hit_counts,
        num_sun_positions=len(light_dirs),
        time_step_hours=step_h,
        computation_time=t3 - t2,
        num_elements=len(processed.face_centroids),
        mesh_vertex_count=len(processed.vertices),
        mesh_face_count=len(processed.faces),
        options=options,
    )
    t4 = time.perf_counter()

    print(
        f"[TIMING] analyze_faces breakdown:\n"
        f"    process_mesh:     {(t1 - t0) * 1000:7.1f} ms\n"
        f"    sun_vectors:      {(t2 - t1) * 1000:7.1f} ms  ({len(light_dirs)} dirs)\n"
        f"    GPU raycast:      {(t3 - t2) * 1000:7.1f} ms\n"
        f"    build_response:   {(t4 - t3) * 1000:7.1f} ms  (colors={options.return_colors})\n"
        f"    TOTAL service:    {(t4 - t0) * 1000:7.1f} ms",
        flush=True,
    )
    return result


def analyze_vertices_separate(
    blocking_mesh_data: MeshData,
    target_points: list[list[float]],
    target_normals: list[list[float]],
    sun: SunConfig,
    options: AnalysisOptions,
) -> dict:
    """
    Compute direct sunlight on arbitrary target vertices using a separate
    blocking mesh for occlusion.
    """
    blocker = process_mesh(blocking_mesh_data)
    pts, nrm = points_and_normals_to_numpy(target_points, target_normals)
    light_dirs, timestamps, step_h = _resolve_sun_vectors(sun)

    t0 = time.perf_counter()
    hit_counts = raycast_directional_batch(
        vertices=blocker.vertices,
        face_indices=blocker.faces,
        start_positions=pts,
        start_normals=nrm,
        light_directions=light_dirs,
        offset_distance=options.offset_distance,
        chunk_size=options.chunk_size,
        use_backface_culling=options.use_backface_culling,
    )
    elapsed = time.perf_counter() - t0

    return _build_response(
        hit_counts=hit_counts,
        num_sun_positions=len(light_dirs),
        time_step_hours=step_h,
        computation_time=elapsed,
        num_elements=len(pts),
        mesh_vertex_count=len(blocker.vertices),
        mesh_face_count=len(blocker.faces),
        options=options,
    )


def analyze_faces_separate(
    blocking_mesh_data: MeshData,
    target_mesh_data: MeshData,
    sun: SunConfig,
    options: AnalysisOptions,
) -> dict:
    """
    Compute direct sunlight on target mesh face centroids using a separate
    blocking mesh for occlusion.
    """
    blocker = process_mesh(blocking_mesh_data)
    target = process_mesh(target_mesh_data)
    light_dirs, timestamps, step_h = _resolve_sun_vectors(sun)

    t0 = time.perf_counter()
    hit_counts = raycast_directional_batch(
        vertices=blocker.vertices,
        face_indices=blocker.faces,
        start_positions=target.face_centroids,
        start_normals=target.face_normals,
        light_directions=light_dirs,
        offset_distance=options.offset_distance,
        chunk_size=options.chunk_size,
        use_backface_culling=options.use_backface_culling,
    )
    elapsed = time.perf_counter() - t0

    return _build_response(
        hit_counts=hit_counts,
        num_sun_positions=len(light_dirs),
        time_step_hours=step_h,
        computation_time=elapsed,
        num_elements=len(target.face_centroids),
        mesh_vertex_count=len(blocker.vertices),
        mesh_face_count=len(blocker.faces),
        options=options,
    )


def analyze_vertices_numpy(
    processed: ProcessedMesh,
    sun: SunConfig,
    options: AnalysisOptions,
) -> dict:
    """Like ``analyze_vertices`` but accepts a pre-built ProcessedMesh."""
    light_dirs, timestamps, step_h = _resolve_sun_vectors(sun)

    t0 = time.perf_counter()
    hit_counts = raycast_directional_batch(
        vertices=processed.vertices,
        face_indices=processed.faces,
        start_positions=processed.vertices,
        start_normals=processed.vertex_normals,
        light_directions=light_dirs,
        offset_distance=options.offset_distance,
        chunk_size=options.chunk_size,
        use_backface_culling=options.use_backface_culling,
    )
    elapsed = time.perf_counter() - t0

    return _build_response(
        hit_counts=hit_counts,
        num_sun_positions=len(light_dirs),
        time_step_hours=step_h,
        computation_time=elapsed,
        num_elements=len(processed.vertices),
        mesh_vertex_count=len(processed.vertices),
        mesh_face_count=len(processed.faces),
        options=options,
    )


def analyze_faces_numpy(
    processed: ProcessedMesh,
    sun: SunConfig,
    options: AnalysisOptions,
) -> dict:
    """Like ``analyze_faces`` but accepts a pre-built ProcessedMesh."""
    light_dirs, timestamps, step_h = _resolve_sun_vectors(sun)

    t0 = time.perf_counter()
    hit_counts = raycast_directional_batch(
        vertices=processed.vertices,
        face_indices=processed.faces,
        start_positions=processed.face_centroids,
        start_normals=processed.face_normals,
        light_directions=light_dirs,
        offset_distance=options.offset_distance,
        chunk_size=options.chunk_size,
        use_backface_culling=options.use_backface_culling,
    )
    elapsed = time.perf_counter() - t0

    return _build_response(
        hit_counts=hit_counts,
        num_sun_positions=len(light_dirs),
        time_step_hours=step_h,
        computation_time=elapsed,
        num_elements=len(processed.face_centroids),
        mesh_vertex_count=len(processed.vertices),
        mesh_face_count=len(processed.faces),
        options=options,
    )


def analyze_vertices_separate_numpy(
    blocker: ProcessedMesh,
    target_points: np.ndarray,
    target_normals: np.ndarray,
    sun: SunConfig,
    options: AnalysisOptions,
) -> dict:
    """Like ``analyze_vertices_separate`` but with pre-built numpy arrays."""
    light_dirs, timestamps, step_h = _resolve_sun_vectors(sun)

    t0 = time.perf_counter()
    hit_counts = raycast_directional_batch(
        vertices=blocker.vertices,
        face_indices=blocker.faces,
        start_positions=target_points,
        start_normals=target_normals,
        light_directions=light_dirs,
        offset_distance=options.offset_distance,
        chunk_size=options.chunk_size,
        use_backface_culling=options.use_backface_culling,
    )
    elapsed = time.perf_counter() - t0

    return _build_response(
        hit_counts=hit_counts,
        num_sun_positions=len(light_dirs),
        time_step_hours=step_h,
        computation_time=elapsed,
        num_elements=len(target_points),
        mesh_vertex_count=len(blocker.vertices),
        mesh_face_count=len(blocker.faces),
        options=options,
    )


def analyze_faces_separate_numpy(
    blocker: ProcessedMesh,
    target: ProcessedMesh,
    sun: SunConfig,
    options: AnalysisOptions,
) -> dict:
    """Like ``analyze_faces_separate`` but with pre-built ProcessedMesh."""
    light_dirs, timestamps, step_h = _resolve_sun_vectors(sun)

    t0 = time.perf_counter()
    hit_counts = raycast_directional_batch(
        vertices=blocker.vertices,
        face_indices=blocker.faces,
        start_positions=target.face_centroids,
        start_normals=target.face_normals,
        light_directions=light_dirs,
        offset_distance=options.offset_distance,
        chunk_size=options.chunk_size,
        use_backface_culling=options.use_backface_culling,
    )
    elapsed = time.perf_counter() - t0

    return _build_response(
        hit_counts=hit_counts,
        num_sun_positions=len(light_dirs),
        time_step_hours=step_h,
        computation_time=elapsed,
        num_elements=len(target.face_centroids),
        mesh_vertex_count=len(blocker.vertices),
        mesh_face_count=len(blocker.faces),
        options=options,
    )


def compute_sun_vectors(sun: SunConfig) -> tuple[np.ndarray, list[datetime]]:
    """Return raw sun direction vectors + timestamps for a SunConfig."""
    light_dirs, timestamps, _ = _resolve_sun_vectors(sun)
    return light_dirs, timestamps
