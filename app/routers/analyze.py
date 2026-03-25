"""
Router for /analyze endpoints.

Provides four JSON endpoints (original API surface):
  - POST /analyze/vertices          (mesh is its own blocker)
  - POST /analyze/faces             (mesh is its own blocker)
  - POST /analyze/vertices/separate (separate blocker + targets)
  - POST /analyze/faces/separate    (separate blocker + targets)

And four binary (numpy) endpoints for maximum throughput:
  - POST /analyze/vertices/numpy
  - POST /analyze/faces/numpy
  - POST /analyze/vertices/separate/numpy
  - POST /analyze/faces/separate/numpy
"""
import io
import logging
import time

import numpy as np
import orjson
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import Response

logger = logging.getLogger("analyze")

from app.models.requests import (
    AnalyzeVerticesRequest,
    AnalyzeFacesRequest,
    AnalyzeVerticesSeparateRequest,
    AnalyzeFacesSeparateRequest,
    SunConfig,
    AnalysisOptions,
)
from app.models.responses import AnalysisResponse
from app.services import analysis as analysis_service
from app.services.mesh_processing import ProcessedMesh, process_mesh_from_numpy

router = APIRouter(prefix="/analyze", tags=["analysis"])


def _numpy_json_response(data: dict) -> Response:
    """Serialise a dict that may contain numpy arrays to JSON via orjson."""
    t0 = time.perf_counter()
    content = orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[TIMING] orjson serialization: {elapsed_ms:.1f} ms  ({len(content) / 1024:.1f} KB)", flush=True)
    return Response(content=content, media_type="application/json")


def _load_npz_mesh(raw_bytes: bytes) -> dict[str, np.ndarray]:
    """Load an in-memory NPZ archive into a dict of numpy arrays."""
    with np.load(io.BytesIO(raw_bytes), allow_pickle=False) as npz:
        return {key: npz[key] for key in npz.files}


# ------------------------------------------------------------------ #
#  Original JSON endpoints (unchanged request format, faster response)
# ------------------------------------------------------------------ #


@router.post(
    "/vertices",
    response_model=AnalysisResponse,
    summary="Vertex-based sunlight analysis",
    description=(
        "Compute annual direct sunlight hours at each mesh vertex. "
        "The submitted mesh acts as both the occlusion geometry and the "
        "analysis target."
    ),
)
async def analyze_vertices(body: AnalyzeVerticesRequest) -> Response:
    t_start = time.perf_counter()
    result = analysis_service.analyze_vertices(
        mesh_data=body.mesh,
        sun=body.sun,
        options=body.options,
    )
    t_service = time.perf_counter()
    resp = _numpy_json_response(result)
    t_end = time.perf_counter()
    print(
        f"[TIMING] /analyze/vertices TOTAL: {(t_end - t_start) * 1000:.1f} ms"
        f"  (service={(t_service - t_start) * 1000:.1f} ms, serialize={(t_end - t_service) * 1000:.1f} ms)",
        flush=True,
    )
    return resp


@router.post(
    "/faces",
    response_model=AnalysisResponse,
    summary="Face-based sunlight analysis",
    description=(
        "Compute annual direct sunlight hours at each face centroid. "
        "The submitted mesh acts as both the occlusion geometry and the "
        "analysis target.  Backface culling is recommended (enabled by default)."
    ),
)
async def analyze_faces(body: AnalyzeFacesRequest) -> Response:
    t_start = time.perf_counter()
    result = analysis_service.analyze_faces(
        mesh_data=body.mesh,
        sun=body.sun,
        options=body.options,
    )
    t_service = time.perf_counter()
    resp = _numpy_json_response(result)
    t_end = time.perf_counter()
    print(
        f"[TIMING] /analyze/faces TOTAL: {(t_end - t_start) * 1000:.1f} ms"
        f"  (service={(t_service - t_start) * 1000:.1f} ms, serialize={(t_end - t_service) * 1000:.1f} ms)",
        flush=True,
    )
    return resp


@router.post(
    "/vertices/separate",
    response_model=AnalysisResponse,
    summary="Vertex-based analysis with separate blocker",
    description=(
        "Compute direct sunlight at explicit target vertices while using "
        "a separate mesh as occlusion geometry.  Useful when the analysis "
        "surface differs from the surrounding buildings / context."
    ),
)
async def analyze_vertices_separate(
    body: AnalyzeVerticesSeparateRequest,
) -> Response:
    result = analysis_service.analyze_vertices_separate(
        blocking_mesh_data=body.blocking_mesh,
        target_points=body.target_points,
        target_normals=body.target_normals,
        sun=body.sun,
        options=body.options,
    )
    return _numpy_json_response(result)


@router.post(
    "/faces/separate",
    response_model=AnalysisResponse,
    summary="Face-based analysis with separate blocker",
    description=(
        "Compute direct sunlight at face centroids of a target mesh while "
        "using a different mesh as occlusion geometry."
    ),
)
async def analyze_faces_separate(
    body: AnalyzeFacesSeparateRequest,
) -> Response:
    result = analysis_service.analyze_faces_separate(
        blocking_mesh_data=body.blocking_mesh,
        target_mesh_data=body.target_mesh,
        sun=body.sun,
        options=body.options,
    )
    return _numpy_json_response(result)


# ------------------------------------------------------------------ #
#  Binary (numpy) endpoints – accept NPZ uploads for max throughput
# ------------------------------------------------------------------ #


@router.post(
    "/vertices/numpy",
    response_model=AnalysisResponse,
    tags=["analysis-numpy"],
    summary="Vertex analysis from numpy arrays",
    description=(
        "Same as /analyze/vertices but accepts the mesh as a binary NPZ "
        "upload (keys: ``points``, ``face_vertex_counts``, "
        "``face_vertex_indices``, optional ``normals``) for faster "
        "parsing on large meshes.\n\n"
        "``sun_config`` and ``options`` are JSON strings in form fields."
    ),
)
async def analyze_vertices_numpy(
    mesh_npz: UploadFile = File(..., description="NPZ archive with mesh arrays"),
    sun_config: str = Form(..., description="SunConfig as JSON string"),
    options: str = Form(default="{}", description="AnalysisOptions as JSON string"),
) -> Response:
    arrays = _load_npz_mesh(await mesh_npz.read())
    processed = process_mesh_from_numpy(arrays)
    sun = SunConfig.model_validate_json(sun_config)
    opts = AnalysisOptions.model_validate_json(options)

    result = analysis_service.analyze_vertices_numpy(processed, sun, opts)
    return _numpy_json_response(result)


@router.post(
    "/faces/numpy",
    response_model=AnalysisResponse,
    tags=["analysis-numpy"],
    summary="Face analysis from numpy arrays",
    description=(
        "Same as /analyze/faces but accepts the mesh as a binary NPZ "
        "upload.  See /analyze/vertices/numpy for NPZ key requirements."
    ),
)
async def analyze_faces_numpy(
    mesh_npz: UploadFile = File(..., description="NPZ archive with mesh arrays"),
    sun_config: str = Form(..., description="SunConfig as JSON string"),
    options: str = Form(default="{}", description="AnalysisOptions as JSON string"),
) -> Response:
    arrays = _load_npz_mesh(await mesh_npz.read())
    processed = process_mesh_from_numpy(arrays)
    sun = SunConfig.model_validate_json(sun_config)
    opts = AnalysisOptions.model_validate_json(options)

    result = analysis_service.analyze_faces_numpy(processed, sun, opts)
    return _numpy_json_response(result)


@router.post(
    "/vertices/separate/numpy",
    response_model=AnalysisResponse,
    tags=["analysis-numpy"],
    summary="Vertex analysis (separate blocker) from numpy arrays",
    description=(
        "Same as /analyze/vertices/separate but accepts the blocking mesh, "
        "target points, and target normals as binary NPZ uploads.\n\n"
        "**blocking_mesh_npz** keys: ``points``, ``face_vertex_counts``, "
        "``face_vertex_indices``, optional ``normals``.\n\n"
        "**targets_npz** keys: ``target_points`` (N,3), "
        "``target_normals`` (N,3)."
    ),
)
async def analyze_vertices_separate_numpy(
    blocking_mesh_npz: UploadFile = File(
        ..., description="NPZ archive with blocking-mesh arrays"
    ),
    targets_npz: UploadFile = File(
        ..., description="NPZ archive with target_points and target_normals"
    ),
    sun_config: str = Form(..., description="SunConfig as JSON string"),
    options: str = Form(default="{}", description="AnalysisOptions as JSON string"),
) -> Response:
    blocker_arrays = _load_npz_mesh(await blocking_mesh_npz.read())
    blocker = process_mesh_from_numpy(blocker_arrays)

    target_arrays = _load_npz_mesh(await targets_npz.read())
    target_pts = np.ascontiguousarray(
        target_arrays["target_points"], dtype=np.float32
    )
    target_nrm = np.ascontiguousarray(
        target_arrays["target_normals"], dtype=np.float32
    )

    sun = SunConfig.model_validate_json(sun_config)
    opts = AnalysisOptions.model_validate_json(options)

    result = analysis_service.analyze_vertices_separate_numpy(
        blocker, target_pts, target_nrm, sun, opts
    )
    return _numpy_json_response(result)


@router.post(
    "/faces/separate/numpy",
    response_model=AnalysisResponse,
    tags=["analysis-numpy"],
    summary="Face analysis (separate blocker) from numpy arrays",
    description=(
        "Same as /analyze/faces/separate but accepts both meshes as binary "
        "NPZ uploads.\n\n"
        "Both **blocking_mesh_npz** and **target_mesh_npz** require keys: "
        "``points``, ``face_vertex_counts``, ``face_vertex_indices``, "
        "optional ``normals``."
    ),
)
async def analyze_faces_separate_numpy(
    blocking_mesh_npz: UploadFile = File(
        ..., description="NPZ archive with blocking-mesh arrays"
    ),
    target_mesh_npz: UploadFile = File(
        ..., description="NPZ archive with target-mesh arrays"
    ),
    sun_config: str = Form(..., description="SunConfig as JSON string"),
    options: str = Form(default="{}", description="AnalysisOptions as JSON string"),
) -> Response:
    blocker = process_mesh_from_numpy(
        _load_npz_mesh(await blocking_mesh_npz.read())
    )
    target = process_mesh_from_numpy(
        _load_npz_mesh(await target_mesh_npz.read())
    )

    sun = SunConfig.model_validate_json(sun_config)
    opts = AnalysisOptions.model_validate_json(options)

    result = analysis_service.analyze_faces_separate_numpy(
        blocker, target, sun, opts
    )
    return _numpy_json_response(result)
