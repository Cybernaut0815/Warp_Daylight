"""
Router for ``/analyze`` endpoints.

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
from enum import Enum

import numpy as np
import orjson
from fastapi import APIRouter, File, Form, Query, UploadFile
from fastapi.responses import Response

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
from app.services.mesh_processing import process_mesh_from_numpy

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["analysis"])


class ResponseFormat(str, Enum):
    json = "json"
    npz = "npz"


_RESPONSE_FORMAT_DOC = (
    "Response format: `json` (default) returns standard JSON; "
    "`npz` returns a binary numpy archive for faster client-side parsing."
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _numpy_json_response(data: dict) -> Response:
    """Serialise a dict that may contain numpy arrays to JSON via orjson."""
    t0 = time.perf_counter()
    content = orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "[TIMING] orjson serialization: %.1f ms  (%.1f KB)",
        elapsed_ms, len(content) / 1024,
    )
    return Response(content=content, media_type="application/json")


def _numpy_npz_response(data: dict) -> Response:
    """Serialise an analysis result dict as a downloadable NPZ archive.

    Array fields (``sunlight_hours``, ``hit_counts``, ``colors``) are stored
    as numpy arrays.  Scalar fields and the ``metadata`` dict are stored as
    a JSON blob under the ``metadata_json`` key.
    """
    t0 = time.perf_counter()
    buf = io.BytesIO()
    arrays: dict[str, np.ndarray] = {}

    for key in ("sunlight_hours", "hit_counts"):
        val = data.get(key)
        if val is not None:
            arrays[key] = np.asarray(val)

    colors = data.get("colors")
    if colors is not None:
        arrays["colors"] = np.asarray(colors, dtype=np.uint8)

    scalars = {
        "total_sun_positions": data.get("total_sun_positions"),
        "max_possible_hours": data.get("max_possible_hours"),
        "metadata": data.get("metadata"),
    }
    arrays["metadata_json"] = np.void(orjson.dumps(scalars))

    np.savez(buf, **arrays)
    content = buf.getvalue()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "[TIMING] NPZ serialization: %.1f ms  (%.1f KB)",
        elapsed_ms, len(content) / 1024,
    )
    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=analysis.npz"},
    )


def _load_npz_mesh(raw_bytes: bytes) -> dict[str, np.ndarray]:
    """Load an in-memory NPZ archive into a dict of numpy arrays."""
    with np.load(io.BytesIO(raw_bytes), allow_pickle=False) as npz:
        return {key: npz[key] for key in npz.files}


def _serialize_response(data: dict, fmt: ResponseFormat) -> Response:
    """Serialize an analysis result dict in the requested format."""
    if fmt == ResponseFormat.npz:
        return _numpy_npz_response(data)
    return _numpy_json_response(data)


def _timed_response(
    label: str,
    service_fn,
    *args,
    response_format: ResponseFormat = ResponseFormat.json,
    **kwargs,
) -> Response:
    """Call *service_fn*, serialise the result, and log timing."""
    t_start = time.perf_counter()
    result = service_fn(*args, **kwargs)
    t_service = time.perf_counter()
    resp = _serialize_response(result, response_format)
    t_end = time.perf_counter()
    logger.info(
        "[TIMING] %s TOTAL: %.1f ms  (service=%.1f ms, serialize=%.1f ms)",
        label,
        (t_end - t_start) * 1000,
        (t_service - t_start) * 1000,
        (t_end - t_service) * 1000,
    )
    return resp


# TODO(roadmap): Add daylight factor from EPW files — a new endpoint
#   (e.g. POST /analyze/daylight-factor) would accept an EPW file upload
#   alongside mesh data, weight each sun direction by its irradiance value,
#   and return lux-hour or daylight-factor results per element.

# ------------------------------------------------------------------
# JSON endpoints
# ------------------------------------------------------------------

@router.post(
    "/vertices",
    response_model=AnalysisResponse,
    summary="Vertex-based sunlight analysis",
    description=(
        "Compute annual direct sunlight hours at each mesh vertex. "
        "The submitted mesh acts as both the occlusion geometry and the "
        "analysis target.\n\n"
        "**Response fields:**\n"
        "- `sunlight_hours` — per-vertex hours of direct sun.\n"
        "- `hit_counts` — per-vertex occlusion count.\n"
        "- `colors` — optional [R,G,B] gradient (0-255).\n\n"
        "**Performance tip:** for meshes >50 k vertices, consider the "
        "`/vertices/numpy` endpoint which accepts binary NPZ uploads."
    ),
)
async def analyze_vertices(
    body: AnalyzeVerticesRequest,
    response_format: ResponseFormat = Query(
        default=ResponseFormat.json, description=_RESPONSE_FORMAT_DOC,
    ),
) -> Response:
    """Vertex-based sunlight analysis (self-blocking mesh)."""
    return _timed_response(
        "/analyze/vertices",
        analysis_service.analyze_vertices,
        mesh_data=body.mesh, sun=body.sun, options=body.options,
        response_format=response_format,
    )


@router.post(
    "/faces",
    response_model=AnalysisResponse,
    summary="Face-based sunlight analysis",
    description=(
        "Compute annual direct sunlight hours at each face centroid. "
        "The submitted mesh acts as both the occlusion geometry and the "
        "analysis target.  Backface culling is recommended (enabled by "
        "default).\n\n"
        "**Performance tip:** for meshes >50 k faces, consider the "
        "`/faces/numpy` endpoint."
    ),
)
async def analyze_faces(
    body: AnalyzeFacesRequest,
    response_format: ResponseFormat = Query(
        default=ResponseFormat.json, description=_RESPONSE_FORMAT_DOC,
    ),
) -> Response:
    """Face-based sunlight analysis (self-blocking mesh)."""
    return _timed_response(
        "/analyze/faces",
        analysis_service.analyze_faces,
        mesh_data=body.mesh, sun=body.sun, options=body.options,
        response_format=response_format,
    )


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
    response_format: ResponseFormat = Query(
        default=ResponseFormat.json, description=_RESPONSE_FORMAT_DOC,
    ),
) -> Response:
    """Vertex-based analysis with a separate blocker mesh."""
    return _timed_response(
        "/analyze/vertices/separate",
        analysis_service.analyze_vertices_separate,
        blocking_mesh_data=body.blocking_mesh,
        target_points=body.target_points,
        target_normals=body.target_normals,
        sun=body.sun, options=body.options,
        response_format=response_format,
    )


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
    response_format: ResponseFormat = Query(
        default=ResponseFormat.json, description=_RESPONSE_FORMAT_DOC,
    ),
) -> Response:
    """Face-based analysis with a separate blocker mesh."""
    return _timed_response(
        "/analyze/faces/separate",
        analysis_service.analyze_faces_separate,
        blocking_mesh_data=body.blocking_mesh,
        target_mesh_data=body.target_mesh,
        sun=body.sun, options=body.options,
        response_format=response_format,
    )


# ------------------------------------------------------------------
# Binary (numpy) endpoints — accept NPZ uploads for max throughput
# ------------------------------------------------------------------

_NPZ_KEY_DOC = (
    "The NPZ archive must contain the keys: `points` (V,3), "
    "`face_vertex_counts` (F,), `face_vertex_indices` (sum(counts),), "
    "and optionally `normals` (V,3)."
)


@router.post(
    "/vertices/numpy",
    response_model=AnalysisResponse,
    tags=["analysis-numpy"],
    summary="Vertex analysis from numpy arrays",
    description=(
        "Same as `/analyze/vertices` but accepts the mesh as a binary NPZ "
        f"upload for faster parsing on large meshes.\n\n{_NPZ_KEY_DOC}\n\n"
        "`sun_config` and `options` are JSON strings in form fields."
    ),
)
async def analyze_vertices_numpy(
    mesh_npz: UploadFile = File(..., description="NPZ archive with mesh arrays"),
    sun_config: str = Form(..., description="SunConfig as JSON string"),
    options: str = Form(default="{}", description="AnalysisOptions as JSON string"),
    response_format: ResponseFormat = Query(
        default=ResponseFormat.json, description=_RESPONSE_FORMAT_DOC,
    ),
) -> Response:
    """Vertex analysis from binary NPZ mesh upload."""
    processed = process_mesh_from_numpy(_load_npz_mesh(await mesh_npz.read()))
    sun = SunConfig.model_validate_json(sun_config)
    opts = AnalysisOptions.model_validate_json(options)
    return _timed_response(
        "/analyze/vertices/numpy",
        analysis_service.analyze_vertices_numpy, processed, sun, opts,
        response_format=response_format,
    )


@router.post(
    "/faces/numpy",
    response_model=AnalysisResponse,
    tags=["analysis-numpy"],
    summary="Face analysis from numpy arrays",
    description=(
        "Same as `/analyze/faces` but accepts the mesh as a binary NPZ "
        f"upload.\n\n{_NPZ_KEY_DOC}"
    ),
)
async def analyze_faces_numpy(
    mesh_npz: UploadFile = File(..., description="NPZ archive with mesh arrays"),
    sun_config: str = Form(..., description="SunConfig as JSON string"),
    options: str = Form(default="{}", description="AnalysisOptions as JSON string"),
    response_format: ResponseFormat = Query(
        default=ResponseFormat.json, description=_RESPONSE_FORMAT_DOC,
    ),
) -> Response:
    """Face analysis from binary NPZ mesh upload."""
    processed = process_mesh_from_numpy(_load_npz_mesh(await mesh_npz.read()))
    sun = SunConfig.model_validate_json(sun_config)
    opts = AnalysisOptions.model_validate_json(options)
    return _timed_response(
        "/analyze/faces/numpy",
        analysis_service.analyze_faces_numpy, processed, sun, opts,
        response_format=response_format,
    )


@router.post(
    "/vertices/separate/numpy",
    response_model=AnalysisResponse,
    tags=["analysis-numpy"],
    summary="Vertex analysis (separate blocker) from numpy arrays",
    description=(
        "Same as `/analyze/vertices/separate` but accepts the blocking mesh, "
        "target points, and target normals as binary NPZ uploads.\n\n"
        "**blocking_mesh_npz** keys: `points`, `face_vertex_counts`, "
        "`face_vertex_indices`, optional `normals`.\n\n"
        "**targets_npz** keys: `target_points` (N,3), "
        "`target_normals` (N,3)."
    ),
)
async def analyze_vertices_separate_numpy(
    blocking_mesh_npz: UploadFile = File(
        ..., description="NPZ archive with blocking-mesh arrays",
    ),
    targets_npz: UploadFile = File(
        ..., description="NPZ archive with target_points and target_normals",
    ),
    sun_config: str = Form(..., description="SunConfig as JSON string"),
    options: str = Form(default="{}", description="AnalysisOptions as JSON string"),
    response_format: ResponseFormat = Query(
        default=ResponseFormat.json, description=_RESPONSE_FORMAT_DOC,
    ),
) -> Response:
    """Vertex analysis (separate blocker) from binary NPZ uploads."""
    blocker = process_mesh_from_numpy(
        _load_npz_mesh(await blocking_mesh_npz.read()),
    )
    target_arrays = _load_npz_mesh(await targets_npz.read())
    target_pts = np.ascontiguousarray(target_arrays["target_points"], dtype=np.float32)
    target_nrm = np.ascontiguousarray(target_arrays["target_normals"], dtype=np.float32)

    sun = SunConfig.model_validate_json(sun_config)
    opts = AnalysisOptions.model_validate_json(options)
    return _timed_response(
        "/analyze/vertices/separate/numpy",
        analysis_service.analyze_vertices_separate_numpy,
        blocker, target_pts, target_nrm, sun, opts,
        response_format=response_format,
    )


@router.post(
    "/faces/separate/numpy",
    response_model=AnalysisResponse,
    tags=["analysis-numpy"],
    summary="Face analysis (separate blocker) from numpy arrays",
    description=(
        "Same as `/analyze/faces/separate` but accepts both meshes as binary "
        "NPZ uploads.\n\n"
        "Both **blocking_mesh_npz** and **target_mesh_npz** require keys: "
        "`points`, `face_vertex_counts`, `face_vertex_indices`, "
        "optional `normals`."
    ),
)
async def analyze_faces_separate_numpy(
    blocking_mesh_npz: UploadFile = File(
        ..., description="NPZ archive with blocking-mesh arrays",
    ),
    target_mesh_npz: UploadFile = File(
        ..., description="NPZ archive with target-mesh arrays",
    ),
    sun_config: str = Form(..., description="SunConfig as JSON string"),
    options: str = Form(default="{}", description="AnalysisOptions as JSON string"),
    response_format: ResponseFormat = Query(
        default=ResponseFormat.json, description=_RESPONSE_FORMAT_DOC,
    ),
) -> Response:
    """Face analysis (separate blocker) from binary NPZ uploads."""
    blocker = process_mesh_from_numpy(
        _load_npz_mesh(await blocking_mesh_npz.read()),
    )
    target = process_mesh_from_numpy(
        _load_npz_mesh(await target_mesh_npz.read()),
    )
    sun = SunConfig.model_validate_json(sun_config)
    opts = AnalysisOptions.model_validate_json(options)
    return _timed_response(
        "/analyze/faces/separate/numpy",
        analysis_service.analyze_faces_separate_numpy,
        blocker, target, sun, opts,
        response_format=response_format,
    )
