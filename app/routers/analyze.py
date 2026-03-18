"""
Router for /analyze endpoints.

Provides four endpoints:
  - POST /analyze/vertices          (mesh is its own blocker)
  - POST /analyze/faces             (mesh is its own blocker)
  - POST /analyze/vertices/separate (separate blocker + targets)
  - POST /analyze/faces/separate    (separate blocker + targets)
"""
from fastapi import APIRouter

from app.models.requests import (
    AnalyzeVerticesRequest,
    AnalyzeFacesRequest,
    AnalyzeVerticesSeparateRequest,
    AnalyzeFacesSeparateRequest,
)
from app.models.responses import AnalysisResponse
from app.services import analysis as analysis_service

router = APIRouter(prefix="/analyze", tags=["analysis"])


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
async def analyze_vertices(body: AnalyzeVerticesRequest) -> AnalysisResponse:
    return analysis_service.analyze_vertices(
        mesh_data=body.mesh,
        sun=body.sun,
        options=body.options,
    )


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
async def analyze_faces(body: AnalyzeFacesRequest) -> AnalysisResponse:
    return analysis_service.analyze_faces(
        mesh_data=body.mesh,
        sun=body.sun,
        options=body.options,
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
) -> AnalysisResponse:
    return analysis_service.analyze_vertices_separate(
        blocking_mesh_data=body.blocking_mesh,
        target_points=body.target_points,
        target_normals=body.target_normals,
        sun=body.sun,
        options=body.options,
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
) -> AnalysisResponse:
    return analysis_service.analyze_faces_separate(
        blocking_mesh_data=body.blocking_mesh,
        target_mesh_data=body.target_mesh,
        sun=body.sun,
        options=body.options,
    )
