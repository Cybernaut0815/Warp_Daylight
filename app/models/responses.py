"""
Pydantic response models for the daylight analysis API.
"""
from pydantic import BaseModel, Field
from typing import Optional


class AnalysisMetadata(BaseModel):
    """Metadata returned alongside analysis results."""
    computation_time_seconds: float = Field(
        description="Wall-clock time for the raycast computation.",
    )
    num_elements: int = Field(
        description="Number of vertices or faces analysed.",
    )
    num_sun_positions: int = Field(
        description="Number of sun direction vectors used.",
    )
    mesh_vertex_count: int = Field(
        description="Total vertices in the blocking mesh.",
    )
    mesh_face_count: int = Field(
        description="Total faces in the blocking mesh.",
    )


class AnalysisResponse(BaseModel):
    """
    Response for /analyze/* endpoints.

    ``sunlight_hours`` and ``hit_counts`` are parallel arrays whose length
    equals the number of analysed elements (vertices or faces).

    If ``return_colors`` was True in the request, ``colors`` contains
    per-element [R, G, B] values (0-255) ready for DCC vertex/face colouring.
    """
    sunlight_hours: list[float] = Field(
        description="Direct sunlight hours per element.",
    )
    hit_counts: list[int] = Field(
        description="Number of sun directions blocked per element.",
    )
    total_sun_positions: int = Field(
        description="Total sun positions (M) tested.",
    )
    max_possible_hours: float = Field(
        description="Maximum theoretically possible sunlight hours.",
    )
    colors: Optional[list[list[int]]] = Field(
        default=None,
        description="Per-element [R,G,B] colors (0-255). "
                    "None if return_colors was False.",
    )
    metadata: AnalysisMetadata


class SunVectorEntry(BaseModel):
    """A single sun direction vector with its timestamp."""
    direction: list[float] = Field(
        description="Light direction vector FROM the sun [x, y, z].",
    )
    timestamp: str = Field(
        description="ISO-8601 timestamp for this sun position.",
    )


class SunVectorsResponse(BaseModel):
    """Response for POST /sun-vectors."""
    vectors: list[SunVectorEntry] = Field(
        description="Sun direction vectors with timestamps.",
    )
    total_count: int = Field(
        description="Number of sun positions returned.",
    )
    coordinate_system: str = Field(
        description="Coordinate system used ('y_up' or 'z_up').",
    )


class HealthResponse(BaseModel):
    """Response for GET /health."""
    status: str = Field(description="'ok' when the service is healthy.")
    gpu_available: bool = Field(
        description="Whether a CUDA-capable GPU was detected.",
    )
    warp_initialized: bool = Field(
        description="Whether NVIDIA Warp has been initialized.",
    )
