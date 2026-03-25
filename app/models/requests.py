"""
Pydantic request models for the daylight analysis API.
"""
from pydantic import BaseModel, Field

from app.models.mesh import MeshData


class SunConfig(BaseModel):
    """Sun / location configuration for daylight analysis.

    Two usage modes:

    1. **Annual** (default): set ``year`` and ``time_step_hours``.
    2. **Custom range**: set ``start_datetime``, ``end_datetime``, and
       optionally ``time_step_minutes`` to override the annual range.

    ``coordinate_system`` should match your DCC tool:

    - ``"y_up"`` for Maya, USD, Omniverse (default).
    - ``"z_up"`` for Rhino, Grasshopper, Ladybug.
    """
    latitude: float = Field(
        ..., ge=-90, le=90,
        description="Location latitude in degrees (-90 to 90).",
    )
    longitude: float = Field(
        ..., ge=-180, le=180,
        description="Location longitude in degrees (-180 to 180).",
    )
    timezone: str | float = Field(
        default="UTC",
        description="IANA timezone string (e.g. 'America/New_York') "
                    "or numeric UTC offset (e.g. -5.0).",
    )
    coordinate_system: str = Field(
        default="y_up",
        pattern=r"^(y_up|z_up)$",
        description="Coordinate system: 'y_up' (Maya/USD) or 'z_up' (Rhino).",
    )
    year: int | None = Field(
        default=None,
        description="Year to analyse. Defaults to current year.",
    )
    time_step_hours: int = Field(
        default=1, ge=1,
        description="Time step in hours for annual analysis.",
    )
    start_datetime: str | None = Field(
        default=None,
        description="Custom range start in ISO-8601 format "
                    "(e.g. '2024-06-21T06:00:00'). Overrides year.",
    )
    end_datetime: str | None = Field(
        default=None,
        description="Custom range end in ISO-8601 format.",
    )
    time_step_minutes: int | None = Field(
        default=None, ge=1,
        description="Time step in minutes for custom range analysis.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "latitude": 47.3769,
                    "longitude": 8.5417,
                    "timezone": "Europe/Zurich",
                    "coordinate_system": "y_up",
                    "year": 2024,
                    "time_step_hours": 1,
                },
                {
                    "latitude": 40.7128,
                    "longitude": -74.006,
                    "timezone": "America/New_York",
                    "start_datetime": "2024-06-21T06:00:00",
                    "end_datetime": "2024-06-21T20:00:00",
                    "time_step_minutes": 30,
                },
            ]
        }
    }


class AnalysisOptions(BaseModel):
    """Tuning knobs for the raycast computation."""
    offset_distance: float = Field(
        default=0.001,
        description="Ray origin offset along surface normal to avoid "
                    "self-intersection (metres).",
    )
    chunk_size: int = Field(
        default=1000, ge=0,
        description="Max sun directions per GPU kernel launch. "
                    "0 = no chunking (process all at once). "
                    "Tune based on available VRAM.",
    )
    use_backface_culling: bool = Field(
        default=True,
        description="Treat back-facing surfaces as occluded.",
    )
    return_colors: bool = Field(
        default=True,
        description="Include pre-computed RGB gradient colors in response.",
    )
    color_gradient: list[list[int]] | None = Field(
        default=None,
        description="Custom gradient as list of [R,G,B] (0-255). "
                    "Needs at least 2 colours.  "
                    "Defaults to a blue-to-red sunlight gradient.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "offset_distance": 0.001,
                    "chunk_size": 1000,
                    "use_backface_culling": True,
                    "return_colors": True,
                    "color_gradient": None,
                }
            ]
        }
    }


class AnalyzeVerticesRequest(BaseModel):
    """Request body for ``POST /analyze/vertices``."""
    mesh: MeshData
    sun: SunConfig
    options: AnalysisOptions = Field(default_factory=AnalysisOptions)


class AnalyzeFacesRequest(BaseModel):
    """Request body for ``POST /analyze/faces``."""
    mesh: MeshData
    sun: SunConfig
    options: AnalysisOptions = Field(default_factory=AnalysisOptions)


class AnalyzeVerticesSeparateRequest(BaseModel):
    """Request body for ``POST /analyze/vertices/separate``.

    ``blocking_mesh`` provides occlusion geometry.
    ``target_points`` / ``target_normals`` define where analysis is computed.
    """
    blocking_mesh: MeshData
    target_points: list[list[float]] = Field(
        ...,
        description="Target vertex positions [[x,y,z], ...].",
        min_length=1,
    )
    target_normals: list[list[float]] = Field(
        ...,
        description="Target vertex normals [[nx,ny,nz], ...].",
        min_length=1,
    )
    sun: SunConfig
    options: AnalysisOptions = Field(default_factory=AnalysisOptions)


class AnalyzeFacesSeparateRequest(BaseModel):
    """Request body for ``POST /analyze/faces/separate``.

    ``blocking_mesh`` provides occlusion geometry.
    ``target_mesh`` faces are where sunlight analysis is computed.
    """
    blocking_mesh: MeshData
    target_mesh: MeshData
    sun: SunConfig
    options: AnalysisOptions = Field(default_factory=AnalysisOptions)


class SunVectorsRequest(BaseModel):
    """Request body for ``POST /sun-vectors``."""
    sun: SunConfig
