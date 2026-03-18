"""
Router for the /sun-vectors utility endpoint.
"""
from fastapi import APIRouter

from app.models.requests import SunVectorsRequest
from app.models.responses import SunVectorsResponse, SunVectorEntry
from app.services.analysis import compute_sun_vectors

router = APIRouter(tags=["sun"])


@router.post(
    "/sun-vectors",
    response_model=SunVectorsResponse,
    summary="Get sun direction vectors",
    description=(
        "Utility endpoint that returns sun direction vectors for a given "
        "location and time configuration.  Useful for previewing / debugging "
        "which sun positions will be used in an analysis."
    ),
)
async def sun_vectors(body: SunVectorsRequest) -> SunVectorsResponse:
    light_dirs, timestamps = compute_sun_vectors(body.sun)

    entries = [
        SunVectorEntry(
            direction=light_dirs[i].tolist(),
            timestamp=timestamps[i].isoformat(),
        )
        for i in range(len(light_dirs))
    ]

    return SunVectorsResponse(
        vectors=entries,
        total_count=len(entries),
        coordinate_system=body.sun.coordinate_system,
    )
