"""
FastAPI application entry point for the Annual Daylight Analysis API.

Start the server with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.models.responses import HealthResponse
from app.routers import analyze, sun

_warp_ready = False
_gpu_available = False


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Initialize NVIDIA Warp once at server startup."""
    global _warp_ready, _gpu_available
    try:
        import warp as wp
        wp.init()
        _warp_ready = True
        _gpu_available = wp.is_cuda_available()
    except Exception:
        _warp_ready = False
        _gpu_available = False
    yield


app = FastAPI(
    title="Annual Daylight Analysis API",
    version="0.1.0",
    description=(
        "GPU-accelerated direct sunlight analysis for 3D meshes.\n\n"
        "Send mesh data in a USD-like format (points, faceVertexCounts, "
        "faceVertexIndices) and receive per-vertex or per-face sunlight "
        "hours plus optional RGB colours for visualisation in any DCC tool "
        "(Maya, Blender, Omniverse, Rhino, etc.)."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router)
app.include_router(sun.router)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Health check",
    description="Returns service health, GPU availability, and Warp init status.",
)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        gpu_available=_gpu_available,
        warp_initialized=_warp_ready,
    )
