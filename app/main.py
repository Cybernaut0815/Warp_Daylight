"""
FastAPI application entry point for the Annual Daylight Analysis API.

Start the server with::

    python -m app.main          # uses settings from .env / environment
    uvicorn app.main:app        # manual override via CLI flags
"""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# TODO(roadmap): Create UIs for different softwares (Maya, Rhino, Blender,
#   Omniverse, ...).  DCC-specific client plugins would consume this API via
#   the /analyze and /sun-vectors endpoints.  CORS is already permissive
#   ("*") so browser-based UIs will work out of the box.

from app.core.config import settings
from app.models.responses import HealthResponse
from app.routers import analyze, sun

# ---------------------------------------------------------------------------
# Logging configuration — ensures timing & debug messages reach the console
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

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
        logger.info("Warp initialized — CUDA available: %s", _gpu_available)
    except Exception:
        _warp_ready = False
        _gpu_available = False
        logger.warning("Warp initialization failed — running without GPU")
    yield


app = FastAPI(
    title="Annual Daylight Analysis API",
    version="0.1.0",
    description=(
        "GPU-accelerated direct sunlight analysis for 3D meshes.\n\n"
        "Send mesh data in a USD-like format (points, faceVertexCounts, "
        "faceVertexIndices) and receive per-vertex or per-face sunlight "
        "hours plus optional RGB colours for visualisation in any DCC tool "
        "(Maya, Blender, Omniverse, Rhino, etc.).\n\n"
        "## Endpoint families\n\n"
        "| Family | Input | Best for |\n"
        "|--------|-------|----------|\n"
        "| `/analyze/*` (JSON) | Pydantic request body | Small-medium meshes |\n"
        "| `/analyze/*/numpy` | NPZ file upload | Large meshes (>50k verts) |\n\n"
        "All endpoints accept `?response_format=npz` to return a binary NPZ "
        "archive instead of JSON.\n"
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


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    """Log wall-clock time for every ``/analyze`` request."""
    if not request.url.path.startswith("/analyze"):
        return await call_next(request)

    t_start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        "[TIMING] %s %s  FULL REQUEST: %.1f ms "
        "(includes JSON parsing + Pydantic validation + service + serialization)",
        request.method, request.url.path, elapsed_ms,
    )
    return response


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
    """Return current service health status."""
    return HealthResponse(
        status="ok",
        gpu_available=_gpu_available,
        warp_initialized=_warp_ready,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level,
    )
