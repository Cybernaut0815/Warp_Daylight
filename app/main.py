"""
FastAPI application entry point for the Annual Daylight Analysis API.

Start the server with:
    python -m app.main          (uses settings from .env / environment)
    uvicorn app.main:app        (manual override via CLI flags)
"""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("analyze")

from app.core.config import settings
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

@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    if not request.url.path.startswith("/analyze"):
        return await call_next(request)

    t_start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t_start) * 1000
    print(
        f"[TIMING] {request.method} {request.url.path}  FULL REQUEST: {elapsed_ms:.1f} ms"
        f"  (includes JSON parsing + Pydantic validation + service + serialization)",
        flush=True,
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
