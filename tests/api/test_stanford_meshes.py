"""
Integration test: Stanford Bunny and Stanford Dragon against the Daylight API.

Prerequisites:
  - The API server must be running (start_server.bat / start_server.ps1).
  - Internet access is needed on the first run to download meshes; subsequent
    runs use the cached files in data/meshes/.
  - If download fails, synthetic meshes are generated as fallback.

Usage:
    python tests/api/test_stanford_meshes.py [--base-url http://localhost:8000]
"""
import sys
import io
import gzip
import time
import argparse
import statistics
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import trimesh
import requests as http_requests

MESH_CACHE_DIR = _PROJECT_ROOT / "data" / "meshes"

MESH_SOURCES = {
    "bunny": {
        "url": "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/stanford-bunny.obj",
        "filename": "stanford_bunny.obj",
    },
    "dragon": {
        "url": "https://raw.githubusercontent.com/hughsk/stanford-dragon/master/models/dragon_vrip.ply.gz",
        "filename": "stanford_dragon.ply.gz",
    },
}

SUN_CONFIG_SHORT = {
    "latitude": 40.7128,
    "longitude": -74.0060,
    "timezone": "America/New_York",
    "coordinate_system": "y_up",
    "start_datetime": "2024-06-21T06:00:00",
    "end_datetime": "2024-06-21T20:00:00",
    "time_step_minutes": 60,
}

OPTIONS = {
    "offset_distance": 0.001,
    "chunk_size": 1000,
    "use_backface_culling": True,
    "return_colors": True,
}


def _generate_fallback_mesh(name: str) -> trimesh.Trimesh:
    """Generate a synthetic mesh when download is unavailable."""
    if name == "bunny":
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=0.1)
        print(f"  Generated fallback icosphere for '{name}' "
              f"({len(mesh.vertices)} verts, {len(mesh.faces)} faces)")
    else:
        mesh = trimesh.creation.icosphere(subdivisions=5, radius=0.15)
        print(f"  Generated fallback icosphere for '{name}' "
              f"({len(mesh.vertices)} verts, {len(mesh.faces)} faces)")
    return mesh


def download_mesh(name: str) -> trimesh.Trimesh:
    """Download (or load from cache) a Stanford mesh and return as trimesh."""
    info = MESH_SOURCES[name]
    cache_path = MESH_CACHE_DIR / info["filename"]

    decompressed_path = MESH_CACHE_DIR / info["filename"].replace(".gz", "")

    if decompressed_path.exists():
        print(f"  Loading cached {name} from {decompressed_path}")
        mesh = trimesh.load(str(decompressed_path))
    elif cache_path.exists():
        print(f"  Loading cached {name} from {cache_path}")
        mesh = trimesh.load(str(cache_path))
    else:
        MESH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            print(f"  Downloading {name} from {info['url']} ...")
            resp = http_requests.get(info["url"], timeout=60)
            resp.raise_for_status()
            raw = resp.content
            if info["filename"].endswith(".gz"):
                raw = gzip.decompress(raw)
                file_type = info["filename"].replace(".gz", "").split(".")[-1]
                save_path = decompressed_path
            else:
                file_type = info["filename"].split(".")[-1]
                save_path = cache_path
            mesh = trimesh.load(io.BytesIO(raw), file_type=file_type)
            mesh.export(str(save_path))
            print(f"  Saved to {save_path}")
        except Exception as exc:
            print(f"  Download failed ({exc}), using generated fallback mesh")
            mesh = _generate_fallback_mesh(name)

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    return mesh


def trimesh_to_mesh_payload(mesh: trimesh.Trimesh) -> dict:
    """Convert a trimesh object to the API's MeshData JSON structure."""
    return {
        "points": mesh.vertices.tolist(),
        "face_vertex_counts": [3] * len(mesh.faces),
        "face_vertex_indices": mesh.faces.flatten().tolist(),
        "normals": mesh.vertex_normals.tolist(),
    }


def print_mesh_stats(name: str, mesh: trimesh.Trimesh) -> None:
    print(f"  {name}: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
    bounds = mesh.bounds
    print(f"    Bounds: [{bounds[0][0]:.4f}, {bounds[0][1]:.4f}, {bounds[0][2]:.4f}] "
          f"-> [{bounds[1][0]:.4f}, {bounds[1][1]:.4f}, {bounds[1][2]:.4f}]")


def print_analysis_result(label: str, data: dict) -> None:
    hours = data["sunlight_hours"]
    meta = data["metadata"]
    print(f"  {label}:")
    print(f"    Elements analysed:  {meta['num_elements']:,}")
    print(f"    Sun positions:      {data['total_sun_positions']}")
    print(f"    Max possible hours: {data['max_possible_hours']:.1f}")
    print(f"    Sunlight hours  -- "
          f"mean: {statistics.mean(hours):.2f}, "
          f"min: {min(hours):.2f}, "
          f"max: {max(hours):.2f}")
    print(f"    Computation time:   {meta['computation_time_seconds']:.4f} s")
    if data.get("colors"):
        print(f"    Colors returned:    {len(data['colors']):,} RGB triplets")


def run_test(base_url: str) -> None:
    separator = "=" * 70

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------
    print(separator)
    print("HEALTH CHECK")
    print(separator)
    resp = http_requests.get(f"{base_url}/health", timeout=10)
    resp.raise_for_status()
    health = resp.json()
    print(f"  Status:           {health['status']}")
    print(f"  GPU available:    {health['gpu_available']}")
    print(f"  Warp initialized: {health['warp_initialized']}")
    assert health["status"] == "ok", "Server is not healthy"
    print()

    # ------------------------------------------------------------------
    # Download / load meshes
    # ------------------------------------------------------------------
    print(separator)
    print("LOADING MESHES")
    print(separator)
    bunny = download_mesh("bunny")
    dragon = download_mesh("dragon")
    print()
    print_mesh_stats("Bunny", bunny)
    print_mesh_stats("Dragon", dragon)

    bunny_payload = trimesh_to_mesh_payload(bunny)
    dragon_payload = trimesh_to_mesh_payload(dragon)
    print()

    # ------------------------------------------------------------------
    # Test 1: Bunny vertex analysis
    # ------------------------------------------------------------------
    print(separator)
    print("TEST 1: Bunny - Vertex Analysis")
    print(separator)
    t0 = time.perf_counter()
    resp = http_requests.post(
        f"{base_url}/analyze/vertices",
        json={"mesh": bunny_payload, "sun": SUN_CONFIG_SHORT, "options": OPTIONS},
        timeout=120,
    )
    wall_time = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    print_analysis_result("Bunny vertices", data)
    print(f"    Total wall time:    {wall_time:.2f} s")
    assert len(data["sunlight_hours"]) == len(bunny.vertices)
    print("    PASSED")
    print()

    # ------------------------------------------------------------------
    # Test 2: Bunny face analysis
    # ------------------------------------------------------------------
    print(separator)
    print("TEST 2: Bunny - Face Analysis")
    print(separator)
    t0 = time.perf_counter()
    resp = http_requests.post(
        f"{base_url}/analyze/faces",
        json={"mesh": bunny_payload, "sun": SUN_CONFIG_SHORT, "options": OPTIONS},
        timeout=120,
    )
    wall_time = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    print_analysis_result("Bunny faces", data)
    print(f"    Total wall time:    {wall_time:.2f} s")
    assert len(data["sunlight_hours"]) == len(bunny.faces)
    print("    PASSED")
    print()

    # ------------------------------------------------------------------
    # Test 3: Dragon vertex analysis
    # ------------------------------------------------------------------
    print(separator)
    print("TEST 3: Dragon - Vertex Analysis")
    print(separator)
    t0 = time.perf_counter()
    resp = http_requests.post(
        f"{base_url}/analyze/vertices",
        json={"mesh": dragon_payload, "sun": SUN_CONFIG_SHORT, "options": OPTIONS},
        timeout=300,
    )
    wall_time = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    print_analysis_result("Dragon vertices", data)
    print(f"    Total wall time:    {wall_time:.2f} s")
    assert len(data["sunlight_hours"]) == len(dragon.vertices)
    print("    PASSED")
    print()

    # ------------------------------------------------------------------
    # Test 4: Dragon face analysis
    # ------------------------------------------------------------------
    print(separator)
    print("TEST 4: Dragon - Face Analysis")
    print(separator)
    t0 = time.perf_counter()
    resp = http_requests.post(
        f"{base_url}/analyze/faces",
        json={"mesh": dragon_payload, "sun": SUN_CONFIG_SHORT, "options": OPTIONS},
        timeout=300,
    )
    wall_time = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    print_analysis_result("Dragon faces", data)
    print(f"    Total wall time:    {wall_time:.2f} s")
    assert len(data["sunlight_hours"]) == len(dragon.faces)
    print("    PASSED")
    print()

    # ------------------------------------------------------------------
    # Test 5: Separate -- Dragon blocks, Bunny is target
    # ------------------------------------------------------------------
    print(separator)
    print("TEST 5: Separate - Dragon (blocker) + Bunny (target faces)")
    print(separator)
    t0 = time.perf_counter()
    resp = http_requests.post(
        f"{base_url}/analyze/faces/separate",
        json={
            "blocking_mesh": dragon_payload,
            "target_mesh": bunny_payload,
            "sun": SUN_CONFIG_SHORT,
            "options": OPTIONS,
        },
        timeout=300,
    )
    wall_time = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    print_analysis_result("Dragon->Bunny separate", data)
    print(f"    Total wall time:    {wall_time:.2f} s")
    assert len(data["sunlight_hours"]) == len(bunny.faces)
    print("    PASSED")
    print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(separator)
    print("ALL TESTS PASSED")
    print(separator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Integration test: Stanford meshes against the Daylight API",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the running API server (default: http://localhost:8000)",
    )
    args = parser.parse_args()
    run_test(args.base_url)
