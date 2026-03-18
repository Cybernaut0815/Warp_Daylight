"""
Tests for POST /analyze/vertices endpoint.
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi.testclient import TestClient
from app.main import app
from tests.api.sample_payloads import SIMPLE_MESH, SIMPLE_SUN_CONFIG, DEFAULT_OPTIONS

client = TestClient(app)


def test_analyze_vertices_returns_200():
    payload = {
        "mesh": SIMPLE_MESH,
        "sun": SIMPLE_SUN_CONFIG,
        "options": DEFAULT_OPTIONS,
    }
    response = client.post("/analyze/vertices", json=payload)
    assert response.status_code == 200, response.text


def test_analyze_vertices_response_schema():
    payload = {
        "mesh": SIMPLE_MESH,
        "sun": SIMPLE_SUN_CONFIG,
        "options": DEFAULT_OPTIONS,
    }
    response = client.post("/analyze/vertices", json=payload)
    data = response.json()

    assert "sunlight_hours" in data
    assert "hit_counts" in data
    assert "total_sun_positions" in data
    assert "max_possible_hours" in data
    assert "colors" in data
    assert "metadata" in data

    num_vertices = len(SIMPLE_MESH["points"])
    assert len(data["sunlight_hours"]) == num_vertices
    assert len(data["hit_counts"]) == num_vertices
    assert data["total_sun_positions"] > 0
    assert data["max_possible_hours"] > 0

    if data["colors"] is not None:
        assert len(data["colors"]) == num_vertices
        assert all(len(c) == 3 for c in data["colors"])


def test_analyze_vertices_without_options():
    payload = {
        "mesh": SIMPLE_MESH,
        "sun": SIMPLE_SUN_CONFIG,
    }
    response = client.post("/analyze/vertices", json=payload)
    assert response.status_code == 200, response.text


def test_analyze_vertices_no_colors():
    payload = {
        "mesh": SIMPLE_MESH,
        "sun": SIMPLE_SUN_CONFIG,
        "options": {**DEFAULT_OPTIONS, "return_colors": False},
    }
    response = client.post("/analyze/vertices", json=payload)
    data = response.json()
    assert data["colors"] is None


def test_analyze_vertices_invalid_mesh_rejected():
    bad_mesh = {**SIMPLE_MESH, "face_vertex_indices": [0, 1, 999, 0, 2, 3]}
    payload = {"mesh": bad_mesh, "sun": SIMPLE_SUN_CONFIG}
    response = client.post("/analyze/vertices", json=payload)
    assert response.status_code == 422


if __name__ == "__main__":
    test_analyze_vertices_returns_200()
    test_analyze_vertices_response_schema()
    test_analyze_vertices_without_options()
    test_analyze_vertices_no_colors()
    test_analyze_vertices_invalid_mesh_rejected()
    print("All /analyze/vertices tests passed.")
