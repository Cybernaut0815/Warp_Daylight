"""
Tests for POST /analyze/vertices/separate and /analyze/faces/separate endpoints.
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi.testclient import TestClient
from app.main import app
from tests.api.sample_payloads import SIMPLE_MESH, SIMPLE_SUN_CONFIG, DEFAULT_OPTIONS

client = TestClient(app)


def test_vertices_separate_returns_200():
    payload = {
        "blocking_mesh": SIMPLE_MESH,
        "target_points": [[0.5, 0.1, 0.5]],
        "target_normals": [[0.0, 1.0, 0.0]],
        "sun": SIMPLE_SUN_CONFIG,
        "options": DEFAULT_OPTIONS,
    }
    response = client.post("/analyze/vertices/separate", json=payload)
    assert response.status_code == 200, response.text


def test_vertices_separate_response_length():
    target_pts = [[0.5, 0.1, 0.5], [0.2, 0.1, 0.2]]
    target_nrm = [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    payload = {
        "blocking_mesh": SIMPLE_MESH,
        "target_points": target_pts,
        "target_normals": target_nrm,
        "sun": SIMPLE_SUN_CONFIG,
        "options": DEFAULT_OPTIONS,
    }
    response = client.post("/analyze/vertices/separate", json=payload)
    data = response.json()
    assert len(data["sunlight_hours"]) == len(target_pts)
    assert len(data["hit_counts"]) == len(target_pts)


def test_faces_separate_returns_200():
    payload = {
        "blocking_mesh": SIMPLE_MESH,
        "target_mesh": SIMPLE_MESH,
        "sun": SIMPLE_SUN_CONFIG,
        "options": DEFAULT_OPTIONS,
    }
    response = client.post("/analyze/faces/separate", json=payload)
    assert response.status_code == 200, response.text


def test_faces_separate_response_schema():
    payload = {
        "blocking_mesh": SIMPLE_MESH,
        "target_mesh": SIMPLE_MESH,
        "sun": SIMPLE_SUN_CONFIG,
        "options": DEFAULT_OPTIONS,
    }
    response = client.post("/analyze/faces/separate", json=payload)
    data = response.json()
    assert "sunlight_hours" in data
    assert "metadata" in data
    assert data["metadata"]["mesh_vertex_count"] == len(SIMPLE_MESH["points"])


if __name__ == "__main__":
    test_vertices_separate_returns_200()
    test_vertices_separate_response_length()
    test_faces_separate_returns_200()
    test_faces_separate_response_schema()
    print("All /analyze/*/separate tests passed.")
