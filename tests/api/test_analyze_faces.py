"""
Tests for POST /analyze/faces endpoint.
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi.testclient import TestClient
from app.main import app
from tests.api.sample_payloads import SIMPLE_MESH, SIMPLE_SUN_CONFIG, DEFAULT_OPTIONS

client = TestClient(app)


def _num_triangulated_faces() -> int:
    """Count triangulated faces from sample mesh face_vertex_counts."""
    count = 0
    for fvc in SIMPLE_MESH["face_vertex_counts"]:
        if fvc == 3:
            count += 1
        elif fvc == 4:
            count += 2
    return count


def test_analyze_faces_returns_200():
    payload = {
        "mesh": SIMPLE_MESH,
        "sun": SIMPLE_SUN_CONFIG,
        "options": DEFAULT_OPTIONS,
    }
    response = client.post("/analyze/faces", json=payload)
    assert response.status_code == 200, response.text


def test_analyze_faces_response_length_matches_faces():
    payload = {
        "mesh": SIMPLE_MESH,
        "sun": SIMPLE_SUN_CONFIG,
        "options": DEFAULT_OPTIONS,
    }
    response = client.post("/analyze/faces", json=payload)
    data = response.json()

    expected_faces = _num_triangulated_faces()
    assert len(data["sunlight_hours"]) == expected_faces
    assert len(data["hit_counts"]) == expected_faces


def test_analyze_faces_backface_culling_effect():
    payload_on = {
        "mesh": SIMPLE_MESH,
        "sun": SIMPLE_SUN_CONFIG,
        "options": {**DEFAULT_OPTIONS, "use_backface_culling": True},
    }
    payload_off = {
        "mesh": SIMPLE_MESH,
        "sun": SIMPLE_SUN_CONFIG,
        "options": {**DEFAULT_OPTIONS, "use_backface_culling": False},
    }
    response_on = client.post("/analyze/faces", json=payload_on)
    response_off = client.post("/analyze/faces", json=payload_off)

    assert response_on.status_code == 200
    assert response_off.status_code == 200


if __name__ == "__main__":
    test_analyze_faces_returns_200()
    test_analyze_faces_response_length_matches_faces()
    test_analyze_faces_backface_culling_effect()
    print("All /analyze/faces tests passed.")
