"""
Tests for POST /sun-vectors endpoint.
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi.testclient import TestClient
from app.main import app
from tests.api.sample_payloads import SIMPLE_SUN_CONFIG

client = TestClient(app)


def test_sun_vectors_returns_200():
    payload = {"sun": SIMPLE_SUN_CONFIG}
    response = client.post("/sun-vectors", json=payload)
    assert response.status_code == 200, response.text


def test_sun_vectors_response_schema():
    payload = {"sun": SIMPLE_SUN_CONFIG}
    response = client.post("/sun-vectors", json=payload)
    data = response.json()

    assert "vectors" in data
    assert "total_count" in data
    assert "coordinate_system" in data
    assert data["total_count"] > 0
    assert data["total_count"] == len(data["vectors"])
    assert data["coordinate_system"] == SIMPLE_SUN_CONFIG["coordinate_system"]

    entry = data["vectors"][0]
    assert "direction" in entry
    assert "timestamp" in entry
    assert len(entry["direction"]) == 3


def test_sun_vectors_annual_mode():
    payload = {
        "sun": {
            "latitude": 40.7128,
            "longitude": -74.0060,
            "timezone": "America/New_York",
            "coordinate_system": "y_up",
            "year": 2024,
            "time_step_hours": 1,
        }
    }
    response = client.post("/sun-vectors", json=payload)
    data = response.json()
    assert response.status_code == 200
    assert data["total_count"] > 100  # a full year has thousands of daylight hours


def test_sun_vectors_z_up():
    payload = {
        "sun": {
            **SIMPLE_SUN_CONFIG,
            "coordinate_system": "z_up",
        }
    }
    response = client.post("/sun-vectors", json=payload)
    data = response.json()
    assert data["coordinate_system"] == "z_up"


if __name__ == "__main__":
    test_sun_vectors_returns_200()
    test_sun_vectors_response_schema()
    test_sun_vectors_annual_mode()
    test_sun_vectors_z_up()
    print("All /sun-vectors tests passed.")
