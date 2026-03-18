"""
Tests for GET /health endpoint.
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_schema():
    response = client.get("/health")
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"
    assert "gpu_available" in data
    assert "warp_initialized" in data
    assert isinstance(data["gpu_available"], bool)
    assert isinstance(data["warp_initialized"], bool)


if __name__ == "__main__":
    test_health_returns_200()
    test_health_response_schema()
    print("All /health tests passed.")
