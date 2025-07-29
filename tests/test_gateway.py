"""Tests for API gateway."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client."""
    import sys
    sys.path.append("services/api-gateway/app")
    from main import app
    return TestClient(app)


def test_healthz(client):
    """Test health check endpoint."""
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_no_route(client):
    """Test request to unmapped route."""
    response = client.post("/unknown/path")
    assert response.status_code == 404