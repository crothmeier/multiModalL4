"""Integration tests for JWT authentication."""

import time

import httpx
import pytest
from jose import jwt

JWT_SECRET = "test-secret-key-for-integration-tests"
JWT_ALGORITHM = "HS256"
JWT_AUDIENCE = "llm-platform"
JWT_ISSUER = "multimodal-stack"


def create_test_token(sub="test-user", exp_delta=3600, **kwargs):
    """Create a test JWT token."""
    payload = {
        "sub": sub,
        "aud": JWT_AUDIENCE,
        "iss": JWT_ISSUER,
        "exp": int(time.time()) + exp_delta,
        **kwargs,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


@pytest.mark.integration
class TestJWTAuthentication:
    """Test JWT authentication in API Gateway."""

    BASE_URL = "http://localhost:8080"

    def test_missing_auth_header(self):
        """Requests without auth header should return 401."""
        response = httpx.post(f"{self.BASE_URL}/chat/completions", json={})
        assert response.status_code == 401
        assert "Authorization header" in response.json()["detail"]

    def test_invalid_token(self):
        """Invalid JWT should return 401."""
        headers = {"Authorization": "Bearer invalid.token.here"}
        response = httpx.post(
            f"{self.BASE_URL}/chat/completions", json={}, headers=headers
        )
        assert response.status_code == 401
        assert "Invalid authentication" in response.json()["detail"]

    def test_expired_token(self):
        """Expired JWT should return 401."""
        token = create_test_token(exp_delta=-3600)  # Expired 1 hour ago
        headers = {"Authorization": f"Bearer {token}"}
        response = httpx.post(
            f"{self.BASE_URL}/chat/completions", json={}, headers=headers
        )
        assert response.status_code == 401

    def test_valid_token(self):
        """Valid JWT should allow request through."""
        token = create_test_token()
        headers = {"Authorization": f"Bearer {token}"}
        response = httpx.post(
            f"{self.BASE_URL}/chat/completions",
            json={
                "model": "/models/mistral-awq",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10,
            },
            headers=headers,
            timeout=30.0,
        )
        # Should either succeed or fail with backend error, not auth error
        assert response.status_code != 401

    def test_wrong_audience(self):
        """JWT with wrong audience should return 401."""
        token = create_test_token(aud="wrong-audience")
        headers = {"Authorization": f"Bearer {token}"}
        response = httpx.post(
            f"{self.BASE_URL}/chat/completions", json={}, headers=headers
        )
        assert response.status_code == 401
