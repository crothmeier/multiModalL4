"""Integration tests for path traversal prevention."""

import httpx
import pytest
from test_auth import create_test_token


@pytest.mark.integration
class TestPathValidation:
    """Test path validation in API Gateway."""

    BASE_URL = "http://localhost:8080"

    @pytest.fixture(autouse=True)
    def auth_headers(self):
        """Provide valid auth headers for all tests."""
        token = create_test_token()
        self.headers = {"Authorization": f"Bearer {token}"}

    def test_path_traversal_dots(self):
        """Path traversal with .. should be blocked."""
        paths = [
            "vision/completions/../admin",
            "chat/../../../etc/passwd",
            "code/completions/..",
            "../admin/secrets",
        ]

        for path in paths:
            response = httpx.post(
                f"{self.BASE_URL}/{path}", json={}, headers=self.headers
            )
            assert response.status_code == 400
            assert "Invalid path" in response.json()["detail"]

    def test_url_encoded_traversal(self):
        """URL encoded path traversal should be blocked."""
        paths = [
            "vision/completions/%2e%2e/admin",
            "chat/%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "code/%252e%252e/secrets",
        ]

        for path in paths:
            response = httpx.post(
                f"{self.BASE_URL}/{path}", json={}, headers=self.headers
            )
            assert response.status_code == 400

    def test_null_byte_injection(self):
        """Null byte injection should be blocked."""
        paths = [
            "chat/completions%00.txt",
            "vision/completions\x00admin",
        ]

        for path in paths:
            response = httpx.post(
                f"{self.BASE_URL}/{path}", json={}, headers=self.headers
            )
            assert response.status_code == 400

    def test_valid_paths(self):
        """Valid paths should be allowed through."""
        paths = [
            "chat/completions",
            "code/completions",
            "vision/completions",
            "chat/completions/stream",
        ]

        for path in paths:
            response = httpx.post(
                f"{self.BASE_URL}/{path}",
                json={"model": "test"},
                headers=self.headers,
                timeout=5.0,
            )
            # Should not be 400 (may be 404 or 503 if backend unavailable)
            assert response.status_code != 400
