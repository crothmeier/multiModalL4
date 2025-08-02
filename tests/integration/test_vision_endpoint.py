"""Integration tests for vision/multimodal endpoint."""

import base64

import httpx
import pytest
from test_auth import create_test_token


@pytest.mark.integration
class TestVisionEndpoint:
    """Test multimodal vision endpoint with LLaVA."""

    BASE_URL = "http://localhost:8080"

    @pytest.fixture(autouse=True)
    def auth_headers(self):
        """Provide valid auth headers for all tests."""
        token = create_test_token()
        self.headers = {"Authorization": f"Bearer {token}"}

    def test_vision_completion_basic(self):
        """Test basic image captioning with LLaVA."""
        # Create a tiny 2x2 red square PNG
        png_data = (
            "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAADklEQVQI12P4z8DwHwAF"
            "AAH/q0m+CAAAAABJRU5ErkJggg=="
        )
        tiny_png = base64.b64decode(png_data)
        image_b64 = base64.b64encode(tiny_png).decode()

        response = httpx.post(
            f"{self.BASE_URL}/vision/completions",
            json={
                "model": "/models/llava-7b",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this image in one sentence.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}"
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 50,
            },
            headers=self.headers,
            timeout=60.0,
        )

        assert response.status_code == 200
        result = response.json()
        assert "choices" in result
        assert len(result["choices"]) > 0
        assert len(result["choices"][0]["message"]["content"]) > 0
