"""API Gateway for multimodal LLM services."""

import os
import sys
from typing import Dict, Set

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

# Add parent directory to path to import route_map
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.security import get_current_user  # noqa: E402
from app.utils import validate_path  # noqa: E402
from route_map import resolve  # noqa: E402

# Parse routing table from environment (kept for backwards compatibility)
ROUTES_RAW = os.getenv("ROUTES", "")
ROUTES: Dict[str, str] = {}
for line in ROUTES_RAW.strip().splitlines():
    if "->" in line:
        route, target = line.split("->", 1)
        ROUTES[route.strip()] = target.strip()

# Headers to pass through from backend responses
ALLOWED_RESPONSE_HEADERS: Set[str] = {
    "content-type",
    "content-length",
    "x-request-id",
    "x-processing-time",
}

app = FastAPI(title="Multimodal LLM Gateway", version="0.1.0")


@app.get("/healthz")
async def health_check():
    """Health check endpoint."""
    return JSONResponse(content={"status": "ok"})


@app.post("/{path:path}")
async def proxy_request(path: str, request: Request):
    """Proxy requests to appropriate backend based on routing table."""

    # Validate path to prevent traversal attacks
    if not validate_path(path):
        raise HTTPException(
            status_code=400,
            detail="Invalid path: contains forbidden characters or patterns",
        )

    # JWT validation - required for all proxied requests
    await get_current_user(request)
    # User info available for logging/tracing if needed

    # Parse request body to check for model and image content
    try:
        body_bytes = await request.body()
        if body_bytes:
            import json

            body = json.loads(body_bytes)

            # Check for image content
            has_image = "image" in body or any(
                isinstance(m.get("content"), list)
                and any(c.get("type") == "image" for c in m.get("content", []))
                for m in body.get("messages", [])
            )

            # Get model name and resolve target
            model = body.get("model", "mistral")
            target = resolve(model, has_image)

            # Extract just the host:port from the target URL
            if target.startswith("http://"):
                target = target[7:]  # Remove http://
            if "/v1" in target:
                target = target.split("/v1")[0]
        else:
            # Fallback to old routing for non-JSON requests
            target = None
            for route_pattern, service_url in ROUTES.items():
                if path.startswith(route_pattern.strip("/")):
                    target = service_url
                    break

            if not target:
                raise HTTPException(
                    status_code=404, detail=f"No backend configured for route: /{path}"
                )
            body_bytes = b""
    except Exception:
        # Fallback to old routing
        target = None
        for route_pattern, service_url in ROUTES.items():
            if path.startswith(route_pattern.strip("/")):
                target = service_url
                break

        if not target:
            raise HTTPException(
                status_code=404, detail=f"No backend configured for route: /{path}"
            )

    # Forward request
    headers = dict(request.headers)
    headers.pop("host", None)  # Remove host header

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            # Ensure path starts with /v1 if it doesn't already
            if not path.startswith("v1/") and not path.startswith("/v1/"):
                path = f"v1/{path}"
            elif path.startswith("/"):
                path = path[1:]  # Remove leading slash

            response = await client.request(
                method=request.method,
                url=f"http://{target}/{path}",
                content=body_bytes,
                headers=headers,
            )

            # Filter response headers
            filtered_headers = {
                k: v
                for k, v in response.headers.items()
                if k.lower() in ALLOWED_RESPONSE_HEADERS
            }
            # Add backend identifier for tracing
            filtered_headers["x-backend"] = target.split(":")[0]

            return JSONResponse(
                content=response.json(),
                status_code=response.status_code,
                headers=filtered_headers,
            )
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=503, detail=f"Backend unavailable: {str(e)}"
            )
