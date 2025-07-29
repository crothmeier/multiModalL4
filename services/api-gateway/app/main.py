"""API Gateway for multimodal LLM services."""
import os
from typing import Dict, Optional, Set

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

# Parse routing table from environment
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


async def validate_jwt(request: Request) -> Optional[Dict]:
    """JWT validation stub - implement with proper verification."""
    # TODO: Implement actual JWT validation
    # Expected: Extract Bearer token, verify signature, check 'aud' claim
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        # Stub: In production, decode and verify JWT here
        return {"sub": "user123", "aud": "llm-platform"}
    return None


@app.post("/{path:path}")
async def proxy_request(path: str, request: Request):
    """Proxy requests to appropriate backend based on routing table."""
    # Find matching route
    target = None
    for route_pattern, service_url in ROUTES.items():
        if path.startswith(route_pattern.strip("/")):
            target = service_url
            break
    
    if not target:
        raise HTTPException(status_code=404, detail=f"No backend configured for route: /{path}")
    
    # JWT validation (stub for now)
    jwt_claims = await validate_jwt(request)
    
    # Forward request
    body = await request.body()
    headers = dict(request.headers)
    headers.pop("host", None)  # Remove host header
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.request(
                method=request.method,
                url=f"http://{target}/{path}",
                content=body,
                headers=headers,
            )
            
            # Filter response headers
            filtered_headers = {
                k: v for k, v in response.headers.items()
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
            raise HTTPException(status_code=503, detail=f"Backend unavailable: {str(e)}")