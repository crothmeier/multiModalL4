import logging

import httpx
from fastapi import APIRouter, HTTPException, Request
from route_map import resolve

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/{path:path}")
async def proxy(path: str, request: Request):
    try:
        body = await request.json()
        has_image = "image" in body or any(
            isinstance(m.get("content"), list)
            and any(c.get("type") == "image" for c in m.get("content", []))
            for m in body.get("messages", [])
        )

        model = body.get("model", "mistral")
        target = resolve(model, has_image)

        logger.info(f"Routing {model} (image={has_image}) to {target}")

        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{target}/v1{path}",
                json=body,
                headers={
                    k: v
                    for k, v in request.headers.items()
                    if k.lower() not in ["host", "content-length"]
                },
            )
            return resp.json()
    except Exception as e:
        logger.error(f"Proxy error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
