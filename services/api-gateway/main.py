import asyncio
import logging
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

ORCHESTRATOR_URL = "http://model-orchestrator:8888"
LLM_URL = "http://host.docker.internal:8000"


async def ensure_model_loaded(model: str) -> bool:
    """Ensure the requested model is loaded via orchestrator."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{ORCHESTRATOR_URL}/ensure_model",
                json={"model": model},
                timeout=60.0,  # Model swaps can take time
            )
            result = response.json()
            if result.get("swapped"):
                logger.info(f"Model swapped to {model}")
                # Wait a bit for model to stabilize
                await asyncio.sleep(2)
            return True
        except Exception as e:
            logger.error(f"Failed to ensure model: {e}")
            raise HTTPException(status_code=503, detail="Model orchestration failed")


async def stream_response(response: httpx.Response) -> AsyncGenerator[bytes, None]:
    """Stream response from LLM."""
    async for chunk in response.aiter_bytes():
        yield chunk


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
@app.post("/v1/completions")
async def proxy_completions(request: Request):
    try:
        body = await request.json()
        model = body.get("model", "mistral")

        # Ensure model is loaded
        await ensure_model_loaded(model)

        # Proxy to LLM
        async with httpx.AsyncClient() as client:
            if body.get("stream", False):
                # Use stream() method for streaming responses
                async with client.stream(
                    "POST",
                    f"{LLM_URL}/v1/chat/completions",
                    json=body,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    timeout=300.0,
                ) as response:
                    return StreamingResponse(
                        stream_response(response), media_type="text/event-stream"
                    )
            else:
                # Use regular post for non-streaming
                response = await client.post(
                    f"{LLM_URL}/v1/chat/completions",
                    json=body,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    timeout=300.0,
                )
                return JSONResponse(content=response.json())

    except httpx.ReadTimeout:
        raise HTTPException(status_code=504, detail="LLM request timeout")
    except Exception as e:
        logger.error(f"Proxy error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    # Check orchestrator health
    try:
        async with httpx.AsyncClient() as client:
            orch_response = await client.get(f"{ORCHESTRATOR_URL}/health", timeout=5.0)
            orch_health = orch_response.json()

            # Check current model if any
            status_response = await client.get(
                f"{ORCHESTRATOR_URL}/status", timeout=5.0
            )
            status = status_response.json()

            return {
                "gateway": "healthy",
                "orchestrator": orch_health.get("status", "unknown"),
                "current_model": status.get("current_model"),
                "loading": status.get("loading", False),
            }
    except Exception:
        return JSONResponse(
            content={"gateway": "healthy", "orchestrator": "unreachable"},
            status_code=503,
        )


@app.get("/status")
async def status():
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{ORCHESTRATOR_URL}/status")
        return response.json()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)  # nosec B104
