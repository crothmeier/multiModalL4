import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Optional

import docker
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
docker_client = docker.from_env()


class RequestQueue:
    def __init__(self, max_wait_time=45):
        self.pending_requests = asyncio.Queue()
        self.max_wait_time = max_wait_time
        self.loading = False
        self.target_model = None
        self.swap_complete_event = asyncio.Event()

    async def wait_for_model(self, model_name):
        """Queue requests during swap, with timeout"""
        if self.loading and self.target_model == model_name:
            try:
                await asyncio.wait_for(
                    self.swap_complete_event.wait(), timeout=self.max_wait_time
                )
            except asyncio.TimeoutError:
                raise HTTPException(503, "Model swap timeout")


class ModelRequest(BaseModel):
    model: str


class ModelStatus(BaseModel):
    current_model: Optional[str]
    loading: bool
    last_swap: Optional[datetime]
    available_models: list


class ModelOrchestrator:
    def __init__(self):
        self.current_model: Optional[str] = None
        self.loading = False
        self.last_swap: Optional[datetime] = None
        self.swap_lock = asyncio.Lock()
        self.request_queue = RequestQueue()
        self.swap_complete_event = asyncio.Event()

        # Model configurations
        self.models = {
            "mistral": {
                "service_name": "mistral-llm",
                "container_name": "multimodal-stack-mistral-llm-1",
                "health_url": "http://mistral-llm:8000/v1/models",
                "gpu_memory": 7,
                "startup_time": 15,
            },
            "llava": {
                "service_name": "llava-llm",
                "container_name": "multimodal-stack-llava-llm-1",
                "health_url": "http://llava-llm:8000/v1/models",
                "gpu_memory": 20,
                "startup_time": 25,
            },
            "coder": {
                "service_name": "coder-llm",
                "container_name": "multimodal-stack-coder-llm-1",
                "health_url": "http://coder-llm:8000/v1/models",
                "gpu_memory": 11,
                "startup_time": 20,
            },
            "deepseek": {  # Alias for coder
                "service_name": "coder-llm",
                "container_name": "multimodal-stack-coder-llm-1",
                "health_url": "http://coder-llm:8000/v1/models",
                "gpu_memory": 11,
                "startup_time": 20,
            },
        }

    async def get_status(self) -> ModelStatus:
        return ModelStatus(
            current_model=self.current_model,
            loading=self.loading,
            last_swap=self.last_swap,
            available_models=list(set(m["service_name"] for m in self.models.values())),
        )

    async def ensure_model(self, model_name: str) -> bool:
        """Ensure the requested model is loaded. Returns True if swap occurred."""
        model_name = model_name.lower()

        # Map model name to service
        if model_name not in self.models:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

        target_service = self.models[model_name]["service_name"]

        # Check if already loaded
        if self.current_model == target_service:
            logger.info(f"Model {target_service} already loaded")
            return False

        # Perform swap
        async with self.swap_lock:
            if self.current_model == target_service:  # Double-check
                return False

            await self._swap_model(target_service)
            return True

    async def _swap_model(self, target_service: str):
        """Perform the actual model swap."""
        self.loading = True
        self.request_queue.loading = True
        self.request_queue.target_model = target_service
        self.swap_complete_event.clear()
        self.request_queue.swap_complete_event.clear()
        start_time = time.time()

        try:
            logger.info(
                f"Starting model swap: {self.current_model} -> {target_service}"
            )

            # Stop current model
            if self.current_model:
                logger.info(f"Stopping {self.current_model}")
                try:
                    container = docker_client.containers.get(
                        self.models[self.current_model]["container_name"]
                    )
                    container.stop(timeout=10)
                except docker.errors.NotFound:
                    logger.warning(f"Container {self.current_model} not found")
                except Exception as e:
                    logger.error(f"Error stopping {self.current_model}: {e}")

            # Wait for GPU memory to clear
            await asyncio.sleep(2)

            # Start target model
            logger.info(f"Starting {target_service}")
            docker_client.api.start(self.models[target_service]["container_name"])

            # Wait for health
            health_url = self.models[target_service]["health_url"]
            max_wait = self.models[target_service]["startup_time"] + 10

            async with httpx.AsyncClient() as client:
                for i in range(max_wait):
                    try:
                        response = await client.get(health_url, timeout=2.0)
                        if response.status_code == 200:
                            logger.info(f"Model {target_service} is healthy")
                            break
                    except:
                        pass
                    await asyncio.sleep(1)
                else:
                    raise Exception(f"Model {target_service} failed to become healthy")

            self.current_model = target_service
            self.last_swap = datetime.now()

            elapsed = time.time() - start_time
            logger.info(f"Model swap completed in {elapsed:.1f}s")

        finally:
            self.loading = False
            self.request_queue.loading = False
            self.swap_complete_event.set()
            self.request_queue.swap_complete_event.set()

    async def initialize(self):
        """Initialize orchestrator by checking which model is currently running."""
        for model_name, config in self.models.items():
            try:
                container = docker_client.containers.get(config["container_name"])
                if container.status == "running":
                    # Verify it's healthy
                    async with httpx.AsyncClient() as client:
                        response = await client.get(config["health_url"], timeout=5.0)
                        if response.status_code == 200:
                            self.current_model = config["service_name"]
                            logger.info(f"Found running model: {self.current_model}")
                            break
            except:
                continue


# Initialize orchestrator
orchestrator = ModelOrchestrator()


@app.on_event("startup")
async def startup():
    await orchestrator.initialize()


@app.get("/status")
async def get_status():
    return await orchestrator.get_status()


@app.post("/ensure_model")
async def ensure_model(request: ModelRequest):
    # Wait if a swap is in progress for this model
    await orchestrator.request_queue.wait_for_model(request.model)

    swapped = await orchestrator.ensure_model(request.model)
    return {
        "model": request.model,
        "swapped": swapped,
        "current_model": orchestrator.current_model,
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "loading": orchestrator.loading}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8888)
