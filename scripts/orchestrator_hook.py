#!/usr/bin/env python3
"""
Integration hooks for Docker orchestrator at phx-ai20
Provides seamless integration with existing infrastructure
"""

import asyncio
import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp
import jwt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ServiceRegistration:
    """Service registration data for orchestrator"""

    service_name: str
    service_type: str
    endpoint: str
    port: int
    health_check_path: str
    capabilities: List[str]
    gpu_required: bool
    gpu_memory_gb: float
    metadata: Dict[str, Any]


class OrchestratorClient:
    """Client for interacting with Docker orchestrator"""

    def __init__(self):
        self.orchestrator_url = os.getenv("ORCHESTRATOR_URL", "http://phx-ai20:8888")
        self.service_name = os.getenv("SERVICE_NAME", "llava-fp8")
        self.jwt_secret = os.getenv("JWT_SECRET_KEY", "")
        self.api_key = os.getenv("ORCHESTRATOR_API_KEY", "")

        # Service configuration
        self.triton_endpoint = os.getenv("TRITON_ENDPOINT", "http://localhost:8000")
        self.gateway_endpoint = os.getenv("GATEWAY_ENDPOINT", "http://localhost:8888")

    def generate_jwt_token(self) -> str:
        """Generate JWT token for authentication"""
        payload = {
            "service": self.service_name,
            "timestamp": datetime.now().isoformat(),
            "capabilities": ["llava", "multimodal", "fp8", "tensorrt"],
        }

        token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
        return token

    async def register_service(self) -> bool:
        """Register LLaVA FP8 service with orchestrator"""
        logger.info("Registering service with orchestrator...")

        registration = ServiceRegistration(
            service_name="llava-fp8",
            service_type="inference",
            endpoint=self.gateway_endpoint,
            port=8888,
            health_check_path="/health",
            capabilities=[
                "image-understanding",
                "visual-qa",
                "multimodal-generation",
                "fp8-quantized",
                "tensorrt-optimized",
            ],
            gpu_required=True,
            gpu_memory_gb=20.0,
            metadata={
                "model": "LLaVA-1.5-7B",
                "quantization": "FP8",
                "gpu": "NVIDIA L4",
                "max_batch_size": 16,
                "supported_formats": ["jpeg", "png", "webp"],
                "max_image_size": [1024, 1024],
                "streaming_enabled": True,
                "version": "1.0.0",
            },
        )

        headers = {
            "Authorization": f"Bearer {self.generate_jwt_token()}",
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.orchestrator_url}/v1/services/register",
                    json=asdict(registration),
                    headers=headers,
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Service registered successfully: {result}")
                        return True
                    else:
                        error = await response.text()
                        logger.error(f"Registration failed: {error}")
                        return False

        except Exception as e:
            logger.error(f"Failed to register service: {e}")
            return False

    async def deregister_service(self) -> bool:
        """Deregister service from orchestrator"""
        logger.info("Deregistering service from orchestrator...")

        headers = {
            "Authorization": f"Bearer {self.generate_jwt_token()}",
            "X-API-Key": self.api_key,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    f"{self.orchestrator_url}/v1/services/{self.service_name}",
                    headers=headers,
                ) as response:
                    if response.status in [200, 204]:
                        logger.info("Service deregistered successfully")
                        return True
                    else:
                        error = await response.text()
                        logger.error(f"Deregistration failed: {error}")
                        return False

        except Exception as e:
            logger.error(f"Failed to deregister service: {e}")
            return False

    async def update_status(self, status: str, metadata: Optional[Dict] = None) -> bool:
        """Update service status with orchestrator"""

        payload = {
            "service_name": self.service_name,
            "status": status,  # "healthy", "degraded", "unhealthy", "maintenance"
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        headers = {
            "Authorization": f"Bearer {self.generate_jwt_token()}",
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.put(
                    f"{self.orchestrator_url}/v1/services/{self.service_name}/status",
                    json=payload,
                    headers=headers,
                ) as response:
                    return response.status == 200

        except Exception as e:
            logger.error(f"Failed to update status: {e}")
            return False

    async def get_routing_config(self) -> Optional[Dict]:
        """Get routing configuration from orchestrator"""

        headers = {
            "Authorization": f"Bearer {self.generate_jwt_token()}",
            "X-API-Key": self.api_key,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.orchestrator_url}/v1/services/{self.service_name}/routing",
                    headers=headers,
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    return None

        except Exception as e:
            logger.error(f"Failed to get routing config: {e}")
            return None

    async def report_metrics(self, metrics: Dict) -> bool:
        """Report performance metrics to orchestrator"""

        payload = {
            "service_name": self.service_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
        }

        headers = {
            "Authorization": f"Bearer {self.generate_jwt_token()}",
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.orchestrator_url}/v1/metrics", json=payload, headers=headers
                ) as response:
                    return response.status in [200, 201, 204]

        except Exception as e:
            logger.error(f"Failed to report metrics: {e}")
            return False


class ServiceLifecycleManager:
    """Manages service lifecycle with orchestrator"""

    def __init__(self):
        self.client = OrchestratorClient()
        self.health_check_interval = 30  # seconds
        self.metrics_report_interval = 60  # seconds

    async def startup(self):
        """Service startup procedure"""
        logger.info("Starting LLaVA FP8 service...")

        # Wait for Triton to be ready
        await self.wait_for_triton()

        # Register with orchestrator
        registered = await self.client.register_service()
        if not registered:
            logger.error("Failed to register with orchestrator")
            sys.exit(1)

        # Start background tasks
        asyncio.create_task(self.health_check_loop())
        asyncio.create_task(self.metrics_report_loop())

        # Update status to healthy
        await self.client.update_status(
            "healthy",
            {
                "startup_time": datetime.now().isoformat(),
                "model_loaded": True,
                "gpu_available": True,
            },
        )

        logger.info("Service startup complete")

    async def shutdown(self):
        """Service shutdown procedure"""
        logger.info("Shutting down LLaVA FP8 service...")

        # Update status to maintenance
        await self.client.update_status(
            "maintenance",
            {"reason": "shutdown", "timestamp": datetime.now().isoformat()},
        )

        # Deregister from orchestrator
        await self.client.deregister_service()

        logger.info("Service shutdown complete")

    async def wait_for_triton(self, max_retries: int = 30):
        """Wait for Triton server to be ready"""
        triton_url = os.getenv("TRITON_ENDPOINT", "http://localhost:8000")

        for i in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{triton_url}/v2/health/ready") as response:
                        if response.status == 200:
                            logger.info("Triton server is ready")
                            return
            except Exception:
                pass

            logger.info(f"Waiting for Triton... ({i + 1}/{max_retries})")
            await asyncio.sleep(2)

        raise RuntimeError("Triton server failed to start")

    async def health_check_loop(self):
        """Continuous health check loop"""
        while True:
            try:
                # Check service health
                is_healthy = await self.check_health()

                # Update orchestrator
                status = "healthy" if is_healthy else "degraded"
                await self.client.update_status(status)

            except Exception as e:
                logger.error(f"Health check error: {e}")

            await asyncio.sleep(self.health_check_interval)

    async def metrics_report_loop(self):
        """Continuous metrics reporting loop"""
        while True:
            try:
                # Collect metrics
                metrics = await self.collect_metrics()

                # Report to orchestrator
                if metrics:
                    await self.client.report_metrics(metrics)

            except Exception as e:
                logger.error(f"Metrics report error: {e}")

            await asyncio.sleep(self.metrics_report_interval)

    async def check_health(self) -> bool:
        """Check service health"""
        gateway_url = os.getenv("GATEWAY_ENDPOINT", "http://localhost:8888")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{gateway_url}/health") as response:
                    return response.status == 200
        except Exception:
            return False

    async def collect_metrics(self) -> Optional[Dict]:
        """Collect service metrics"""
        triton_metrics_url = os.getenv(
            "TRITON_METRICS_URL", "http://localhost:8002/metrics"
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(triton_metrics_url) as response:
                    if response.status == 200:
                        metrics_text = await response.text()

                        # Parse key metrics
                        metrics = {
                            "inference_count": 0,
                            "inference_latency_ms": 0,
                            "gpu_memory_mb": 0,
                            "queue_size": 0,
                        }

                        for line in metrics_text.split("\n"):
                            if "nv_inference_count" in line:
                                metrics["inference_count"] = int(
                                    float(line.split()[-1])
                                )
                            elif "nv_inference_compute_infer_duration_us" in line:
                                metrics["inference_latency_ms"] = (
                                    float(line.split()[-1]) / 1000
                                )
                            elif "nv_gpu_memory_used_bytes" in line:
                                metrics["gpu_memory_mb"] = float(line.split()[-1]) / (
                                    1024 * 1024
                                )
                            elif "nv_inference_queue_size" in line:
                                metrics["queue_size"] = int(float(line.split()[-1]))

                        return metrics

        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            return None


async def main():
    """Main entry point for orchestrator integration"""
    manager = ServiceLifecycleManager()

    # Handle signals for graceful shutdown
    import signal

    def signal_handler(sig, frame):
        asyncio.create_task(manager.shutdown())
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start service
    await manager.startup()

    # Keep running
    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
