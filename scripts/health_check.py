#!/usr/bin/env python3
"""
Health check script for LLaVA FP8 deployment
Monitors model performance, accuracy, and system resources
"""

import asyncio
import base64
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import aiohttp
import GPUtil
import psutil
import requests
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    push_to_gateway,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class HealthMetrics:
    """Health metrics for the system"""

    timestamp: str
    model_available: bool
    inference_latency_ms: float
    throughput_rps: float
    gpu_memory_used_gb: float
    gpu_utilization_percent: float
    cpu_percent: float
    memory_percent: float
    accuracy_score: float
    error_rate: float
    queue_size: int
    active_connections: int


class HealthChecker:
    """Main health checking system"""

    def __init__(self):
        self.triton_url = os.getenv("TRITON_HTTP_ENDPOINT", "http://localhost:8000")
        self.gateway_url = os.getenv("GATEWAY_URL", "http://localhost:8888")
        self.metrics_url = os.getenv(
            "TRITON_METRICS_URL", "http://localhost:8002/metrics"
        )

        # Thresholds
        self.latency_threshold = float(os.getenv("LATENCY_P99_THRESHOLD", "500"))
        self.throughput_threshold = float(os.getenv("THROUGHPUT_THRESHOLD", "100"))
        self.accuracy_threshold = float(os.getenv("ACCURACY_THRESHOLD", "0.95"))
        self.memory_threshold = float(os.getenv("MEMORY_THRESHOLD", "20"))

        # Prometheus metrics
        self.registry = CollectorRegistry()
        self.setup_prometheus_metrics()

        # Test data for accuracy checks
        self.test_images = self.load_test_images()
        self.baseline_responses = {}

    def setup_prometheus_metrics(self):
        """Setup Prometheus metrics collectors"""
        self.latency_gauge = Gauge(
            "llava_inference_latency_ms",
            "Inference latency in milliseconds",
            registry=self.registry,
        )
        self.throughput_gauge = Gauge(
            "llava_throughput_rps",
            "Throughput in requests per second",
            registry=self.registry,
        )
        self.accuracy_gauge = Gauge(
            "llava_accuracy_score", "Model accuracy score", registry=self.registry
        )
        self.gpu_memory_gauge = Gauge(
            "llava_gpu_memory_gb", "GPU memory usage in GB", registry=self.registry
        )
        self.gpu_util_gauge = Gauge(
            "llava_gpu_utilization_percent",
            "GPU utilization percentage",
            registry=self.registry,
        )
        self.error_counter = Counter(
            "llava_errors_total", "Total number of errors", registry=self.registry
        )
        self.health_status_gauge = Gauge(
            "llava_health_status",
            "Overall health status (1=healthy, 0=unhealthy)",
            registry=self.registry,
        )

    def load_test_images(self) -> List[Dict]:
        """Load test images for accuracy validation"""
        test_data = [
            {
                "url": "http://images.cocodataset.org/val2017/000000397133.jpg",
                "expected_objects": ["person", "kitchen", "food"],
                "prompt": "What objects are in this image?",
            },
            {
                "url": "http://images.cocodataset.org/val2017/000000037777.jpg",
                "expected_objects": ["airplane", "sky", "runway"],
                "prompt": "Describe what you see in this image",
            },
        ]
        return test_data

    async def check_triton_health(self) -> bool:
        """Check if Triton server is healthy"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.triton_url}/v2/health/ready"
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Triton health check failed: {e}")
            return False

    async def measure_inference_latency(self) -> Tuple[float, bool]:
        """Measure inference latency with a test request"""
        try:
            # Prepare test request
            test_image = self.test_images[0]

            start_time = time.time()
            async with aiohttp.ClientSession() as session:
                # Download test image
                async with session.get(test_image["url"]) as img_response:
                    img_data = await img_response.read()

                # Prepare inference request
                img_base64 = base64.b64encode(img_data).decode("utf-8")
                payload = {
                    "inputs": [
                        {
                            "name": "images",
                            "shape": [1, 3, 336, 336],
                            "datatype": "FP16",
                            "data": img_base64,
                        },
                        {
                            "name": "prompt",
                            "shape": [1, 1],
                            "datatype": "BYTES",
                            "data": [test_image["prompt"]],
                        },
                    ]
                }

                # Send inference request
                async with session.post(
                    f"{self.triton_url}/v2/models/ensemble_llava/infer",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        latency_ms = (time.time() - start_time) * 1000
                        return latency_ms, True
                    else:
                        logger.error(f"Inference failed with status {response.status}")
                        return 0, False

        except Exception as e:
            logger.error(f"Latency measurement failed: {e}")
            return 0, False

    async def check_accuracy(self) -> float:
        """Check model accuracy against baseline"""
        correct_predictions = 0
        total_predictions = 0

        for test_image in self.test_images[:5]:  # Check first 5 test images
            try:
                async with aiohttp.ClientSession() as session:
                    # Get image
                    async with session.get(test_image["url"]) as img_response:
                        img_data = await img_response.read()

                    # Prepare request
                    img_base64 = base64.b64encode(img_data).decode("utf-8")
                    payload = {"image": img_base64, "prompt": test_image["prompt"]}

                    # Get prediction through gateway
                    async with session.post(
                        f"{self.gateway_url}/v1/predict",
                        json=payload,
                        headers={
                            "Authorization": f"Bearer {os.getenv('JWT_TOKEN', '')}"
                        },
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            generated_text = result.get("generated_text", "").lower()

                            # Check if expected objects are mentioned
                            matches = sum(
                                1
                                for obj in test_image["expected_objects"]
                                if obj.lower() in generated_text
                            )
                            if matches >= len(test_image["expected_objects"]) * 0.6:
                                correct_predictions += 1
                            total_predictions += 1

            except Exception as e:
                logger.error(f"Accuracy check failed for image: {e}")

        return correct_predictions / max(total_predictions, 1)

    def get_gpu_metrics(self) -> Tuple[float, float]:
        """Get GPU memory and utilization metrics"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                memory_used_gb = (gpu.memoryTotal - gpu.memoryFree) / 1024
                utilization_percent = gpu.load * 100
                return memory_used_gb, utilization_percent
        except Exception as e:
            logger.error(f"Failed to get GPU metrics: {e}")
        return 0, 0

    def get_system_metrics(self) -> Tuple[float, float]:
        """Get CPU and memory metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            return cpu_percent, memory_percent
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
        return 0, 0

    async def get_triton_metrics(self) -> Dict:
        """Get metrics from Triton metrics endpoint"""
        metrics = {
            "queue_size": 0,
            "active_connections": 0,
            "throughput_rps": 0,
            "error_rate": 0,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.metrics_url) as response:
                    if response.status == 200:
                        text = await response.text()
                        # Parse Prometheus format metrics
                        for line in text.split("\n"):
                            if "nv_inference_queue_size" in line:
                                metrics["queue_size"] = int(float(line.split()[-1]))
                            elif "nv_inference_request_success" in line:
                                # Calculate throughput (simplified)
                                metrics["throughput_rps"] = float(line.split()[-1]) / 60
                            elif "nv_inference_request_failure" in line:
                                failures = float(line.split()[-1])
                                if "nv_inference_request_success" in text:
                                    success_line = [
                                        l
                                        for l in text.split("\n")
                                        if "nv_inference_request_success" in l
                                    ][0]
                                    successes = float(success_line.split()[-1])
                                    total = successes + failures
                                    metrics["error_rate"] = failures / max(total, 1)
        except Exception as e:
            logger.error(f"Failed to get Triton metrics: {e}")

        return metrics

    async def run_health_check(self) -> HealthMetrics:
        """Run complete health check"""
        logger.info("Running health check...")

        # Check Triton availability
        model_available = await self.check_triton_health()

        # Measure latency
        latency_ms, inference_success = await self.measure_inference_latency()

        # Check accuracy
        accuracy_score = await self.check_accuracy() if inference_success else 0

        # Get GPU metrics
        gpu_memory_gb, gpu_utilization = self.get_gpu_metrics()

        # Get system metrics
        cpu_percent, memory_percent = self.get_system_metrics()

        # Get Triton metrics
        triton_metrics = await self.get_triton_metrics()

        # Create health metrics
        metrics = HealthMetrics(
            timestamp=datetime.now().isoformat(),
            model_available=model_available,
            inference_latency_ms=latency_ms,
            throughput_rps=triton_metrics["throughput_rps"],
            gpu_memory_used_gb=gpu_memory_gb,
            gpu_utilization_percent=gpu_utilization,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            accuracy_score=accuracy_score,
            error_rate=triton_metrics["error_rate"],
            queue_size=triton_metrics["queue_size"],
            active_connections=triton_metrics["active_connections"],
        )

        # Update Prometheus metrics
        self.update_prometheus_metrics(metrics)

        return metrics

    def update_prometheus_metrics(self, metrics: HealthMetrics):
        """Update Prometheus metrics"""
        self.latency_gauge.set(metrics.inference_latency_ms)
        self.throughput_gauge.set(metrics.throughput_rps)
        self.accuracy_gauge.set(metrics.accuracy_score)
        self.gpu_memory_gauge.set(metrics.gpu_memory_used_gb)
        self.gpu_util_gauge.set(metrics.gpu_utilization_percent)

        # Determine overall health
        is_healthy = (
            metrics.model_available
            and metrics.inference_latency_ms < self.latency_threshold
            and metrics.throughput_rps > self.throughput_threshold
            and metrics.accuracy_score >= self.accuracy_threshold
            and metrics.gpu_memory_used_gb < self.memory_threshold
            and metrics.error_rate < 0.05
        )

        self.health_status_gauge.set(1 if is_healthy else 0)

        # Push to Prometheus gateway if configured
        pushgateway_url = os.getenv("PROMETHEUS_PUSHGATEWAY")
        if pushgateway_url:
            try:
                push_to_gateway(
                    pushgateway_url, job="llava_health", registry=self.registry
                )
            except Exception as e:
                logger.error(f"Failed to push metrics to Prometheus: {e}")

    def evaluate_health(self, metrics: HealthMetrics) -> Tuple[bool, List[str]]:
        """Evaluate overall system health"""
        issues = []

        if not metrics.model_available:
            issues.append("Model not available")

        if metrics.inference_latency_ms > self.latency_threshold:
            issues.append(
                f"High latency: {metrics.inference_latency_ms:.1f}ms > {self.latency_threshold}ms"
            )

        if metrics.throughput_rps < self.throughput_threshold:
            issues.append(
                f"Low throughput: {metrics.throughput_rps:.1f} < {self.throughput_threshold} rps"
            )

        if metrics.accuracy_score < self.accuracy_threshold:
            issues.append(
                f"Low accuracy: {metrics.accuracy_score:.2%} < {self.accuracy_threshold:.2%}"
            )

        if metrics.gpu_memory_used_gb > self.memory_threshold:
            issues.append(
                f"High GPU memory: {metrics.gpu_memory_used_gb:.1f}GB > {self.memory_threshold}GB"
            )

        if metrics.error_rate > 0.05:
            issues.append(f"High error rate: {metrics.error_rate:.2%}")

        is_healthy = len(issues) == 0
        return is_healthy, issues

    def trigger_rollback(self, issues: List[str]):
        """Trigger rollback to FP16 if needed"""
        if os.getenv("ROLLBACK_ENABLED", "false").lower() == "true":
            logger.warning(f"Triggering rollback due to issues: {issues}")
            rollback_command = os.getenv(
                "ROLLBACK_COMMAND", "/scripts/rollback_to_fp16.sh"
            )

            try:
                import subprocess

                result = subprocess.run(
                    [rollback_command], capture_output=True, text=True
                )
                if result.returncode == 0:
                    logger.info("Rollback initiated successfully")
                else:
                    logger.error(f"Rollback failed: {result.stderr}")
            except Exception as e:
                logger.error(f"Failed to trigger rollback: {e}")

        # Send alert
        self.send_alert(issues)

    def send_alert(self, issues: List[str]):
        """Send alert via webhook"""
        webhook_url = os.getenv("ALERT_WEBHOOK_URL")
        if webhook_url:
            try:
                payload = {
                    "text": "LLaVA FP8 Health Check Failed",
                    "issues": issues,
                    "timestamp": datetime.now().isoformat(),
                    "severity": "critical",
                }
                requests.post(webhook_url, json=payload, timeout=5)
            except Exception as e:
                logger.error(f"Failed to send alert: {e}")


async def main():
    """Main health check loop"""
    checker = HealthChecker()
    check_interval = int(os.getenv("CHECK_INTERVAL", "60"))

    consecutive_failures = 0
    max_failures = 3

    while True:
        try:
            # Run health check
            metrics = await checker.run_health_check()

            # Evaluate health
            is_healthy, issues = checker.evaluate_health(metrics)

            # Log results
            if is_healthy:
                logger.info("Health check PASSED")
                consecutive_failures = 0
                # Write success marker for Docker health check
                with open("/tmp/health_check_status", "w") as f:
                    f.write("healthy")
            else:
                logger.warning(f"Health check FAILED: {issues}")
                consecutive_failures += 1

                # Trigger rollback if failures exceed threshold
                if consecutive_failures >= max_failures:
                    checker.trigger_rollback(issues)
                    consecutive_failures = 0  # Reset after rollback

                # Write failure marker
                with open("/tmp/health_check_status", "w") as f:
                    f.write("unhealthy")

            # Save metrics to file
            with open("/logs/health_metrics.json", "w") as f:
                json.dump(asdict(metrics), f, indent=2)

        except Exception as e:
            logger.error(f"Health check error: {e}")
            consecutive_failures += 1

        # Wait for next check
        await asyncio.sleep(check_interval)


if __name__ == "__main__":
    asyncio.run(main())
