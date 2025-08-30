#!/usr/bin/env python3
"""
Production monitoring script for LLaVA FP8 deployment
Tracks performance metrics and manages A/B testing
"""

import asyncio
import json
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import aiohttp
import numpy as np
import redis
from prometheus_client import Counter, Gauge, Histogram, start_http_server

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking"""

    timestamp: datetime
    model_version: str  # "fp8" or "fp16"
    latency_p50: float
    latency_p95: float
    latency_p99: float
    throughput: float
    accuracy: float
    gpu_memory_mb: int
    batch_size: int
    queue_depth: int


@dataclass
class ABTestResults:
    """A/B test results between FP8 and FP16"""

    fp8_metrics: List[PerformanceMetrics] = field(default_factory=list)
    fp16_metrics: List[PerformanceMetrics] = field(default_factory=list)
    fp8_wins: int = 0
    fp16_wins: int = 0
    inconclusive: int = 0


class PerformanceMonitor:
    """Main performance monitoring system"""

    def __init__(self):
        # Configuration
        self.fp8_endpoint = os.getenv("FP8_ENDPOINT", "http://triton-server-fp8:8000")
        self.fp16_endpoint = os.getenv(
            "FP16_ENDPOINT", "http://triton-server-fp16:8000"
        )
        self.gateway_endpoint = os.getenv("GATEWAY_ENDPOINT", "http://api-gateway:8888")
        self.redis_host = os.getenv("REDIS_HOST", "redis-cache")
        self.redis_port = int(os.getenv("REDIS_PORT", "6379"))

        # A/B testing configuration
        self.ab_test_enabled = os.getenv("ENABLE_AB_TESTING", "false").lower() == "true"
        self.ab_test_ratio = float(os.getenv("AB_TEST_RATIO", "0.1"))  # 10% to FP16

        # Performance thresholds
        self.latency_sla_ms = float(os.getenv("LATENCY_SLA_MS", "500"))
        self.throughput_sla = float(os.getenv("THROUGHPUT_SLA", "100"))

        # Metrics storage
        self.metrics_window = deque(maxlen=1000)  # Keep last 1000 measurements
        self.ab_test_results = ABTestResults()

        # Redis client for distributed state
        self.redis_client = redis.Redis(
            host=self.redis_host, port=self.redis_port, decode_responses=True
        )

        # Prometheus metrics
        self.setup_prometheus_metrics()

    def setup_prometheus_metrics(self):
        """Setup Prometheus metrics collectors"""
        # Latency metrics
        self.latency_histogram = Histogram(
            "llava_request_latency_seconds",
            "Request latency in seconds",
            ["model_version", "endpoint"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
        )

        # Throughput metrics
        self.throughput_gauge = Gauge(
            "llava_throughput_rps",
            "Current throughput in requests per second",
            ["model_version"],
        )

        # Accuracy metrics
        self.accuracy_gauge = Gauge(
            "llava_accuracy_score", "Model accuracy score", ["model_version"]
        )

        # Resource metrics
        self.gpu_memory_gauge = Gauge(
            "llava_gpu_memory_mb", "GPU memory usage in MB", ["model_version"]
        )

        # A/B test metrics
        self.ab_test_comparison = Gauge(
            "llava_ab_test_winner",
            "A/B test winner (1=FP8, -1=FP16, 0=tie)",
        )

        # Error metrics
        self.error_counter = Counter(
            "llava_errors_total",
            "Total number of errors",
            ["model_version", "error_type"],
        )

        # SLA violation metrics
        self.sla_violation_counter = Counter(
            "llava_sla_violations_total",
            "Total SLA violations",
            ["violation_type", "model_version"],
        )

    async def collect_metrics(
        self, endpoint: str, model_version: str
    ) -> Optional[PerformanceMetrics]:
        """Collect performance metrics from an endpoint"""
        try:
            async with aiohttp.ClientSession() as session:
                # Get metrics from Triton metrics endpoint
                metrics_url = endpoint.replace(":8000", ":8002") + "/metrics"
                async with session.get(metrics_url) as response:
                    if response.status != 200:
                        return None

                    metrics_text = await response.text()

                    # Parse Prometheus metrics
                    metrics = self.parse_prometheus_metrics(metrics_text)

                    # Calculate latency percentiles from recent requests
                    latencies = await self.get_recent_latencies(model_version)
                    if latencies:
                        p50 = np.percentile(latencies, 50)
                        p95 = np.percentile(latencies, 95)
                        p99 = np.percentile(latencies, 99)
                    else:
                        p50 = p95 = p99 = 0

                    # Get GPU memory from metrics
                    gpu_memory_mb = metrics.get("nv_gpu_memory_used_bytes", 0) / (
                        1024 * 1024
                    )

                    # Calculate throughput
                    throughput = await self.calculate_throughput(model_version)

                    # Get accuracy (from cached test results)
                    accuracy = await self.get_cached_accuracy(model_version)

                    return PerformanceMetrics(
                        timestamp=datetime.now(),
                        model_version=model_version,
                        latency_p50=p50,
                        latency_p95=p95,
                        latency_p99=p99,
                        throughput=throughput,
                        accuracy=accuracy,
                        gpu_memory_mb=int(gpu_memory_mb),
                        batch_size=metrics.get("batch_size", 1),
                        queue_depth=metrics.get("nv_inference_queue_size", 0),
                    )

        except Exception as e:
            logger.error(f"Failed to collect metrics from {endpoint}: {e}")
            self.error_counter.labels(
                model_version=model_version, error_type="metrics_collection"
            ).inc()
            return None

    def parse_prometheus_metrics(self, metrics_text: str) -> Dict:
        """Parse Prometheus format metrics"""
        metrics = {}
        for line in metrics_text.split("\n"):
            if line and not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 2:
                    metric_name = parts[0].split("{")[0]
                    try:
                        value = float(parts[-1])
                        metrics[metric_name] = value
                    except ValueError:
                        pass
        return metrics

    async def get_recent_latencies(self, model_version: str) -> List[float]:
        """Get recent latency measurements from Redis"""
        key = f"latencies:{model_version}"
        latencies = self.redis_client.lrange(key, 0, 100)
        return [float(l) for l in latencies]

    async def calculate_throughput(self, model_version: str) -> float:
        """Calculate current throughput"""
        key = f"request_count:{model_version}"

        # Get request count for last minute
        current_count = int(self.redis_client.get(key) or 0)

        # Get count from 1 minute ago
        prev_key = f"{key}:prev"
        prev_count = int(self.redis_client.get(prev_key) or 0)

        # Calculate RPS
        rps = (current_count - prev_count) / 60.0

        # Update previous count
        self.redis_client.set(prev_key, current_count)

        return rps

    async def get_cached_accuracy(self, model_version: str) -> float:
        """Get cached accuracy score"""
        key = f"accuracy:{model_version}"
        accuracy = self.redis_client.get(key)
        return float(accuracy) if accuracy else 0.95  # Default to baseline

    async def run_ab_test(self):
        """Run A/B test comparison between FP8 and FP16"""
        if not self.ab_test_enabled:
            return

        logger.info("Running A/B test comparison...")

        # Collect metrics from both versions
        fp8_metrics = await self.collect_metrics(self.fp8_endpoint, "fp8")
        fp16_metrics = await self.collect_metrics(self.fp16_endpoint, "fp16")

        if not fp8_metrics or not fp16_metrics:
            logger.warning("Could not collect metrics for A/B test")
            return

        # Store metrics
        self.ab_test_results.fp8_metrics.append(fp8_metrics)
        self.ab_test_results.fp16_metrics.append(fp16_metrics)

        # Compare performance
        fp8_score = self.calculate_performance_score(fp8_metrics)
        fp16_score = self.calculate_performance_score(fp16_metrics)

        # Determine winner
        if fp8_score > fp16_score * 1.05:  # FP8 needs to be 5% better
            self.ab_test_results.fp8_wins += 1
            self.ab_test_comparison.set(1)
            logger.info(f"FP8 wins: score {fp8_score:.2f} vs {fp16_score:.2f}")
        elif fp16_score > fp8_score:
            self.ab_test_results.fp16_wins += 1
            self.ab_test_comparison.set(-1)
            logger.info(f"FP16 wins: score {fp16_score:.2f} vs {fp8_score:.2f}")
        else:
            self.ab_test_results.inconclusive += 1
            self.ab_test_comparison.set(0)
            logger.info(f"Tie: scores {fp8_score:.2f} vs {fp16_score:.2f}")

        # Check if we should switch models
        await self.evaluate_model_switch()

    def calculate_performance_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate overall performance score for a model"""
        # Weighted scoring: lower latency and higher throughput is better
        latency_score = max(0, 1 - (metrics.latency_p99 / self.latency_sla_ms))
        throughput_score = min(1, metrics.throughput / self.throughput_sla)
        accuracy_score = metrics.accuracy
        memory_score = max(0, 1 - (metrics.gpu_memory_mb / 20000))  # 20GB threshold

        # Weighted average
        score = (
            latency_score * 0.3
            + throughput_score * 0.3
            + accuracy_score * 0.3
            + memory_score * 0.1
        )

        return score

    async def evaluate_model_switch(self):
        """Evaluate if we should switch the primary model"""
        # Need at least 100 comparisons
        total_tests = (
            self.ab_test_results.fp8_wins
            + self.ab_test_results.fp16_wins
            + self.ab_test_results.inconclusive
        )

        if total_tests < 100:
            return

        # Calculate win rates
        fp8_win_rate = self.ab_test_results.fp8_wins / total_tests
        fp16_win_rate = self.ab_test_results.fp16_wins / total_tests

        # Log results
        logger.info(f"A/B Test Results after {total_tests} tests:")
        logger.info(f"  FP8 wins: {fp8_win_rate:.1%}")
        logger.info(f"  FP16 wins: {fp16_win_rate:.1%}")
        logger.info(f"  Ties: {(self.ab_test_results.inconclusive / total_tests):.1%}")

        # Switch if FP16 consistently wins
        if fp16_win_rate > 0.6:  # FP16 wins >60% of the time
            logger.warning("FP16 consistently outperforms FP8. Consider rolling back.")
            await self.trigger_rollback_recommendation()

        # Reset counters after evaluation
        self.ab_test_results = ABTestResults()

    async def trigger_rollback_recommendation(self):
        """Trigger recommendation to rollback to FP16"""
        # Send alert
        alert = {
            "severity": "warning",
            "message": "FP16 consistently outperforms FP8 in A/B testing",
            "recommendation": "Consider rolling back to FP16",
            "timestamp": datetime.now().isoformat(),
        }

        # Store in Redis for other services
        self.redis_client.set("rollback_recommendation", json.dumps(alert), ex=3600)

        # Send webhook if configured
        webhook_url = os.getenv("ALERT_WEBHOOK_URL")
        if webhook_url:
            try:
                async with aiohttp.ClientSession() as session:
                    await session.post(webhook_url, json=alert)
            except Exception as e:
                logger.error(f"Failed to send alert: {e}")

    async def check_sla_violations(self, metrics: PerformanceMetrics):
        """Check for SLA violations"""
        violations = []

        if metrics.latency_p99 > self.latency_sla_ms:
            violations.append("latency")
            self.sla_violation_counter.labels(
                violation_type="latency", model_version=metrics.model_version
            ).inc()

        if metrics.throughput < self.throughput_sla:
            violations.append("throughput")
            self.sla_violation_counter.labels(
                violation_type="throughput", model_version=metrics.model_version
            ).inc()

        if violations:
            logger.warning(f"SLA violations for {metrics.model_version}: {violations}")

    async def monitor_loop(self):
        """Main monitoring loop"""
        logger.info("Starting performance monitoring...")

        # Start Prometheus metrics server
        start_http_server(9091)

        while True:
            try:
                # Collect metrics from primary (FP8) endpoint
                fp8_metrics = await self.collect_metrics(self.fp8_endpoint, "fp8")

                if fp8_metrics:
                    # Update Prometheus metrics
                    self.latency_histogram.labels(
                        model_version="fp8", endpoint="inference"
                    ).observe(fp8_metrics.latency_p99 / 1000)  # Convert to seconds

                    self.throughput_gauge.labels(model_version="fp8").set(
                        fp8_metrics.throughput
                    )
                    self.accuracy_gauge.labels(model_version="fp8").set(
                        fp8_metrics.accuracy
                    )
                    self.gpu_memory_gauge.labels(model_version="fp8").set(
                        fp8_metrics.gpu_memory_mb
                    )

                    # Check SLA violations
                    await self.check_sla_violations(fp8_metrics)

                    # Store metrics
                    self.metrics_window.append(fp8_metrics)

                # Run A/B test if enabled
                if self.ab_test_enabled:
                    await self.run_ab_test()

                # Generate performance report every 5 minutes
                if len(self.metrics_window) > 0 and len(self.metrics_window) % 300 == 0:
                    await self.generate_performance_report()

            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                self.error_counter.labels(
                    model_version="unknown", error_type="monitoring"
                ).inc()

            # Wait before next iteration
            await asyncio.sleep(1)

    async def generate_performance_report(self):
        """Generate performance report"""
        if not self.metrics_window:
            return

        # Calculate statistics
        recent_metrics = list(self.metrics_window)[-300:]  # Last 5 minutes

        avg_latency = np.mean([m.latency_p99 for m in recent_metrics])
        avg_throughput = np.mean([m.throughput for m in recent_metrics])
        avg_accuracy = np.mean([m.accuracy for m in recent_metrics])
        avg_gpu_memory = np.mean([m.gpu_memory_mb for m in recent_metrics])

        report = {
            "timestamp": datetime.now().isoformat(),
            "period": "5_minutes",
            "metrics": {
                "avg_latency_p99_ms": round(avg_latency, 2),
                "avg_throughput_rps": round(avg_throughput, 2),
                "avg_accuracy": round(avg_accuracy, 4),
                "avg_gpu_memory_mb": round(avg_gpu_memory, 0),
            },
            "sla_compliance": {
                "latency": avg_latency <= self.latency_sla_ms,
                "throughput": avg_throughput >= self.throughput_sla,
            },
        }

        # Save report
        report_path = (
            f"/logs/performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Performance report generated: {report_path}")


async def main():
    """Main entry point"""
    monitor = PerformanceMonitor()
    await monitor.monitor_loop()


if __name__ == "__main__":
    asyncio.run(main())
