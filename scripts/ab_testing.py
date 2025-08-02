#!/usr/bin/env python3
"""
A/B Testing Configuration and Management for FP8 vs FP16
Handles traffic routing, performance comparison, and automatic decisions
"""

import asyncio
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import redis
from scipy import stats

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ABTestConfig:
    """A/B test configuration"""

    test_name: str = "fp8_vs_fp16"
    variant_a: str = "fp8"
    variant_b: str = "fp16"
    traffic_split: float = 0.9  # 90% to FP8, 10% to FP16
    min_sample_size: int = 1000
    confidence_level: float = 0.95
    performance_threshold: float = 0.05  # 5% performance difference
    test_duration_hours: int = 24
    metrics_to_track: List[str] = field(
        default_factory=lambda: [
            "latency_p50",
            "latency_p95",
            "latency_p99",
            "throughput",
            "accuracy",
            "gpu_memory",
            "error_rate",
        ]
    )


@dataclass
class VariantMetrics:
    """Metrics for a variant"""

    variant_name: str
    request_count: int = 0
    latency_samples: List[float] = field(default_factory=list)
    accuracy_samples: List[float] = field(default_factory=list)
    error_count: int = 0
    gpu_memory_samples: List[float] = field(default_factory=list)
    throughput_samples: List[float] = field(default_factory=list)

    def get_statistics(self) -> Dict:
        """Calculate statistics for the variant"""
        return {
            "request_count": self.request_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "latency_p50": (
                np.percentile(self.latency_samples, 50) if self.latency_samples else 0
            ),
            "latency_p95": (
                np.percentile(self.latency_samples, 95) if self.latency_samples else 0
            ),
            "latency_p99": (
                np.percentile(self.latency_samples, 99) if self.latency_samples else 0
            ),
            "accuracy_mean": (
                np.mean(self.accuracy_samples) if self.accuracy_samples else 0
            ),
            "accuracy_std": (
                np.std(self.accuracy_samples) if self.accuracy_samples else 0
            ),
            "gpu_memory_mean": (
                np.mean(self.gpu_memory_samples) if self.gpu_memory_samples else 0
            ),
            "throughput_mean": (
                np.mean(self.throughput_samples) if self.throughput_samples else 0
            ),
        }


class ABTestManager:
    """Manages A/B testing between FP8 and FP16 models"""

    def __init__(self, config: ABTestConfig):
        self.config = config
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "redis-cache"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            decode_responses=True,
        )

        # Endpoints
        self.fp8_endpoint = os.getenv("FP8_ENDPOINT", "http://triton-server-fp8:8000")
        self.fp16_endpoint = os.getenv(
            "FP16_ENDPOINT", "http://triton-server-fp16:8000"
        )

        # Test state
        self.test_start_time = datetime.now()
        self.variant_a_metrics = VariantMetrics(config.variant_a)
        self.variant_b_metrics = VariantMetrics(config.variant_b)

    def should_route_to_variant_b(self, request_id: str) -> bool:
        """Determine if request should go to variant B (FP16)"""
        # Use consistent hashing for deterministic routing
        hash_value = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        threshold = int(self.config.traffic_split * (2**128))
        return hash_value >= threshold

    async def route_request(self, request_data: Dict) -> Tuple[str, Dict]:
        """Route request to appropriate variant and collect metrics"""
        request_id = request_data.get("request_id", str(datetime.now().timestamp()))

        # Determine routing
        use_variant_b = self.should_route_to_variant_b(request_id)
        variant = self.config.variant_b if use_variant_b else self.config.variant_a
        endpoint = self.fp16_endpoint if use_variant_b else self.fp8_endpoint

        # Record routing decision
        self.redis_client.hincrby(
            f"ab_test:{self.config.test_name}:routing", variant, 1
        )

        # Execute request and measure performance
        start_time = asyncio.get_event_loop().time()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{endpoint}/v2/models/ensemble_llava/infer",
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000

                    if response.status == 200:
                        result = await response.json()

                        # Record metrics
                        await self.record_metrics(variant, latency_ms, success=True)

                        # Add variant metadata to response
                        result["_ab_test_variant"] = variant
                        result["_ab_test_latency_ms"] = latency_ms

                        return variant, result
                    else:
                        await self.record_metrics(variant, latency_ms, success=False)
                        raise Exception(f"Request failed with status {response.status}")

        except Exception as e:
            logger.error(f"Request to {variant} failed: {e}")
            await self.record_metrics(variant, 0, success=False, error=str(e))
            raise

    async def record_metrics(
        self,
        variant: str,
        latency_ms: float,
        success: bool,
        error: Optional[str] = None,
    ):
        """Record metrics for a variant"""
        metrics = (
            self.variant_a_metrics
            if variant == self.config.variant_a
            else self.variant_b_metrics
        )

        metrics.request_count += 1

        if success:
            metrics.latency_samples.append(latency_ms)

            # Store in Redis for persistence
            self.redis_client.lpush(f"ab_test:{variant}:latencies", latency_ms)
            self.redis_client.ltrim(
                f"ab_test:{variant}:latencies", 0, 9999
            )  # Keep last 10k
        else:
            metrics.error_count += 1

        # Update Redis counters
        self.redis_client.hincrby(
            f"ab_test:{self.config.test_name}:requests", variant, 1
        )
        if not success:
            self.redis_client.hincrby(
                f"ab_test:{self.config.test_name}:errors", variant, 1
            )

    async def collect_variant_metrics(self, variant: str) -> Dict:
        """Collect comprehensive metrics for a variant"""
        endpoint = (
            self.fp8_endpoint
            if variant == self.config.variant_a
            else self.fp16_endpoint
        )
        metrics_url = endpoint.replace(":8000", ":8002") + "/metrics"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(metrics_url) as response:
                    if response.status == 200:
                        metrics_text = await response.text()

                        # Parse Prometheus metrics
                        parsed_metrics = {}
                        for line in metrics_text.split("\n"):
                            if "nv_inference_compute_infer_duration_us" in line:
                                parsed_metrics["inference_latency_us"] = float(
                                    line.split()[-1]
                                )
                            elif "nv_gpu_memory_used_bytes" in line:
                                parsed_metrics["gpu_memory_mb"] = float(
                                    line.split()[-1]
                                ) / (1024 * 1024)
                            elif "nv_inference_request_success" in line:
                                parsed_metrics["success_count"] = int(
                                    float(line.split()[-1])
                                )
                            elif "nv_inference_request_failure" in line:
                                parsed_metrics["failure_count"] = int(
                                    float(line.split()[-1])
                                )

                        return parsed_metrics

        except Exception as e:
            logger.error(f"Failed to collect metrics for {variant}: {e}")
            return {}

    def perform_statistical_test(self) -> Dict:
        """Perform statistical significance testing"""
        a_stats = self.variant_a_metrics.get_statistics()
        b_stats = self.variant_b_metrics.get_statistics()

        results = {"variant_a": a_stats, "variant_b": b_stats, "tests": {}}

        # T-test for latency (lower is better)
        if (
            self.variant_a_metrics.latency_samples
            and self.variant_b_metrics.latency_samples
        ):
            t_stat, p_value = stats.ttest_ind(
                self.variant_a_metrics.latency_samples,
                self.variant_b_metrics.latency_samples,
            )

            results["tests"]["latency"] = {
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < (1 - self.config.confidence_level),
                "winner": (
                    self.config.variant_a if t_stat < 0 else self.config.variant_b
                ),
            }

        # Z-test for error rate (lower is better)
        a_errors = self.variant_a_metrics.error_count
        b_errors = self.variant_b_metrics.error_count
        a_total = self.variant_a_metrics.request_count
        b_total = self.variant_b_metrics.request_count

        if a_total > 0 and b_total > 0:
            a_rate = a_errors / a_total
            b_rate = b_errors / b_total
            pooled_rate = (a_errors + b_errors) / (a_total + b_total)

            se = np.sqrt(pooled_rate * (1 - pooled_rate) * (1 / a_total + 1 / b_total))
            if se > 0:
                z_stat = (a_rate - b_rate) / se
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

                results["tests"]["error_rate"] = {
                    "z_statistic": z_stat,
                    "p_value": p_value,
                    "significant": p_value < (1 - self.config.confidence_level),
                    "winner": (
                        self.config.variant_a if z_stat < 0 else self.config.variant_b
                    ),
                }

        return results

    def calculate_winner(self, test_results: Dict) -> str:
        """Determine overall winner based on test results"""
        votes = {"fp8": 0, "fp16": 0}

        # Weight different metrics
        weights = {"latency": 0.4, "error_rate": 0.3, "accuracy": 0.3}

        for metric, weight in weights.items():
            if metric in test_results["tests"]:
                test = test_results["tests"][metric]
                if test["significant"]:
                    votes[test["winner"]] += weight

        # Consider resource usage (GPU memory)
        a_memory = test_results["variant_a"].get("gpu_memory_mean", 0)
        b_memory = test_results["variant_b"].get("gpu_memory_mean", 0)

        if a_memory > 0 and b_memory > 0:
            memory_advantage = (b_memory - a_memory) / b_memory
            if abs(memory_advantage) > 0.1:  # >10% difference
                winner = "fp8" if memory_advantage > 0 else "fp16"
                votes[winner] += 0.1

        # Determine winner
        if votes["fp8"] > votes["fp16"]:
            return "fp8"
        elif votes["fp16"] > votes["fp8"]:
            return "fp16"
        else:
            return "tie"

    async def evaluate_test(self) -> Dict:
        """Evaluate A/B test results"""
        # Check if we have enough samples
        total_requests = (
            self.variant_a_metrics.request_count + self.variant_b_metrics.request_count
        )

        if total_requests < self.config.min_sample_size:
            return {
                "status": "insufficient_data",
                "message": (
                    f"Need {self.config.min_sample_size - total_requests} "
                    f"more samples"
                ),
                "current_samples": total_requests,
            }

        # Perform statistical tests
        test_results = self.perform_statistical_test()

        # Determine winner
        winner = self.calculate_winner(test_results)

        # Calculate confidence
        confidence = self.calculate_confidence(test_results)

        # Generate recommendation
        recommendation = self.generate_recommendation(winner, confidence, test_results)

        return {
            "status": "complete",
            "test_name": self.config.test_name,
            "duration_hours": (datetime.now() - self.test_start_time).total_seconds()
            / 3600,
            "total_requests": total_requests,
            "winner": winner,
            "confidence": confidence,
            "test_results": test_results,
            "recommendation": recommendation,
        }

    def calculate_confidence(self, test_results: Dict) -> float:
        """Calculate overall confidence in results"""
        confidences = []

        for test in test_results["tests"].values():
            if "p_value" in test:
                # Convert p-value to confidence
                confidence = 1 - test["p_value"]
                confidences.append(confidence)

        return np.mean(confidences) if confidences else 0

    def generate_recommendation(
        self, winner: str, confidence: float, test_results: Dict
    ) -> Dict:
        """Generate deployment recommendation"""
        recommendation = {
            "action": "continue_testing",
            "reason": "",
            "confidence": confidence,
        }

        if winner == "fp8" and confidence > self.config.confidence_level:
            recommendation["action"] = "deploy_fp8"
            recommendation["reason"] = "FP8 shows significant performance improvements"

            # Calculate improvements
            improvements = []
            if "latency" in test_results["tests"]:
                a_p99 = test_results["variant_a"]["latency_p99"]
                b_p99 = test_results["variant_b"]["latency_p99"]
                improvement = (b_p99 - a_p99) / b_p99 * 100
                improvements.append(f"{improvement:.1f}% latency reduction")

            recommendation["improvements"] = improvements

        elif winner == "fp16" and confidence > self.config.confidence_level:
            recommendation["action"] = "rollback_to_fp16"
            recommendation["reason"] = "FP16 performs better than FP8"
            recommendation["warnings"] = ["Consider investigating FP8 calibration"]

        elif winner == "tie":
            recommendation["action"] = "deploy_fp8"
            recommendation["reason"] = "Similar performance but FP8 uses less memory"

        else:
            recommendation["action"] = "continue_testing"
            recommendation["reason"] = f"Insufficient confidence ({confidence:.2%})"

        return recommendation

    async def automated_decision(self):
        """Make automated deployment decision based on test results"""
        evaluation = await self.evaluate_test()

        if evaluation["status"] != "complete":
            logger.info(f"Test not complete: {evaluation['message']}")
            return

        recommendation = evaluation["recommendation"]
        logger.info(f"A/B Test Recommendation: {recommendation['action']}")
        logger.info(f"Reason: {recommendation['reason']}")

        # Store decision in Redis
        self.redis_client.set(
            f"ab_test:{self.config.test_name}:decision",
            json.dumps(
                {
                    "timestamp": datetime.now().isoformat(),
                    "evaluation": evaluation,
                    "executed": False,
                }
            ),
            ex=86400,  # Expire after 24 hours
        )

        # Execute automated action if configured
        if os.getenv("AB_TEST_AUTO_DEPLOY", "false").lower() == "true":
            await self.execute_recommendation(recommendation)

    async def execute_recommendation(self, recommendation: Dict):
        """Execute the deployment recommendation"""
        action = recommendation["action"]

        if action == "deploy_fp8":
            logger.info("Deploying FP8 as primary model...")
            await self.deploy_variant("fp8")

        elif action == "rollback_to_fp16":
            logger.info("Rolling back to FP16...")
            await self.deploy_variant("fp16")

        elif action == "continue_testing":
            logger.info("Continuing A/B testing...")

        # Update Redis with execution status
        decision_key = f"ab_test:{self.config.test_name}:decision"
        decision = json.loads(self.redis_client.get(decision_key) or "{}")
        decision["executed"] = True
        decision["execution_time"] = datetime.now().isoformat()
        self.redis_client.set(decision_key, json.dumps(decision), ex=86400)

    async def deploy_variant(self, variant: str):
        """Deploy a specific variant as primary"""
        # Update orchestrator configuration
        orchestrator_url = os.getenv("ORCHESTRATOR_URL", "http://phx-ai20:8888")

        config_update = {
            "service": "llava",
            "primary_variant": variant,
            "routing": {
                "fp8": 1.0 if variant == "fp8" else 0.0,
                "fp16": 1.0 if variant == "fp16" else 0.0,
            },
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.put(
                    f"{orchestrator_url}/v1/services/llava/config", json=config_update
                ) as response:
                    if response.status == 200:
                        logger.info(f"Successfully deployed {variant} as primary")
                    else:
                        logger.error(
                            f"Failed to deploy {variant}: {await response.text()}"
                        )

        except Exception as e:
            logger.error(f"Deployment error: {e}")


async def main():
    """Main A/B testing loop"""
    config = ABTestConfig()
    manager = ABTestManager(config)

    logger.info(f"Starting A/B test: {config.test_name}")
    logger.info(
        f"Traffic split: {config.traffic_split*100:.0f}% to "
        f"{config.variant_a}, {(1-config.traffic_split)*100:.0f}% to "
        f"{config.variant_b}"
    )

    # Run continuous evaluation
    evaluation_interval = 300  # 5 minutes

    while True:
        try:
            # Wait for evaluation interval
            await asyncio.sleep(evaluation_interval)

            # Evaluate test
            await manager.automated_decision()

            # Check if test duration exceeded
            test_duration = (
                datetime.now() - manager.test_start_time
            ).total_seconds() / 3600
            if test_duration > config.test_duration_hours:
                logger.info(f"Test duration exceeded ({test_duration:.1f} hours)")

                # Final evaluation
                final_evaluation = await manager.evaluate_test()
                logger.info(
                    f"Final test results: {json.dumps(final_evaluation, indent=2)}"
                )

                # Execute final recommendation
                if final_evaluation["status"] == "complete":
                    await manager.execute_recommendation(
                        final_evaluation["recommendation"]
                    )

                break

        except Exception as e:
            logger.error(f"A/B testing error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
