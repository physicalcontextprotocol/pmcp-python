#!/usr/bin/env python3
"""
PCP Chaos Engineering Tests
==============================

Chaos engineering scenarios for PCP services.

Usage:
    python -m tests.chaos.test_chaos
"""

import asyncio
import random
import unittest
from typing import Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum


class FailureType(Enum):
    """Types of failures to inject."""
    NETWORK_TIMEOUT = "timeout"
    NETWORK_ERROR = "error"
    SERVICE_UNAVAILABLE = "unavailable"
    CACHE_MISS = "cache_miss"
    RATE_LIMIT = "rate_limit"


@dataclass
class ChaosConfig:
    """Chaos experiment configuration."""
    failure_type: FailureType
    probability: float
    duration_seconds: int
    affected_endpoints: List[str]


@dataclass
class ChaosResult:
    """Result of chaos experiment."""
    experiment_id: str
    total_requests: int
    affected_requests: int
    recovered_requests: int
    success_rate: float
    avg_recovery_time: float


class ChaosEngine:
    """Engine for chaos engineering experiments."""

    def __init__(self):
        self._experiments: List[ChaosResult] = []
        self._active_failures: Dict[str, bool] = {}

    def inject_failure(self, endpoint: str, config: ChaosConfig) -> bool:
        """Inject failure into endpoint."""
        if random.random() > config.probability:
            return False

        self._active_failures[endpoint] = True
        return True

    def should_fail(self, endpoint: str) -> bool:
        """Check if request should fail."""
        return self._active_failures.get(endpoint, False)

    def recover_endpoint(self, endpoint: str):
        """Recover endpoint from failure."""
        if endpoint in self._active_failures:
            del self._active_failures[endpoint]

    async def run_experiment(
        self,
        experiment_id: str,
        endpoints: List[str],
        config: ChaosConfig,
        num_requests: int
    ) -> ChaosResult:
        """Run chaos experiment."""
        results = []

        for _ in range(num_requests):
            endpoint = random.choice(endpoints)

            if self.inject_failure(endpoint, config):
                await asyncio.sleep(config.duration_seconds / 1000)
                self.recover_endpoint(endpoint)

            is_affected = self.should_fail(endpoint)
            results.append({
                "endpoint": endpoint,
                "affected": is_affected,
            })

            await asyncio.sleep(0.01)

        affected = sum(1 for r in results if r["affected"])
        recovered = len(results) - affected

        return ChaosResult(
            experiment_id=experiment_id,
            total_requests=num_requests,
            affected_requests=affected,
            recovered_requests=recovered,
            success_rate=recovered / num_requests * 100,
            avg_recovery_time=config.duration_seconds / 1000,
        )


class TestChaosEngineering(unittest.TestCase):
    """Test chaos engineering scenarios."""

    def setUp(self):
        self.engine = ChaosEngine()

    def test_failure_injection(self):
        config = ChaosConfig(
            failure_type=FailureType.NETWORK_TIMEOUT,
            probability=1.0,
            duration_seconds=100,
            affected_endpoints=["/api/robots"],
        )

        result = self.engine.inject_failure("/api/robots", config)
        self.assertTrue(result)

    def test_failure_probability(self):
        config = ChaosConfig(
            failure_type=FailureType.NETWORK_ERROR,
            probability=0.0,
            duration_seconds=100,
            affected_endpoints=["/api/status"],
        )

        failures = 0
        for _ in range(100):
            if self.engine.inject_failure("/api/status", config):
                failures += 1

        self.assertEqual(failures, 0)

    def test_endpoint_recovery(self):
        config = ChaosConfig(
            failure_type=FailureType.SERVICE_UNAVAILABLE,
            probability=1.0,
            duration_seconds=50,
            affected_endpoints=["/api/health"],
        )

        self.engine.inject_failure("/api/health", config)
        self.assertTrue(self.engine.should_fail("/api/health"))

        self.engine.recover_endpoint("/api/health")
        self.assertFalse(self.engine.should_fail("/api/health"))

    def test_chaos_experiment(self):
        config = ChaosConfig(
            failure_type=FailureType.RATE_LIMIT,
            probability=0.3,
            duration_seconds=100,
            affected_endpoints=["/api/robots", "/api/zones"],
        )

        result = asyncio.run(
            self.engine.run_experiment(
                "exp-001",
                ["/api/robots", "/api/zones"],
                config,
                100
            )
        )

        self.assertEqual(result.total_requests, 100)
        self.assertGreater(result.success_rate, 50)


if __name__ == "__main__":
    unittest.main()