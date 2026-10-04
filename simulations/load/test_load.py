#!/usr/bin/env python3
"""
PCP Load Testing
==================

Load testing scenarios for PCP services.

Usage:
    python -m tests.load.test_load
"""

import asyncio
import random
import time
import unittest
from typing import Dict, List, Any
from dataclasses import dataclass, field


@dataclass
class LoadTestResult:
    """Load test result."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    throughput: float
    error_rate: float
    duration: float


class LoadGenerator:
    """Generate load for testing."""

    def __init__(self):
        self._results: List[LoadTestResult] = []

    async def simulate_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate a single request."""
        start_time = time.time()

        await asyncio.sleep(random.uniform(0.01, 0.1))

        response_time = time.time() - start_time
        success = random.random() > 0.05

        return {
            "endpoint": endpoint,
            "success": success,
            "response_time": response_time,
            "payload": payload,
        }

    async def run_load_test(
        self,
        endpoint: str,
        concurrent_users: int,
        requests_per_user: int
    ) -> LoadTestResult:
        """Run load test."""
        start_time = time.time()
        results: List[Dict[str, Any]] = []

        async def user_workflow():
            for _ in range(requests_per_user):
                result = await self.simulate_request(endpoint, {"test": "data"})
                results.append(result)

        tasks = [user_workflow() for _ in range(concurrent_users)]
        await asyncio.gather(*tasks)

        duration = time.time() - start_time
        response_times = [r["response_time"] for r in results if r["success"]]

        return LoadTestResult(
            total_requests=len(results),
            successful_requests=sum(1 for r in results if r["success"]),
            failed_requests=sum(1 for r in results if not r["success"]),
            avg_response_time=sum(response_times) / len(response_times) if response_times else 0,
            min_response_time=min(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            throughput=len(results) / duration,
            error_rate=sum(1 for r in results if not r["success"]) / len(results) * 100,
            duration=duration,
        )


class TestLoadGeneration(unittest.TestCase):
    """Test load generation."""

    def setUp(self):
        self.generator = LoadGenerator()

    def test_single_request(self):
        result = asyncio.run(
            self.generator.simulate_request("/api/test", {"key": "value"})
        )
        self.assertIn("response_time", result)

    def test_concurrent_load(self):
        result = asyncio.run(
            self.generator.run_load_test("/api/robots", 10, 5)
        )
        self.assertEqual(result.total_requests, 50)
        self.assertGreater(result.throughput, 0)

    def test_response_time_stats(self):
        result = asyncio.run(
            self.generator.run_load_test("/api/status", 5, 10)
        )
        self.assertGreater(result.avg_response_time, 0)
        self.assertLessEqual(result.min_response_time, result.avg_response_time)
        self.assertGreaterEqual(result.max_response_time, result.avg_response_time)


if __name__ == "__main__":
    unittest.main()