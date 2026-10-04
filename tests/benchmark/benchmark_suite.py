#!/usr/bin/env python3
"""
PCP Performance Benchmarks
============================

Benchmark suite for PCP services.

Usage:
    python -m tests.benchmark.benchmark_suite
"""

import asyncio
import time
import unittest
from typing import Dict, List, Any, Callable
from dataclasses import dataclass, field
import statistics


@dataclass
class BenchmarkResult:
    """Benchmark result."""
    name: str
    iterations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    stddev: float
    ops_per_second: float


class BenchmarkSuite:
    """Performance benchmark suite."""

    def __init__(self):
        self._results: List[BenchmarkResult] = []

    def benchmark(
        self,
        name: str,
        func: Callable,
        iterations: int = 1000,
        warmup: int = 10
    ) -> BenchmarkResult:
        """Run benchmark."""
        for _ in range(warmup):
            func()

        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = statistics.mean(times)
        stddev = statistics.stdev(times) if len(times) > 1 else 0

        result = BenchmarkResult(
            name=name,
            iterations=iterations,
            total_time=sum(times),
            avg_time=avg_time,
            min_time=min(times),
            max_time=max(times),
            stddev=stddev,
            ops_per_second=1 / avg_time if avg_time > 0 else 0,
        )

        self._results.append(result)
        return result

    def async_benchmark(
        self,
        name: str,
        func: Callable,
        iterations: int = 100,
        warmup: int = 10
    ) -> BenchmarkResult:
        """Run async benchmark."""
        async def warmup_func():
            for _ in range(warmup):
                await func()

        async def run_func():
            start = time.perf_counter()
            await func()
            return time.perf_counter() - start

        asyncio.run(warmup_func())

        times = []
        for _ in range(iterations):
            elapsed = asyncio.run(run_func())
            times.append(elapsed)

        avg_time = statistics.mean(times)

        result = BenchmarkResult(
            name=name,
            iterations=iterations,
            total_time=sum(times),
            avg_time=avg_time,
            min_time=min(times),
            max_time=max(times),
            stddev=statistics.stdev(times) if len(times) > 1 else 0,
            ops_per_second=1 / avg_time if avg_time > 0 else 0,
        )

        self._results.append(result)
        return result

    def get_results(self) -> List[BenchmarkResult]:
        """Get all benchmark results."""
        return self._results

    def print_results(self):
        """Print benchmark results."""
        print("\n=== Benchmark Results ===")
        print(f"{'Name':<30} {'Ops/sec':<15} {'Avg (ms)':<12} {'StdDev':<12}")
        print("-" * 70)
        for r in self._results:
            print(f"{r.name:<30} {r.ops_per_second:>10.2f}   {r.avg_time * 1000:>8.4f}   {r.stddev * 1000:>8.4f}")


class TestBenchmarkSuite(unittest.TestCase):
    """Test benchmark suite."""

    def setUp(self):
        self.suite = BenchmarkSuite()

    def test_sync_benchmark(self):
        def simple_function():
            result = 0
            for i in range(100):
                result += i
            return result

        result = self.suite.benchmark("simple_calc", simple_function, iterations=100)
        self.assertGreater(result.ops_per_second, 0)
        self.assertLess(result.avg_time, 1.0)

    def test_string_operations(self):
        def string_concat():
            s = ""
            for i in range(100):
                s += str(i)
            return s

        result = self.suite.benchmark("string_concat", string_concat, iterations=50)
        self.assertIsNotNone(result)

    def test_dict_operations(self):
        def dict_operations():
            d = {}
            for i in range(100):
                d[f"key_{i}"] = i
            return len(d)

        result = self.suite.benchmark("dict_ops", dict_operations, iterations=100)
        self.assertGreater(result.ops_per_second, 0)

    def test_list_operations(self):
        def list_operations():
            lst = []
            for i in range(100):
                lst.append(i)
            lst.sort()
            return len(lst)

        result = self.suite.benchmark("list_ops", list_operations, iterations=100)
        self.assertGreater(result.ops_per_second, 0)

    def test_json_serialization(self):
        import json

        data = {"key": "value", "number": 123, "nested": {"a": 1, "b": 2}}

        def json_serialize():
            return json.dumps(data)

        result = self.suite.benchmark("json_serialize", json_serialize, iterations=200)
        self.assertGreater(result.ops_per_second, 100)

    def test_async_benchmark(self):
        async def async_operation():
            await asyncio.sleep(0.001)
            return True

        result = self.suite.async_benchmark("async_sleep", async_operation, iterations=50)
        self.assertGreater(result.ops_per_second, 0)


class TestPerformance(unittest.TestCase):
    """Test performance characteristics."""

    def test_comparison(self):
        """Compare different implementations."""
        suite = BenchmarkSuite()

        def implementation_a():
            result = 0
            for i in range(1000):
                result += i
            return result

        def implementation_b():
            return sum(range(1000))

        result_a = suite.benchmark("impl_a_loop", implementation_a, iterations=100)
        result_b = suite.benchmark("impl_b_sum", implementation_b, iterations=100)

        self.assertGreater(result_b.ops_per_second, result_a.ops_per_second)


if __name__ == "__main__":
    unittest.main()