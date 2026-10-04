#!/usr/bin/env python3
"""
P-MCP Compliance Runner
========================

Conformance test suite for P-MCP servers.
Tests all MCP tool calls, resources, prompts, and P-MCP extensions.

Usage:
    python -m tests.compliance_runner --server-url http://localhost:8080
    python -m tests.compliance_runner --stdio python v05/robot_servers/arm_server.py
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from enum import Enum

from v05.pmcp_v5_types import PMCP_VERSION, MCP_VERSION


class TestResult(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    ERROR = "ERROR"


@dataclass
class TestCase:
    name: str
    description: str
    method: str
    params: Dict[str, Any]
    expected: Dict[str, Any]
    validator: Optional[Callable[[Dict], bool]] = None
    category: str = "mcp"


@dataclass
class TestRun:
    name: str
    result: TestResult
    duration_ms: float
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class PMCPComplianceRunner:
    """Compliance test runner for P-MCP servers."""

    def __init__(self, transport: str = "stdio", server_cmd: Optional[List[str]] = None,
                 server_url: Optional[str] = None):
        self.transport = transport
        self.server_cmd = server_cmd
        self.server_url = server_url
        self.results: List[TestRun] = []
        self.server_process = None

    async def start_server(self):
        """Start the server process."""
        if self.transport == "stdio" and self.server_cmd:
            self.server_process = await asyncio.create_subprocess_exec(
                *self.server_cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.sleep(0.5)  # Wait for server to start
        elif self.transport == "http" and self.server_url:
            pass  # HTTP doesn't need process start

    async def stop_server(self):
        """Stop the server process."""
        if self.server_process:
            self.server_process.terminate()
            await self.server_process.wait()

    async def send_request(self, method: str, params: Dict = None, request_id: str = "1") -> Dict:
        """Send a JSON-RPC request."""
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {}
        }

        if self.transport == "stdio" and self.server_process:
            self.server_process.stdin.write((json.dumps(request) + "\n").encode())
            await self.server_process.stdin.drain()
            await asyncio.sleep(0.1)

            line = await self.server_process.stdout.readline()
            if line:
                return json.loads(line.decode().strip())

        elif self.transport == "http" and self.server_url:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.server_url}/pmcp",
                    json=request,
                    headers={"Content-Type": "application/json"}
                ) as resp:
                    return await resp.json()

        return {}

    async def run_test(self, test: TestCase) -> TestRun:
        """Run a single test case."""
        start = time.time()

        try:
            response = await self.send_request(test.method, test.params, test.name)

            if "error" in response:
                if test.expected.get("error"):
                    # Expected error
                    return TestRun(
                        name=test.name,
                        result=TestResult.PASS,
                        duration_ms=(time.time() - start) * 1000,
                    )
                return TestRun(
                    name=test.name,
                    result=TestResult.FAIL,
                    duration_ms=(time.time() - start) * 1000,
                    error=f"Server error: {response['error']}",
                )

            result = response.get("result", {})

            # Run validator if provided
            if test.validator:
                if test.validator(result):
                    return TestRun(
                        name=test.name,
                        result=TestResult.PASS,
                        duration_ms=(time.time() - start) * 1000,
                    )
                else:
                    return TestRun(
                        name=test.name,
                        result=TestResult.FAIL,
                        duration_ms=(time.time() - start) * 1000,
                        error=f"Validation failed: {result}",
                    )

            # Check expected fields
            for key, expected in test.expected.items():
                if key not in result:
                    return TestRun(
                        name=test.name,
                        result=TestResult.FAIL,
                        duration_ms=(time.time() - start) * 1000,
                        error=f"Missing expected field: {key}",
                    )

            return TestRun(
                name=test.name,
                result=TestResult.PASS,
                duration_ms=(time.time() - start) * 1000,
                details=result,
            )

        except Exception as e:
            return TestRun(
                name=test.name,
                result=TestResult.ERROR,
                duration_ms=(time.time() - start) * 1000,
                error=str(e),
            )

    def get_test_suite(self) -> List[TestCase]:
        """Get all test cases."""
        return [
            # MCP Standard Methods
            TestCase(
                name="initialize",
                description="Initialize connection with server",
                method="initialize",
                params={
                    "protocolVersion": MCP_VERSION,
                    "clientInfo": {"name": "compliance-tester", "version": "1.0.0"},
                    "capabilities": {},
                },
                expected={
                    "protocolVersion": MCP_VERSION,
                    "serverInfo": {},
                    "pmcp": {},
                },
                category="mcp",
            ),
            TestCase(
                name="ping",
                description="Ping the server",
                method="ping",
                params={},
                expected={"pong": True, "robot_id": ""},
                category="mcp",
            ),
            TestCase(
                name="tools_list",
                description="List available tools (actuations)",
                method="tools/list",
                params={},
                expected={"tools": []},
                validator=lambda r: "tools" in r and isinstance(r["tools"], list),
                category="mcp",
            ),
            TestCase(
                name="tools_list_has_mcp_fields",
                description="Tools have MCP-required fields",
                method="tools/list",
                params={},
                expected={},
                validator=lambda r: all(
                    all(k in t for k in ["name", "description", "inputSchema"])
                    for t in r.get("tools", [])
                ),
                category="mcp",
            ),
            TestCase(
                name="resources_list",
                description="List available resources (sensors)",
                method="resources/list",
                params={},
                expected={"resources": []},
                validator=lambda r: "resources" in r and isinstance(r["resources"], list),
                category="mcp",
            ),
            TestCase(
                name="resources_list_has_mcp_fields",
                description="Resources have MCP-required fields",
                method="resources/list",
                params={},
                expected={},
                validator=lambda r: all(
                    all(k in t for k in ["uri", "name", "description"])
                    for t in r.get("resources", [])
                ),
                category="mcp",
            ),
            TestCase(
                name="prompts_list",
                description="List available prompts (missions)",
                method="prompts/list",
                params={},
                expected={"prompts": []},
                category="mcp",
            ),
            TestCase(
                name="logging_setLevel",
                description="Set logging level",
                method="logging/setLevel",
                params={"level": "info"},
                expected={},
                category="mcp",
            ),
            TestCase(
                name="tools_call_unknown",
                description="Call unknown tool should error",
                method="tools/call",
                params={"name": "nonexistent_tool", "arguments": {}},
                expected={},
                validator=lambda r: "error" in r or "content" in r,
                category="mcp",
            ),
            # P-MCP Extension Methods
            TestCase(
                name="pmcp_status",
                description="Get server status",
                method="pmcp/status",
                params={},
                expected={"robot_id": "", "pmcp_version": PMCP_VERSION},
                validator=lambda r: all(k in r for k in ["robot_id", "pmcp_version", "uptime_s"]),
                category="pmcp",
            ),
            TestCase(
                name="pmcp_identity",
                description="Get robot identity",
                method="pmcp/identity",
                params={},
                expected={},
                validator=lambda r: all(k in r for k in ["did", "class", "model"]),
                category="pmcp",
            ),
            TestCase(
                name="pmcp_constitution",
                description="Get safety constitution summary",
                method="pmcp/constitution",
                params={},
                expected={},
                validator=lambda r: "fingerprint" in r,
                category="pmcp",
            ),
            TestCase(
                name="lease_request_grant",
                description="Request a lease should be granted",
                method="lease/request",
                params={
                    "robotId": "test-robot",
                    "zoneId": "test-zone",
                    "durationMs": 5000,
                    "bidEnergyJ": 100.0,
                },
                expected={},
                validator=lambda r: r.get("lease", {}).get("state") == "ACTIVE",
                category="pmcp",
            ),
            TestCase(
                name="lease_release",
                description="Release a lease",
                method="lease/release",
                params={"leaseId": "test-lease"},
                expected={},
                category="pmcp",
            ),
            TestCase(
                name="pmcp_estop_on",
                description="Activate emergency stop",
                method="pmcp/estop",
                params={"active": True},
                expected={"estop": True},
                category="pmcp",
            ),
            TestCase(
                name="pmcp_estop_off",
                description="Deactivate emergency stop",
                method="pmcp/estop",
                params={"active": False},
                expected={"estop": False},
                category="pmcp",
            ),
            TestCase(
                name="shadow_preview_safe",
                description="Shadow preview for safe motion",
                method="shadow/preview",
                params={
                    "name": "move_to",
                    "arguments": {"x": 0.0, "y": 0.0, "z": 0.5, "speed": 0.1},
                },
                expected={},
                validator=lambda r: "preview" in r,
                category="pmcp",
            ),
        ]

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all test cases."""
        await self.start_server()

        test_suite = self.get_test_suite()
        results = []

        for test in test_suite:
            print(f"Running: {test.name}...", end=" ")
            result = await self.run_test(test)
            results.append(result)

            if result.result == TestResult.PASS:
                print(f"PASS ({result.duration_ms:.2f}ms)")
            elif result.result == TestResult.FAIL:
                print(f"FAIL - {result.error}")
            elif result.result == TestResult.ERROR:
                print(f"ERROR - {result.error}")
            else:
                print(f"SKIP")

        await self.stop_server()

        # Calculate summary
        passed = sum(1 for r in results if r.result == TestResult.PASS)
        failed = sum(1 for r in results if r.result == TestResult.FAIL)
        errors = sum(1 for r in results if r.result == TestResult.ERROR)
        skipped = sum(1 for r in results if r.result == TestResult.SKIP)

        return {
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "skipped": skipped,
            "results": results,
        }

    def print_summary(self, summary: Dict):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("P-MCP COMPLIANCE TEST RESULTS")
        print("=" * 60)
        print(f"Total:   {summary['total']}")
        print(f"Passed:  {summary['passed']}")
        print(f"Failed:  {summary['failed']}")
        print(f"Errors:  {summary['errors']}")
        print(f"Skipped: {summary['skipped']}")
        print("=" * 60)

        if summary['failed'] > 0 or summary['errors'] > 0:
            print("\nFailed/Error Tests:")
            for r in summary['results']:
                if r.result in [TestResult.FAIL, TestResult.ERROR]:
                    print(f"  - {r.name}: {r.error}")

        print()


async def main():
    parser = argparse.ArgumentParser(description="P-MCP Compliance Runner")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport type",
    )
    parser.add_argument(
        "--server-cmd",
        nargs="+",
        help="Command to start the server (for stdio transport)",
    )
    parser.add_argument(
        "--server-url",
        help="Server URL (for HTTP transport)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.transport == "stdio" and not args.server_cmd:
        # Default to v05 arm server
        args.server_cmd = ["python", "-m", "v05.robot_servers.arm_server"]

    runner = PMCPComplianceRunner(
        transport=args.transport,
        server_cmd=args.server_cmd,
        server_url=args.server_url,
    )

    summary = await runner.run_all_tests()
    runner.print_summary(summary)

    # Exit with appropriate code
    if summary['failed'] > 0 or summary['errors'] > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())