"""
P-MCP SDK — PCPClient
======================
The LLM / orchestration side of P-MCP.  Analogous to MCP's mcp.client.Client.

Usage:

    from pcp.client import PCPClient

    async with PCPClient() as client:
        # Connect to a local robot server via stdio
        await client.connect_stdio(["python", "my_robot_server.py"])

        # Or connect over HTTP
        await client.connect_http("http://arm-01.local:8080")

        # Discover capabilities
        actuations = await client.list_actuations()
        sensors    = await client.list_sensors()

        # Safety: preview before executing
        preview = await client.shadow_preview("move_to", arguments={"x": 0.3, "y": 0, "z": 0.5})
        if preview["preview"]["safe"]:
            lease  = await client.request_lease(zone_id="workspace-A")
            result = await client.call_actuation("move_to", arguments={"x": 0.3, "y": 0, "z": 0.5},
                                                  lease_token=lease["lease"]["lease_id"])
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from pcp.types import (
    PCP_VERSION,
    ClientInfo,
    PCPErrorCode,
)

log = logging.getLogger("pcp.client")


class PCPClientError(Exception):
    """Raised when the server returns an error response."""

    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


# ─────────────────────────────────────────────────────────────────────────────
#  TRANSPORT ABSTRACTIONS
# ─────────────────────────────────────────────────────────────────────────────


class _Transport:
    async def send(self, message: dict) -> dict:
        raise NotImplementedError

    async def close(self):
        raise NotImplementedError


class _StdioTransport(_Transport):
    """Launch a subprocess and communicate over its stdin/stdout."""

    def __init__(self, argv: List[str]):
        self._argv = argv
        self._proc: Optional[asyncio.subprocess.Process] = None

    async def connect(self):
        self._proc = await asyncio.create_subprocess_exec(
            *self._argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def send(self, message: dict) -> dict:
        if not self._proc:
            raise RuntimeError("Not connected")
        proc = self._proc
        # Requested with stdin=PIPE, stdout=PIPE above, so these are
        # guaranteed non-None -- asserted explicitly (rather than relying on
        # the self._proc None-check above, which mypy can't use to narrow
        # self._proc.stdin/.stdout: instance attributes aren't narrowed
        # across statements the way local variables are, since another
        # coroutine could in principle reassign them in between).
        assert proc.stdin is not None
        assert proc.stdout is not None
        line = (json.dumps(message) + "\n").encode()
        proc.stdin.write(line)
        await proc.stdin.drain()
        raw = await proc.stdout.readline()
        return json.loads(raw.decode().strip())

    async def close(self):
        if self._proc:
            self._proc.terminate()
            await self._proc.wait()


class _HTTPTransport(_Transport):
    """HTTP POST transport — communicates with a P-MCP HTTP server."""

    def __init__(self, base_url: str):
        self._base = base_url.rstrip("/")
        self._client = None

    async def connect(self):
        try:
            import httpx

            self._client = httpx.AsyncClient(base_url=self._base, timeout=30.0)
        except ImportError:
            # Fallback: use urllib (stdlib only, slower)
            self._client = None

    async def send(self, message: dict) -> dict:
        if self._client is not None:
            # httpx path
            resp = await self._client.post("/pcp", json=message)
            return resp.json()
        else:
            # stdlib urllib path
            import urllib.request

            url = self._base + "/pcp"
            # Reject file:// and other unexpected schemes -- self._base is
            # developer-configured, but validating avoids ever silently
            # honoring a misconfigured non-http(s) connection string.
            if not url.startswith(("http://", "https://")):
                raise ValueError(f"Refusing non-http(s) URL: {url!r}")
            data = json.dumps(message).encode()
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(
                req, timeout=30
            ) as r:  # nosec B310 -- scheme validated above (http/https only)
                return json.loads(r.read())

    async def close(self):
        if self._client is not None:
            await self._client.aclose()


class _InProcessTransport(_Transport):
    """
    In-process transport for testing.
    Passes messages directly to a PCPServer instance.
    """

    def __init__(self, server):
        self._server = server

    async def connect(self):
        pass

    async def send(self, message: dict) -> dict:
        resp = await self._server.handle_message(message)
        return resp or {}

    async def close(self):
        pass


# ─────────────────────────────────────────────────────────────────────────────
#  PCP CLIENT
# ─────────────────────────────────────────────────────────────────────────────


class PCPClient:
    """
    P-MCP Client — the LLM / orchestration side.

    Context manager:
        async with PCPClient() as client:
            await client.connect_http("http://arm-01.local:8080")
            ...
    """

    def __init__(self, name: str = "pcp-client", version: str = "0.4.0"):
        self._info = ClientInfo(name=name, version=version)
        self._transport: Optional[_Transport] = None
        self._server_info: dict = {}
        self._capabilities: dict = {}
        self._initialized: bool = False

    # ── Connection methods ────────────────────────────────────────────────────

    async def connect_stdio(self, argv: List[str]) -> "PCPClient":
        """Launch a robot server subprocess and connect via stdio."""
        transport = _StdioTransport(argv)
        await transport.connect()
        self._transport = transport
        await self._initialize()
        return self

    async def connect_http(self, base_url: str) -> "PCPClient":
        """Connect to a P-MCP HTTP server."""
        transport = _HTTPTransport(base_url)
        await transport.connect()
        self._transport = transport
        await self._initialize()
        return self

    async def connect_inprocess(self, server) -> "PCPClient":
        """
        Connect directly to a PCPServer in the same process.
        Ideal for testing and embedded deployments.

            server = PCPServer("test-robot")
            client = PCPClient()
            await client.connect_inprocess(server)
        """
        transport = _InProcessTransport(server)
        await transport.connect()
        self._transport = transport
        await self._initialize()
        return self

    async def disconnect(self):
        if self._transport:
            await self._transport.close()
            self._transport = None
            self._initialized = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def _initialize(self):
        resp = await self._rpc(
            "initialize",
            {
                "protocolVersion": PCP_VERSION,
                "clientInfo": self._info.to_dict(),
                "capabilities": {},
            },
        )
        self._server_info = resp.get("serverInfo", {})
        self._capabilities = resp.get("capabilities", {})
        self._initialized = True

        # Send initialized notification
        await self._notify("notifications/initialized", {})
        log.info(
            f"  🤝  Connected to P-MCP server: "
            f"{self._server_info.get('name', '?')} "
            f"v{self._server_info.get('version', '?')}"
        )

    # ── Actuations ────────────────────────────────────────────────────────────

    async def list_actuations(self) -> List[dict]:
        resp = await self._rpc("actuations/list", {})
        return resp.get("actuations", [])

    async def call_actuation(
        self,
        name: str,
        arguments: Dict[str, Any],
        robot_id: str = "",
        lease_token: str = "",
        fence_token: Optional[int] = None,
    ) -> dict:
        """
        Execute a physical actuation on the robot.

        fence_token should be the "fence_token" returned with the lease
        grant (see request_lease). Presenting a stale fence token is
        rejected by the server even if lease_token still lexically matches
        -- see PCPServer._handle_actuations_call.

        Returns the ActuationResult dict on success.
        Raises PCPClientError on safety block or hardware failure.
        """
        params: dict = {"name": name, "arguments": arguments}
        if robot_id:
            params["robot_id"] = robot_id
        if lease_token:
            params["lease_token"] = lease_token
        if fence_token is not None:
            params["fence_token"] = fence_token
        resp = await self._rpc("actuations/call", params)
        content = resp.get("content", [{}])
        return content[0].get("data", resp) if content else resp

    # ── Sensors ───────────────────────────────────────────────────────────────

    async def list_sensors(self) -> List[dict]:
        resp = await self._rpc("sensors/list", {})
        return resp.get("sensors", [])

    async def read_sensor(self, name: str) -> dict:
        resp = await self._rpc("sensors/read", {"name": name})
        contents = resp.get("contents", [])
        return contents[0].get("data", {}) if contents else {}

    # ── Prompts ───────────────────────────────────────────────────────────────

    async def list_prompts(self) -> List[dict]:
        resp = await self._rpc("prompts/list", {})
        return resp.get("prompts", [])

    async def get_prompt(self, name: str, arguments: Dict[str, Any] = {}) -> dict:
        return await self._rpc("prompts/get", {"name": name, "arguments": arguments})

    # ── Physical Safety Layer ─────────────────────────────────────────────────

    async def shadow_preview(
        self,
        actuation_name: str,
        arguments: Dict[str, Any] = {},
        robot_id: str = "",
    ) -> dict:
        """
        Run a ghost simulation of the actuation without touching hardware.
        Returns a ShadowPreview dict — check ['preview']['safe'] before executing.

            preview = await client.shadow_preview("move_to", {"x":0.3,"y":0,"z":0.5})
            assert preview["preview"]["safe"], "Trajectory blocked by shadow!"
        """
        params: dict = {"actuation_name": actuation_name, "arguments": arguments}
        if robot_id:
            params["robot_id"] = robot_id
        return await self._rpc("shadow/preview", params)

    async def request_lease(
        self,
        zone_id: str,
        robot_id: str = "",
        duration_ms: int = 10_000,
        bid_energy_j: float = 100.0,
    ) -> dict:
        """
        Request exclusive temporal zone lease (collision prevention).
        Returns a LeaseGrant dict -- check ['lease']['state'] == 'ACTIVE'.
        Hold onto ['lease']['fence_token'] and re-present it on every
        subsequent call_actuation() against this zone (see fence_token
        docs on call_actuation).
        """
        return await self._rpc(
            "lease/request",
            {
                "robot_id": robot_id,
                "zone_id": zone_id,
                "duration_ms": duration_ms,
                "bid_energy_j": bid_energy_j,
            },
        )

    async def release_lease(self, lease_id: str) -> dict:
        return await self._rpc("lease/release", {"lease_id": lease_id})

    # ── E-Stop ───────────────────────────────────────────────────────────────

    async def estop(self, robot_id: str = "", source: Optional[str] = None) -> dict:
        """
        Trigger an emergency stop. First-class, lease-independent -- bypasses
        the lease/constitution/shadow gate pipeline entirely on the server
        side. source: one of "hardware_button", "software_watchdog",
        "operator_console", "gate_failure_escalation".
        """
        params: dict = {}
        if robot_id:
            params["robot_id"] = robot_id
        if source:
            params["source"] = source
        return await self._rpc("pcp/estop", params)

    async def estop_reset(self, robot_id: str = "") -> dict:
        """Clear the E-Stop latch. Always a distinct, explicit, auditable call."""
        params: dict = {}
        if robot_id:
            params["robot_id"] = robot_id
        return await self._rpc("pcp/estop_reset", params)

    # ── Convenience: Safe Actuation (shadow + lease + call) ───────────────────

    async def safe_actuation(
        self,
        name: str,
        arguments: Dict[str, Any],
        zone_id: str = "default",
        robot_id: str = "",
    ) -> Tuple[dict, dict]:
        """
        Full P-MCP safe execution flow in one call:
          1. shadow/preview  — simulate
          2. lease/request   — claim zone
          3. actuations/call — execute on hardware
          4. lease/release   — free zone

        Returns (ActuationResult dict, ShadowPreview dict).
        Raises PCPClientError if shadow blocks the trajectory.

            result, shadow = await client.safe_actuation(
                "move_to", {"x": 0.3, "y": 0, "z": 0.5},
                zone_id="workspace-A")
        """
        # 1. Shadow preview
        shadow_resp = await self.shadow_preview(name, arguments, robot_id)
        preview = shadow_resp.get("preview", {})
        if not preview.get("safe", True):
            raise PCPClientError(
                int(PCPErrorCode.SHADOW_BLOCKED),
                f"Shadow preview blocked trajectory: {preview.get('violations', [])}",
                data=preview,
            )

        # 2. Request lease
        lease_resp = await self.request_lease(zone_id=zone_id, robot_id=robot_id)
        lease = lease_resp.get("lease", {})
        if lease.get("state") != "ACTIVE":
            raise PCPClientError(
                int(PCPErrorCode.LEASE_REQUIRED),
                f"Zone lease denied: {lease.get('deny_reason', 'unknown')}",
                data=lease,
            )
        lease_id = lease.get("lease_id", "")
        fence_token = lease.get("fence_token")

        # 3. Execute
        try:
            result = await self.call_actuation(
                name, arguments, robot_id, lease_id, fence_token=fence_token
            )
        finally:
            # 4. Release lease (even on failure)
            if lease_id:
                await self.release_lease(lease_id)

        return result, preview

    async def batch_execute(
        self,
        actuations: List[Dict[str, Any]],
        zone_id: str = "default",
        robot_id: str = "",
        atomic: bool = True,
    ) -> dict:
        """
        Execute multiple actuations atomically under a shared lease.

        ``actuations`` is a list of ``{"name": ..., "arguments": {...}}`` dicts.
        When ``atomic=True`` (default), the first safety failure aborts the batch.

            result = await client.batch_execute([
                {"name": "move_to",  "arguments": {"x": 0.3, "y": 0, "z": 0.5}},
                {"name": "grip",     "arguments": {"force_n": 20}},
                {"name": "move_to",  "arguments": {"x": 0.0, "y": 0, "z": 0.8}},
            ], zone_id="workspace-A")
        """
        params: dict = {
            "actuations": actuations,
            "zone_id": zone_id,
            "atomic": atomic,
        }
        if robot_id:
            params["robot_id"] = robot_id
        resp = await self._rpc("actuations/batch", params)
        return resp.get("batch", resp)

    # ── Observability ─────────────────────────────────────────────────────────

    async def get_metrics(self) -> dict:
        """
        Retrieve a point-in-time telemetry snapshot from the server.

        Returns a MetricsSnapshot dict with call counts, block rates,
        active leases, and registration totals.

            metrics = await client.get_metrics()
            print(f"Block rate: {metrics['callsBlocked']} / {metrics['callsTotal']}")
        """
        resp = await self._rpc("metrics/get", {})
        return resp.get("metrics", resp)

    async def get_audit_log(
        self,
        limit: int = 100,
        robot_id: str = "",
        event_type: str = "",
        since: float = 0.0,
    ) -> List[dict]:
        """
        Retrieve recent audit log entries from the server.

        Supports filtering by robot_id, event_type, and minimum timestamp.
        Returns the most recent ``limit`` matching entries.

            entries = await client.get_audit_log(event_type="constitution_blocked")
            for e in entries:
                print(e["timestamp"], e["violations"])
        """
        params: dict = {"limit": limit}
        if robot_id:
            params["robot_id"] = robot_id
        if event_type:
            params["event_type"] = event_type
        if since:
            params["since"] = since
        resp = await self._rpc("audit/list", params)
        return resp.get("entries", [])

    # ── Utils ─────────────────────────────────────────────────────────────────

    async def ping(self) -> dict:
        return await self._rpc("ping", {})

    @property
    def server_info(self) -> dict:
        return self._server_info

    @property
    def capabilities(self) -> dict:
        return self._capabilities

    # ── Internal RPC ─────────────────────────────────────────────────────────

    async def _rpc(self, method: str, params: dict) -> dict:
        if not self._transport:
            raise RuntimeError("Client not connected. Call connect_* first.")
        msg_id = str(uuid.uuid4())[:8]
        request = {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}
        resp = await self._transport.send(request)
        if "error" in resp and resp["error"]:
            err = resp["error"]
            raise PCPClientError(err.get("code", -1), err.get("message", ""), err.get("data"))
        return resp.get("result", resp)

    async def _notify(self, method: str, params: dict):
        if not self._transport:
            return
        note = {"jsonrpc": "2.0", "method": method, "params": params}
        try:
            await self._transport.send(note)
        except Exception:
            pass  # Notifications can silently fail

    # ── Context manager ───────────────────────────────────────────────────────

    async def __aenter__(self) -> "PCPClient":
        return self

    async def __aexit__(self, *_):
        await self.disconnect()
