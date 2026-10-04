"""
PCP SDK — PCPClient

.. deprecated::
    This ``sdk/`` package predates the ``pcp/`` package restructuring and
    is kept only for backward compatibility. The canonical client is
    :class:`pcp.client.PCPClient`. New code should import from ``pcp``,
    not ``sdk``.
======================
The LLM / orchestration side of PCP.  Analogous to MCP's mcp.client.Client.

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
                                                  lease_token=lease["lease"]["leaseId"])
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
import warnings as _warnings
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from pcp.types import (
    ClientInfo, PCP_VERSION,
    PCPError, PCPErrorCode,
)

_warnings.warn(
    "sdk.client is deprecated and will be removed in a future release; "
    "use pcp.client.PCPClient instead.",
    DeprecationWarning,
    stacklevel=2,
)

log = logging.getLogger("pcp.client")


class PCPClientError(Exception):
    """Raised when the server returns an error response."""
    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code    = code
        self.message = message
        self.data    = data


# ─────────────────────────────────────────────────────────────────────────────
#  TRANSPORT ABSTRACTIONS
# ─────────────────────────────────────────────────────────────────────────────

class _Transport:
    async def send(self, message: dict) -> dict: ...
    async def close(self): ...


class _StdioTransport(_Transport):
    """Launch a subprocess and communicate over its stdin/stdout."""

    def __init__(self, argv: List[str]):
        self._argv    = argv
        self._proc:   Optional[asyncio.subprocess.Process] = None

    async def connect(self):
        self._proc = await asyncio.create_subprocess_exec(
            *self._argv,
            stdin  = asyncio.subprocess.PIPE,
            stdout = asyncio.subprocess.PIPE,
            stderr = asyncio.subprocess.PIPE,
        )

    async def send(self, message: dict) -> dict:
        if not self._proc:
            raise RuntimeError("Not connected")
        line = (json.dumps(message) + "\n").encode()
        self._proc.stdin.write(line)
        await self._proc.stdin.drain()
        raw = await self._proc.stdout.readline()
        return json.loads(raw.decode().strip())

    async def close(self):
        if self._proc:
            self._proc.terminate()
            await self._proc.wait()


class _HTTPTransport(_Transport):
    """HTTP POST transport — communicates with a PCP HTTP server."""

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
            data = json.dumps(message).encode()
            req  = urllib.request.Request(
                self._base + "/pcp", data=data,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
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
    PCP Client — the LLM / orchestration side.

    Context manager:
        async with PCPClient() as client:
            await client.connect_http("http://arm-01.local:8080")
            ...
    """

    def __init__(self, name: str = "pcp-client", version: str = "0.4.0"):
        self._info      = ClientInfo(name=name, version=version)
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
        """Connect to a PCP HTTP server."""
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
            self._transport   = None
            self._initialized = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def _initialize(self):
        resp = await self._rpc("initialize", {
            "protocolVersion": PCP_VERSION,
            "clientInfo":      self._info.to_dict(),
            "capabilities":    {},
        })
        self._server_info   = resp.get("serverInfo", {})
        self._capabilities  = resp.get("capabilities", {})
        self._initialized   = True

        # Send initialized notification
        await self._notify("notifications/initialized", {})
        log.info(f"  🤝  Connected to PCP server: "
                 f"{self._server_info.get('name', '?')} "
                 f"v{self._server_info.get('version', '?')}")

    # ── Actuations ────────────────────────────────────────────────────────────

    async def list_actuations(self) -> List[dict]:
        resp = await self._rpc("actuations/list", {})
        return resp.get("actuations", [])

    async def call_actuation(
        self,
        name:        str,
        arguments:   Dict[str, Any],
        robot_id:    str = "",
        lease_token: str = "",
    ) -> dict:
        """
        Execute a physical actuation on the robot.

        Returns the ActuationResult dict on success.
        Raises PCPClientError on safety block or hardware failure.
        """
        params: dict = {"name": name, "arguments": arguments}
        if robot_id:     params["robotId"]    = robot_id
        if lease_token:  params["leaseToken"] = lease_token
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
        arguments:      Dict[str, Any] = {},
        robot_id:       str = "",
    ) -> dict:
        """
        Run a ghost simulation of the actuation without touching hardware.
        Returns a ShadowPreview dict — check ['preview']['safe'] before executing.

            preview = await client.shadow_preview("move_to", {"x":0.3,"y":0,"z":0.5})
            assert preview["preview"]["safe"], "Trajectory blocked by shadow!"
        """
        params: dict = {"actuationName": actuation_name, "arguments": arguments}
        if robot_id: params["robotId"] = robot_id
        return await self._rpc("shadow/preview", params)

    async def request_lease(
        self,
        zone_id:     str,
        robot_id:    str = "",
        duration_ms: int = 10_000,
        bid_energy_j: float = 100.0,
    ) -> dict:
        """
        Request exclusive temporal zone lease (collision prevention).
        Returns a LeaseGrant dict — check ['lease']['granted'].
        """
        return await self._rpc("lease/request", {
            "robotId":     robot_id,
            "zoneId":      zone_id,
            "durationMs":  duration_ms,
            "bidEnergyJ":  bid_energy_j,
        })

    async def release_lease(self, lease_id: str) -> dict:
        return await self._rpc("lease/release", {"leaseId": lease_id})

    # ── Convenience: Safe Actuation (shadow + lease + call) ───────────────────

    async def safe_actuation(
        self,
        name:        str,
        arguments:   Dict[str, Any],
        zone_id:     str = "default",
        robot_id:    str = "",
    ) -> Tuple[dict, dict]:
        """
        Full PCP safe execution flow in one call:
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
        preview     = shadow_resp.get("preview", {})
        if not preview.get("safe", True):
            raise PCPClientError(
                int(PCPErrorCode.SHADOW_BLOCKED),
                f"Shadow preview blocked trajectory: {preview.get('violations', [])}",
                data=preview,
            )

        # 2. Request lease
        lease_resp = await self.request_lease(zone_id=zone_id, robot_id=robot_id)
        lease      = lease_resp.get("lease", {})
        if not lease.get("granted", True):
            raise PCPClientError(
                int(PCPErrorCode.LEASE_REQUIRED),
                f"Zone lease denied: {lease.get('denyReason', 'unknown')}",
                data=lease,
            )
        lease_id = lease.get("leaseId", "")

        # 3. Execute
        try:
            result = await self.call_actuation(name, arguments, robot_id, lease_id)
        finally:
            # 4. Release lease (even on failure)
            if lease_id:
                await self.release_lease(lease_id)

        return result, preview

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
        msg_id  = str(uuid.uuid4())[:8]
        request = {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}
        resp    = await self._transport.send(request)
        if "error" in resp and resp["error"]:
            err = resp["error"]
            raise PCPClientError(err.get("code", -1), err.get("message", ""),
                                  err.get("data"))
        return resp.get("result", resp)

    async def _notify(self, method: str, params: dict):
        if not self._transport:
            return
        note = {"jsonrpc": "2.0", "method": method, "params": params}
        try:
            await self._transport.send(note)
        except Exception:
            pass   # Notifications can silently fail

    # ── Context manager ───────────────────────────────────────────────────────

    async def __aenter__(self) -> "PCPClient":
        return self

    async def __aexit__(self, *_):
        await self.disconnect()
