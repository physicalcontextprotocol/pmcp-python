"""
P-MCP v0.5 — PCPClient
========================
LLM-side MCP-compatible client for robot servers.

Works with any P-MCP server (stdio or HTTP) and any standard MCP server.

NOTE: This is the reference implementation backing the ``pcp-server`` /
``pcp-demo`` console-script entry points (see pyproject.toml) and the
v0.5 CLI demo, so it is kept fully functional and is NOT deprecated.
For new application code, prefer the top-level, package-exported client
at :class:`pcp.client.PCPClient` (``from pcp.client import PCPClient``),
which is the canonical public SDK entry point. This module remains the
internal implementation used by the CLI/demo tooling.

Usage:

    from v05.pcp_v5_client import PCPClient

    async with PCPClient("my-agent") as client:
        # Connect to robot via stdio
        await client.connect_stdio(["python", "my_robot_server.py"])

        # Or connect over HTTP
        await client.connect_http("http://arm-01.local:8080")

        # Discover capabilities
        tools    = await client.list_tools()
        sensors  = await client.list_sensors()
        missions = await client.list_missions()

        # Safety-first: preview before calling
        preview = await client.shadow_preview("move_to", {"x": 0.3, "y": 0, "z": 0.5})
        if preview["preview"]["safe"]:
            lease  = await client.request_lease("workspace-A")
            result = await client.call_tool("move_to",
                                            {"x": 0.3, "y": 0, "z": 0.5},
                                            lease_token=lease["lease"]["leaseId"])
            print(result)
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

log = logging.getLogger("pcp.client_v5")


class PCPClientError(Exception):
    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


# ─────────────────────────────────────────────────────────────────────────────
#  TRANSPORT LAYER
# ─────────────────────────────────────────────────────────────────────────────


class _Transport:
    async def send(self, msg: dict) -> dict: ...
    async def close(self): ...


class _StdioTransport(_Transport):
    def __init__(self, argv: List[str]):
        self._argv = argv
        self._proc: Optional[asyncio.subprocess.Process] = None

    async def connect(self):
        self._proc = await asyncio.create_subprocess_exec(
            *self._argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

    async def send(self, msg: dict) -> dict:
        assert self._proc
        line = (json.dumps(msg) + "\n").encode()
        self._proc.stdin.write(line)
        await self._proc.stdin.drain()
        raw = await asyncio.wait_for(self._proc.stdout.readline(), timeout=30.0)
        return json.loads(raw.decode().strip())

    async def close(self):
        if self._proc:
            self._proc.terminate()
            await self._proc.wait()


class _HTTPTransport(_Transport):
    def __init__(self, base_url: str):
        self._base = base_url.rstrip("/")
        self._session = None

    async def connect(self):
        try:
            import aiohttp

            self._session = aiohttp.ClientSession()
        except ImportError:
            self._session = None  # fallback to urllib

    async def send(self, msg: dict) -> dict:
        url = self._base + "/pcp"
        if self._session is not None:
            async with self._session.post(url, json=msg) as resp:
                return await resp.json()
        else:
            import urllib.request

            # Reject file:// and other unexpected schemes -- url is
            # developer-configured, but validating avoids ever silently
            # honoring a misconfigured non-http(s) connection string.
            if not url.startswith(("http://", "https://")):
                raise ValueError(f"Refusing non-http(s) URL: {url!r}")
            data = json.dumps(msg).encode()
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(
                req, timeout=30
            ) as r:  # nosec B310 -- scheme validated above (http/https only)
                return json.loads(r.read().decode())

    async def close(self):
        if self._session:
            await self._session.close()


class _InProcessTransport(_Transport):
    """Direct in-process transport — no serialization, for testing."""

    def __init__(self, server):
        self._server = server

    async def connect(self):
        pass

    async def send(self, msg: dict) -> dict:
        resp = await self._server.handle_message(msg)
        return resp or {}

    async def close(self):
        pass


# ─────────────────────────────────────────────────────────────────────────────
#  PCP CLIENT
# ─────────────────────────────────────────────────────────────────────────────


class PCPClient:
    """
    P-MCP Client — the LLM/orchestrator side of the protocol.

    Discovers and calls robot actuations with built-in safety workflow:
      1. list_tools()      — discover what the robot can do
      2. shadow_preview()  — simulate the action first
      3. request_lease()   — claim the workspace zone
      4. call_tool()       — execute with lease token
    """

    def __init__(self, client_name: str = "pcp-client", client_version: str | None = None):
        self._name = client_name
        if client_version is None:
            from importlib.metadata import PackageNotFoundError
            from importlib.metadata import version as _dist_version

            try:
                client_version = _dist_version("pcp")
            except PackageNotFoundError:
                client_version = "0.0.0+unknown"
        self._version = client_version
        self._transport: Optional[_Transport] = None
        self._server_info: Optional[dict] = None
        self._server_caps: Optional[dict] = None
        self._initialized = False

    # ── Connection factory methods ────────────────────────────────────────────

    async def connect_stdio(self, argv: List[str]) -> "PCPClient":
        self._transport = _StdioTransport(argv)
        await self._transport.connect()
        await self._do_initialize()
        return self

    async def connect_http(self, base_url: str) -> "PCPClient":
        self._transport = _HTTPTransport(base_url)
        await self._transport.connect()
        await self._do_initialize()
        return self

    async def connect_server(self, server) -> "PCPClient":
        """Connect to an in-process PCPServer (for testing)."""
        self._transport = _InProcessTransport(server)
        await self._transport.connect()
        await self._do_initialize()
        return self

    async def close(self) -> None:
        if self._transport:
            await self._transport.close()

    async def __aenter__(self) -> "PCPClient":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    # ── Internal RPC helper ───────────────────────────────────────────────────

    async def _rpc(self, method: str, params: Optional[dict] = None) -> Any:
        req = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4())[:8],
            "method": method,
            "params": params or {},
        }
        resp = await self._transport.send(req)
        if "error" in resp and resp["error"] is not None:
            err = resp["error"]
            raise PCPClientError(err["code"], err["message"], err.get("data"))
        return resp.get("result")

    async def _do_initialize(self) -> None:
        result = await self._rpc(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
                "clientInfo": {"name": self._name, "version": self._version},
            },
        )
        self._server_info = result.get("serverInfo", {})
        self._server_caps = result.get("capabilities", {})
        pcp_meta = result.get("pcp", {})
        log.info(
            f"[Client] Connected to {self._server_info.get('name','?')} "
            f"v{self._server_info.get('version','?')} "
            f"(P-MCP {pcp_meta.get('version','?')})"
        )
        self._initialized = True

    # ── MCP-standard methods ──────────────────────────────────────────────────

    async def list_tools(self) -> List[dict]:
        """List all robot actuations (MCP tools/list)."""
        result = await self._rpc("tools/list")
        return result.get("tools", [])

    async def call_tool(
        self,
        name: str,
        arguments: Optional[dict] = None,
        lease_token: Optional[str] = None,
        zone_id: Optional[str] = None,
        fence_token: Optional[int] = None,
    ) -> dict:
        """Execute a robot actuation (MCP tools/call).

        fence_token should be the "fenceToken" returned with the lease grant
        (see request_lease). Presenting a stale fence token is rejected by
        the server even if lease_token still lexically matches.
        """
        params: Dict[str, Any] = {
            "name": name,
            "arguments": arguments or {},
        }
        if lease_token:
            params["_lease_token"] = lease_token
        if zone_id:
            params["_zone_id"] = zone_id
        if fence_token is not None:
            params["_fence_token"] = fence_token
        result = await self._rpc("tools/call", params)
        return result

    async def list_sensors(self) -> List[dict]:
        """List all robot sensors (MCP resources/list)."""
        result = await self._rpc("resources/list")
        return result.get("resources", [])

    async def read_sensor(self, sensor_name: str) -> dict:
        """Read a sensor value by name (MCP resources/read)."""
        # Build URI from name
        robot_id = (self._server_info or {}).get("name", "robot")
        uri = f"pcp://{robot_id}/sensors/{sensor_name}"
        result = await self._rpc("resources/read", {"uri": uri})
        return result

    async def list_missions(self) -> List[dict]:
        """List all mission templates (MCP prompts/list)."""
        result = await self._rpc("prompts/list")
        return result.get("prompts", [])

    async def get_mission(self, name: str, arguments: Optional[dict] = None) -> dict:
        """Get a filled mission template (MCP prompts/get)."""
        return await self._rpc("prompts/get", {"name": name, "arguments": arguments or {}})

    async def ping(self) -> dict:
        """Ping the robot server."""
        return await self._rpc("ping")

    # ── P-MCP Physical extension methods ─────────────────────────────────────

    async def shadow_preview(self, actuation_name: str, arguments: Optional[dict] = None) -> dict:
        """Run pre-flight simulation without executing (P-MCP extension)."""
        return await self._rpc(
            "shadow/preview", {"name": actuation_name, "arguments": arguments or {}}
        )

    async def request_lease(
        self,
        zone_id: str,
        duration_ms: int = 10_000,
        bid_energy_j: float = 100.0,
        priority: int = 5,
    ) -> dict:
        """Request a temporal zone lease (P-MCP extension)."""
        return await self._rpc(
            "lease/request",
            {
                "zoneId": zone_id,
                "durationMs": duration_ms,
                "bidEnergyJ": bid_energy_j,
                "priority": priority,
            },
        )

    async def release_lease(self, lease_id: str) -> dict:
        """Release a temporal zone lease (P-MCP extension)."""
        return await self._rpc("lease/release", {"leaseId": lease_id})

    async def estop(self, active: bool = True) -> dict:
        """Activate or deactivate emergency stop (P-MCP extension)."""
        return await self._rpc("pcp/estop", {"active": active})

    async def get_status(self) -> dict:
        """Get server status (P-MCP extension)."""
        return await self._rpc("pcp/status")

    async def get_identity(self) -> dict:
        """Get robot identity (DID) (P-MCP extension)."""
        return await self._rpc("pcp/identity")

    async def get_constitution(self) -> dict:
        """Get loaded safety constitution (P-MCP extension)."""
        return await self._rpc("pcp/constitution")

    # ── High-level safe_call helper ───────────────────────────────────────────

    async def safe_call(
        self,
        actuation_name: str,
        arguments: Optional[dict] = None,
        zone_id: Optional[str] = None,
        lease_duration_ms: int = 15_000,
    ) -> dict:
        """
        Full safe call workflow:
          1. shadow_preview — simulate
          2. request_lease  — claim zone
          3. call_tool      — execute
          4. release_lease  — release zone

        Returns the tool result dict or raises PCPClientError.
        """
        args = arguments or {}

        # 1. Preview
        preview_resp = await self.shadow_preview(actuation_name, args)
        preview = preview_resp.get("preview", {})
        if not preview.get("safe", True):
            raise PCPClientError(
                -33001,
                f"Shadow preview rejected: {preview.get('status')} "
                f"warnings={preview.get('warnings',[])}",
                preview,
            )

        lease_token = None
        lease_id = None
        fence_token = None

        # 2. Lease (if zone specified)
        if zone_id:
            lease_resp = await self.request_lease(zone_id, lease_duration_ms)
            lease = lease_resp.get("lease", {})
            if lease.get("state") != "ACTIVE":
                raise PCPClientError(-33003, f"Lease denied for zone={zone_id}", lease)
            lease_token = lease.get("leaseId")
            lease_id = lease_token
            fence_token = lease.get("fenceToken")

        # 3. Execute
        try:
            result = await self.call_tool(
                actuation_name,
                args,
                lease_token=lease_token,
                zone_id=zone_id,
                fence_token=fence_token,
            )
        finally:
            # 4. Release lease regardless
            if lease_id:
                await self.release_lease(lease_id)

        return result

    @property
    def server_info(self) -> Optional[dict]:
        return self._server_info

    @property
    def server_caps(self) -> Optional[dict]:
        return self._server_caps
