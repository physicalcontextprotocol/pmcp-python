"""
P-MCP Python SDK — pmcp-client package
=======================================
High-level async client for connecting to P-MCP robot servers.

.. deprecated::
    This module is superseded by :mod:`pmcp.client` (``PMCPClient``),
    which is the canonical client exported from ``pmcp.__init__``.
    ``client_v2`` is kept for backward compatibility during the v0.5
    restructuring and will be removed in a future major version.
    New code should use::

        from pmcp.client import PMCPClient
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import warnings as _warnings

from pmcp import __version__
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

import aiohttp

from pmcp.types import (
    PMCP_VERSION,
    ActuationResult,
    ActuationSpec,
    AuditEntry,
    LeaseGrant,
    LeaseState,
    MetricsSnapshot,
    PMCPError,
    PMCPErrorCode,
    SensorReading,
    SensorSpec,
)

_warnings.warn(
    "pmcp.client_v2 is deprecated and will be removed in a future release; "
    "use pmcp.client.PMCPClient instead.",
    DeprecationWarning,
    stacklevel=2,
)

log = logging.getLogger("pmcp.client")


# ─────────────────────────────────────────────────────────────────────────────
#  CONNECTION CONFIG
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ClientConfig:
    host: str = "localhost"
    port: int = 8080
    transport: str = "http"  # "http" | "stdio" | "tcp" | "ws"
    tls: bool = False
    timeout_s: float = 30.0
    retry_attempts: int = 3
    retry_delay_s: float = 0.5
    bearer_token: Optional[str] = None
    did: Optional[str] = None
    api_path: str = "/mcp"

    @property
    def base_url(self) -> str:
        scheme = "https" if self.tls else "http"
        return f"{scheme}://{self.host}:{self.port}"

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}{self.api_path}"


# ─────────────────────────────────────────────────────────────────────────────
#  PMCP CLIENT
# ─────────────────────────────────────────────────────────────────────────────


class PMCPClient:
    """
    JSON-RPC 2.0 client for P-MCP robot servers.

    Example::

        async with PMCPClient(ClientConfig("my-robot.local", 8080)) as client:
            caps = await client.initialize()
            result = await client.actuate("move_to", x=0.5, y=0.0, z=0.3)
            print(result.final_pose)
    """

    def __init__(self, config: ClientConfig):
        self._config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._req_counter = 0
        self._server_info: Optional[Dict[str, Any]] = None
        self._capabilities: Optional[Dict[str, Any]] = None
        self._initialized = False
        self._notification_handlers: Dict[str, List[Callable]] = {}

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def __aenter__(self) -> "PMCPClient":
        connector = aiohttp.TCPConnector(ssl=self._config.tls)
        timeout = aiohttp.ClientTimeout(total=self._config.timeout_s)
        self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        await self.initialize()
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    # ── initialize ───────────────────────────────────────────────────────────

    async def initialize(self) -> Dict[str, Any]:
        """Perform MCP initialization handshake."""
        result = await self._call(
            "initialize",
            {
                "protocolVersion": PMCP_VERSION,
                "clientInfo": {
                    "name": "pmcp-python-client",
                    "version": __version__,
                },
                "capabilities": {
                    "notifications": True,
                    "batching": True,
                },
            },
        )
        self._server_info = result.get("serverInfo", {})
        self._capabilities = result.get("capabilities", {})
        self._initialized = True
        log.debug(
            "Initialized with server: %s %s",
            self._server_info.get("name"),
            self._server_info.get("version"),
        )
        return result

    # ── actuations ───────────────────────────────────────────────────────────

    async def list_actuations(self) -> List[ActuationSpec]:
        """Discover available actuations from the server."""
        result = await self._call("actuations/list", {})
        return [_parse_actuation_spec(a) for a in result.get("actuations", [])]

    async def actuate(
        self,
        name: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ActuationResult:
        """Execute an actuation. Kwargs are merged into params."""
        merged = {**(params or {}), **kwargs}
        result = await self._call(
            "actuations/execute",
            {
                "name": name,
                "params": merged,
            },
        )
        return ActuationResult(
            success=result.get("success", False),
            final_pose=result.get("final_pose"),
            energy_consumed_j=result.get("energy_consumed_j", 0.0),
            duration_ms=result.get("duration_ms", 0),
            error_msg=result.get("error_msg"),
        )

    async def batch_actuate(
        self,
        actuations: List[Tuple[str, Dict[str, Any]]],
        atomic: bool = True,
    ) -> List[ActuationResult]:
        """Execute multiple actuations as a batch."""
        items = [{"name": n, "params": p} for n, p in actuations]
        result = await self._call(
            "actuations/batch",
            {
                "actuations": items,
                "atomic": atomic,
            },
        )
        return [
            ActuationResult(
                success=r.get("success", False),
                energy_consumed_j=r.get("energy_consumed_j", 0.0),
                duration_ms=r.get("duration_ms", 0),
                error_msg=r.get("error_msg"),
            )
            for r in result.get("results", [])
        ]

    # ── sensors ──────────────────────────────────────────────────────────────

    async def list_sensors(self) -> List[SensorSpec]:
        result = await self._call("sensors/list", {})
        return [_parse_sensor_spec(s) for s in result.get("sensors", [])]

    async def read_sensor(self, name: str) -> SensorReading:
        result = await self._call("sensors/read", {"name": name})
        return SensorReading(
            value=result.get("value"),
            unit=result.get("unit", ""),
            timestamp_ms=result.get("timestamp_ms", int(time.time() * 1000)),
            quality=result.get("quality", 1.0),
        )

    async def subscribe_sensor(
        self,
        name: str,
        callback: Callable[[SensorReading], None],
        interval_ms: int = 100,
    ) -> asyncio.Task:
        """Subscribe to a sensor by polling at a given interval."""

        async def _poll():
            while True:
                try:
                    reading = await self.read_sensor(name)
                    callback(reading)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    log.warning("Sensor %s poll error: %s", name, e)
                await asyncio.sleep(interval_ms / 1000.0)

        return asyncio.create_task(_poll())

    # ── leases ───────────────────────────────────────────────────────────────

    async def acquire_lease(
        self,
        zone_id: str,
        duration_ms: int = 30_000,
    ) -> LeaseGrant:
        result = await self._call(
            "leases/acquire",
            {
                "zone_id": zone_id,
                "duration_ms": duration_ms,
            },
        )
        return LeaseGrant(
            lease_id=result.get("lease_id", ""),
            zone_id=zone_id,
            robot_id=result.get("robot_id", ""),
            expires_at=time.time() + result.get("expires_ms", 0) / 1000.0,
            state=LeaseState.ACTIVE if result.get("granted", False) else LeaseState.DENIED,
            fence_token=result.get("fence_token", 0),
        )

    @asynccontextmanager
    async def leased_zone(
        self,
        zone_id: str,
        duration_ms: int = 30_000,
    ) -> AsyncIterator[LeaseGrant]:
        """Context manager: acquire a zone lease and release it on exit."""
        grant = await self.acquire_lease(zone_id, duration_ms)
        if not grant.granted:
            raise PMCPError(
                code=PMCPErrorCode.LEASE_REQUIRED,
                message=f"Could not acquire lease for zone {zone_id}",
            )
        try:
            yield grant
        finally:
            await self._call("leases/release", {"lease_id": grant.lease_id})

    # ── safety / shadow ──────────────────────────────────────────────────────

    async def preview_shadow(
        self,
        actuation_name: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run a shadow simulation without executing."""
        return await self._call(
            "safety/preview",
            {
                "name": actuation_name,
                "params": params,
            },
        )

    async def constitution_check(
        self,
        actuation_name: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        return await self._call(
            "safety/constitution_check",
            {
                "name": actuation_name,
                "params": params,
            },
        )

    async def get_audit_log(
        self,
        limit: int = 50,
        since_ms: Optional[int] = None,
    ) -> List[AuditEntry]:
        result = await self._call(
            "audit/list",
            {
                "limit": limit,
                "since_ms": since_ms,
            },
        )
        return result.get("entries", [])

    # ── metrics ──────────────────────────────────────────────────────────────

    async def get_metrics(self) -> MetricsSnapshot:
        result = await self._call("pmcp/metrics", {})
        return MetricsSnapshot(**result) if result else MetricsSnapshot()

    async def ping(self) -> float:
        """Returns round-trip time in milliseconds."""
        start = time.perf_counter()
        await self._call("pmcp/ping", {})
        return (time.perf_counter() - start) * 1000

    # ── notifications ────────────────────────────────────────────────────────

    def on_notification(
        self,
        method: str,
        handler: Callable[[Dict[str, Any]], None],
    ) -> None:
        """Register a handler for a JSON-RPC notification method."""
        self._notification_handlers.setdefault(method, []).append(handler)

    # ── internal ─────────────────────────────────────────────────────────────

    async def _call(
        self,
        method: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        self._req_counter += 1
        req_id = self._req_counter

        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": req_id,
        }

        headers = {"Content-Type": "application/json"}
        if self._config.bearer_token:
            headers["Authorization"] = f"Bearer {self._config.bearer_token}"
        if self._config.did:
            headers["X-PMCP-DID"] = self._config.did

        last_exc: Optional[Exception] = None
        for attempt in range(self._config.retry_attempts):
            try:
                if self._session is None:
                    self._session = aiohttp.ClientSession()

                async with self._session.post(
                    self._config.endpoint,
                    json=payload,
                    headers=headers,
                ) as resp:
                    resp.raise_for_status()
                    body = await resp.json()

                # Handle notifications mixed into response
                if "method" in body and "id" not in body:
                    self._dispatch_notification(body.get("method", ""), body.get("params", {}))
                    return {}

                if "error" in body:
                    err = body["error"]
                    raise PMCPError(
                        code=PMCPErrorCode(err.get("code", -32603)),
                        message=err.get("message", "Unknown RPC error"),
                    )

                return body.get("result", {})

            except PMCPError:
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exc = e
                if attempt < self._config.retry_attempts - 1:
                    await asyncio.sleep(self._config.retry_delay_s * (2**attempt))

        raise ConnectionError(
            f"Failed to call {method} on {self._config.endpoint} "
            f"after {self._config.retry_attempts} attempts: {last_exc}"
        )

    def _dispatch_notification(self, method: str, params: Dict[str, Any]) -> None:
        for handler in self._notification_handlers.get(method, []):
            try:
                handler(params)
            except Exception as e:
                log.warning("Notification handler error for %s: %s", method, e)


# ─────────────────────────────────────────────────────────────────────────────
#  STDIO CLIENT  (for local/subprocess communication)
# ─────────────────────────────────────────────────────────────────────────────


class StdioClient:
    """
    Connect to a P-MCP server via stdio (subprocess).

    Example::

        async with StdioClient(["python", "my_robot_server.py"]) as client:
            await client.actuate("move_to", x=0.5, y=0.0, z=0.3)
    """

    def __init__(self, command: List[str]):
        self._command = command
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._req_counter = 0
        self._pending: Dict[int, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None

    async def __aenter__(self) -> "StdioClient":
        self._proc = await asyncio.create_subprocess_exec(
            *self._command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._reader_task = asyncio.create_task(self._read_loop())
        return self

    async def __aexit__(self, *_) -> None:
        if self._reader_task:
            self._reader_task.cancel()
        if self._proc:
            self._proc.terminate()
            await self._proc.wait()

    async def actuate(
        self,
        name: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ActuationResult:
        merged = {**(params or {}), **kwargs}
        result = await self._call("actuations/execute", {"name": name, "params": merged})
        return ActuationResult(
            success=result.get("success", False),
            final_pose=result.get("final_pose"),
            energy_consumed_j=result.get("energy_consumed_j", 0.0),
            duration_ms=result.get("duration_ms", 0),
        )

    async def _call(
        self,
        method: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        self._req_counter += 1
        req_id = self._req_counter
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = fut

        payload = (
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": method,
                    "params": params,
                    "id": req_id,
                }
            )
            + "\n"
        )

        assert self._proc and self._proc.stdin
        self._proc.stdin.write(payload.encode())
        await self._proc.stdin.drain()

        return await asyncio.wait_for(fut, timeout=30.0)

    async def _read_loop(self) -> None:
        assert self._proc and self._proc.stdout
        async for line in self._proc.stdout:
            try:
                msg = json.loads(line.decode().strip())
            except json.JSONDecodeError:
                continue

            req_id = msg.get("id")
            if req_id is not None and req_id in self._pending:
                fut = self._pending.pop(req_id)
                if "error" in msg:
                    fut.set_exception(Exception(msg["error"].get("message", "error")))
                else:
                    fut.set_result(msg.get("result", {}))


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _parse_actuation_spec(raw: Dict[str, Any]) -> ActuationSpec:
    from pmcp.types import ActuationParameter, ActuationSpec

    return ActuationSpec(
        name=raw.get("name", ""),
        description=raw.get("description", ""),
        parameters=[
            ActuationParameter(
                name=p.get("name", ""),
                param_type=p.get("type", "string"),
                description=p.get("description", ""),
                required=p.get("required", False),
            )
            for p in raw.get("parameters", [])
        ],
        robot_id=raw.get("robot_id", ""),
        max_speed_m_s=raw.get("max_speed_m_s", 1.0),
        max_force_n=raw.get("max_force_n", 100.0),
        max_energy_j=raw.get("max_energy_j", 500.0),
    )


def _parse_sensor_spec(raw: Dict[str, Any]) -> SensorSpec:
    from pmcp.types import SensorSpec, SensorType

    return SensorSpec(
        name=raw.get("name", ""),
        description=raw.get("description", ""),
        sensor_type=SensorType(raw.get("sensor_type", "custom")),
        unit=raw.get("unit", ""),
        robot_id=raw.get("robot_id", ""),
        sample_rate_hz=raw.get("sample_rate_hz", 1.0),
    )
