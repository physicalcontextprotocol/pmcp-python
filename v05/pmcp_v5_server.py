"""
P-MCP v0.5 — PMCPServer
========================
MCP-wire-compatible robot server.

Any MCP client (Claude Desktop, Cursor, custom LLM apps) can connect to a
PMCPServer and call robot actuations as standard MCP Tools.

Wire protocol: JSON-RPC 2.0 over stdio or HTTP.

Usage:

    from v05.pmcp_v5_server import PMCPServer
    from v05.pmcp_v5_types  import ActuationResult, SensorReading, SensorType

    server = PMCPServer("ur5-arm", version="1.0.0")

    @server.actuation("move_to", description="Move TCP to XYZ position",
                      max_speed_m_s=1.0, category="motion")
    async def move_to(x: float, y: float, z: float, speed: float = 0.3):
        # ... call real hardware here ...
        return ActuationResult(success=True, output={"x": x, "y": y, "z": z})

    @server.sensor("joint_angles", description="Current joint angles",
                   sensor_type=SensorType.JOINT_STATES, unit="rad")
    async def joint_angles():
        return SensorReading(sensor_name="joint_angles", robot_id=server.robot_id,
                             value=[0.0, -1.57, 1.57, 0.0, 1.57, 0.0], unit="rad")

    if __name__ == "__main__":
        import asyncio
        asyncio.run(server.run())                           # stdio
        # or: asyncio.run(server.run("http", port=8080))   # HTTP
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import sys
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional

from v05.pmcp_safety_v5 import SafetyConstitution, SafetyMiddleware, ShadowSimulator
from v05.pmcp_v5_types import (
    MCP_VERSION,
    PMCP_VERSION,
    ActuationParameter,
    ActuationResult,
    ActuationSpec,
    Capabilities,
    LeaseGrant,
    LeaseRequest,
    LeaseState,
    MissionArgument,
    MissionResult,
    MissionSpec,
    PMCPError,
    PMCPErrorCode,
    PMCPResponse,
    RobotIdentity,
    SensorReading,
    SensorSpec,
    SensorType,
)

log = logging.getLogger("pmcp.server_v5")


# ─────────────────────────────────────────────────────────────────────────────
#  INTERNAL REGISTRATION CONTAINERS
# ─────────────────────────────────────────────────────────────────────────────


class _ActuationEntry:
    __slots__ = ("spec", "fn")

    def __init__(self, spec: ActuationSpec, fn: Callable):
        self.spec = spec
        self.fn = fn


class _SensorEntry:
    __slots__ = ("spec", "fn")

    def __init__(self, spec: SensorSpec, fn: Callable):
        self.spec = spec
        self.fn = fn


# ─────────────────────────────────────────────────────────────────────────────
#  LEASE MANAGER (in-process; replace with distributed for multi-robot)
# ─────────────────────────────────────────────────────────────────────────────


class _LeaseManager:
    """
    Fencing tokens (Kleppmann 2016): a monotonically increasing counter is
    issued on every grant and renewal, and MUST be presented (as
    ``_fence_token``) on every subsequent actuation call, not only at
    lease-request time. This closes the classic paused-holder-plus-clock-skew
    failure mode: even if a caller's original grant is still lexically valid
    (lease_id, TTL), a stale fence token proves the caller's view of zone
    ownership is out of date and the execution gate rejects it.

    Today this counter is a single-process monotonic int, which is honest
    about current maturity (v05 has no Raft-backed multi-replica lease
    authority yet). SAFETY_ARCHITECTURE.md's Raft term numbers are the
    intended source once that ships; the wire-level contract established
    here (fence token required on every actuation) does not change when the
    counter's source does.
    """

    def __init__(self):
        self._leases: Dict[str, LeaseGrant] = {}  # zone_id → LeaseGrant
        # Per-zone, not global: fencing only needs to be monotonic within a
        # given zone's ownership history, not across unrelated zones. Kept
        # consistent with the canonical pmcp.safety.SafetyMiddleware and the
        # Rust LeaseManager, both of which use per-zone counters for the
        # same reason.
        self._fence_counters: Dict[str, int] = defaultdict(int)

    def _next_fence_token(self, zone_id: str) -> int:
        self._fence_counters[zone_id] += 1
        return self._fence_counters[zone_id]

    def request(self, req: LeaseRequest) -> LeaseGrant:
        existing = self._leases.get(req.zone_id)
        if existing and existing.valid:
            if existing.robot_id == req.robot_id:
                # Renew — bump the fence token too. A caller holding a
                # reference to the pre-renewal grant must re-fetch the new
                # token; this is deliberate, not an oversight, since a
                # "renewal" the caller doesn't know about is exactly the
                # stale-holder scenario fencing exists to catch.
                existing.expires_at = time.time() + req.duration_ms / 1000
                existing.fence_token = self._next_fence_token(req.zone_id)
                return existing
            # Another robot holds the zone — deny lower-bid requests
            if req.bid_energy_j <= existing.bid_energy_j:
                return LeaseGrant(
                    robot_id=req.robot_id,
                    zone_id=req.zone_id,
                    state=LeaseState.DENIED,
                    expires_at=0.0,
                )
            # Higher bid wins
            log.info(f"[Lease] zone={req.zone_id}: {req.robot_id} outbid {existing.robot_id}")

        grant = LeaseGrant(
            robot_id=req.robot_id,
            zone_id=req.zone_id,
            state=LeaseState.ACTIVE,
            expires_at=time.time() + req.duration_ms / 1000,
            bid_energy_j=req.bid_energy_j,
            fence_token=self._next_fence_token(req.zone_id),
        )
        self._leases[req.zone_id] = grant
        log.info(
            f"[Lease] ACTIVE zone={req.zone_id} robot={req.robot_id} "
            f"fence={grant.fence_token} ({req.duration_ms}ms)"
        )
        return grant

    def release(self, lease_id: str) -> bool:
        for zid, grant in list(self._leases.items()):
            if grant.lease_id == lease_id:
                grant.state = LeaseState.FREE
                del self._leases[zid]
                log.info(f"[Lease] RELEASED {lease_id} zone={zid}")
                return True
        return False

    def check(
        self, lease_token: Optional[str], zone_id: str, fence_token: Optional[int] = None
    ) -> tuple:
        grant = self._leases.get(zone_id)
        if not grant:
            return False, f"No lease held for zone={zone_id}"
        if not grant.valid:
            return False, f"Lease for zone={zone_id} has expired"
        if lease_token and grant.lease_id != lease_token:
            return False, f"Lease token mismatch for zone={zone_id}"
        if fence_token is not None and fence_token != grant.fence_token:
            return False, (
                f"Stale fence token for zone={zone_id}: presented "
                f"{fence_token}, current is {grant.fence_token} "
                f"(lease was renewed or re-granted since this token was issued)"
            )
        return True, ""

    def revoke_zone(self, zone_id: str) -> None:
        if zone_id in self._leases:
            del self._leases[zone_id]
            log.warning(f"[Lease] zone={zone_id} REVOKED (safety event)")


# ─────────────────────────────────────────────────────────────────────────────
#  PMCP SERVER
# ─────────────────────────────────────────────────────────────────────────────


class PMCPServer:
    """
    P-MCP Server — robot-side MCP-compatible endpoint.

    Exposes robot actuations as MCP Tools, sensors as MCP Resources,
    and missions as MCP Prompts.  Adds a mandatory safety pipeline
    (constitution → shadow → execute) before every actuation.
    """

    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        robot_id: Optional[str] = None,
        robot_class: str = "arm",
        model: str = "generic",
        serial: str = "000000",
        location: str = "lab-01",
        capabilities: Optional[Capabilities] = None,
        safety: Optional[SafetyMiddleware] = None,
    ):
        self.name = name
        self.version = version
        self.robot_id = robot_id or name
        self.identity = RobotIdentity.new(robot_class, model, serial, location)
        self.caps = capabilities or Capabilities()

        # Safety pipeline
        self._constitution = SafetyConstitution(self.robot_id)
        self._lease_mgr = _LeaseManager()
        self.safety = safety or SafetyMiddleware(
            self._constitution,
            ShadowSimulator(),
            self._lease_mgr,
        )
        # If a custom SafetyMiddleware was provided, sync the lease manager
        # so that lease/request and lease/release handlers use the same store
        if safety and hasattr(safety, "_leases") and safety._leases is not None:
            self._lease_mgr = safety._leases
        if safety and hasattr(safety, "_constitution"):
            self._constitution = safety._constitution

        self._actuations: Dict[str, _ActuationEntry] = {}
        self._sensors: Dict[str, _SensorEntry] = {}
        self._missions: Dict[str, tuple] = {}

        self._call_count = 0
        self._blocked_count = 0
        self._started_at = 0.0

        # JSON-RPC method dispatch table
        self._methods: Dict[str, Callable] = {
            # MCP-standard methods
            "initialize": self._h_initialize,
            "ping": self._h_ping,
            "tools/list": self._h_tools_list,
            "tools/call": self._h_tools_call,
            "resources/list": self._h_resources_list,
            "resources/read": self._h_resources_read,
            "prompts/list": self._h_prompts_list,
            "prompts/get": self._h_prompts_get,
            "logging/setLevel": self._h_logging_set_level,
            # P-MCP physical extensions
            "shadow/preview": self._h_shadow_preview,
            "lease/request": self._h_lease_request,
            "lease/release": self._h_lease_release,
            "pmcp/estop": self._h_estop,
            "pmcp/status": self._h_status,
            "pmcp/identity": self._h_identity,
            "pmcp/constitution": self._h_constitution,
        }

    # ── Decorator API ─────────────────────────────────────────────────────────

    def actuation(
        self,
        name: str,
        description: str = "",
        category: str = "motion",
        max_speed_m_s: float = 1.0,
        max_force_n: float = 100.0,
        max_energy_j: float = 500.0,
        est_duration_s: float = 2.0,
        requires_lease: bool = True,
        shadow_required: bool = True,
        iso_class: str = "ISO10218",
    ) -> Callable:
        """
        Register a physical actuation as an MCP Tool.

            @server.actuation("move_to", description="Move to XYZ", max_speed_m_s=1.0)
            async def move_to(x: float, y: float, z: float, speed: float = 0.3):
                ...
                return ActuationResult(success=True, output={"reached": True})
        """

        def decorator(fn: Callable) -> Callable:
            params = self._extract_params(fn)
            spec = ActuationSpec(
                name=name,
                description=description or fn.__doc__ or "",
                parameters=params,
                robot_id=self.robot_id,
                category=category,
                max_speed_m_s=max_speed_m_s,
                max_force_n=max_force_n,
                max_energy_j=max_energy_j,
                est_duration_s=est_duration_s,
                requires_lease=requires_lease,
                shadow_required=shadow_required,
                iso_class=iso_class,
            )
            self._actuations[name] = _ActuationEntry(spec, fn)
            log.debug(f"[Server] Registered actuation: {name}")
            return fn

        return decorator

    def sensor(
        self,
        name: str,
        description: str = "",
        sensor_type: SensorType = SensorType.CUSTOM,
        unit: str = "",
        hz: float = 10.0,
        is_stream: bool = False,
    ) -> Callable:
        """
        Register a sensor as an MCP Resource.

            @server.sensor("joint_angles", sensor_type=SensorType.JOINT_STATES, unit="rad")
            async def joint_angles():
                return SensorReading(sensor_name="joint_angles", ...)
        """

        def decorator(fn: Callable) -> Callable:
            spec = SensorSpec(
                name=name,
                description=description or fn.__doc__ or "",
                robot_id=self.robot_id,
                sensor_type=sensor_type,
                unit=unit,
                hz=hz,
                is_stream=is_stream,
            )
            self._sensors[name] = _SensorEntry(spec, fn)
            log.debug(f"[Server] Registered sensor: {name}")
            return fn

        return decorator

    def mission(self, name: str, description: str = "", robot_class: str = "any") -> Callable:
        """
        Register a mission template as an MCP Prompt.

            @server.mission("pick_and_place", description="Pick object from A, place at B")
            async def pick_and_place(source: str, destination: str):
                return MissionResult(mission_name="pick_and_place", messages=[...])
        """

        def decorator(fn: Callable) -> Callable:
            args = [
                MissionArgument(
                    name=p.name,
                    description=f"{p.name} argument",
                    required=p.default is inspect.Parameter.empty,
                )
                for p in inspect.signature(fn).parameters.values()
            ]
            spec = MissionSpec(
                name=name,
                description=description or fn.__doc__ or "",
                robot_class=robot_class,
                arguments=args,
            )
            self._missions[name] = (spec, fn)
            return fn

        return decorator

    # ── JSON-RPC Dispatch ─────────────────────────────────────────────────────

    async def handle_message(self, raw: dict) -> Optional[dict]:
        """Dispatch a JSON-RPC message and return response dict (or None for notifications)."""
        # Handle notification (no id)
        if "id" not in raw:
            return None

        req_id = raw.get("id", "")
        method = raw.get("method", "")

        handler = self._methods.get(method)
        if not handler:
            err = PMCPError(PMCPErrorCode.METHOD_NOT_FOUND, f"Method not found: {method}")
            return PMCPResponse(id=req_id, error=err).to_dict()

        try:
            params = raw.get("params", {})
            result = await handler(params)
            return PMCPResponse(id=req_id, result=result).to_dict()
        except PMCPError as e:
            return PMCPResponse(id=req_id, error=e).to_dict()
        except Exception as exc:
            log.exception(f"[Server] Error in {method}: {exc}")
            err = PMCPError(PMCPErrorCode.INTERNAL_ERROR, str(exc))
            return PMCPResponse(id=req_id, error=err).to_dict()

    # ── MCP Standard Handlers ─────────────────────────────────────────────────

    async def _h_initialize(self, params: dict) -> dict:
        self._started_at = time.time()
        client_info = params.get("clientInfo", {})
        log.info(
            f"[Server] initialize from {client_info.get('name','?')} "
            f"v{client_info.get('version','?')}"
        )
        return {
            "protocolVersion": MCP_VERSION,
            "capabilities": self.caps.to_mcp_dict(),
            "serverInfo": {
                "name": self.name,
                "version": self.version,
            },
            "pmcp": {
                "version": PMCP_VERSION,
                "robotId": self.robot_id,
                "identity": self.identity.to_dict(),
                "constitution": self._constitution.fingerprint[:16] + "...",
            },
        }

    async def _h_ping(self, params: dict) -> dict:
        return {"pong": True, "ts": time.time(), "robot_id": self.robot_id}

    async def _h_tools_list(self, params: dict) -> dict:
        """MCP tools/list — returns actuations as MCP Tools."""
        tools = [e.spec.to_mcp_tool() for e in self._actuations.values()]
        return {"tools": tools}

    async def _h_tools_call(self, params: dict) -> dict:
        """MCP tools/call — executes an actuation through the safety pipeline."""
        name = params.get("name", "")
        arguments = params.get("arguments", {})
        lease_token = params.get("_lease_token")  # P-MCP extension
        zone_id = params.get("_zone_id")
        fence_token = params.get("_fence_token")  # P-MCP extension — see LeaseGrant.fence_token

        entry = self._actuations.get(name)
        if not entry:
            raise PMCPError(PMCPErrorCode.METHOD_NOT_FOUND, f"Actuation not found: {name}")

        self._call_count += 1

        # ── Safety pipeline ──────────────────────────────────────────────────
        skip_shadow = not entry.spec.shadow_required
        safe, preview, violations = self.safety.check(
            name,
            arguments,
            lease_token=lease_token,
            zone_id=zone_id,
            fence_token=fence_token,
            skip_shadow=skip_shadow,
        )

        if not safe:
            self._blocked_count += 1
            raise PMCPError(
                PMCPErrorCode.SHADOW_BLOCKED if preview else PMCPErrorCode.CONSTITUTION_BLOCKED,
                "; ".join(violations),
                {"violations": violations, "shadow": preview.to_dict() if preview else None},
            )

        # ── Execute actuation ────────────────────────────────────────────────
        t0 = time.time()
        try:
            if asyncio.iscoroutinefunction(entry.fn):
                result = await entry.fn(**arguments)
            else:
                result = entry.fn(**arguments)
        except Exception as exc:
            raise PMCPError(PMCPErrorCode.INTERNAL_ERROR, f"Actuation {name} raised: {exc}")

        duration = time.time() - t0

        # ── Build MCP-compatible response ────────────────────────────────────
        if isinstance(result, ActuationResult):
            result.duration_s = duration
            result.robot_id = self.robot_id
            result.actuation_name = name
            if preview:
                result.shadow_delta_m = 0.0  # real delta would be computed from sensors
            content = result.to_mcp_content()
            is_error = not result.success
        else:
            content = [{"type": "text", "text": json.dumps(result, indent=2)}]
            is_error = False

        response = {"content": content, "isError": is_error}
        if preview:
            response["_shadow"] = preview.to_dict()  # P-MCP extension field
        return response

    async def _h_resources_list(self, params: dict) -> dict:
        """MCP resources/list — returns sensors as MCP Resources."""
        resources = [e.spec.to_mcp_resource() for e in self._sensors.values()]
        return {"resources": resources}

    async def _h_resources_read(self, params: dict) -> dict:
        """MCP resources/read — reads a sensor value."""
        uri = params.get("uri", "")
        # Find by URI
        for name, entry in self._sensors.items():
            if entry.spec.uri == uri or name == uri.split("/")[-1]:
                if asyncio.iscoroutinefunction(entry.fn):
                    reading = await entry.fn()
                else:
                    reading = entry.fn()
                if isinstance(reading, SensorReading):
                    return {"contents": reading.to_mcp_content()}
                return {"contents": [{"type": "text", "text": json.dumps(reading)}]}
        raise PMCPError(PMCPErrorCode.METHOD_NOT_FOUND, f"Sensor not found: {uri}")

    async def _h_prompts_list(self, params: dict) -> dict:
        """MCP prompts/list — returns missions as MCP Prompts."""
        return {"prompts": [spec.to_mcp_prompt() for spec, _ in self._missions.values()]}

    async def _h_prompts_get(self, params: dict) -> dict:
        """MCP prompts/get — returns a filled mission template."""
        name = params.get("name", "")
        args = params.get("arguments", {})
        entry = self._missions.get(name)
        if not entry:
            raise PMCPError(PMCPErrorCode.METHOD_NOT_FOUND, f"Mission not found: {name}")
        spec, fn = entry
        if asyncio.iscoroutinefunction(fn):
            result = await fn(**args)
        else:
            result = fn(**args)
        if isinstance(result, MissionResult):
            return result.to_dict()
        return {
            "description": name,
            "messages": [{"role": "user", "content": [{"type": "text", "text": str(result)}]}],
        }

    async def _h_logging_set_level(self, params: dict) -> dict:
        level = params.get("level", "info").upper()
        logging.getLogger("pmcp").setLevel(getattr(logging, level, logging.INFO))
        return {}

    # ── P-MCP Physical Extension Handlers ────────────────────────────────────

    async def _h_shadow_preview(self, params: dict) -> dict:
        """shadow/preview — run pre-flight simulation without executing."""
        name = params.get("name", "")
        arguments = params.get("arguments", {})
        if name not in self._actuations:
            raise PMCPError(PMCPErrorCode.METHOD_NOT_FOUND, f"Actuation not found: {name}")
        preview = self.safety._sim.preview(name, arguments)
        return {"preview": preview.to_dict()}

    async def _h_lease_request(self, params: dict) -> dict:
        req = LeaseRequest(
            robot_id=params.get("robotId", self.robot_id),
            zone_id=params.get("zoneId", "default"),
            duration_ms=params.get("durationMs", 10_000),
            bid_energy_j=params.get("bidEnergyJ", 100.0),
            priority=params.get("priority", 5),
        )
        grant = self._lease_mgr.request(req)
        return {"lease": grant.to_dict()}

    async def _h_lease_release(self, params: dict) -> dict:
        lease_id = params.get("leaseId", "")
        released = self._lease_mgr.release(lease_id)
        return {"released": released}

    async def _h_estop(self, params: dict) -> dict:
        active = params.get("active", True)
        self.safety.set_estop(active)
        return {"estop": active, "ts": time.time()}

    async def _h_status(self, params: dict) -> dict:
        uptime = round(time.time() - self._started_at, 1) if self._started_at else 0
        return {
            "robot_id": self.robot_id,
            "pmcp_version": PMCP_VERSION,
            "uptime_s": uptime,
            "call_count": self._call_count,
            "blocked_count": self._blocked_count,
            "safety_stats": self.safety.stats,
            "actuations": list(self._actuations.keys()),
            "sensors": list(self._sensors.keys()),
            "missions": list(self._missions.keys()),
        }

    async def _h_identity(self, params: dict) -> dict:
        return self.identity.to_dict()

    async def _h_constitution(self, params: dict) -> dict:
        return self._constitution.summary()

    # ── Transport: stdio ──────────────────────────────────────────────────────

    async def _run_stdio(self) -> None:
        """JSON-RPC over stdin/stdout — compatible with MCP stdio transport."""
        log.info(f"[Server] {self.name} v{self.version} — stdio transport ready")
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        loop = asyncio.get_event_loop()
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

        writer_transport, writer_protocol = await loop.connect_write_pipe(
            asyncio.BaseProtocol, sys.stdout
        )

        async def write_json(obj: dict) -> None:
            line = json.dumps(obj) + "\n"
            sys.stdout.write(line)
            sys.stdout.flush()

        while True:
            try:
                line = await reader.readline()
                if not line:
                    break
                raw = json.loads(line.decode().strip())
                response = await self.handle_message(raw)
                if response is not None:
                    await write_json(response)
            except (json.JSONDecodeError, UnicodeDecodeError):
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"},
                }
                await write_json(err_resp)
            except Exception as exc:
                log.exception(f"[Server] stdio error: {exc}")
                break

    # ── Transport: HTTP ───────────────────────────────────────────────────────

    async def _run_http(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        """JSON-RPC over HTTP POST /pmcp — with optional SSE stream."""
        try:
            from aiohttp import web
        except ImportError:
            log.error("[Server] aiohttp not installed. Run: pip install aiohttp")
            return

        app = web.Application()

        async def handle_post(request: web.Request) -> web.Response:
            try:
                raw = await request.json()
            except Exception:
                return web.json_response(
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": "Parse error"},
                    },
                    status=400,
                )
            response = await self.handle_message(raw)
            return web.json_response(response or {})

        async def handle_health(request: web.Request) -> web.Response:
            status = await self._h_status({})
            return web.json_response(status)

        app.router.add_post("/pmcp", handle_post)
        app.router.add_post("/mcp", handle_post)  # MCP compat alias
        app.router.add_get("/health", handle_health)
        app.router.add_get("/", handle_health)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        log.info(f"[Server] {self.name} listening on http://{host}:{port}/pmcp")
        try:
            await asyncio.Event().wait()
        finally:
            await runner.cleanup()

    # ── Public run() entry point ──────────────────────────────────────────────

    async def run(
        self, transport: str = "stdio", host: str = "127.0.0.1", port: int = 8080
    ) -> None:
        """
        Start the server.

          await server.run()                        # stdio (default)
          await server.run("http", port=8080)       # HTTP
        """
        self._started_at = time.time()
        if transport == "http":
            await self._run_http(host, port)
        else:
            await self._run_stdio()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_params(fn: Callable) -> List[ActuationParameter]:
        """Infer ActuationParameters from function signature annotations."""
        sig = inspect.signature(fn)
        params: List[ActuationParameter] = []
        for pname, p in sig.parameters.items():
            if pname in ("self", "cls"):
                continue
            ann = p.annotation
            if ann == inspect.Parameter.empty:
                ptype = "string"
            elif ann in (float, int):
                ptype = "number" if ann is float else "integer"
            elif ann is bool:
                ptype = "boolean"
            elif ann is list:
                ptype = "array"
            elif ann is dict:
                ptype = "object"
            else:
                ptype = "string"
            required = p.default is inspect.Parameter.empty
            default = None if required else p.default
            params.append(
                ActuationParameter(
                    name=pname,
                    type=ptype,
                    description=f"{pname} parameter",
                    required=required,
                    default=default,
                )
            )
        return params


# ── Console-script entry point: `pmcp-server` ────────────────────────────────


def main() -> None:
    """CLI entry point — run a default stdio server with a sample actuation.

    For real robots, write your own server module and import PMCPServer.
    """
    import argparse

    from v05.pmcp_v5_types import ActuationResult

    parser = argparse.ArgumentParser(description="P-MCP v0.5 reference server")
    parser.add_argument("--name", default="pmcp-server", help="server name")
    parser.add_argument("--transport", default="stdio", choices=["stdio", "http"])
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    server = PMCPServer(args.name, robot_class="arm", model="ReferenceArm")

    @server.actuation("ping", description="Health check — returns the current pose")
    async def ping():
        return ActuationResult(success=True, output={"pong": True, "ts": time.time()})

    asyncio.run(server.run(args.transport, port=args.port))
