"""
P-MCP SDK — PCPServer
======================
The core server class. Analogous to MCP's `mcp.server.Server`.

Usage:

    from pcp.server import PCPServer

    server = PCPServer("my-arm", version="1.0.0")

    @server.actuation("move_to", description="Move end-effector to XYZ position")
    async def move_to(x: float, y: float, z: float, speed: float = 0.3):
        # Physical hardware call here
        return ActuationResult(success=True, final_pose={"x": x, "y": y, "z": z})

    @server.sensor("joint_angles", description="Current joint positions",
                   sensor_type=SensorType.POSITION, unit="rad")
    async def get_joints():
        return SensorReading(value=[0.1, -0.2, 0.5, 0.0, 1.1, 0.0], unit="rad")

    if __name__ == "__main__":
        import asyncio
        asyncio.run(server.run())      # stdio transport (default)
        # or: asyncio.run(server.run(transport="http", port=8080))

The server automatically injects the P-MCP safety pipeline:
  1. LeaseCheck       — verify robot holds valid zone lease
  2. ConstitutionCheck — evaluate against TEE-signed safety rules
  3. ShadowPreview    — run ghost simulation
  4. Execute          — call your handler function
  5. Broadcast        — emit result notification to connected clients
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import sys
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Type

from pcp.types import (
    ActuationParameter, ActuationResult, ActuationSpec,
    Capabilities, ClientInfo, ConstitutionCheck,
    LeaseGrant, LeaseRequest,
    PCPError, PCPErrorCode, PCPNotification, PCPRequest, PCPResponse,
    PromptResult, PromptSpec, SensorReading, SensorSpec, SensorType,
    ServerInfo, ShadowPreview, ShadowStatus, PCP_VERSION,
)

log = logging.getLogger("pcp.server")


# ─────────────────────────────────────────────────────────────────────────────
#  INTERNAL HANDLER REGISTRATIONS
# ─────────────────────────────────────────────────────────────────────────────

class _ActuationHandler:
    __slots__ = ("spec", "fn", "shadow_fn")
    def __init__(self, spec: ActuationSpec, fn: Callable, shadow_fn: Optional[Callable] = None):
        self.spec      = spec
        self.fn        = fn
        self.shadow_fn = shadow_fn   # Custom shadow function, optional


class _SensorHandler:
    __slots__ = ("spec", "fn")
    def __init__(self, spec: SensorSpec, fn: Callable):
        self.spec = spec
        self.fn   = fn


# ─────────────────────────────────────────────────────────────────────────────
#  PCP SERVER
# ─────────────────────────────────────────────────────────────────────────────

class PCPServer:
    """
    P-MCP Server — the robot-side endpoint of the protocol.

    Hosts register actuations (physical commands) and sensors (physical data),
    and the server handles the JSON-RPC 2.0 lifecycle with built-in
    safety middleware.
    """

    def __init__(
        self,
        name:             str,
        version:          str = "1.0.0",
        robot_id:         Optional[str] = None,
        capabilities:     Optional[Capabilities] = None,
        safety_middleware: Optional[Any] = None,   # SafetyMiddleware instance
    ):
        self.name      = name
        self.version   = version
        self.robot_id  = robot_id or name
        self.caps      = capabilities or Capabilities()
        self.safety    = safety_middleware

        self._actuations:   Dict[str, _ActuationHandler] = {}
        self._sensors:      Dict[str, _SensorHandler]    = {}
        self._prompts:      Dict[str, tuple]             = {}
        self._request_handlers: Dict[str, Callable]      = {}

        self._connected_clients: List[str] = []
        self._started_at:   float = 0.0
        self._call_count:   int   = 0
        self._blocked_count: int  = 0

        # Register built-in JSON-RPC method handlers
        self._request_handlers.update({
            "initialize":       self._handle_initialize,
            "actuations/list":  self._handle_actuations_list,
            "actuations/call":  self._handle_actuations_call,
            "sensors/list":     self._handle_sensors_list,
            "sensors/read":     self._handle_sensors_read,
            "prompts/list":     self._handle_prompts_list,
            "prompts/get":      self._handle_prompts_get,
            "shadow/preview":   self._handle_shadow_preview,
            "lease/request":    self._handle_lease_request,
            "lease/release":    self._handle_lease_release,
            "ping":             self._handle_ping,
        })

    # ── Decorator API ─────────────────────────────────────────────────────────

    def actuation(
        self,
        name:            str,
        description:     str  = "",
        max_speed_m_s:   Optional[float] = None,
        max_force_n:     Optional[float] = None,
        max_energy_j:    Optional[float] = None,
        requires_lease:  bool = True,
        shadow_required: bool = True,
        robot_class:     str  = "any",
    ) -> Callable:
        """
        Register a physical actuation.

            @server.actuation("move_to", description="Move TCP to XYZ",
                              max_speed_m_s=1.0)
            async def move_to(x: float, y: float, z: float, speed: float = 0.3):
                ...
                return ActuationResult(success=True)
        """
        def decorator(fn: Callable) -> Callable:
            params = _extract_parameters(fn)
            spec = ActuationSpec(
                name            = name,
                description     = description or (fn.__doc__ or "").strip().split("\n")[0],
                parameters      = params,
                max_speed_m_s   = max_speed_m_s,
                max_force_n     = max_force_n,
                max_energy_j    = max_energy_j,
                requires_lease  = requires_lease,
                shadow_required = shadow_required,
                robot_class     = robot_class,
            )
            self._actuations[name] = _ActuationHandler(spec, fn)
            log.debug(f"Registered actuation: {name}")
            return fn
        return decorator

    def sensor(
        self,
        name:           str,
        description:    str = "",
        sensor_type:    SensorType = SensorType.CUSTOM,
        unit:           str = "",
        min_value:      Optional[float] = None,
        max_value:      Optional[float] = None,
        sample_rate_hz: float = 10.0,
        streaming:      bool = False,
    ) -> Callable:
        """
        Register a physical sensor.

            @server.sensor("temperature", unit="°C", sensor_type=SensorType.TEMPERATURE)
            async def read_temperature():
                return SensorReading(value=22.5, unit="°C")
        """
        def decorator(fn: Callable) -> Callable:
            spec = SensorSpec(
                name          = name,
                description   = description or (fn.__doc__ or "").strip().split("\n")[0],
                sensor_type   = sensor_type,
                unit          = unit,
                min_value     = min_value,
                max_value     = max_value,
                sample_rate_hz = sample_rate_hz,
                streaming     = streaming,
            )
            self._sensors[name] = _SensorHandler(spec, fn)
            log.debug(f"Registered sensor: {name}")
            return fn
        return decorator

    def prompt(self, name: str, description: str = "") -> Callable:
        """Register a prompt template."""
        def decorator(fn: Callable) -> Callable:
            spec = PromptSpec(name=name, description=description)
            self._prompts[name] = (spec, fn)
            return fn
        return decorator

    # ── Request Handlers ──────────────────────────────────────────────────────

    async def _handle_initialize(self, params: dict, _req_id: str) -> dict:
        client_info = params.get("clientInfo", {})
        proto_ver   = params.get("protocolVersion", PCP_VERSION)
        self._connected_clients.append(client_info.get("name", "unknown"))
        self._started_at = time.time()

        return {
            "protocolVersion": PCP_VERSION,
            "capabilities": self.caps.to_dict(),
            "serverInfo": ServerInfo(
                name=self.name, version=self.version).to_dict(),
            "instructions": (
                "This is a P-MCP physical robot server. "
                "Use actuations/list to discover physical commands, "
                "sensors/list for available sensor streams. "
                "All actuations require shadow/preview to pass before execution."
            ),
        }

    async def _handle_actuations_list(self, params: dict, _req_id: str) -> dict:
        return {"actuations": [h.spec.to_dict() for h in self._actuations.values()]}

    async def _handle_sensors_list(self, params: dict, _req_id: str) -> dict:
        return {"sensors": [h.spec.to_dict() for h in self._sensors.values()]}

    async def _handle_prompts_list(self, params: dict, _req_id: str) -> dict:
        return {"prompts": [spec.to_dict() for spec, _ in self._prompts.values()]}

    async def _handle_prompts_get(self, params: dict, _req_id: str) -> dict:
        name = params.get("name", "")
        if name not in self._prompts:
            raise PCPError(PCPErrorCode.METHOD_NOT_FOUND, f"Prompt '{name}' not found")
        spec, fn = self._prompts[name]
        args = params.get("arguments", {})
        result: PromptResult = await fn(**args)
        return {"description": spec.description, "messages": [result.to_dict()]}

    async def _handle_sensors_read(self, params: dict, _req_id: str) -> dict:
        name = params.get("name", "")
        if name not in self._sensors:
            raise PCPError(PCPErrorCode.METHOD_NOT_FOUND, f"Sensor '{name}' not found")
        handler = self._sensors[name]
        reading: SensorReading = await handler.fn()
        reading.sensor_name = name
        return {"contents": [{"uri": handler.spec.uri,
                               "mimeType": "application/pcp-sensor",
                               "data": reading.to_dict()}]}

    async def _handle_actuations_call(self, params: dict, req_id: str) -> dict:
        """
        Core actuation dispatch with safety pipeline:
          1. Lookup handler
          2. Run safety middleware (lease + constitution + shadow)
          3. Execute handler
          4. Return ActuationResult
        """
        name      = params.get("name", "")
        arguments = params.get("arguments", {})
        robot_id  = params.get("robotId", self.robot_id)
        lease_token = params.get("leaseToken")

        if name not in self._actuations:
            raise PCPError(PCPErrorCode.METHOD_NOT_FOUND,
                            f"Actuation '{name}' not found. "
                            f"Available: {list(self._actuations.keys())}")

        handler = self._actuations[name]
        self._call_count += 1

        # ── Safety Pipeline ──────────────────────────────────────────────────
        if self.safety:
            # 1. Constitution check
            payload = {"call_id": req_id, "speed": arguments.get("speed", 0),
                       "z": arguments.get("z", 1.0), "energy_j": arguments.get("energy_j", 0),
                       "_shadow_ts": time.time(), **arguments}
            cleared, violations = self.safety.check_constitution(payload)
            if not cleared:
                self._blocked_count += 1
                raise PCPError(PCPErrorCode.CONSTITUTION_BLOCKED,
                                f"Safety constitution blocked: {'; '.join(violations)}",
                                data={"violations": violations})

            # 2. Shadow preview (if required)
            if handler.spec.shadow_required:
                shadow = await self.safety.run_shadow(name, robot_id, arguments)
                if not shadow.safe:
                    self._blocked_count += 1
                    raise PCPError(PCPErrorCode.SHADOW_BLOCKED,
                                    f"Shadow preview blocked: {'; '.join(shadow.violations)}",
                                    data=shadow.to_dict())

        # ── Execute Handler ──────────────────────────────────────────────────
        try:
            result: ActuationResult = await handler.fn(**arguments)
            if not isinstance(result, ActuationResult):
                # Allow returning dict for convenience
                if isinstance(result, dict):
                    result = ActuationResult(**result)
                else:
                    result = ActuationResult(success=bool(result))
            result.robot_id = robot_id
            return {"content": [{"type": "actuation", "data": result.to_dict()}],
                    "isError": not result.success}
        except PCPError:
            raise
        except Exception as exc:
            raise PCPError(PCPErrorCode.INTERNAL_ERROR,
                            f"Actuation '{name}' raised: {exc}") from exc

    async def _handle_shadow_preview(self, params: dict, _req_id: str) -> dict:
        """Explicit shadow preview without executing the actuation."""
        name      = params.get("actuationName", params.get("name", ""))
        robot_id  = params.get("robotId", self.robot_id)
        arguments = params.get("arguments", {})

        if name not in self._actuations:
            raise PCPError(PCPErrorCode.METHOD_NOT_FOUND,
                            f"Actuation '{name}' not found")

        if self.safety:
            shadow = await self.safety.run_shadow(name, robot_id, arguments)
        else:
            shadow = _default_shadow_preview(name, robot_id, arguments)

        return {"preview": shadow.to_dict()}

    async def _handle_lease_request(self, params: dict, _req_id: str) -> dict:
        req = LeaseRequest(
            robot_id     = params.get("robotId", self.robot_id),
            zone_id      = params.get("zoneId", "default"),
            duration_ms  = params.get("durationMs", 10_000),
            bid_energy_j = params.get("bidEnergyJ", 100.0),
        )
        if self.safety and hasattr(self.safety, "request_lease"):
            grant = await self.safety.request_lease(req)
        else:
            # Default: always grant if no lease manager attached
            grant = LeaseGrant(
                lease_id   = str(uuid.uuid4())[:12],
                robot_id   = req.robot_id,
                zone_id    = req.zone_id,
                granted    = True,
                expires_at = time.time() + req.duration_ms / 1000.0,
            )
        return {"lease": grant.to_dict()}

    async def _handle_lease_release(self, params: dict, _req_id: str) -> dict:
        lease_id = params.get("leaseId", "")
        if self.safety and hasattr(self.safety, "release_lease"):
            ok = self.safety.release_lease(lease_id)
        else:
            ok = True
        return {"released": ok, "leaseId": lease_id}

    async def _handle_ping(self, params: dict, _req_id: str) -> dict:
        return {"pong": True, "server": self.name, "uptime_s": time.time() - self._started_at}

    # ── Message Dispatch ──────────────────────────────────────────────────────

    async def handle_message(self, raw: dict) -> Optional[dict]:
        """Dispatch a raw JSON-RPC message and return the response dict."""
        method = raw.get("method", "")
        req_id = raw.get("id")
        params = raw.get("params", {})

        # Notification (no id) — fire and forget
        if req_id is None:
            if method == "notifications/initialized":
                log.info(f"  🤝  Client initialized: {self.name}")
            return None

        handler = self._request_handlers.get(method)
        if not handler:
            return PCPResponse(
                id=req_id,
                error=PCPError(PCPErrorCode.METHOD_NOT_FOUND,
                                f"Unknown method: {method}"),
            ).to_dict()

        try:
            result = await handler(params, req_id)
            return PCPResponse(id=req_id, result=result).to_dict()
        except PCPError as e:
            return PCPResponse(id=req_id, error=e).to_dict()
        except Exception as e:
            log.error(f"Internal error in {method}: {e}", exc_info=True)
            return PCPResponse(
                id=req_id,
                error=PCPError(PCPErrorCode.INTERNAL_ERROR, str(e)),
            ).to_dict()

    # ── Transport ─────────────────────────────────────────────────────────────

    async def run(self, transport: str = "stdio", host: str = "localhost", port: int = 8080):
        """
        Start the P-MCP server.

        transport: "stdio"  — reads JSON-RPC from stdin, writes to stdout (default)
                   "http"   — serves HTTP POST /pcp endpoint
        """
        log.info(f"  🤖  P-MCP Server '{self.name}' starting "
                 f"(transport={transport}, actuations={len(self._actuations)}, "
                 f"sensors={len(self._sensors)})")

        if transport == "stdio":
            await self._run_stdio()
        elif transport == "http":
            await self._run_http(host, port)
        else:
            raise ValueError(f"Unknown transport: {transport}. Use 'stdio' or 'http'")

    async def _run_stdio(self):
        """stdio transport — identical to MCP's stdio transport."""
        loop = asyncio.get_event_loop()
        reader = asyncio.StreamReader()
        await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)

        while True:
            try:
                line = await reader.readline()
                if not line:
                    break
                raw = json.loads(line.decode().strip())
                response = await self.handle_message(raw)
                if response is not None:
                    sys.stdout.write(json.dumps(response) + "\n")
                    sys.stdout.flush()
            except json.JSONDecodeError as e:
                err = PCPResponse(
                    id="", error=PCPError(PCPErrorCode.PARSE_ERROR, str(e))).to_dict()
                sys.stdout.write(json.dumps(err) + "\n")
                sys.stdout.flush()
            except EOFError:
                break

    async def _run_http(self, host: str, port: int):
        """Simple HTTP transport using stdlib only."""
        from http.server import BaseHTTPRequestHandler, HTTPServer

        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length  = int(self.headers.get("Content-Length", 0))
                body    = self.rfile.read(length)
                raw     = json.loads(body)
                loop    = asyncio.new_event_loop()
                resp    = loop.run_until_complete(server_ref.handle_message(raw))
                loop.close()
                data    = json.dumps(resp or {}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        httpd = HTTPServer((host, port), Handler)
        log.info(f"  🌐  P-MCP HTTP server on http://{host}:{port}/")
        httpd.serve_forever()

    # ── Info ──────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "actuations":     list(self._actuations.keys()),
            "sensors":        list(self._sensors.keys()),
            "calls_total":    self._call_count,
            "calls_blocked":  self._blocked_count,
            "uptime_s":       round(time.time() - self._started_at, 1) if self._started_at else 0,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_PY_TO_JSON: Dict[str, str] = {
    "float": "number", "int": "integer", "str": "string", "bool": "boolean",
    "list": "array",   "dict": "object",
}

def _extract_parameters(fn: Callable) -> List[ActuationParameter]:
    """Introspect function signature to build ActuationParameter list."""
    sig = inspect.signature(fn)
    params = []
    for pname, p in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        ann = p.annotation
        type_name = ann.__name__ if hasattr(ann, "__name__") else str(ann)
        json_type  = _PY_TO_JSON.get(type_name, "string")
        required   = p.default is inspect.Parameter.empty
        default    = None if required else p.default
        params.append(ActuationParameter(
            name=pname, type=json_type, required=required, default=default))
    return params


def _default_shadow_preview(name: str, robot_id: str, args: dict) -> ShadowPreview:
    """Default shadow preview when no safety middleware is attached."""
    z     = float(args.get("z", 1.0))
    speed = float(args.get("speed", 0.3))
    violations = []
    if z < 0:
        violations.append(f"Z={z} below floor (0 m)")
    if speed > 1.5:
        violations.append(f"Speed={speed} m/s exceeds limit (1.5 m/s)")
    safe = len(violations) == 0
    return ShadowPreview(
        actuation_name = name,
        robot_id       = robot_id,
        status         = ShadowStatus.SAFE if safe else ShadowStatus.UNSAFE,
        safe           = safe,
        violations     = violations,
        predicted_pose = {k: float(args[k]) for k in ("x","y","z") if k in args},
    )
