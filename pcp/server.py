"""
PCP SDK — PCPServer
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

The server automatically injects the PCP safety pipeline:
  1. RateCheck        — per-robot actuation frequency cap (v0.5)
  2. LeaseCheck       — verify robot holds valid zone lease
  3. ConstitutionCheck — evaluate against TEE-signed safety rules
  4. ShadowPreview    — run ghost simulation
  5. Execute          — call your handler function
  6. Broadcast        — emit result notification to connected clients
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import sys
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from pcp.types import (
    PCP_VERSION,
    ActuationParameter,
    ActuationResult,
    ActuationSpec,
    AuditEntry,
    AuditEventType,
    BatchActuationResult,
    Capabilities,
    EStopMessage,
    EStopSource,
    LeaseGrant,
    LeaseRequest,
    LeaseState,
    MetricsSnapshot,
    PCPError,
    PCPErrorCode,
    PCPResponse,
    PromptResult,
    PromptSpec,
    SensorReading,
    SensorSpec,
    SensorType,
    ServerInfo,
    ShadowPreview,
    ShadowStatus,
)

log = logging.getLogger("pcp.server")


# ─────────────────────────────────────────────────────────────────────────────
#  INTERNAL HANDLER REGISTRATIONS
# ─────────────────────────────────────────────────────────────────────────────


class _ActuationHandler:
    __slots__ = ("spec", "fn", "shadow_fn")

    def __init__(self, spec: ActuationSpec, fn: Callable, shadow_fn: Optional[Callable] = None):
        self.spec = spec
        self.fn = fn
        self.shadow_fn = shadow_fn  # Custom shadow function, optional


class _SensorHandler:
    __slots__ = ("spec", "fn")

    def __init__(self, spec: SensorSpec, fn: Callable):
        self.spec = spec
        self.fn = fn


# ─────────────────────────────────────────────────────────────────────────────
#  PCP SERVER
# ─────────────────────────────────────────────────────────────────────────────


class PCPServer:
    """
    PCP Server — the robot-side endpoint of the protocol.

    Hosts register actuations (physical commands) and sensors (physical data),
    and the server handles the JSON-RPC 2.0 lifecycle with built-in
    safety middleware.
    """

    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        robot_id: Optional[str] = None,
        capabilities: Optional[Capabilities] = None,
        safety_middleware: Optional[Any] = None,  # SafetyMiddleware instance
    ):
        self.name = name
        self.version = version
        self.robot_id = robot_id or name
        self.caps = capabilities or Capabilities()
        self.safety = safety_middleware

        self._actuations: Dict[str, _ActuationHandler] = {}
        self._sensors: Dict[str, _SensorHandler] = {}
        self._prompts: Dict[str, tuple] = {}
        self._request_handlers: Dict[str, Callable] = {}

        self._connected_clients: List[str] = []
        self._started_at: float = 0.0
        self._call_count: int = 0
        self._blocked_count: int = 0
        self._executed_count: int = 0
        self._shadow_block_count: int = 0
        self._constitution_block_count: int = 0
        self._lease_denial_count: int = 0

        # In-memory lease store: lease_id -> LeaseGrant
        self._active_leases: Dict[str, LeaseGrant] = {}

        # E-Stop state (SAFETY_ARCHITECTURE.md sec 2, 3 Inv 5, 4). This is a
        # real, checked latch -- not just an audit log entry. While set,
        # every actuation call is rejected before it reaches the safety
        # pipeline, and it can ONLY be cleared via pcp/estop_reset, never
        # implicitly by a lease expiring or a new lease/actuation call.
        self._estopped: bool = False
        self._last_estop: Optional[EStopMessage] = None

        # ISO 10218 / IEC 62443 structured safety audit log
        self._audit_log: List[AuditEntry] = []
        self._max_audit_entries: int = 10_000  # Circular buffer limit

        # Register built-in JSON-RPC method handlers
        self._request_handlers.update(
            {
                "initialize": self._handle_initialize,
                "actuations/list": self._handle_actuations_list,
                "actuations/call": self._handle_actuations_call,
                "actuations/batch": self._handle_actuations_batch,
                "sensors/list": self._handle_sensors_list,
                "sensors/read": self._handle_sensors_read,
                "prompts/list": self._handle_prompts_list,
                "prompts/get": self._handle_prompts_get,
                "shadow/preview": self._handle_shadow_preview,
                "lease/request": self._handle_lease_request,
                "lease/release": self._handle_lease_release,
                "metrics/get": self._handle_metrics_get,
                "audit/list": self._handle_audit_list,
                "ping": self._handle_ping,
                "pcp/estop": self._handle_estop,
                "pcp/estop_reset": self._handle_estop_reset,
            }
        )

    # ── Decorator API ─────────────────────────────────────────────────────────

    def actuation(
        self,
        name: str,
        description: str = "",
        max_speed_m_s: Optional[float] = None,
        max_force_n: Optional[float] = None,
        max_energy_j: Optional[float] = None,
        requires_lease: bool = True,
        shadow_required: bool = True,
        robot_class: str = "any",
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
                name=name,
                description=description or (fn.__doc__ or "").strip().split("\n")[0],
                parameters=params,
                max_speed_m_s=max_speed_m_s,
                max_force_n=max_force_n,
                max_energy_j=max_energy_j,
                requires_lease=requires_lease,
                shadow_required=shadow_required,
                robot_class=robot_class,
            )
            self._actuations[name] = _ActuationHandler(spec, fn)
            log.debug(f"Registered actuation: {name}")
            return fn

        return decorator

    def sensor(
        self,
        name: str,
        description: str = "",
        sensor_type: SensorType = SensorType.CUSTOM,
        unit: str = "",
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        sample_rate_hz: float = 10.0,
        streaming: bool = False,
    ) -> Callable:
        """
        Register a physical sensor.

            @server.sensor("temperature", unit="°C", sensor_type=SensorType.TEMPERATURE)
            async def read_temperature():
                return SensorReading(value=22.5, unit="°C")
        """

        def decorator(fn: Callable) -> Callable:
            spec = SensorSpec(
                name=name,
                description=description or (fn.__doc__ or "").strip().split("\n")[0],
                sensor_type=sensor_type,
                unit=unit,
                min_value=min_value,
                max_value=max_value,
                sample_rate_hz=sample_rate_hz,
                streaming=streaming,
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
        params.get("protocolVersion", PCP_VERSION)
        self._connected_clients.append(client_info.get("name", "unknown"))
        self._started_at = time.time()

        return {
            "protocolVersion": PCP_VERSION,
            "capabilities": self.caps.to_dict(),
            "serverInfo": ServerInfo(name=self.name, version=self.version).to_dict(),
            "instructions": (
                "This is a PCP physical robot server. "
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
        return {
            "contents": [
                {
                    "uri": handler.spec.uri,
                    "mimeType": "application/pcp-sensor",
                    "data": reading.to_dict(),
                }
            ]
        }

    async def _handle_actuations_call(self, params: dict, req_id: str) -> dict:
        """
        Core actuation dispatch with full safety pipeline:
          0. E-Stop latch check (real block, not advisory -- see __init__)
          1. Rate limit check
          2. Lease enforcement (if requires_lease=True)
          3. Constitution check
          4. Shadow preview (if shadow_required=True)
          5. Execute handler
          6. Audit log entry
        """
        name = params.get("name", "")
        arguments = params.get("arguments", {})
        robot_id = params.get("robot_id", self.robot_id)

        # ── Layer 0: E-Stop latch ────────────────────────────────────────────
        # Checked before anything else, including lease/rate-limit lookups,
        # so an E-Stop cannot be starved by contention on those paths.
        if self._estopped:
            self._blocked_count += 1
            self._append_audit(
                AuditEntry(
                    event_type=AuditEventType.ESTOP_TRIGGERED,
                    robot_id=robot_id,
                    actuation_name=name,
                    call_id=req_id,
                    outcome="blocked",
                    violations=["E-Stop is active -- call pcp/estop_reset to clear"],
                )
            )
            raise PCPError(
                PCPErrorCode.ESTOP_ACTIVE,
                f"E-Stop is active for '{self.robot_id}'. "
                f"All actuations are blocked until pcp/estop_reset "
                f"is called by an authorized operator.",
            )
        lease_token = params.get("lease_token", "")
        fence_token = params.get("fence_token")

        if name not in self._actuations:
            raise PCPError(
                PCPErrorCode.METHOD_NOT_FOUND,
                f"Actuation '{name}' not found. " f"Available: {list(self._actuations.keys())}",
            )

        handler = self._actuations[name]
        self._call_count += 1
        t_start = time.time()

        # ── 1. Rate limit check ──────────────────────────────────────────────
        if self.safety:
            rate_violation = self.safety.check_rate_limit(robot_id, name)
            if rate_violation:
                self._blocked_count += 1
                self._constitution_block_count += 1
                self._append_audit(
                    AuditEntry(
                        event_type=AuditEventType.RATE_LIMITED,
                        robot_id=robot_id,
                        actuation_name=name,
                        call_id=req_id,
                        outcome="blocked",
                        violations=[rate_violation],
                        constitution_fp=self.safety.fingerprint,
                    )
                )
                raise PCPError(
                    PCPErrorCode.CONSTITUTION_BLOCKED, f"Rate limit exceeded: {rate_violation}"
                )

        # ── 2. Lease enforcement ─────────────────────────────────────────────
        if handler.spec.requires_lease:
            self._evict_expired_leases()
            if not lease_token:
                self._blocked_count += 1
                self._lease_denial_count += 1
                self._append_audit(
                    AuditEntry(
                        event_type=AuditEventType.LEASE_DENIED,
                        robot_id=robot_id,
                        actuation_name=name,
                        call_id=req_id,
                        outcome="blocked",
                        violations=["lease_token missing — actuation requires zone lease"],
                    )
                )
                raise PCPError(
                    PCPErrorCode.LEASE_REQUIRED,
                    f"Actuation '{name}' requires a valid zone lease "
                    f"(lease_token missing). Call lease/request first.",
                )
            grant = self._active_leases.get(lease_token)
            if grant is None:
                self._blocked_count += 1
                self._lease_denial_count += 1
                self._append_audit(
                    AuditEntry(
                        event_type=AuditEventType.LEASE_DENIED,
                        robot_id=robot_id,
                        actuation_name=name,
                        call_id=req_id,
                        outcome="blocked",
                        violations=[f"Lease '{lease_token}' not found or already released"],
                    )
                )
                raise PCPError(
                    PCPErrorCode.LEASE_REQUIRED,
                    f"Lease '{lease_token}' not found. " f"It may have expired or been released.",
                )
            if not grant.is_valid():
                self._active_leases.pop(lease_token, None)
                self._blocked_count += 1
                self._lease_denial_count += 1
                self._append_audit(
                    AuditEntry(
                        event_type=AuditEventType.LEASE_EXPIRED,
                        robot_id=robot_id,
                        actuation_name=name,
                        call_id=req_id,
                        outcome="blocked",
                        violations=[f"Lease '{lease_token}' expired at {grant.expires_at:.3f}"],
                        lease_id=lease_token,
                    )
                )
                raise PCPError(
                    PCPErrorCode.LEASE_EXPIRED,
                    f"Lease '{lease_token}' expired. Request a new lease.",
                )
            # Fencing (Kleppmann 2016): a lexically valid, unexpired lease
            # is not enough -- the fence token proves the caller's view of
            # zone ownership is current, catching the paused-holder /
            # renewed-elsewhere case even when lease_token itself still
            # matches.
            if fence_token is not None and fence_token != grant.fence_token:
                self._blocked_count += 1
                self._lease_denial_count += 1
                self._append_audit(
                    AuditEntry(
                        event_type=AuditEventType.LEASE_EXPIRED,
                        robot_id=robot_id,
                        actuation_name=name,
                        call_id=req_id,
                        outcome="blocked",
                        violations=[
                            f"Stale fence token for lease '{lease_token}': "
                            f"presented {fence_token}, current is {grant.fence_token}"
                        ],
                        lease_id=lease_token,
                    )
                )
                raise PCPError(
                    PCPErrorCode.LEASE_EXPIRED,
                    f"Stale fence token for lease '{lease_token}'. "
                    f"The lease was renewed or re-granted since this token "
                    f"was issued -- re-fetch the current lease.",
                )

        # ── 3. Constitution + 4. Shadow ──────────────────────────────────────
        constitution_fp = ""
        if self.safety:
            constitution_fp = self.safety.fingerprint
            payload = {
                "call_id": req_id,
                "speed": arguments.get("speed", 0),
                "z": arguments.get("z", 1.0),
                "energy_j": arguments.get("energy_j", 0),
                "_shadow_ts": time.time(),
                **arguments,
            }
            cleared, violations = self.safety.check_constitution(payload)
            if not cleared:
                self._blocked_count += 1
                self._constitution_block_count += 1
                self._append_audit(
                    AuditEntry(
                        event_type=AuditEventType.CONSTITUTION_BLOCKED,
                        robot_id=robot_id,
                        actuation_name=name,
                        call_id=req_id,
                        outcome="blocked",
                        violations=violations,
                        constitution_fp=constitution_fp,
                        lease_id=lease_token,
                    )
                )
                raise PCPError(
                    PCPErrorCode.CONSTITUTION_BLOCKED,
                    f"Safety constitution blocked: {'; '.join(violations)}",
                    data={"violations": violations},
                )

            if handler.spec.shadow_required:
                shadow = await self.safety.run_shadow(name, robot_id, arguments)
                if not shadow.safe:
                    self._blocked_count += 1
                    self._shadow_block_count += 1
                    self._append_audit(
                        AuditEntry(
                            event_type=AuditEventType.SHADOW_BLOCKED,
                            robot_id=robot_id,
                            actuation_name=name,
                            call_id=req_id,
                            outcome="blocked",
                            violations=shadow.violations,
                            constitution_fp=constitution_fp,
                            lease_id=lease_token,
                        )
                    )
                    raise PCPError(
                        PCPErrorCode.SHADOW_BLOCKED,
                        f"Shadow preview blocked: {'; '.join(shadow.violations)}",
                        data=shadow.to_dict(),
                    )

        # ── 5. Execute Handler ───────────────────────────────────────────────
        try:
            result: ActuationResult = await handler.fn(**arguments)
            if not isinstance(result, ActuationResult):
                if isinstance(result, dict):
                    result = ActuationResult(**result)
                else:
                    result = ActuationResult(success=bool(result))
            result.robot_id = robot_id
            result.duration_s = round(time.time() - t_start, 4)
            self._executed_count += 1
            self._append_audit(
                AuditEntry(
                    event_type=AuditEventType.ACTUATION_EXECUTED,
                    robot_id=robot_id,
                    actuation_name=name,
                    call_id=req_id,
                    outcome="executed",
                    constitution_fp=constitution_fp,
                    lease_id=lease_token,
                    duration_s=result.duration_s,
                    energy_j=result.energy_j,
                )
            )
            return {
                "content": [{"type": "actuation", "data": result.to_dict()}],
                "isError": not result.success,
            }
        except PCPError:
            raise
        except Exception as exc:
            self._append_audit(
                AuditEntry(
                    event_type=AuditEventType.ACTUATION_FAILED,
                    robot_id=robot_id,
                    actuation_name=name,
                    call_id=req_id,
                    outcome="failed",
                    violations=[str(exc)],
                    lease_id=lease_token,
                )
            )
            raise PCPError(
                PCPErrorCode.INTERNAL_ERROR, f"Actuation '{name}' raised: {exc}"
            ) from exc

    async def _handle_shadow_preview(self, params: dict, _req_id: str) -> dict:
        """Explicit shadow preview without executing the actuation."""
        name = params.get("actuation_name", params.get("name", ""))
        robot_id = params.get("robot_id", self.robot_id)
        arguments = params.get("arguments", {})

        if name not in self._actuations:
            raise PCPError(PCPErrorCode.METHOD_NOT_FOUND, f"Actuation '{name}' not found")

        if self.safety:
            shadow = await self.safety.run_shadow(name, robot_id, arguments)
        else:
            shadow = _default_shadow_preview(name, robot_id, arguments)

        return {"preview": shadow.to_dict()}

    async def _handle_lease_request(self, params: dict, _req_id: str) -> dict:
        req = LeaseRequest(
            robot_id=params.get("robot_id", self.robot_id),
            zone_id=params.get("zone_id", "default"),
            duration_ms=params.get("duration_ms", 10_000),
            bid_energy_j=params.get("bid_energy_j", 100.0),
        )
        if self.safety and hasattr(self.safety, "request_lease"):
            grant = await self.safety.request_lease(req)
        else:
            grant = LeaseGrant(
                lease_id=str(uuid.uuid4())[:12],
                robot_id=req.robot_id,
                zone_id=req.zone_id,
                state=LeaseState.ACTIVE,
                expires_at=time.time() + req.duration_ms / 1000.0,
                fence_token=1,
            )

        # Store the grant so actuation calls can validate it
        if grant.granted:
            self._active_leases[grant.lease_id] = grant
            self._append_audit(
                AuditEntry(
                    event_type=AuditEventType.LEASE_GRANTED,
                    robot_id=req.robot_id,
                    call_id=_req_id,
                    outcome="granted",
                    lease_id=grant.lease_id,
                )
            )
        else:
            self._lease_denial_count += 1
            self._append_audit(
                AuditEntry(
                    event_type=AuditEventType.LEASE_DENIED,
                    robot_id=req.robot_id,
                    call_id=_req_id,
                    outcome="denied",
                    violations=[grant.deny_reason],
                )
            )

        return {"lease": grant.to_dict()}

    async def _handle_lease_release(self, params: dict, _req_id: str) -> dict:
        lease_id = params.get("lease_id", "")
        self._active_leases.pop(lease_id, None)
        if self.safety and hasattr(self.safety, "release_lease"):
            ok = self.safety.release_lease(lease_id)
        else:
            ok = True
        self._append_audit(
            AuditEntry(
                event_type=AuditEventType.LEASE_RELEASED,
                call_id=_req_id,
                outcome="released",
                lease_id=lease_id,
            )
        )
        return {"released": ok, "lease_id": lease_id}

    async def _handle_actuations_batch(self, params: dict, req_id: str) -> dict:
        """
        Execute multiple actuations atomically under a shared lease and safety check.

        If atomic=True (default), the first safety failure aborts the entire batch.
        """
        actuations = params.get("actuations", [])
        zone_id = params.get("zone_id", "default")
        robot_id = params.get("robot_id", self.robot_id)
        atomic = params.get("atomic", True)

        if not actuations:
            raise PCPError(
                PCPErrorCode.INVALID_PARAMS, "actuations/batch requires at least one actuation"
            )

        # Request a shared lease for the batch
        lease_resp = await self._handle_lease_request(
            {
                "robot_id": robot_id,
                "zone_id": zone_id,
                "duration_ms": 30_000,
            },
            req_id + "-batch-lease",
        )
        lease = lease_resp.get("lease", {})
        lease_id = lease.get("lease_id", "")

        if not lease.get("state") == "ACTIVE":
            raise PCPError(
                PCPErrorCode.LEASE_REQUIRED,
                f"Batch lease denied: {lease.get('deny_reason', 'zone occupied')}",
            )

        batch_result = BatchActuationResult(batch_id=req_id + "-batch")
        try:
            for idx, act in enumerate(actuations):
                act_name = act.get("name", "")
                act_args = act.get("arguments", {})
                try:
                    result = await self._handle_actuations_call(
                        {
                            "name": act_name,
                            "arguments": act_args,
                            "robot_id": robot_id,
                            "lease_token": lease_id,
                            "fence_token": lease.get("fence_token"),
                        },
                        req_id=f"{req_id}-{idx}",
                    )
                    content = result.get("content", [{}])
                    batch_result.results.append(
                        content[0].get("data", result) if content else result
                    )
                except PCPError as e:
                    batch_result.results.append({"error": e.message, "index": idx})
                    if atomic:
                        batch_result.success = False
                        batch_result.failed_at = idx
                        batch_result.error = e.message
                        self._append_audit(
                            AuditEntry(
                                event_type=AuditEventType.BATCH_BLOCKED,
                                robot_id=robot_id,
                                actuation_name=act_name,
                                call_id=req_id,
                                outcome="blocked",
                                violations=[e.message],
                                lease_id=lease_id,
                            )
                        )
                        break
        finally:
            await self._handle_lease_release({"lease_id": lease_id}, req_id)

        return {"batch": batch_result.to_dict()}

        if batch_result.success:
            self._append_audit(
                AuditEntry(
                    event_type=AuditEventType.BATCH_EXECUTED,
                    robot_id=robot_id,
                    call_id=req_id,
                    outcome="executed",
                    lease_id=lease_id,
                )
            )

        return {"batch": batch_result.to_dict()}

    async def _handle_metrics_get(self, params: dict, _req_id: str) -> dict:
        """Return a point-in-time telemetry snapshot."""
        self._evict_expired_leases()
        snap = MetricsSnapshot(
            server_name=self.name,
            uptime_s=round(time.time() - self._started_at, 1) if self._started_at else 0.0,
            calls_total=self._call_count,
            calls_blocked=self._blocked_count,
            calls_executed=self._executed_count,
            shadow_blocks=self._shadow_block_count,
            constitution_blocks=self._constitution_block_count,
            lease_denials=self._lease_denial_count,
            active_leases=len(self._active_leases),
            actuations_registered=len(self._actuations),
            sensors_registered=len(self._sensors),
            audit_entries=len(self._audit_log),
        )
        return {"metrics": snap.to_dict()}

    async def _handle_audit_list(self, params: dict, _req_id: str) -> dict:
        """
        Return recent audit log entries, optionally filtered.

        Params:
          limit      — max entries to return (default 100)
          robot_id   — filter by robot
          event_type — filter by event type string
          since      — Unix timestamp; return entries after this time
        """
        limit = int(params.get("limit", 100))
        robot_filter = params.get("robot_id", "")
        type_filter = params.get("event_type", "")
        since = float(params.get("since", 0.0))

        entries = self._audit_log
        if robot_filter:
            entries = [e for e in entries if e.robot_id == robot_filter]
        if type_filter:
            entries = [e for e in entries if e.event_type.value == type_filter]
        if since:
            entries = [e for e in entries if e.timestamp >= since]

        # Return the most recent `limit` entries
        return {
            "entries": [e.to_dict() for e in entries[-limit:]],
            "total": len(self._audit_log),
        }

    async def _handle_ping(self, params: dict, _req_id: str) -> dict:
        return {"pong": True, "server": self.name, "uptime_s": time.time() - self._started_at}

    async def _handle_estop(self, params: dict, req_id: str) -> dict:
        """
        Emergency stop. First-class, lease-independent -- does not touch
        SafetyMiddleware, the lease store, or the actuation dispatch path,
        so it cannot inherit their latency or failure modes. Sets a real
        latch checked at the very top of _handle_actuations_call.
        """
        robot_id = params.get("robot_id", self.robot_id)
        source_raw = params.get("source")
        source = EStopSource(source_raw) if source_raw else None

        estop = EStopMessage(robot_id=robot_id, source=source)
        self._estopped = True
        self._last_estop = estop

        self._append_audit(
            AuditEntry(
                event_type=AuditEventType.ESTOP_TRIGGERED,
                robot_id=robot_id,
                call_id=req_id,
                outcome="estopped",
                violations=[
                    f"E-Stop triggered (source={source.value if source else 'unspecified'})"
                ],
            )
        )
        log.warning(
            f"  🛑  E-STOP TRIGGERED robot={robot_id} "
            f"source={source.value if source else 'unspecified'}"
        )
        return {"estop": estop.to_dict()}

    async def _handle_estop_reset(self, params: dict, req_id: str) -> dict:
        """
        Clear the E-Stop latch. Deliberately a separate, explicit method
        (never implicit via a new lease or actuation call) so clearing an
        E-Stop is always a distinct, auditable operator action.
        """
        was_estopped = self._estopped
        self._estopped = False
        self._append_audit(
            AuditEntry(
                event_type=AuditEventType.ESTOP_TRIGGERED,
                robot_id=params.get("robot_id", self.robot_id),
                call_id=req_id,
                outcome="estop_reset",
                violations=[],
            )
        )
        log.warning(f"  ✅  E-Stop reset for '{self.robot_id}'")
        return {"was_estopped": was_estopped, "estopped": self._estopped}

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
                error=PCPError(PCPErrorCode.METHOD_NOT_FOUND, f"Unknown method: {method}"),
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
        Start the PCP server.

        transport: "stdio"  — reads JSON-RPC from stdin, writes to stdout (default)
                   "http"   — serves HTTP POST /pcp endpoint
        """
        log.info(
            f"  🤖  PCP Server '{self.name}' starting "
            f"(transport={transport}, actuations={len(self._actuations)}, "
            f"sensors={len(self._sensors)})"
        )

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
                err = PCPResponse(id="", error=PCPError(PCPErrorCode.PARSE_ERROR, str(e))).to_dict()
                sys.stdout.write(json.dumps(err) + "\n")
                sys.stdout.flush()
            except EOFError:
                break

    async def _run_http(self, host: str, port: int):
        """
        Pure-asyncio HTTP transport.

        Replaces the previous stdlib BaseHTTPRequestHandler approach which
        created a new event loop per request (broken under Python 3.10+).
        Uses asyncio.start_server for a proper async connection pipeline.
        """
        server_ref = self

        async def handle_connection(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ):
            try:
                # ── Read HTTP headers ────────────────────────────────────────
                raw_headers = b""
                while b"\r\n\r\n" not in raw_headers:
                    chunk = await reader.read(4096)
                    if not chunk:
                        return
                    raw_headers += chunk

                header_section, body_start = raw_headers.split(b"\r\n\r\n", 1)
                headers_str = header_section.decode(errors="replace")

                # Parse Content-Length
                content_length = 0
                for line in headers_str.splitlines()[1:]:
                    if line.lower().startswith("content-length:"):
                        content_length = int(line.split(":", 1)[1].strip())
                        break

                # ── Read body ────────────────────────────────────────────────
                body = body_start
                while len(body) < content_length:
                    chunk = await reader.read(content_length - len(body))
                    if not chunk:
                        break
                    body += chunk

                # ── Handle OPTIONS preflight ─────────────────────────────────
                first_line = headers_str.splitlines()[0] if headers_str else ""
                if first_line.startswith("OPTIONS"):
                    writer.write(
                        b"HTTP/1.1 204 No Content\r\n"
                        b"Access-Control-Allow-Origin: *\r\n"
                        b"Access-Control-Allow-Methods: POST, OPTIONS\r\n"
                        b"Access-Control-Allow-Headers: Content-Type\r\n"
                        b"Connection: close\r\n\r\n"
                    )
                    await writer.drain()
                    return

                # ── Dispatch JSON-RPC ────────────────────────────────────────
                raw = json.loads(body)
                resp = await server_ref.handle_message(raw)
                data = json.dumps(resp or {}).encode()

                response = (
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: application/json\r\n"
                    b"Access-Control-Allow-Origin: *\r\n"
                    + f"Content-Length: {len(data)}\r\n".encode()
                    + b"Connection: close\r\n\r\n"
                    + data
                )
                writer.write(response)
                await writer.drain()

            except json.JSONDecodeError as e:
                err_body = json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": f"Parse error: {e}"},
                    }
                ).encode()
                writer.write(
                    b"HTTP/1.1 400 Bad Request\r\n"
                    b"Content-Type: application/json\r\n"
                    + f"Content-Length: {len(err_body)}\r\n".encode()
                    + b"Connection: close\r\n\r\n"
                    + err_body
                )
                await writer.drain()
            except Exception as e:
                log.error(f"HTTP handler error: {e}", exc_info=True)
            finally:
                try:
                    writer.close()
                    await writer.wait_closed()
                except Exception:
                    pass

        srv = await asyncio.start_server(handle_connection, host, port)
        log.info(f"  🌐  PCP HTTP server on http://{host}:{port}/")
        async with srv:
            await srv.serve_forever()

    # ── Info ──────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        self._evict_expired_leases()
        return {
            "actuations": list(self._actuations.keys()),
            "sensors": list(self._sensors.keys()),
            "calls_total": self._call_count,
            "calls_blocked": self._blocked_count,
            "calls_executed": self._executed_count,
            "shadow_blocks": self._shadow_block_count,
            "constitution_blocks": self._constitution_block_count,
            "lease_denials": self._lease_denial_count,
            "active_leases": len(self._active_leases),
            "audit_entries": len(self._audit_log),
            "uptime_s": round(time.time() - self._started_at, 1) if self._started_at else 0,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _evict_expired_leases(self):
        """Remove expired leases from the in-memory store."""
        now = time.time()
        expired = [lid for lid, g in self._active_leases.items() if g.expires_at <= now]
        for lid in expired:
            self._active_leases.pop(lid, None)
            log.debug(f"Evicted expired lease: {lid}")

    def _append_audit(self, entry: AuditEntry):
        """Append an audit entry, evicting oldest if the circular buffer is full."""
        if len(self._audit_log) >= self._max_audit_entries:
            self._audit_log = self._audit_log[-(self._max_audit_entries // 2) :]
        self._audit_log.append(entry)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_PY_TO_JSON: Dict[str, str] = {
    "float": "number",
    "int": "integer",
    "str": "string",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
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
        json_type = _PY_TO_JSON.get(type_name, "string")
        required = p.default is inspect.Parameter.empty
        default = None if required else p.default
        params.append(
            ActuationParameter(name=pname, type=json_type, required=required, default=default)
        )
    return params


def _default_shadow_preview(name: str, robot_id: str, args: dict) -> ShadowPreview:
    """Default shadow preview when no safety middleware is attached."""
    z = float(args.get("z", 1.0))
    speed = float(args.get("speed", 0.3))
    violations = []
    if z < 0:
        violations.append(f"Z={z} below floor (0 m)")
    if speed > 1.5:
        violations.append(f"Speed={speed} m/s exceeds limit (1.5 m/s)")
    safe = len(violations) == 0
    return ShadowPreview(
        actuation_name=name,
        robot_id=robot_id,
        status=ShadowStatus.SAFE if safe else ShadowStatus.UNSAFE,
        safe=safe,
        violations=violations,
        predicted_pose={k: float(args[k]) for k in ("x", "y", "z") if k in args},
    )
