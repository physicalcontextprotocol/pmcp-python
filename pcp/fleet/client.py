"""
P-MCP Python SDK – Fleet Client
================================
High-level fleet management client for orchestrating multiple robots.

Features:
- Discover robots on the network
- Send coordinated multi-robot commands
- Monitor fleet health and telemetry
- Manage zone leases across a fleet
- Support for async/await patterns
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from pcp import __version__

try:
    import aiohttp

    _AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the 'http' extra
    aiohttp = None  # type: ignore[assignment]
    _AIOHTTP_AVAILABLE = False

from pcp.types import (
    PCP_VERSION,
    ActuationResult,
    BatchActuationResult,
    LeaseGrant,
    LeaseState,
    MetricsSnapshot,
    PCPError,
    PCPErrorCode,
    SensorReading,
)

log = logging.getLogger("pcp.fleet")


# ─────────────────────────────────────────────────────────────────────────────
#  ROBOT CONNECTION
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RobotEndpoint:
    robot_id: str
    host: str
    port: int
    transport: str = "http"  # "http" | "ws" | "tcp"
    tls: bool = False
    did: Optional[str] = None
    token: Optional[str] = None

    @property
    def base_url(self) -> str:
        scheme = "https" if self.tls else "http"
        return f"{scheme}://{self.host}:{self.port}"


@dataclass
class RobotStatus:
    robot_id: str
    endpoint: RobotEndpoint
    online: bool
    last_seen_ms: int
    actuations: List[str] = field(default_factory=list)
    sensors: List[str] = field(default_factory=list)
    active_lease: Optional[str] = None
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
#  FLEET CLIENT
# ─────────────────────────────────────────────────────────────────────────────


class FleetClient:
    """
    Manages connections to a fleet of P-MCP robots.

    Example::

        async with FleetClient() as fleet:
            await fleet.add_robot(RobotEndpoint("arm-01", "192.168.1.10", 8080))
            await fleet.add_robot(RobotEndpoint("arm-02", "192.168.1.11", 8080))

            # Parallel actuation
            results = await fleet.actuate_all("move_to", {
                "arm-01": {"x": 0.5, "y": 0.0, "z": 0.3},
                "arm-02": {"x": -0.5, "y": 0.0, "z": 0.3},
            })
    """

    def __init__(
        self,
        session_timeout_s: float = 30.0,
        retry_attempts: int = 3,
        retry_delay_s: float = 1.0,
    ):
        if not _AIOHTTP_AVAILABLE:
            raise ImportError(
                "FleetClient requires aiohttp for HTTP transport. "
                "Install it with: pip install pcp[http]  (or pcp[full])"
            )
        self._endpoints: Dict[str, RobotEndpoint] = {}
        self._status: Dict[str, RobotStatus] = {}
        self._session: Optional["aiohttp.ClientSession"] = None
        self._timeout = aiohttp.ClientTimeout(total=session_timeout_s)
        self._retry_attempts = retry_attempts
        self._retry_delay = retry_delay_s
        self._health_task: Optional[asyncio.Task] = None
        self._req_id_counter = 0

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def __aenter__(self) -> "FleetClient":
        self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    async def close(self) -> None:
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        if self._session:
            await self._session.close()

    # ── robot management ─────────────────────────────────────────────────────

    async def add_robot(
        self,
        endpoint: RobotEndpoint,
        probe: bool = True,
    ) -> RobotStatus:
        """Register a robot endpoint. Optionally probe it for capabilities."""
        self._endpoints[endpoint.robot_id] = endpoint
        status = RobotStatus(
            robot_id=endpoint.robot_id,
            endpoint=endpoint,
            online=False,
            last_seen_ms=0,
        )
        self._status[endpoint.robot_id] = status

        if probe:
            await self._probe_robot(endpoint.robot_id)

        return self._status[endpoint.robot_id]

    def remove_robot(self, robot_id: str) -> None:
        self._endpoints.pop(robot_id, None)
        self._status.pop(robot_id, None)

    def list_robots(self) -> List[RobotStatus]:
        return list(self._status.values())

    def online_robots(self) -> List[RobotStatus]:
        return [s for s in self._status.values() if s.online]

    # ── capability probe ─────────────────────────────────────────────────────

    async def _probe_robot(self, robot_id: str) -> None:
        try:
            endpoint = self._endpoints[robot_id]
            result = await self._rpc(
                endpoint,
                "initialize",
                {
                    "protocolVersion": PCP_VERSION,
                    "clientInfo": {"name": "pcp-fleet-client", "version": __version__},
                },
            )
            status = self._status[robot_id]
            status.online = True
            status.last_seen_ms = _now_ms()

            caps = result.get("capabilities", {})
            status.actuations = list(caps.get("actuations", {}).keys())
            status.sensors = list(caps.get("sensors", {}).keys())

        except Exception as e:
            log.warning("Failed to probe robot %s: %s", robot_id, e)
            self._status[robot_id].online = False
            self._status[robot_id].error = str(e)

    # ── health loop ──────────────────────────────────────────────────────────

    async def start_health_loop(self, interval_s: float = 30.0) -> None:
        """Periodically check all robot health in the background."""

        async def _loop():
            while True:
                await asyncio.sleep(interval_s)
                for robot_id in list(self._endpoints.keys()):
                    try:
                        await self._ping(robot_id)
                    except Exception:
                        pass

        self._health_task = asyncio.create_task(_loop())

    async def _ping(self, robot_id: str) -> bool:
        try:
            endpoint = self._endpoints[robot_id]
            await self._rpc(endpoint, "pcp/ping", {})
            self._status[robot_id].online = True
            self._status[robot_id].last_seen_ms = _now_ms()
            return True
        except Exception:
            self._status[robot_id].online = False
            return False

    # ── actuation ────────────────────────────────────────────────────────────

    async def actuate(
        self,
        robot_id: str,
        actuation_name: str,
        params: Dict[str, Any],
        timeout_s: Optional[float] = None,
    ) -> ActuationResult:
        """Send a single actuation to one robot."""
        endpoint = self._endpoints.get(robot_id)
        if endpoint is None:
            raise ValueError(f"Robot {robot_id!r} not registered")

        result = await self._rpc(
            endpoint,
            "actuations/execute",
            {"name": actuation_name, "params": params},
            timeout_s=timeout_s,
        )
        return ActuationResult(
            success=result.get("success", False),
            final_pose=result.get("final_pose"),
            energy_j=result.get("energy_j", 0.0),
            duration_s=result.get("duration_s", 0.0),
            error=result.get("error"),
        )

    async def actuate_all(
        self,
        actuation_name: str,
        robot_params: Dict[str, Dict[str, Any]],
        timeout_s: Optional[float] = None,
        fail_fast: bool = False,
    ) -> Dict[str, ActuationResult]:
        """Execute an actuation on multiple robots in parallel."""
        tasks = {
            robot_id: asyncio.create_task(self.actuate(robot_id, actuation_name, params, timeout_s))
            for robot_id, params in robot_params.items()
        }

        results: Dict[str, ActuationResult] = {}
        failed = False

        for robot_id, task in tasks.items():
            try:
                results[robot_id] = await task
                if fail_fast and not results[robot_id].success:
                    failed = True
                    break
            except Exception as e:
                results[robot_id] = ActuationResult(
                    success=False,
                    error=str(e),
                )
                if fail_fast:
                    failed = True
                    break

        if failed:
            # Cancel remaining tasks
            for t in tasks.values():
                t.cancel()

        return results

    async def batch_actuate(
        self,
        robot_id: str,
        actuations: List[Tuple[str, Dict[str, Any]]],
        atomic: bool = True,
    ) -> BatchActuationResult:
        """Execute multiple actuations on one robot atomically."""
        endpoint = self._endpoints.get(robot_id)
        if endpoint is None:
            raise ValueError(f"Robot {robot_id!r} not registered")

        items = [{"name": name, "params": params} for name, params in actuations]
        result = await self._rpc(
            endpoint,
            "actuations/batch",
            {"actuations": items, "atomic": atomic},
        )
        return BatchActuationResult(
            success=result.get("success", False),
            results=[
                ActuationResult(
                    success=r.get("success", False),
                    energy_j=r.get("energy_j", 0.0),
                    duration_s=r.get("duration_s", 0.0),
                    error=r.get("error"),
                ).to_dict()
                for r in result.get("results", [])
            ],
        )

    # ── sensors ──────────────────────────────────────────────────────────────

    async def read_sensor(
        self,
        robot_id: str,
        sensor_name: str,
    ) -> SensorReading:
        """Read a sensor value from a robot."""
        endpoint = self._endpoints.get(robot_id)
        if endpoint is None:
            raise ValueError(f"Robot {robot_id!r} not registered")

        result = await self._rpc(
            endpoint,
            "sensors/read",
            {"name": sensor_name},
        )
        return SensorReading(
            sensor_name=sensor_name,
            value=result.get("value"),
            unit=result.get("unit", ""),
            timestamp=result.get("timestamp", time.time()),
            quality=result.get("quality", 1.0),
        )

    async def read_all_sensors(
        self,
        sensor_name: str,
        robot_ids: Optional[List[str]] = None,
    ) -> Dict[str, SensorReading]:
        """Read the same sensor from multiple robots."""
        targets = robot_ids or list(self._endpoints.keys())
        tasks = {
            rid: asyncio.create_task(self.read_sensor(rid, sensor_name))
            for rid in targets
            if rid in self._endpoints
        }
        results = {}
        for rid, task in tasks.items():
            try:
                results[rid] = await task
            except Exception:
                results[rid] = SensorReading(
                    sensor_name=sensor_name,
                    value=None,
                    unit="",
                    timestamp=time.time(),
                    quality=0.0,
                )
        return results

    # ── lease management ─────────────────────────────────────────────────────

    async def acquire_lease(
        self,
        robot_id: str,
        zone_id: str,
        duration_ms: int = 30_000,
    ) -> LeaseGrant:
        """Request a zone lease from a robot's lease manager."""
        endpoint = self._endpoints.get(robot_id)
        if endpoint is None:
            raise ValueError(f"Robot {robot_id!r} not registered")

        result = await self._rpc(
            endpoint,
            "leases/acquire",
            {"zone_id": zone_id, "duration_ms": duration_ms},
        )
        return LeaseGrant(
            lease_id=result.get("lease_id", ""),
            zone_id=zone_id,
            robot_id=robot_id,
            expires_at=time.time() + result.get("expires_ms", 0) / 1000.0,
            state=LeaseState.ACTIVE if result.get("granted", False) else LeaseState.DENIED,
            fence_token=result.get("fence_token", 0),
        )

    async def release_lease(
        self,
        robot_id: str,
        lease_id: str,
    ) -> bool:
        endpoint = self._endpoints.get(robot_id)
        if endpoint is None:
            return False
        result = await self._rpc(
            endpoint,
            "leases/release",
            {"lease_id": lease_id},
        )
        return result.get("released", False)

    # ── metrics ──────────────────────────────────────────────────────────────

    async def get_metrics(self, robot_id: str) -> MetricsSnapshot:
        """Fetch telemetry snapshot from a robot."""
        endpoint = self._endpoints.get(robot_id)
        if endpoint is None:
            raise ValueError(f"Robot {robot_id!r} not registered")

        result = await self._rpc(endpoint, "pcp/metrics", {})
        return MetricsSnapshot(**result) if result else _empty_metrics_snapshot(robot_id)

    async def get_fleet_metrics(self) -> Dict[str, MetricsSnapshot]:
        tasks = {rid: asyncio.create_task(self.get_metrics(rid)) for rid in self._endpoints}
        out = {}
        for rid, task in tasks.items():
            try:
                out[rid] = await task
            except Exception:
                out[rid] = _empty_metrics_snapshot(rid)
        return out

    # ── internal RPC ─────────────────────────────────────────────────────────

    async def _rpc(
        self,
        endpoint: RobotEndpoint,
        method: str,
        params: Dict[str, Any],
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        self._req_id_counter += 1
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": self._req_id_counter,
        }

        headers = {}
        if endpoint.token:
            headers["Authorization"] = f"Bearer {endpoint.token}"

        url = f"{endpoint.base_url}/mcp"
        timeout = aiohttp.ClientTimeout(total=timeout_s) if timeout_s else None

        last_exc: Optional[Exception] = None
        for attempt in range(self._retry_attempts):
            try:
                session = self._session
                if session is None:
                    session = aiohttp.ClientSession(timeout=self._timeout)

                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=timeout,
                ) as resp:
                    resp.raise_for_status()
                    body = await resp.json()

                if "error" in body:
                    err = body["error"]
                    raise PCPError(
                        code=PCPErrorCode(err.get("code", -32603)),
                        message=err.get("message", "RPC error"),
                    )

                return body.get("result", {})

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exc = e
                if attempt < self._retry_attempts - 1:
                    await asyncio.sleep(self._retry_delay * (2**attempt))

        raise ConnectionError(
            f"Failed to reach {endpoint.robot_id} at {endpoint.base_url} "
            f"after {self._retry_attempts} attempts: {last_exc}"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  FLEET ORCHESTRATOR  (higher-level mission sequencing)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MissionStep:
    step_id: str
    robot_id: str
    actuation: str
    params: Dict[str, Any]
    depends_on: List[str] = field(default_factory=list)
    timeout_s: float = 30.0
    retry_on_fail: bool = False


@dataclass
class MissionResult:
    mission_id: str
    success: bool
    step_results: Dict[str, ActuationResult]
    total_duration_ms: int
    failed_steps: List[str]


class FleetOrchestrator:
    """
    Execute multi-robot missions with dependency ordering and error recovery.

    Example::

        orch = FleetOrchestrator(fleet_client)
        mission_id = await orch.run_mission([
            MissionStep("pick", "arm-01", "pick_object", {"object_id": "box-1"}),
            MissionStep("place", "arm-02", "place_object", {"object_id": "box-1"},
                        depends_on=["pick"]),
        ])
    """

    def __init__(self, fleet: FleetClient):
        self._fleet = fleet
        self._history: List[MissionResult] = []

    async def run_mission(
        self,
        steps: List[MissionStep],
        mission_id: Optional[str] = None,
    ) -> MissionResult:
        mission_id = mission_id or str(uuid.uuid4())[:8]
        log.info("Starting mission %s with %d steps", mission_id, len(steps))

        start_ms = _now_ms()
        step_results: Dict[str, ActuationResult] = {}
        failed_steps: List[str] = []

        # Build dependency graph
        {s.step_id: s for s in steps}
        completed: Set[str] = set()
        in_progress: Set[str] = set()

        while len(completed) + len(failed_steps) < len(steps):
            # Find steps whose dependencies are met
            ready = [
                s
                for s in steps
                if s.step_id not in completed
                and s.step_id not in in_progress
                and s.step_id not in failed_steps
                and all(dep in completed for dep in s.depends_on)
            ]

            if not ready:
                if not in_progress:
                    # Deadlock — remaining steps can never run
                    log.error("Mission %s deadlocked; aborting", mission_id)
                    break
                await asyncio.sleep(0.1)
                continue

            # Launch all ready steps in parallel
            tasks = {}
            for step in ready:
                in_progress.add(step.step_id)
                tasks[step.step_id] = asyncio.create_task(self._execute_step(step))

            # Collect results
            for step_id, task in tasks.items():
                try:
                    result = await task
                    step_results[step_id] = result
                    if result.success:
                        completed.add(step_id)
                    else:
                        log.warning("Step %s failed: %s", step_id, result.error)
                        failed_steps.append(step_id)
                        # Mark all downstream steps as failed too
                        self._cascade_fail(step_id, steps, failed_steps)
                except Exception as e:
                    log.error("Step %s exception: %s", step_id, e)
                    step_results[step_id] = ActuationResult(
                        success=False,
                        error=str(e),
                    )
                    failed_steps.append(step_id)
                finally:
                    in_progress.discard(step_id)

        mission_result = MissionResult(
            mission_id=mission_id,
            success=not failed_steps,
            step_results=step_results,
            total_duration_ms=_now_ms() - start_ms,
            failed_steps=failed_steps,
        )
        self._history.append(mission_result)
        log.info(
            "Mission %s %s in %dms",
            mission_id,
            "succeeded" if mission_result.success else "FAILED",
            mission_result.total_duration_ms,
        )
        return mission_result

    async def _execute_step(self, step: MissionStep) -> ActuationResult:
        attempts = 2 if step.retry_on_fail else 1
        last_result = ActuationResult(success=False)
        for _ in range(attempts):
            last_result = await self._fleet.actuate(
                step.robot_id,
                step.actuation,
                step.params,
                timeout_s=step.timeout_s,
            )
            if last_result.success:
                return last_result
        return last_result

    def _cascade_fail(
        self,
        failed_id: str,
        steps: List[MissionStep],
        failed_steps: List[str],
    ) -> None:
        for step in steps:
            if failed_id in step.depends_on and step.step_id not in failed_steps:
                failed_steps.append(step.step_id)
                self._cascade_fail(step.step_id, steps, failed_steps)

    def mission_history(self) -> List[MissionResult]:
        return list(self._history)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _now_ms() -> int:
    return int(time.time() * 1000)


def _empty_metrics_snapshot(robot_id: str) -> MetricsSnapshot:
    """
    MetricsSnapshot has no field defaults (12 required positional fields) --
    calling MetricsSnapshot() with no args, as this file's fallback paths
    previously did, would crash with a missing-arguments TypeError at
    exactly the moment a fallback is needed (e.g. inside an `except`
    handler). Explicit all-zero snapshot for "no metrics available".
    """
    return MetricsSnapshot(
        server_name=robot_id,
        uptime_s=0.0,
        calls_total=0,
        calls_blocked=0,
        calls_executed=0,
        shadow_blocks=0,
        constitution_blocks=0,
        lease_denials=0,
        active_leases=0,
        actuations_registered=0,
        sensors_registered=0,
        audit_entries=0,
    )
