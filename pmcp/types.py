"""
P-MCP SDK — Types
=================
All data types for the Physical Model Context Protocol.

Mirrors MCP's type system (JSON-RPC 2.0 envelopes + physical primitives),
then extends it with the physical safety layer:
  Actuations  ≈  MCP Tools      (physical movement commands)
  Sensors     ≈  MCP Resources  (physical data streams)
  Prompts     =  MCP Prompts    (safety templates / workflows)

New vs MCP:
  ShadowPreview       — pre-flight 3D simulation result
  ConstitutionCheck   — TEE-signed rule evaluation
  LeaseRequest/Grant  — temporal zone ownership token
  BatchActuation      — atomic multi-actuation execution (v0.5)
  AuditEntry          — ISO 10218-compliant safety event log (v0.5)
  MetricsSnapshot     — server telemetry snapshot (v0.5)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
#  PROTOCOL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

PMCP_VERSION = "0.5"
JSONRPC_VERSION = "2.0"
SPEC_DATE = "2025-05-04"


# ─────────────────────────────────────────────────────────────────────────────
#  ERROR CODES (JSON-RPC standard + P-MCP extensions)
# ─────────────────────────────────────────────────────────────────────────────


class PMCPErrorCode(int, Enum):
    # JSON-RPC 2.0 standard
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # P-MCP Physical Safety (-33000 range)
    SHADOW_BLOCKED = -33001  # Shadow validator rejected the trajectory
    CONSTITUTION_BLOCKED = -33002  # Safety constitution rule violated
    LEASE_REQUIRED = -33003  # No valid lease for this zone
    LEASE_EXPIRED = -33004  # Lease expired mid-execution
    ESTOP_ACTIVE = -33005  # Emergency stop is active
    FLOOR_GUARD = -33006  # Z target below floor limit
    SPEED_LIMIT = -33007  # Speed exceeds hard limit
    ENERGY_BUDGET = -33008  # Energy budget exhausted
    HUMAN_PROXIMITY = -33009  # Human too close to workspace
    ZK_PROOF_INVALID = -33010  # ZK safety proof failed verification


@dataclass
class PMCPError(Exception):
    """
    JSON-RPC error payload, also directly raise-able as a Python exception.

    NOTE: this must inherit from Exception -- pmcp/server.py raises and
    catches PMCPError throughout its dispatch/error-handling path. Before
    this fix it was a plain dataclass, so every `raise PMCPError(...)` and
    `except PMCPError` in that file crashed with an unrelated
    "exceptions must derive from BaseException" TypeError -- meaning every
    safety-block error path (rate limit, lease denied, constitution
    violation, shadow blocked, etc.) was broken whenever actually exercised
    end-to-end. Caught via a real handle_message() round trip, not by the
    existing unit tests, which apparently never exercised the full path.
    """

    code: PMCPErrorCode
    message: str
    data: Optional[Any] = None

    def __post_init__(self):
        Exception.__init__(self, self.message)

    def to_dict(self) -> dict:
        d = {"code": int(self.code), "message": self.message}
        if self.data is not None:
            d["data"] = self.data
        return d


# ─────────────────────────────────────────────────────────────────────────────
#  JSON-RPC 2.0 ENVELOPE TYPES
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PMCPRequest:
    """JSON-RPC 2.0 request message."""

    method: str
    params: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    jsonrpc: str = JSONRPC_VERSION

    def to_dict(self) -> dict:
        return {
            "jsonrpc": self.jsonrpc,
            "id": self.id,
            "method": self.method,
            "params": self.params,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PMCPRequest":
        return cls(
            method=d["method"],
            params=d.get("params", {}),
            id=d.get("id", ""),
            jsonrpc=d.get("jsonrpc", JSONRPC_VERSION),
        )


@dataclass
class PMCPResponse:
    """JSON-RPC 2.0 response message."""

    id: str
    result: Optional[Any] = None
    error: Optional[PMCPError] = None
    jsonrpc: str = JSONRPC_VERSION

    def to_dict(self) -> dict:
        d: dict = {"jsonrpc": self.jsonrpc, "id": self.id}
        if self.error:
            d["error"] = self.error.to_dict()
        else:
            d["result"] = self.result
        return d


@dataclass
class PMCPNotification:
    """JSON-RPC 2.0 notification (no response expected)."""

    method: str
    params: Dict[str, Any] = field(default_factory=dict)
    jsonrpc: str = JSONRPC_VERSION

    def to_dict(self) -> dict:
        return {"jsonrpc": self.jsonrpc, "method": self.method, "params": self.params}


# ─────────────────────────────────────────────────────────────────────────────
#  LIFECYCLE TYPES
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ServerInfo:
    name: str
    version: str
    protocol_version: str = PMCP_VERSION

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "protocolVersion": self.protocol_version,
        }


@dataclass
class ClientInfo:
    name: str
    version: str

    def to_dict(self) -> dict:
        return {"name": self.name, "version": self.version}


@dataclass
class Capabilities:
    """Declared during initialize handshake."""

    actuations: bool = True  # Server can execute physical actuations
    sensors: bool = True  # Server exposes sensor streams
    prompts: bool = False  # Server provides prompt templates
    shadow: bool = True  # Server supports shadow preview
    constitution: bool = True  # Server enforces safety constitution
    leases: bool = True  # Server uses temporal zone leases
    streaming: bool = False  # Server supports sensor streaming
    zk_proofs: bool = False  # Server generates ZK safety proofs

    def to_dict(self) -> dict:
        caps: dict = {}
        if self.actuations:
            caps["actuations"] = {"listChanged": True}
        if self.sensors:
            caps["sensors"] = {"streaming": self.streaming}
        if self.prompts:
            caps["prompts"] = {}
        if self.shadow:
            caps["shadow"] = {}
        if self.constitution:
            caps["constitution"] = {}
        if self.leases:
            caps["leases"] = {}
        if self.zk_proofs:
            caps["zkProofs"] = {}
        return caps


# ─────────────────────────────────────────────────────────────────────────────
#  PHYSICAL PRIMITIVES
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. Actuations (≈ MCP Tools) ──────────────────────────────────────────────


@dataclass
class ActuationParameter:
    """JSON-Schema parameter descriptor for an actuation."""

    name: str
    type: str  # "number" | "integer" | "string" | "boolean"
    description: str = ""
    required: bool = True
    default: Any = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    unit: str = ""  # "m/s", "m", "rad", "N", "°C" etc.

    def to_schema_property(self) -> dict:
        prop: dict = {"type": self.type, "description": self.description}
        if self.unit:
            prop["x-unit"] = self.unit
        if self.minimum is not None:
            prop["minimum"] = self.minimum
        if self.maximum is not None:
            prop["maximum"] = self.maximum
        if self.default is not None:
            prop["default"] = self.default
        return prop


@dataclass
class ActuationSpec:
    """
    Describes a physical actuation (P-MCP primitive #1).

    Analogous to MCP Tool, but with physical safety metadata:
      - max_speed_m_s     hard velocity limit
      - max_force_n       hard force limit
      - requires_lease    zone ownership required before execute
      - shadow_required   shadow preview must pass before execute
    """

    name: str
    description: str
    parameters: List[ActuationParameter] = field(default_factory=list)
    max_speed_m_s: Optional[float] = None
    max_force_n: Optional[float] = None
    max_energy_j: Optional[float] = None
    requires_lease: bool = True
    shadow_required: bool = True
    robot_class: str = ""  # "arm" | "mobile" | "plc" | "any"
    safety_category: str = "motion"

    def to_dict(self) -> dict:
        props = {p.name: p.to_schema_property() for p in self.parameters}
        required_params = [p.name for p in self.parameters if p.required]
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {"type": "object", "properties": props, "required": required_params},
            "physical": {
                "max_speed": self.max_speed_m_s,
                "max_force": self.max_force_n,
                "max_energy": self.max_energy_j,
                "requires_lease": self.requires_lease,
                "shadow_required": self.shadow_required,
                "robot_class": self.robot_class,
                "safety_category": self.safety_category,
            },
        }


@dataclass
class ActuationResult:
    """Result of executing a physical actuation."""

    success: bool
    actuation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    robot_id: str = ""
    final_pose: Optional[Dict[str, float]] = None  # {x, y, z, roll, pitch, yaw}
    duration_s: float = 0.0
    energy_j: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "actuation_id": self.actuation_id,
            "robot_id": self.robot_id,
            "final_pose": self.final_pose,
            "duration_s": self.duration_s,
            "energy_j": self.energy_j,
            "error": self.error,
            "metadata": self.metadata,
        }


# ── 2. Sensors (≈ MCP Resources) ─────────────────────────────────────────────


class SensorType(str, Enum):
    POSITION = "position"  # Encoder / GPS / odometry
    FORCE = "force"  # Force/torque sensor
    TEMPERATURE = "temperature"  # Thermal
    PROXIMITY = "proximity"  # Sonar / IR / LiDAR
    VISION = "vision"  # Camera / depth
    IMU = "imu"  # Accelerometer / gyro
    CURRENT = "current"  # Motor current
    VOLTAGE = "voltage"  # Battery voltage
    PRESSURE = "pressure"  # Pneumatic / hydraulic
    CUSTOM = "custom"


@dataclass
class SensorSpec:
    """
    Describes a physical sensor (P-MCP primitive #2).

    Analogous to MCP Resource, but exposes real-time physical data.
    URI format:  pmcp://sensor/{robot_id}/{sensor_name}
    """

    name: str
    description: str
    sensor_type: SensorType
    unit: str  # "m", "N", "°C", "m/s²", etc.
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    sample_rate_hz: float = 10.0
    streaming: bool = False

    @property
    def uri(self) -> str:
        return f"pmcp://sensor/{self.name}"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "uri": self.uri,
            "description": self.description,
            "mimeType": "application/pmcp-sensor",
            "physical": {
                "sensor_type": self.sensor_type.value,
                "unit": self.unit,
                "min_value": self.min_value,
                "max_value": self.max_value,
                "sample_rate_hz": self.sample_rate_hz,
                "streaming": self.streaming,
            },
        }


@dataclass
class SensorReading:
    """A single sensor measurement."""

    sensor_name: str = ""
    value: Any = None
    unit: str = ""
    timestamp: float = field(default_factory=time.time)
    quality: float = 1.0  # 0=bad, 1=good
    raw: Optional[Any] = None

    def to_dict(self) -> dict:
        return {
            "sensor_name": self.sensor_name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp,
            "quality": self.quality,
        }


# ── 3. Prompts (= MCP Prompts) ────────────────────────────────────────────────


@dataclass
class PromptSpec:
    """A reusable safety/interaction template (same as MCP Prompt)."""

    name: str
    description: str
    arguments: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description, "arguments": self.arguments}


@dataclass
class PromptResult:
    role: str  # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": {"type": "text", "text": self.content}}


# ─────────────────────────────────────────────────────────────────────────────
#  PHYSICAL SAFETY LAYER  (P-MCP extension beyond MCP)
# ─────────────────────────────────────────────────────────────────────────────

# ── Shadow Preview ────────────────────────────────────────────────────────────


class ShadowStatus(str, Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    COLLISION = "collision"
    FLOOR_GUARD = "floor_guard"
    SPEED_LIMIT = "speed_limit"
    HUMAN_NEARBY = "human_nearby"
    PENDING = "pending"


class ShadowVerdict(str, Enum):
    """
    pmcp-spec/schema/v0.6.0 ShadowResult.verdict — set by the non-ML monitor,
    not the HNN directly (SAFETY_ARCHITECTURE.md sec 10.1). FAIL = hard limit
    violated. INDETERMINATE = confidence/OOD/ensemble check exceeded threshold
    ("I don't know" != "it's fine"). Both FAIL and INDETERMINATE block
    actuation. This SDK version derives verdict from the kinematic/physics
    ShadowStatus below; it does not yet run the HNN-Simplex monitor that
    would produce CONDITIONAL_PASS/INDETERMINATE from ensemble disagreement
    or conformal-prediction confidence intervals (see ConformalConfidence /
    MonitoringBlock / DeterminismBlock in the schema, and the open research
    items in pmcp-spec on cross-hardware determinism budgets and conformal
    prediction under adversarial input — those remain unimplemented anywhere
    in this org today, so this SDK never fabricates them).
    """

    PASS = "PASS"
    CONDITIONAL_PASS = "CONDITIONAL_PASS"
    FAIL = "FAIL"
    INDETERMINATE = "INDETERMINATE"


def _status_to_verdict(status: "ShadowStatus", safe: bool) -> "ShadowVerdict":
    if safe and status == ShadowStatus.SAFE:
        return ShadowVerdict.PASS
    if status == ShadowStatus.PENDING:
        return ShadowVerdict.INDETERMINATE
    return ShadowVerdict.FAIL


@dataclass
class ShadowPreview:
    """
    Pre-flight 3D simulation result (P-MCP primitive #4).

    Before any actuation executes on hardware, the Shadow layer
    runs a ghost simulation and returns this object.
    status == SAFE is required for actuation to proceed.
    """

    actuation_name: str
    robot_id: str
    status: ShadowStatus
    safe: bool
    predicted_pose: Optional[Dict[str, float]] = None
    duration_s: float = 0.0
    energy_j: float = 0.0
    collisions: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    shadow_token: str = field(default_factory=lambda: str(uuid.uuid4())[:16])
    timestamp: float = field(default_factory=time.time)

    @property
    def verdict(self) -> ShadowVerdict:
        return _status_to_verdict(self.status, self.safe)

    def to_dict(self) -> dict:
        return {
            # schema/v0.6.0 ShadowResult field names (canonical, snake_case).
            "verdict": self.verdict.value,
            # predicted_trajectory is schema-optional in this SDK version:
            # the physics/geometric simulator predicts a single resulting
            # pose, not a full trajectory, so we report that honestly under
            # predicted_pose rather than fabricate trajectory samples.
            "predicted_trajectory": None,
            # confidence / monitoring / determinism are schema-optional and
            # deliberately omitted: they describe the HNN-Simplex monitor's
            # conformal-prediction and ensemble-disagreement output, which
            # is not implemented anywhere in this org yet (open research
            # item, see pmcp-spec). Do not populate these with fake numbers.
            "confidence": None,
            "monitoring": None,
            "determinism": None,
            # Additional fields beyond the schema's core five — real,
            # working output from this SDK's physics/geometric simulator.
            "actuation_name": self.actuation_name,
            "robot_id": self.robot_id,
            "status": self.status.value,
            "safe": self.safe,
            "predicted_pose": self.predicted_pose,
            "duration_s": self.duration_s,
            "energy_j": self.energy_j,
            "collisions": self.collisions,
            "violations": self.violations,
            "shadow_token": self.shadow_token,
            "timestamp": self.timestamp,
        }


# ── Constitution Check ────────────────────────────────────────────────────────


@dataclass
class ConstitutionCheck:
    """TEE-signed safety constitution evaluation result."""

    cleared: bool
    violations: List[str] = field(default_factory=list)
    rule_ids: List[str] = field(default_factory=list)
    fingerprint: str = ""  # Constitution bundle fingerprint
    tee_sig: Optional[str] = None  # Ed25519 signature

    def to_dict(self) -> dict:
        # First violated rule id, if any -- extracted from "[RULE-ID] msg"
        # formatted violation strings. schema/v0.6.0 ConstitutionCheck models
        # a single violated_rule_id; this SDK evaluates the full ruleset and
        # reports every violation, which is strictly more informative, so we
        # keep the full list under the additional "violations"/"reasons"
        # fields rather than discard information to fit the narrower schema
        # field.
        violated_rule_id = None
        if self.violations:
            first = self.violations[0]
            if first.startswith("[") and "]" in first:
                violated_rule_id = first[1 : first.index("]")]
        return {
            # schema/v0.6.0 ConstitutionCheck field names (canonical):
            "passed": self.cleared,
            "rule_ids_evaluated": self.rule_ids,
            "violated_rule_id": violated_rule_id,
            "reason": "; ".join(self.violations) if self.violations else None,
            # Additional fields: full violation list + fingerprint, real
            # data this SDK produces that the core schema fields alone
            # would lose.
            "cleared": self.cleared,
            "violations": self.violations,
            "rule_ids": self.rule_ids,
            "fingerprint": self.fingerprint,
        }


# ── Temporal Lease ────────────────────────────────────────────────────────────


class LeaseState(str, Enum):
    """
    Matches pmcp-spec/schema/v0.6.0 LeaseState exactly. FREE and PENDING are
    included for schema conformance even though this SDK's non-Raft lease
    manager doesn't yet produce PENDING grants (Raft-backed multi-replica
    arbitration is a later version; see SAFETY_ARCHITECTURE.md).
    """

    FREE = "FREE"
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    EXPIRED = "EXPIRED"
    DENIED = "DENIED"


@dataclass
class LeaseRequest:
    robot_id: str
    zone_id: str
    duration_ms: int = 10_000
    bid_energy_j: float = 100.0

    def to_dict(self) -> dict:
        return {
            "robot_id": self.robot_id,
            "zone_id": self.zone_id,
            "duration_ms": self.duration_ms,
            "bid_energy_j": self.bid_energy_j,
        }


@dataclass
class LeaseGrant:
    """Temporal zone lease token."""

    lease_id: str
    robot_id: str
    zone_id: str
    state: LeaseState
    expires_at: float  # Unix timestamp
    token: str = ""  # Signed lease token
    deny_reason: str = ""
    # Monotonically increasing fencing token (Kleppmann 2016). Must be
    # re-presented on every actuation call against this zone, not only at
    # lease-request time -- see SafetyMiddleware.check_lease / the
    # _fence_token wire parameter on tools/call.
    fence_token: int = 0

    @property
    def granted(self) -> bool:
        """Backward-compatible convenience accessor."""
        return self.state == LeaseState.ACTIVE

    def is_valid(self) -> bool:
        return self.granted and time.time() < self.expires_at

    def to_dict(self) -> dict:
        return {
            "lease_id": self.lease_id,
            "robot_id": self.robot_id,
            "zone_id": self.zone_id,
            "state": self.state.value,
            "expires_at": self.expires_at,
            "valid": self.is_valid(),
            "deny_reason": self.deny_reason or None,
            "fence_token": self.fence_token,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  BATCH ACTUATION  (P-MCP v0.5 extension)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class BatchActuationRequest:
    """
    Atomic multi-actuation request.

    All actuations in the batch share one lease and one shadow validation pass.
    If any member fails the safety check, the entire batch is rejected when
    atomic=True (default).  Use atomic=False for best-effort execution.
    """

    actuations: List[Dict[str, Any]]  # [{"name": ..., "arguments": {...}}, ...]
    zone_id: str = "default"
    robot_id: str = ""
    atomic: bool = True  # All-or-nothing execution

    def to_dict(self) -> dict:
        return {
            "actuations": self.actuations,
            "zone_id": self.zone_id,
            "robot_id": self.robot_id,
            "atomic": self.atomic,
        }


@dataclass
class BatchActuationResult:
    """Result of a batch actuation."""

    batch_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    success: bool = True
    results: List[Dict[str, Any]] = field(default_factory=list)
    failed_at: Optional[int] = None  # Index of first failure (atomic=True)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "success": self.success,
            "results": self.results,
            "failed_at": self.failed_at,
            "error": self.error,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  AUDIT LOG  (ISO 10218 / IEC 62443 compliance)
# ─────────────────────────────────────────────────────────────────────────────

# ── E-Stop ─────────────────────────────────────────────────────────────────
# docs/SAFETY_ARCHITECTURE.md sec 2, 3 (Inv 5), 4: first-class, lease-
# independent, bypasses Lease->Constitution->Shadow entirely. NOT modeled as
# an ErrorCode -- this is a distinct message type precisely so it cannot
# inherit the gate pipeline's latency/failure modes.


class StopCategory(int, Enum):
    """
    IEC 60204-1 stop categories, as referenced by SAFETY_ARCHITECTURE.md.
    EStopMessage is always CATEGORY_0 (schema: stop_category is a hardcoded
    const 0 on that message type) -- controlled-deceleration (1) and
    graceful (2) stops are represented by other, non-E-Stop mechanisms.
    """

    CATEGORY_0 = 0  # Immediate, uncontrolled power removal (E-Stop only)
    CATEGORY_1 = 1  # Controlled deceleration, then power removal
    CATEGORY_2 = 2  # Graceful stop, power maintained


class EStopSource(str, Enum):
    HARDWARE_BUTTON = "hardware_button"
    SOFTWARE_WATCHDOG = "software_watchdog"
    OPERATOR_CONSOLE = "operator_console"
    GATE_FAILURE_ESCALATION = "gate_failure_escalation"


@dataclass
class EStopMessage:
    """
    Emergency stop event. stop_category is always CATEGORY_0 for this
    message type (schema/v0.6.0 EStopMessage.stop_category: const 0).
    """

    robot_id: str
    triggered_at: float = field(default_factory=time.time)  # Unix epoch seconds
    stop_category: StopCategory = StopCategory.CATEGORY_0
    source: Optional[EStopSource] = None

    def to_dict(self) -> dict:
        d = {
            "robot_id": self.robot_id,
            "triggered_at": self.triggered_at,
            "stop_category": int(self.stop_category),
        }
        if self.source is not None:
            d["source"] = self.source.value
        return d


class AuditEventType(str, Enum):
    ACTUATION_CALLED = "actuation_called"
    ACTUATION_EXECUTED = "actuation_executed"
    ACTUATION_FAILED = "actuation_failed"
    SHADOW_BLOCKED = "shadow_blocked"
    CONSTITUTION_BLOCKED = "constitution_blocked"
    LEASE_GRANTED = "lease_granted"
    LEASE_DENIED = "lease_denied"
    LEASE_EXPIRED = "lease_expired"
    LEASE_RELEASED = "lease_released"
    RATE_LIMITED = "rate_limited"
    ESTOP_TRIGGERED = "estop_triggered"
    BATCH_EXECUTED = "batch_executed"
    BATCH_BLOCKED = "batch_blocked"


@dataclass
class AuditEntry:
    """
    Structured safety audit event.

    Every safety-relevant decision is persisted here for post-incident analysis
    and regulatory compliance (ISO 10218, IEC 62443).
    """

    event_type: AuditEventType
    timestamp: float = field(default_factory=time.time)
    robot_id: str = ""
    actuation_name: str = ""
    call_id: str = ""
    outcome: str = ""  # "passed" | "blocked" | "executed" | "failed"
    violations: List[str] = field(default_factory=list)
    constitution_fp: str = ""  # Constitution fingerprint at time of check
    lease_id: str = ""
    duration_s: float = 0.0
    energy_j: float = 0.0

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "robot_id": self.robot_id,
            "actuation_name": self.actuation_name,
            "call_id": self.call_id,
            "outcome": self.outcome,
            "violations": self.violations,
            "constitution_fp": self.constitution_fp,
            "lease_id": self.lease_id,
            "duration_s": self.duration_s,
            "energy_j": self.energy_j,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  METRICS SNAPSHOT  (observability)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MetricsSnapshot:
    """Point-in-time server telemetry for dashboards and health checks."""

    server_name: str
    uptime_s: float
    calls_total: int
    calls_blocked: int
    calls_executed: int
    shadow_blocks: int
    constitution_blocks: int
    lease_denials: int
    active_leases: int
    actuations_registered: int
    sensors_registered: int
    audit_entries: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "server_name": self.server_name,
            "uptime_s": self.uptime_s,
            "calls_total": self.calls_total,
            "calls_blocked": self.calls_blocked,
            "calls_executed": self.calls_executed,
            "shadow_blocks": self.shadow_blocks,
            "constitution_blocks": self.constitution_blocks,
            "lease_denials": self.lease_denials,
            "active_leases": self.active_leases,
            "actuations_registered": self.actuations_registered,
            "sensors_registered": self.sensors_registered,
            "audit_entries": self.audit_entries,
            "timestamp": self.timestamp,
        }
