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
  ShadowPreview      — pre-flight 3D simulation result
  ConstitutionCheck  — TEE-signed rule evaluation
  LeaseRequest/Grant — temporal zone ownership token
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


# ─────────────────────────────────────────────────────────────────────────────
#  PROTOCOL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

PMCP_VERSION       = "0.4"
JSONRPC_VERSION    = "2.0"
SPEC_DATE          = "2025-05-04"


# ─────────────────────────────────────────────────────────────────────────────
#  ERROR CODES (JSON-RPC standard + P-MCP extensions)
# ─────────────────────────────────────────────────────────────────────────────

class PMCPErrorCode(int, Enum):
    # JSON-RPC 2.0 standard
    PARSE_ERROR      = -32700
    INVALID_REQUEST  = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS   = -32602
    INTERNAL_ERROR   = -32603

    # P-MCP Physical Safety (-33000 range)
    SHADOW_BLOCKED       = -33001   # Shadow validator rejected the trajectory
    CONSTITUTION_BLOCKED = -33002   # Safety constitution rule violated
    LEASE_REQUIRED       = -33003   # No valid lease for this zone
    LEASE_EXPIRED        = -33004   # Lease expired mid-execution
    ESTOP_ACTIVE         = -33005   # Emergency stop is active
    FLOOR_GUARD          = -33006   # Z target below floor limit
    SPEED_LIMIT          = -33007   # Speed exceeds hard limit
    ENERGY_BUDGET        = -33008   # Energy budget exhausted
    HUMAN_PROXIMITY      = -33009   # Human too close to workspace
    ZK_PROOF_INVALID     = -33010   # ZK safety proof failed verification


@dataclass
class PMCPError:
    code:    PMCPErrorCode
    message: str
    data:    Optional[Any] = None

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
    method:  str
    params:  Dict[str, Any]          = field(default_factory=dict)
    id:      str                     = field(default_factory=lambda: str(uuid.uuid4())[:8])
    jsonrpc: str                     = JSONRPC_VERSION

    def to_dict(self) -> dict:
        return {"jsonrpc": self.jsonrpc, "id": self.id,
                "method": self.method, "params": self.params}

    @classmethod
    def from_dict(cls, d: dict) -> "PMCPRequest":
        return cls(method=d["method"], params=d.get("params", {}),
                   id=d.get("id", ""), jsonrpc=d.get("jsonrpc", JSONRPC_VERSION))


@dataclass
class PMCPResponse:
    """JSON-RPC 2.0 response message."""
    id:      str
    result:  Optional[Any]       = None
    error:   Optional[PMCPError] = None
    jsonrpc: str                 = JSONRPC_VERSION

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
    method:  str
    params:  Dict[str, Any] = field(default_factory=dict)
    jsonrpc: str             = JSONRPC_VERSION

    def to_dict(self) -> dict:
        return {"jsonrpc": self.jsonrpc, "method": self.method, "params": self.params}


# ─────────────────────────────────────────────────────────────────────────────
#  LIFECYCLE TYPES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ServerInfo:
    name:             str
    version:          str
    protocol_version: str = PMCP_VERSION

    def to_dict(self) -> dict:
        return {"name": self.name, "version": self.version,
                "protocolVersion": self.protocol_version}


@dataclass
class ClientInfo:
    name:    str
    version: str

    def to_dict(self) -> dict:
        return {"name": self.name, "version": self.version}


@dataclass
class Capabilities:
    """Declared during initialize handshake."""
    actuations:   bool = True    # Server can execute physical actuations
    sensors:      bool = True    # Server exposes sensor streams
    prompts:      bool = False   # Server provides prompt templates
    shadow:       bool = True    # Server supports shadow preview
    constitution: bool = True    # Server enforces safety constitution
    leases:       bool = True    # Server uses temporal zone leases
    streaming:    bool = False   # Server supports sensor streaming
    zk_proofs:    bool = False   # Server generates ZK safety proofs

    def to_dict(self) -> dict:
        caps: dict = {}
        if self.actuations:   caps["actuations"]   = {"listChanged": True}
        if self.sensors:      caps["sensors"]      = {"streaming": self.streaming}
        if self.prompts:      caps["prompts"]      = {}
        if self.shadow:       caps["shadow"]       = {}
        if self.constitution: caps["constitution"] = {}
        if self.leases:       caps["leases"]       = {}
        if self.zk_proofs:    caps["zkProofs"]     = {}
        return caps


# ─────────────────────────────────────────────────────────────────────────────
#  PHYSICAL PRIMITIVES
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. Actuations (≈ MCP Tools) ──────────────────────────────────────────────

@dataclass
class ActuationParameter:
    """JSON-Schema parameter descriptor for an actuation."""
    name:        str
    type:        str            # "number" | "integer" | "string" | "boolean"
    description: str  = ""
    required:    bool = True
    default:     Any  = None
    minimum:     Optional[float] = None
    maximum:     Optional[float] = None
    unit:        str  = ""      # "m/s", "m", "rad", "N", "°C" etc.

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
    name:               str
    description:        str
    parameters:         List[ActuationParameter] = field(default_factory=list)
    max_speed_m_s:      Optional[float] = None
    max_force_n:        Optional[float] = None
    max_energy_j:       Optional[float] = None
    requires_lease:     bool = True
    shadow_required:    bool = True
    robot_class:        str  = ""    # "arm" | "mobile" | "plc" | "any"
    safety_category:    str  = "motion"

    def to_dict(self) -> dict:
        props = {p.name: p.to_schema_property() for p in self.parameters}
        required_params = [p.name for p in self.parameters if p.required]
        return {
            "name":            self.name,
            "description":     self.description,
            "inputSchema":     {"type": "object", "properties": props,
                                "required": required_params},
            "physical": {
                "maxSpeed":       self.max_speed_m_s,
                "maxForce":       self.max_force_n,
                "maxEnergy":      self.max_energy_j,
                "requiresLease":  self.requires_lease,
                "shadowRequired": self.shadow_required,
                "robotClass":     self.robot_class,
                "safetyCategory": self.safety_category,
            },
        }


@dataclass
class ActuationResult:
    """Result of executing a physical actuation."""
    success:        bool
    actuation_id:   str  = field(default_factory=lambda: str(uuid.uuid4())[:12])
    robot_id:       str  = ""
    final_pose:     Optional[Dict[str, float]] = None  # {x, y, z, roll, pitch, yaw}
    duration_s:     float = 0.0
    energy_j:       float = 0.0
    error:          Optional[str] = None
    metadata:       Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "success":      self.success,
            "actuationId":  self.actuation_id,
            "robotId":      self.robot_id,
            "finalPose":    self.final_pose,
            "durationS":    self.duration_s,
            "energyJ":      self.energy_j,
            "error":        self.error,
            "metadata":     self.metadata,
        }


# ── 2. Sensors (≈ MCP Resources) ─────────────────────────────────────────────

class SensorType(str, Enum):
    POSITION     = "position"      # Encoder / GPS / odometry
    FORCE        = "force"         # Force/torque sensor
    TEMPERATURE  = "temperature"   # Thermal
    PROXIMITY    = "proximity"     # Sonar / IR / LiDAR
    VISION       = "vision"        # Camera / depth
    IMU          = "imu"           # Accelerometer / gyro
    CURRENT      = "current"       # Motor current
    VOLTAGE      = "voltage"       # Battery voltage
    PRESSURE     = "pressure"      # Pneumatic / hydraulic
    CUSTOM       = "custom"


@dataclass
class SensorSpec:
    """
    Describes a physical sensor (P-MCP primitive #2).

    Analogous to MCP Resource, but exposes real-time physical data.
    URI format:  pmcp://sensor/{robot_id}/{sensor_name}
    """
    name:          str
    description:   str
    sensor_type:   SensorType
    unit:          str            # "m", "N", "°C", "m/s²", etc.
    min_value:     Optional[float] = None
    max_value:     Optional[float] = None
    sample_rate_hz: float = 10.0
    streaming:     bool   = False

    @property
    def uri(self) -> str:
        return f"pmcp://sensor/{self.name}"

    def to_dict(self) -> dict:
        return {
            "name":         self.name,
            "uri":          self.uri,
            "description":  self.description,
            "mimeType":     "application/pmcp-sensor",
            "physical": {
                "sensorType":    self.sensor_type.value,
                "unit":          self.unit,
                "minValue":      self.min_value,
                "maxValue":      self.max_value,
                "sampleRateHz":  self.sample_rate_hz,
                "streaming":     self.streaming,
            },
        }


@dataclass
class SensorReading:
    """A single sensor measurement."""
    sensor_name:  str   = ""
    value:        Any   = None
    unit:         str   = ""
    timestamp:    float = field(default_factory=time.time)
    quality:      float = 1.0       # 0=bad, 1=good
    raw:          Optional[Any] = None

    def to_dict(self) -> dict:
        return {
            "sensorName": self.sensor_name,
            "value":      self.value,
            "unit":       self.unit,
            "timestamp":  self.timestamp,
            "quality":    self.quality,
        }


# ── 3. Prompts (= MCP Prompts) ────────────────────────────────────────────────

@dataclass
class PromptSpec:
    """A reusable safety/interaction template (same as MCP Prompt)."""
    name:        str
    description: str
    arguments:   List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description,
                "arguments": self.arguments}


@dataclass
class PromptResult:
    role:    str   # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": {"type": "text", "text": self.content}}


# ─────────────────────────────────────────────────────────────────────────────
#  PHYSICAL SAFETY LAYER  (P-MCP extension beyond MCP)
# ─────────────────────────────────────────────────────────────────────────────

# ── Shadow Preview ────────────────────────────────────────────────────────────

class ShadowStatus(str, Enum):
    SAFE          = "safe"
    UNSAFE        = "unsafe"
    COLLISION     = "collision"
    FLOOR_GUARD   = "floor_guard"
    SPEED_LIMIT   = "speed_limit"
    HUMAN_NEARBY  = "human_nearby"
    PENDING       = "pending"


@dataclass
class ShadowPreview:
    """
    Pre-flight 3D simulation result (P-MCP primitive #4).

    Before any actuation executes on hardware, the Shadow layer
    runs a ghost simulation and returns this object.
    status == SAFE is required for actuation to proceed.
    """
    actuation_name:  str
    robot_id:        str
    status:          ShadowStatus
    safe:            bool
    predicted_pose:  Optional[Dict[str, float]] = None
    duration_s:      float = 0.0
    energy_j:        float = 0.0
    collisions:      List[str] = field(default_factory=list)
    violations:      List[str] = field(default_factory=list)
    shadow_token:    str  = field(default_factory=lambda: str(uuid.uuid4())[:16])
    timestamp:       float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "actuationName": self.actuation_name,
            "robotId":       self.robot_id,
            "status":        self.status.value,
            "safe":          self.safe,
            "predictedPose": self.predicted_pose,
            "durationS":     self.duration_s,
            "energyJ":       self.energy_j,
            "collisions":    self.collisions,
            "violations":    self.violations,
            "shadowToken":   self.shadow_token,
            "timestamp":     self.timestamp,
        }


# ── Constitution Check ────────────────────────────────────────────────────────

@dataclass
class ConstitutionCheck:
    """TEE-signed safety constitution evaluation result."""
    cleared:     bool
    violations:  List[str]     = field(default_factory=list)
    rule_ids:    List[str]     = field(default_factory=list)
    fingerprint: str           = ""     # Constitution bundle fingerprint
    tee_sig:     Optional[str] = None   # Ed25519 signature

    def to_dict(self) -> dict:
        return {
            "cleared":     self.cleared,
            "violations":  self.violations,
            "ruleIds":     self.rule_ids,
            "fingerprint": self.fingerprint,
        }


# ── Temporal Lease ────────────────────────────────────────────────────────────

@dataclass
class LeaseRequest:
    robot_id:    str
    zone_id:     str
    duration_ms: int   = 10_000
    bid_energy_j: float = 100.0

    def to_dict(self) -> dict:
        return {"robotId": self.robot_id, "zoneId": self.zone_id,
                "durationMs": self.duration_ms, "bidEnergyJ": self.bid_energy_j}


@dataclass
class LeaseGrant:
    """Temporal zone lease token."""
    lease_id:    str
    robot_id:    str
    zone_id:     str
    granted:     bool
    expires_at:  float          # Unix timestamp
    token:       str = ""       # Signed lease token
    deny_reason: str = ""

    def is_valid(self) -> bool:
        return self.granted and time.time() < self.expires_at

    def to_dict(self) -> dict:
        return {
            "leaseId":    self.lease_id,
            "robotId":    self.robot_id,
            "zoneId":     self.zone_id,
            "granted":    self.granted,
            "expiresAt":  self.expires_at,
            "valid":      self.is_valid(),
            "denyReason": self.deny_reason,
        }
