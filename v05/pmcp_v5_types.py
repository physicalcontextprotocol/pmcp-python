"""
P-MCP v0.5 — Types
===================
All data types for the Physical Model Context Protocol v0.5.

Maps to MCP primitives:
  Actuations  ↔  MCP Tools      (physical movement commands)
  Sensors     ↔  MCP Resources  (physical data streams)
  Missions    ↔  MCP Prompts    (multi-step robot workflows)

Physical-only extensions (no MCP equivalent):
  ShadowPreview      — pre-flight 3D simulation
  ConstitutionCheck  — TEE-signed rule evaluation
  LeaseGrant         — temporal zone ownership token
  SafetyEnvelope     — per-call safety bounds
  RobotIdentity      — W3C DID-based hardware identity
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
#  PROTOCOL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

PMCP_VERSION = "0.5"
JSONRPC_VERSION = "2.0"
SPEC_DATE = "2026-05-04"
MCP_VERSION = "2024-11-05"  # Anthropic MCP spec version we align with


# ─────────────────────────────────────────────────────────────────────────────
#  ERROR CODES  (JSON-RPC standard + P-MCP physical extensions)
# ─────────────────────────────────────────────────────────────────────────────


class PMCPErrorCode(int, Enum):
    # JSON-RPC 2.0 standard
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # P-MCP Physical Safety  (-33000 range)
    SHADOW_BLOCKED = -33001  # Shadow validator rejected trajectory
    CONSTITUTION_BLOCKED = -33002  # Safety constitution rule violated
    LEASE_REQUIRED = -33003  # No valid lease for this zone
    LEASE_EXPIRED = -33004  # Lease expired mid-execution
    ESTOP_ACTIVE = -33005  # Emergency stop is active
    FLOOR_GUARD = -33006  # Z target below floor limit
    SPEED_LIMIT = -33007  # Speed exceeds hard limit
    ENERGY_BUDGET = -33008  # Energy budget exhausted
    HUMAN_PROXIMITY = -33009  # Human too close to workspace
    ZK_PROOF_INVALID = -33010  # ZK safety proof failed
    JOINT_LIMIT = -33011  # Joint position/velocity out of range
    TORQUE_LIMIT = -33012  # Torque exceeds hardware limit
    WORKSPACE_VIOLATION = -33013  # Target outside defined workspace
    COLLISION_DETECTED = -33014  # Shadow simulation found collision
    ROBOT_FAULT = -33015  # Hardware fault state


@dataclass
class PMCPError(Exception):
    code: PMCPErrorCode
    message: str
    data: Optional[Any] = None

    def to_dict(self) -> dict:
        d = {"code": int(self.code), "message": self.message}
        if self.data is not None:
            d["data"] = self.data
        return d


# ─────────────────────────────────────────────────────────────────────────────
#  JSON-RPC 2.0 ENVELOPES
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PMCPRequest:
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
    id: str
    result: Optional[Any] = None
    error: Optional[PMCPError] = None
    jsonrpc: str = JSONRPC_VERSION

    def to_dict(self) -> dict:
        d: Dict[str, Any] = {"jsonrpc": self.jsonrpc, "id": self.id}
        if self.error is not None:
            d["error"] = self.error.to_dict()
        else:
            d["result"] = self.result
        return d


@dataclass
class PMCPNotification:
    method: str
    params: Dict[str, Any] = field(default_factory=dict)
    jsonrpc: str = JSONRPC_VERSION

    def to_dict(self) -> dict:
        return {"jsonrpc": self.jsonrpc, "method": self.method, "params": self.params}


# ─────────────────────────────────────────────────────────────────────────────
#  ROBOT IDENTITY  (W3C DID-based hardware identity)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RobotIdentity:
    """W3C DID-based hardware identity for a robot."""

    did: str  # e.g. "did:pmcp:arm:ur5:farm01:abc123"
    robot_class: str  # "arm" | "mobile" | "drone" | "plc" | "custom"
    model: str  # "UR5e", "TurtleBot4", "Spot", etc.
    serial: str  # Hardware serial number
    firmware_ver: str = "unknown"
    cert_hash: str = ""  # TEE safety cert fingerprint
    public_key: str = ""  # Ed25519 public key (base64url)
    location: str = "unknown"  # Physical location / cell ID

    @classmethod
    def new(
        cls, robot_class: str, model: str, serial: str, location: str = "lab-01"
    ) -> "RobotIdentity":
        uid = str(uuid.uuid4())[:8]
        did = f"did:pmcp:{robot_class}:{model.lower()}:{location}:{uid}"
        return cls(did=did, robot_class=robot_class, model=model, serial=serial, location=location)

    def to_dict(self) -> dict:
        return {
            "did": self.did,
            "class": self.robot_class,
            "model": self.model,
            "serial": self.serial,
            "firmware": self.firmware_ver,
            "location": self.location,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  SAFETY ENVELOPE  (per-call physical bounds)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SafetyEnvelope:
    """Physical safety constraints for a single tool call."""

    max_speed_m_s: float = 1.0  # TCP / base speed limit
    max_force_n: float = 100.0  # End-effector force limit
    max_torque_nm: float = 50.0  # Joint torque limit
    max_energy_j: float = 500.0  # Energy budget per call
    min_human_clearance_m: float = 0.5  # ISO 10218 minimum safe distance
    workspace_box: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None
    joint_limits_rad: Optional[List[Tuple[float, float]]] = None
    requires_estop_check: bool = True
    iso_mode: str = "ISO10218"  # "ISO10218" | "IEC62443" | "custom"

    def to_dict(self) -> dict:
        return {
            "max_speed_m_s": self.max_speed_m_s,
            "max_force_n": self.max_force_n,
            "max_torque_nm": self.max_torque_nm,
            "max_energy_j": self.max_energy_j,
            "min_human_clearance_m": self.min_human_clearance_m,
            "iso_mode": self.iso_mode,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  ACTUATION SCHEMA  (MCP Tool equivalent for physical commands)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ActuationParameter:
    name: str
    type: str  # "number" | "integer" | "string" | "boolean" | "array" | "object"
    description: str
    required: bool = True
    default: Any = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    enum: Optional[List[Any]] = None
    unit: str = ""

    def to_json_schema(self) -> dict:
        prop: Dict[str, Any] = {
            "type": self.type,
            "description": f"{self.description}" + (f" [{self.unit}]" if self.unit else ""),
        }
        if self.minimum is not None:
            prop["minimum"] = self.minimum
        if self.maximum is not None:
            prop["maximum"] = self.maximum
        if self.enum is not None:
            prop["enum"] = self.enum
        if self.default is not None:
            prop["default"] = self.default
        return prop


@dataclass
class ActuationSpec:
    """
    Describes a single physical actuation — the P-MCP equivalent of an MCP Tool.

    Wire format is MCP-compatible: exposed via tools/list and tools/call.
    Extra fields in inputSchema annotations carry the physical metadata.
    """

    name: str
    description: str
    parameters: List[ActuationParameter]
    robot_id: str

    # Physical metadata (serialized into MCP tool annotations)
    category: str = "motion"  # motion|manipulation|sensing|navigation|utility
    max_speed_m_s: float = 1.0
    max_force_n: float = 100.0
    max_energy_j: float = 500.0
    est_duration_s: float = 2.0
    requires_lease: bool = True
    shadow_required: bool = True
    iso_class: str = "ISO10218"

    def to_mcp_tool(self) -> dict:
        """Serialize as a standard MCP Tool object (tools/list response)."""
        required = [p.name for p in self.parameters if p.required]
        properties = {p.name: p.to_json_schema() for p in self.parameters}
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
            # P-MCP physical extension — stored in annotations
            "annotations": {
                "robot_id": self.robot_id,
                "category": self.category,
                "max_speed_m_s": self.max_speed_m_s,
                "max_force_n": self.max_force_n,
                "max_energy_j": self.max_energy_j,
                "est_duration_s": self.est_duration_s,
                "requires_lease": self.requires_lease,
                "shadow_required": self.shadow_required,
                "iso_class": self.iso_class,
                "protocol": "pmcp/0.5",
            },
        }


@dataclass
class ActuationResult:
    """Result of executing a physical actuation."""

    success: bool
    robot_id: str = ""
    actuation_name: str = ""
    output: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    duration_s: float = 0.0
    energy_consumed_j: float = 0.0
    final_pose: Optional[Dict[str, Any]] = None
    shadow_delta_m: float = 0.0  # error vs shadow simulation (meters)
    timestamp: float = field(default_factory=time.time)

    def to_mcp_content(self) -> List[dict]:
        """Serialize as MCP tool result content blocks."""
        body: Dict[str, Any] = {
            "success": self.success,
            "robot_id": self.robot_id,
            "actuation": self.actuation_name,
            "output": self.output,
            "metrics": {
                "duration_s": self.duration_s,
                "energy_consumed_j": self.energy_consumed_j,
                "shadow_delta_m": self.shadow_delta_m,
            },
        }
        if not self.success:
            body["error"] = self.error_message
        if self.final_pose:
            body["final_pose"] = self.final_pose
        return [{"type": "text", "text": __import__("json").dumps(body, indent=2)}]


# ─────────────────────────────────────────────────────────────────────────────
#  SENSOR SCHEMA  (MCP Resource equivalent for physical data streams)
# ─────────────────────────────────────────────────────────────────────────────


class SensorType(str, Enum):
    JOINT_STATES = "joint_states"
    END_EFFECTOR = "end_effector"
    FORCE_TORQUE = "force_torque"
    CAMERA_RGB = "camera_rgb"
    CAMERA_DEPTH = "camera_depth"
    LIDAR = "lidar"
    IMU = "imu"
    BATTERY = "battery"
    TEMPERATURE = "temperature"
    PROXIMITY = "proximity"
    GPS = "gps"
    ODOMETRY = "odometry"
    PLANT_HEALTH = "plant_health"
    ENERGY_METER = "energy_meter"
    CUSTOM = "custom"


@dataclass
class SensorSpec:
    """
    Describes a sensor data stream — P-MCP equivalent of an MCP Resource.

    Wire format: exposed via resources/list and resources/read.
    """

    name: str
    description: str
    robot_id: str
    sensor_type: SensorType
    unit: str = ""
    hz: float = 10.0  # update frequency
    is_stream: bool = False  # true → SSE subscription available

    @property
    def uri(self) -> str:
        return f"pmcp://{self.robot_id}/sensors/{self.name}"

    def to_mcp_resource(self) -> dict:
        """Serialize as a standard MCP Resource object."""
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": "application/json",
            "annotations": {
                "robot_id": self.robot_id,
                "sensor_type": self.sensor_type.value,
                "unit": self.unit,
                "hz": self.hz,
                "is_stream": self.is_stream,
                "protocol": "pmcp/0.5",
            },
        }


@dataclass
class SensorReading:
    """A single sensor reading."""

    sensor_name: str
    robot_id: str
    value: Any
    unit: str = ""
    timestamp: float = field(default_factory=time.time)
    quality: float = 1.0  # 0.0-1.0 data quality estimate

    def to_mcp_content(self) -> List[dict]:
        return [
            {
                "type": "text",
                "text": __import__("json").dumps(
                    {
                        "sensor": self.sensor_name,
                        "robot_id": self.robot_id,
                        "value": self.value,
                        "unit": self.unit,
                        "timestamp": self.timestamp,
                        "quality": self.quality,
                    },
                    indent=2,
                ),
            }
        ]


# ─────────────────────────────────────────────────────────────────────────────
#  MISSION SCHEMA  (MCP Prompt equivalent for multi-step robot workflows)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MissionArgument:
    name: str
    description: str
    required: bool = True


@dataclass
class MissionSpec:
    """
    Describes a reusable robot mission template — P-MCP equivalent of MCP Prompt.

    Missions are structured robot programs that an LLM fills in and the
    PMCPServer expands into a sequence of actuation calls.
    """

    name: str
    description: str
    robot_class: str  # arm|mobile|drone|any
    arguments: List[MissionArgument] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)

    def to_mcp_prompt(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "arguments": [
                {"name": a.name, "description": a.description, "required": a.required}
                for a in self.arguments
            ],
        }


@dataclass
class MissionResult:
    """Result of running a mission."""

    mission_name: str
    messages: List[dict]  # MCP-style message list for LLM

    def to_dict(self) -> dict:
        return {"description": self.mission_name, "messages": self.messages}


# ─────────────────────────────────────────────────────────────────────────────
#  SHADOW PREVIEW  (pre-flight 3D simulation result)
# ─────────────────────────────────────────────────────────────────────────────


class ShadowStatus(str, Enum):
    SAFE = "SAFE"
    COLLISION = "COLLISION"
    JOINT_LIMIT = "JOINT_LIMIT"
    WORKSPACE_VIOLATION = "WORKSPACE_VIOLATION"
    SPEED_EXCEEDED = "SPEED_EXCEEDED"
    ENERGY_EXCEEDED = "ENERGY_EXCEEDED"
    SIMULATED = "SIMULATED"  # geometric fallback (no physics engine)
    SKIPPED = "SKIPPED"  # shadow disabled for this actuation


@dataclass
class ShadowPreview:
    """Result of a pre-flight shadow simulation."""

    actuation_name: str
    arguments: Dict[str, Any]
    status: ShadowStatus
    safe: bool
    risk_score: float = 0.0  # 0.0 (safe) → 1.0 (certain collision)
    sim_duration_s: float = 0.0  # time the simulation ran
    est_duration_s: float = 0.0  # predicted execution duration
    est_energy_j: float = 0.0  # predicted energy consumption
    collision_body: str = ""  # colliding object name if any
    warnings: List[str] = field(default_factory=list)
    engine: str = "geometric"  # "pybullet" | "mujoco" | "geometric"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "actuation": self.actuation_name,
            "status": self.status.value,
            "safe": self.safe,
            "risk_score": round(self.risk_score, 3),
            "est_duration_s": self.est_duration_s,
            "est_energy_j": self.est_energy_j,
            "warnings": self.warnings,
            "engine": self.engine,
            "collision_body": self.collision_body or None,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  LEASE SYSTEM  (temporal zone ownership)
# ─────────────────────────────────────────────────────────────────────────────


class LeaseState(str, Enum):
    # NOTE: values match pmcp-spec/schema/v0.6.0/pmcp.schema.json's LeaseState
    # enum exactly (FREE/PENDING/ACTIVE/EXPIRED/DENIED). Earlier v05 code used
    # a non-conformant "GRANTED" value; that was a drift bug, not a deliberate
    # naming choice -- confirmed against pmcp-labs/v04/pmcp_temporal_lease.py,
    # which already used ACTIVE. FREE and PENDING are included for schema
    # conformance even though v05's non-Raft lease manager doesn't yet produce
    # PENDING grants (Raft-backed arbitration lands in a later version).
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
    priority: int = 5  # 1 (highest) … 10 (lowest)


@dataclass
class LeaseGrant:
    lease_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    robot_id: str = ""
    zone_id: str = ""
    state: LeaseState = LeaseState.ACTIVE
    expires_at: float = field(default_factory=lambda: time.time() + 10.0)
    bid_energy_j: float = 0.0
    # Monotonically increasing fencing token (Kleppmann 2016). Must be
    # re-presented on every actuation call against this zone, not only at
    # lease-request time. See _LeaseManager in pmcp_v5_server.py.
    fence_token: int = 0

    @property
    def remaining_ms(self) -> float:
        return max(0.0, (self.expires_at - time.time()) * 1000)

    @property
    def valid(self) -> bool:
        return self.state == LeaseState.ACTIVE and self.expires_at > time.time()

    def to_dict(self) -> dict:
        return {
            "leaseId": self.lease_id,
            "robotId": self.robot_id,
            "zoneId": self.zone_id,
            "state": self.state.value,
            "expiresAt": self.expires_at,
            "remainingMs": round(self.remaining_ms, 1),
            "fenceToken": self.fence_token,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  MCP PROTOCOL MESSAGES  (capability negotiation)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ServerInfo:
    name: str
    version: str = "1.0.0"


@dataclass
class ClientInfo:
    name: str
    version: str = "1.0.0"


@dataclass
class Capabilities:
    """MCP-compatible capabilities negotiation object."""

    # MCP standard capabilities
    tools: bool = True
    resources: bool = True
    prompts: bool = True
    logging: bool = True
    sampling: bool = False  # server-initiated LLM sampling

    # P-MCP physical extensions
    shadow: bool = True  # shadow/preview method available
    leases: bool = True  # lease/request + lease/release available
    estop: bool = True  # pmcp/estop available
    constitution: bool = True  # TEE safety constitution loaded
    streaming: bool = False  # SSE sensor streaming

    def to_mcp_dict(self) -> dict:
        """MCP-compatible capabilities dict for initialize response."""
        caps: Dict[str, Any] = {}
        if self.tools:
            caps["tools"] = {"listChanged": True}
        if self.resources:
            caps["resources"] = {"subscribe": self.streaming, "listChanged": True}
        if self.prompts:
            caps["prompts"] = {"listChanged": False}
        if self.logging:
            caps["logging"] = {}
        if self.sampling:
            caps["sampling"] = {}
        # P-MCP extension namespace
        caps["experimental"] = {
            "pmcp": {
                "version": PMCP_VERSION,
                "shadow": self.shadow,
                "leases": self.leases,
                "estop": self.estop,
                "constitution": self.constitution,
            }
        }
        return caps
