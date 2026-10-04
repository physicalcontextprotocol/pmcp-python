"""
PhysOS Protocol v1
==================
Universal protocol for Hamiltonian-based physical agents.

Message types:
    - STATE_REPORT: (q, p, H) state report
    - TRAJECTORY_REQUEST: request symplectic trajectory
    - ENERGY_QUERY: query system energy
    - VIOLATION_ALERT: safety violation notification
    - SKILL_TRANSFER: HNN model exchange

All physical state is represented in phase space coordinates (q, p)
with Hamiltonian energy H as the canonical invariant.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MessageType(str, Enum):
    STATE_REPORT = "physos.state_report"
    TRAJECTORY_REQUEST = "physos.trajectory_request"
    TRAJECTORY_RESPONSE = "physos.trajectory_response"
    ENERGY_QUERY = "physos.energy_query"
    ENERGY_RESPONSE = "physos.energy_response"
    VIOLATION_ALERT = "physos.violation_alert"
    SKILL_TRANSFER = "physos.skill_transfer"
    INITIALIZE = "physos.initialize"
    HEARTBEAT = "physos.heartbeat"


@dataclass
class PhysOSMessage:
    """
    PhysOS message envelope.

    All physical state is phase space coordinates:
        q = generalized positions
        p = generalized momenta
        H = Hamiltonian energy (invariant)
    """

    msg_type: MessageType
    sender_id: str
    robot_id: str
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])

    q: Optional[List[float]] = None
    p: Optional[List[float]] = None
    H: Optional[float] = None

    trajectory_q: Optional[List[List[float]]] = None
    trajectory_p: Optional[List[List[float]]] = None

    energy_drift: Optional[float] = None
    safety_status: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def to_dict(self) -> dict:
        d = {
            "msg_type": self.msg_type.value,
            "sender_id": self.sender_id,
            "robot_id": self.robot_id,
            "timestamp": self.timestamp,
            "message_id": self.message_id,
        }
        if self.q is not None:
            d["q"] = self.q
        if self.p is not None:
            d["p"] = self.p
        if self.H is not None:
            d["H"] = self.H
        if self.trajectory_q is not None:
            d["trajectory_q"] = self.trajectory_q
        if self.trajectory_p is not None:
            d["trajectory_p"] = self.trajectory_p
        if self.energy_drift is not None:
            d["energy_drift"] = self.energy_drift
        if self.safety_status is not None:
            d["safety_status"] = self.safety_status
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PhysOSMessage":
        return cls(
            msg_type=MessageType(d["msg_type"]),
            sender_id=d["sender_id"],
            robot_id=d["robot_id"],
            timestamp=d.get("timestamp", time.time()),
            message_id=d.get("message_id", str(uuid.uuid4())[:12]),
            q=d.get("q"),
            p=d.get("p"),
            H=d.get("H"),
            trajectory_q=d.get("trajectory_q"),
            trajectory_p=d.get("trajectory_p"),
            energy_drift=d.get("energy_drift"),
            safety_status=d.get("safety_status"),
            metadata=d.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "PhysOSMessage":
        return cls.from_dict(json.loads(json_str))


class PhysOSProtocol:
    """
    Protocol handler for PhysOS messages.

    Usage:
        protocol = PhysOSProtocol("my-robot")
        msg = protocol.create_state_report(q=q, p=p, H=H)
        serialized = msg.to_json()
        # send over MQTT/ROS2/CAN
        received = PhysOSMessage.from_json(recv_str)
        response = protocol.handle_message(received)
    """

    def __init__(self, sender_id: str):
        self.sender_id = sender_id

    def create_state_report(
        self, robot_id: str, q: List[float], p: List[float], H: float
    ) -> PhysOSMessage:
        return PhysOSMessage(
            msg_type=MessageType.STATE_REPORT,
            sender_id=self.sender_id,
            robot_id=robot_id,
            q=q,
            p=p,
            H=H,
        )

    def create_violation_alert(
        self,
        robot_id: str,
        drift: float,
        safety_status: str,
        q: Optional[List[float]] = None,
        p: Optional[List[float]] = None,
    ) -> PhysOSMessage:
        return PhysOSMessage(
            msg_type=MessageType.VIOLATION_ALERT,
            sender_id=self.sender_id,
            robot_id=robot_id,
            q=q,
            p=p,
            energy_drift=drift,
            safety_status=safety_status,
        )

    def handle_message(self, msg: PhysOSMessage) -> Optional[PhysOSMessage]:
        if msg.msg_type == MessageType.ENERGY_QUERY:
            return PhysOSMessage(
                msg_type=MessageType.ENERGY_RESPONSE,
                sender_id=self.sender_id,
                robot_id=msg.robot_id,
                H=msg.H,
                metadata={"query_id": msg.message_id},
            )
        return None
