"""
Phase 5 — Protocol Tools
=========================
P-MCP actuations for PhysOS protocol communication.
Encode/decode robot state as PhysOS messages, handle protocol handshake.
"""

from __future__ import annotations

from typing import Dict

from pcp.protocol.physos_v1 import PhysOSMessage, PhysOSProtocol
from pcp.server import PCPServer
from pcp.tools.hamiltonian_tools import _get_encoder, _get_hnn, _get_sensor_data
from pcp.types import ActuationResult, SensorReading, SensorType

_protocol_registry: Dict[str, PhysOSProtocol] = {}


def _get_protocol(robot_id: str) -> PhysOSProtocol:
    if robot_id not in _protocol_registry:
        _protocol_registry[robot_id] = PhysOSProtocol(sender_id=robot_id)
    return _protocol_registry[robot_id]


def register_protocol_tools(server: PCPServer):

    @server.actuation(
        "create_state_report",
        description="Create a PhysOS STATE_REPORT message from current (q,p,H) state",
        max_speed_m_s=None,
        max_energy_j=None,
    )
    async def create_state_report(robot_id: str) -> ActuationResult:
        protocol = _get_protocol(robot_id)
        sensor_data = await _get_sensor_data(robot_id)
        q, p = _get_encoder(robot_id).encode(sensor_data)
        H = _get_hnn(robot_id).hamiltonian(q, p)

        msg = protocol.create_state_report(robot_id, q.tolist(), p.tolist(), float(H))
        msg_dict = msg.to_dict()

        return ActuationResult(success=True, metadata={"message": msg_dict})

    @server.actuation(
        "parse_physos_message",
        description="Parse and handle an incoming PhysOS protocol message",
        max_speed_m_s=None,
        max_energy_j=None,
    )
    async def parse_physos_message(message_json: str) -> ActuationResult:
        try:
            msg = PhysOSMessage.from_json(message_json)
            protocol = _get_protocol(msg.robot_id)
            response = protocol.handle_message(msg)

            return ActuationResult(
                success=True,
                metadata={
                    "parsed": msg.to_dict(),
                    "response": response.to_dict() if response else None,
                },
            )
        except Exception as e:
            return ActuationResult(success=False, metadata={"error": str(e)})

    @server.actuation(
        "create_violation_alert",
        description="Create a PhysOS VIOLATION_ALERT message from conservation violation",
        max_speed_m_s=None,
        max_energy_j=None,
    )
    async def create_violation_alert(
        robot_id: str, drift: float, safety_status: str
    ) -> ActuationResult:
        protocol = _get_protocol(robot_id)
        sensor_data = await _get_sensor_data(robot_id)
        q, p = _get_encoder(robot_id).encode(sensor_data)

        msg = protocol.create_violation_alert(
            robot_id, drift, safety_status, q.tolist(), p.tolist()
        )

        return ActuationResult(success=True, metadata={"message": msg.to_dict()})

    @server.sensor(
        "physos_state",
        description="Current PhysOS state message as JSON",
        sensor_type=SensorType.CUSTOM,
        unit="",
        sample_rate_hz=1.0,
    )
    async def physos_state_sensor(robot_id: str = "default") -> SensorReading:
        protocol = _get_protocol(robot_id)
        sensor_data = await _get_sensor_data(robot_id)
        q, p = _get_encoder(robot_id).encode(sensor_data)
        H = _get_hnn(robot_id).hamiltonian(q, p)

        msg = protocol.create_state_report(robot_id, q.tolist(), p.tolist(), float(H))
        return SensorReading(
            sensor_name="physos_state",
            value=msg.message_id,
            unit="",
            quality=1.0,
            metadata=msg.to_dict(),
        )

    return server
