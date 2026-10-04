"""
Phase 2 — Digital Twin Tools
=============================
PCP actuations for the Hamiltonian digital twin layer.
Bidirectional sync: real sensors → twin state, twin prediction → real execution.
Violation detection → safety FSM events.
"""

from __future__ import annotations

import time
from typing import Dict

from pcp.server import PCPServer
from pcp.tools.hamiltonian_tools import (
    _get_encoder,
    _get_hnn,
    _get_sensor_data,
)
from pcp.twin.hamiltonian_twin.sync import TwinSynchronizer
from pcp.twin.hamiltonian_twin.violation_detector import HamiltonianViolationDetector
from pcp.types import ActuationResult, SensorReading, SensorType

_twin_registry: Dict[str, TwinSynchronizer] = {}
_violation_detector = HamiltonianViolationDetector(threshold=0.05)
_fleet_state_store: Dict[str, Dict] = {}


def _get_twin(robot_id: str) -> TwinSynchronizer:
    if robot_id not in _twin_registry:
        _twin_registry[robot_id] = TwinSynchronizer(
            robot_id=robot_id,
            hnn_model=_get_hnn(robot_id),
            encoder=_get_encoder(robot_id),
        )
    return _twin_registry[robot_id]


def update_fleet_state(robot_id: str, q, p):
    import numpy as np

    _fleet_state_store[robot_id] = {
        "q": np.asarray(q),
        "p": np.asarray(p),
        "ts": time.time(),
    }


def get_fleet_state() -> Dict[str, Dict]:
    return _fleet_state_store


def register_twin_tools(server: PCPServer):

    @server.actuation(
        "sync_twin",
        description="Sync Hamiltonian digital twin with real sensor data",
        max_speed_m_s=None,
        max_energy_j=None,
    )
    async def sync_twin(robot_id: str) -> ActuationResult:
        twin = _get_twin(robot_id)
        sensor_data = await _get_sensor_data(robot_id)
        state = await twin.sync_from_sensors(sensor_data)

        update_fleet_state(robot_id, state.q_real, state.p_real)

        return ActuationResult(
            success=True,
            metadata={
                **state.to_dict(),
                "synchronized": twin.is_synchronized,
            },
        )

    @server.actuation(
        "check_twin_violation",
        description="Check if twin deviation exceeds safety threshold — triggers ISO 10218 event",
        max_speed_m_s=None,
        max_energy_j=None,
    )
    async def check_twin_violation(robot_id: str) -> ActuationResult:
        twin = _get_twin(robot_id)
        if twin.last_state is None:
            return ActuationResult(success=True, metadata={"status": "no_data"})

        deviation = twin.last_state.deviation
        is_safe = twin.is_synchronized

        metadata = {
            "deviation": deviation,
            "safe": is_safe,
            "threshold": twin.deviation_threshold,
        }
        if not is_safe:
            metadata["safety_event"] = "TWIN_DEVIATION_VIOLATION"

        return ActuationResult(success=True, metadata=metadata)

    @server.actuation(
        "recalibrate_twin",
        description="Recalibrate twin to current real sensor state (reset deviation to zero)",
        max_speed_m_s=None,
        max_energy_j=None,
    )
    async def recalibrate_twin(robot_id: str) -> ActuationResult:
        twin = _get_twin(robot_id)
        sensor_data = await _get_sensor_data(robot_id)
        await twin.recalibrate(sensor_data)

        return ActuationResult(success=True, metadata={"robot_id": robot_id, "recalibrated": True})

    @server.sensor(
        "twin_deviation",
        description="Real-time deviation between physical robot and Hamiltonian twin",
        sensor_type=SensorType.CUSTOM,
        unit="m",
        sample_rate_hz=10.0,
    )
    async def twin_deviation_sensor(robot_id: str = "default") -> SensorReading:
        twin = _get_twin(robot_id)
        deviation = twin.last_state.deviation if twin.last_state else 0.0
        return SensorReading(
            sensor_name="twin_deviation",
            value=deviation,
            unit="m",
            quality=1.0 if deviation < twin.deviation_threshold else 0.3,
        )

    @server.sensor(
        "twin_energy",
        description="Hamiltonian energy of the digital twin",
        sensor_type=SensorType.CUSTOM,
        unit="J",
        sample_rate_hz=10.0,
    )
    async def twin_energy_sensor(robot_id: str = "default") -> SensorReading:
        twin = _get_twin(robot_id)
        if twin.last_state:
            H = twin.last_state.H_twin
            H_real = twin.last_state.H_real
        else:
            H = 0.0
            H_real = 0.0
        return SensorReading(
            sensor_name="twin_energy",
            value=H,
            unit="J",
            quality=1.0,
            metadata={"H_real": H_real, "H_twin": H},
        )

    return server
