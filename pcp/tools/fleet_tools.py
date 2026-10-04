"""
Phase 3 — Fleet Coordination Tools
====================================
P-MCP actuations for system Hamiltonian + fleet-wide planning.
H_sys = Σ H_i + Σ V_ij — coordinates all robots as one physical system.
"""

from __future__ import annotations

from pcp.fleet.system_hamiltonian.system_h import SystemHamiltonian
from pcp.server import PCPServer
from pcp.tools.hamiltonian_tools import hnn_registry
from pcp.tools.twin_tools import get_fleet_state
from pcp.types import ActuationResult, SensorReading, SensorType

_system_h: SystemHamiltonian | None = None


def _get_system_h() -> SystemHamiltonian:
    global _system_h
    if _system_h is None:
        _system_h = SystemHamiltonian(hnn_per_robot=hnn_registry)
    return _system_h


def register_fleet_tools(server: PCPServer):

    @server.actuation(
        "compute_system_hamiltonian",
        description="Compute fleet-wide system Hamiltonian H_sys = ΣH_i + ΣV_ij",
        max_speed_m_s=None,
        max_energy_j=None,
    )
    async def compute_system_hamiltonian() -> ActuationResult:
        fleet_state = get_fleet_state()
        system_h = _get_system_h()
        H_total = system_h.compute_system_h(fleet_state)

        return ActuationResult(
            success=True,
            metadata={
                "H_sys": float(H_total),
                "num_robots": len(fleet_state),
                "robot_ids": list(fleet_state.keys()),
            },
        )

    @server.actuation(
        "allocate_energy_budget",
        description="Distribute energy budget across fleet proportional to individual H",
        max_speed_m_s=None,
        max_energy_j=None,
    )
    async def allocate_energy_budget(total_joules: float) -> ActuationResult:
        system_h = _get_system_h()
        fleet_state = get_fleet_state()
        budget = system_h.get_energy_budget(total_joules, fleet_state)

        return ActuationResult(
            success=True,
            metadata={
                "budget": {rid: float(b) for rid, b in budget.items()},
                "total_joules": total_joules,
            },
        )

    @server.actuation(
        "compute_fleet_gradient",
        description="Compute dH_sys/dq_i for each robot (HJB optimal control gradient)",
        max_speed_m_s=None,
        max_energy_j=None,
    )
    async def compute_fleet_gradient() -> ActuationResult:
        system_h = _get_system_h()
        fleet_state = get_fleet_state()
        gradients = system_h.compute_gradient(fleet_state)

        return ActuationResult(
            success=True,
            metadata={
                "gradients": {rid: g.tolist() for rid, g in gradients.items()},
            },
        )

    @server.sensor(
        "system_hamiltonian",
        description="Current system Hamiltonian energy for the entire fleet",
        sensor_type=SensorType.CUSTOM,
        unit="J",
        sample_rate_hz=1.0,
    )
    async def system_hamiltonian_sensor() -> SensorReading:
        system_h = _get_system_h()
        H_total = system_h.compute_system_h(get_fleet_state())
        return SensorReading(
            sensor_name="system_hamiltonian",
            value=float(H_total),
            unit="J",
            quality=1.0,
        )

    @server.sensor(
        "fleet_energy_distribution",
        description="Per-robot energy contribution to system Hamiltonian",
        sensor_type=SensorType.CUSTOM,
        unit="J",
        sample_rate_hz=1.0,
    )
    async def fleet_energy_sensor() -> SensorReading:
        system_h = _get_system_h()
        fleet_state = get_fleet_state()
        energy_per_robot = {}
        for rid in fleet_state:
            if rid in system_h.hnn_per_robot:
                energy_per_robot[rid] = float(
                    system_h.hnn_per_robot[rid].hamiltonian(
                        fleet_state[rid]["q"], fleet_state[rid]["p"]
                    )
                )
        return SensorReading(
            sensor_name="fleet_energy_distribution",
            value=sum(energy_per_robot.values()),
            unit="J",
            quality=1.0,
            metadata={"per_robot": energy_per_robot},
        )

    return server
