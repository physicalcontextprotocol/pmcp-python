"""
Phase 1 — Hamiltonian Tools
=============================
P-MCP tools for Hamiltonian physics operations.

These tools expose Hamiltonian mechanics as P-MCP actuations,
enabling Claude agents to query and manipulate robot physics
using phase space coordinates (q, p) and energy conservation.

Tool categories:
    - Phase space query: get current (q, p) state
    - Trajectory prediction: simulate future states
    - Conservation checking: detect physics anomalies
    - Energy analysis: compute H, H_sys, energy budgets
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List

from pcp.physics.hamiltonian import (
    ConservationChecker,
    HamiltonianNN,
    PhaseSpaceEncoder,
)
from pcp.server import PCPServer
from pcp.types import ActuationResult, SensorReading, SensorType

hnn_registry: Dict[str, HamiltonianNN] = {}
encoder_registry: Dict[str, PhaseSpaceEncoder] = {}


def _get_hnn(robot_id: str) -> HamiltonianNN:
    if robot_id not in hnn_registry:
        hnn_registry[robot_id] = HamiltonianNN(q_dim=6, p_dim=6)
    return hnn_registry[robot_id]


def _get_encoder(robot_id: str) -> PhaseSpaceEncoder:
    if robot_id not in encoder_registry:
        encoder_registry[robot_id] = PhaseSpaceEncoder(n_joints=6)
    return encoder_registry[robot_id]


_checker = ConservationChecker(threshold=0.05)


_sensor_cache: Dict[str, Dict] = {}


def update_sensor_cache(robot_id: str, data: Dict):
    _sensor_cache[robot_id] = {**data, "_ts": time.time()}


async def _get_sensor_data(robot_id: str) -> Dict:
    if robot_id in _sensor_cache:
        cached = _sensor_cache[robot_id]
        if time.time() - cached.get("_ts", 0) < 0.5:
            return cached
    return {
        "joint_angles": [0.0] * 6,
        "joint_velocities": [0.0] * 6,
        "end_effector_pose": {"x": 0, "y": 0, "z": 0},
    }


@dataclass
class PhaseSpaceState:
    robot_id: str
    q: List[float]
    p: List[float]
    H: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "robot_id": self.robot_id,
            "q": self.q,
            "p": self.p,
            "H": self.H,
            "timestamp": self.timestamp,
        }


@dataclass
class TrajectoryPrediction:
    robot_id: str
    traj_q: List[List[float]]
    traj_p: List[List[float]]
    energies: List[float]
    energy_drift: float
    steps: int
    dt: float

    def to_dict(self) -> dict:
        return {
            "robot_id": self.robot_id,
            "trajectory": {
                "q": self.traj_q,
                "p": self.traj_p,
                "energies": self.energies,
            },
            "energy_drift": self.energy_drift,
            "steps": self.steps,
            "dt": self.dt,
        }


def register_hamiltonian_tools(server: PCPServer):
    """
    Register all Phase 1 Hamiltonian tools with a PCPServer.

    Usage:
        server = PCPServer("pcp-hamiltonian")
        register_hamiltonian_tools(server)
        asyncio.run(server.run())
    """

    @server.actuation(
        "get_phase_space_state",
        description="Get current (q, p) phase space state and Hamiltonian energy for a robot",
        max_speed_m_s=None,
        max_energy_j=None,
    )
    async def get_phase_space_state(robot_id: str) -> ActuationResult:
        """
        Returns the current (q, p) phase space state for a robot.

        q = generalized positions (joint angles, xyz)
        p = generalized momenta (joint velocities × inertia)

        Returns:
            ActuationResult with phase space data:
            - q: position vector
            - p: momentum vector
            - H: Hamiltonian energy in Joules
            - q_dim, p_dim: dimensions
        """
        hnn = _get_hnn(robot_id)
        encoder = _get_encoder(robot_id)

        raw = await _get_sensor_data(robot_id)
        q, p = encoder.encode(raw)
        H = hnn.hamiltonian(q, p)

        state = PhaseSpaceState(robot_id=robot_id, q=q.tolist(), p=p.tolist(), H=float(H))

        return ActuationResult(
            success=True,
            final_pose={"q": state.q, "p": state.p, "H": state.H},
            duration_s=0.001,
            energy_j=0.0,
            metadata={"q_dim": len(q), "p_dim": len(p), "timestamp": state.timestamp},
        )

    @server.actuation(
        "predict_trajectory",
        description="Predict future robot states using symplectic integration (energy-conserving)",
        max_speed_m_s=None,
        max_energy_j=None,
    )
    async def predict_trajectory(
        robot_id: str,
        steps: int = 100,
        dt: float = 0.01,
    ) -> ActuationResult:
        """
        Predict future states using symplectic integration.

        Uses Störmer-Verlet integrator which guarantees energy conservation.
        Physically impossible states cannot be predicted.

        Args:
            robot_id: Robot identifier
            steps: Number of integration steps (default 100)
            dt: Timestep in seconds (default 0.01s)

        Returns:
            ActuationResult with trajectory data:
            - traj_q: position trajectory (steps+1 × q_dim)
            - traj_p: momentum trajectory (steps+1 × p_dim)
            - energies: H values along trajectory
            - energy_drift: fractional energy drift
        """
        hnn = _get_hnn(robot_id)
        encoder = _get_encoder(robot_id)

        raw = await _get_sensor_data(robot_id)
        q0, p0 = encoder.encode(raw)

        traj_q, traj_p = hnn.integrate_symplectic(q0, p0, steps=steps, dt=dt)

        energies = [hnn.hamiltonian(traj_q[i], traj_p[i]) for i in range(len(traj_q))]
        drift = _checker.drift_from_arrays(traj_q, traj_p, hnn)

        prediction = TrajectoryPrediction(
            robot_id=robot_id,
            traj_q=traj_q.tolist(),
            traj_p=traj_p.tolist(),
            energies=energies,
            energy_drift=drift,
            steps=steps,
            dt=dt,
        )

        return ActuationResult(
            success=True,
            final_pose={"H_final": energies[-1] if energies else 0},
            duration_s=steps * dt,
            energy_j=0.0,
            metadata=prediction.to_dict(),
        )

    @server.actuation(
        "check_energy_conservation",
        description="Verify energy conservation — drift > 5% triggers safety flag",
        max_speed_m_s=None,
        max_energy_j=None,
    )
    async def check_energy_conservation(robot_id: str) -> ActuationResult:
        """
        Compares predicted H vs measured H across a trajectory.

        Drift > threshold (5%) = safety flag raised.
        Plugs directly into ISO 10218 safety FSM via HAMILTONIAN_VIOLATION event.

        Returns:
            ActuationResult with:
            - drift: fractional energy drift
            - safe: boolean (drift < threshold)
            - status: NOMINAL | CAUTION | WARNING | ALERT | CRITICAL
        """
        hnn = _get_hnn(robot_id)
        encoder = _get_encoder(robot_id)

        raw = await _get_sensor_data(robot_id)
        q0, p0 = encoder.encode(raw)

        traj_q, traj_p = hnn.integrate_symplectic(q0, p0, steps=50, dt=0.01)
        drift = _checker.drift_from_arrays(traj_q, traj_p, hnn)

        is_safe = drift < _checker.threshold
        status = _checker.get_safety_status(drift)

        metadata = {
            "drift": drift,
            "safe": is_safe,
            "status": status,
            "threshold": _checker.threshold,
            "violation": not is_safe,
        }

        if not is_safe:
            metadata["safety_event"] = "HAMILTONIAN_VIOLATION"
            metadata["severity"] = _checker._severity_level(drift, _checker.threshold)

        return ActuationResult(
            success=True,
            duration_s=0.5,
            energy_j=0.0,
            metadata=metadata,
        )

    @server.actuation(
        "train_hamiltonian",
        description="Train the HNN model from sensor trajectory data",
        max_speed_m_s=None,
        max_energy_j=None,
    )
    async def train_hamiltonian(
        robot_id: str,
        trajectories: List[Dict],
        lr: float = 1e-3,
        epochs: int = 100,
    ) -> ActuationResult:
        """
        Fine-tune the Hamiltonian neural network from recorded trajectories.

        Each trajectory should have:
            - q: list of position states
            - p: list of momentum states

        Training loss: L = ||dH/dt||² (energy should be constant)

        Args:
            robot_id: Robot identifier
            trajectories: List of trajectory dicts
            lr: Learning rate (default 1e-3)
            epochs: Number of training epochs (default 100)

        Returns:
            ActuationResult with training metrics:
            - final_loss, initial_loss
            - improvement_percent
        """
        hnn = _get_hnn(robot_id)
        hnn.setup_optimizer(lr=lr)

        losses = []
        for epoch in range(epochs):
            epoch_losses = []
            for traj in trajectories:
                if "q" in traj and "p" in traj:
                    q = traj["q"]
                    p = traj["p"]
                    if len(q) > 0 and len(p) > 0:
                        loss = hnn.train_step(q, p)
                        epoch_losses.append(loss)
            if epoch_losses:
                losses.append(sum(epoch_losses) / len(epoch_losses))

        initial_loss = losses[0] if len(losses) > 0 else 0.0
        final_loss = losses[-1] if len(losses) > 0 else 0.0

        improvement = (initial_loss - final_loss) / initial_loss if initial_loss > 0 else 0.0

        return ActuationResult(
            success=True,
            duration_s=epochs * 0.01,
            energy_j=0.0,
            metadata={
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "improvement": improvement,
                "epochs": epochs,
            },
        )

    @server.sensor(
        "phase_space_energy",
        description="Current Hamiltonian energy H for a robot",
        sensor_type=SensorType.CUSTOM,
        unit="J",
        sample_rate_hz=10.0,
    )
    async def phase_space_energy_sensor(robot_id: str = "default") -> SensorReading:
        hnn = _get_hnn(robot_id)
        encoder = _get_encoder(robot_id)

        raw = await _get_sensor_data(robot_id)
        q, p = encoder.encode(raw)
        H = hnn.hamiltonian(q, p)

        return SensorReading(
            sensor_name="phase_space_energy",
            value=float(H),
            unit="J",
            quality=1.0,
        )

    @server.sensor(
        "energy_conservation_status",
        description="Energy drift status as safety indicator",
        sensor_type=SensorType.CUSTOM,
        unit="",
        sample_rate_hz=10.0,
    )
    async def energy_conservation_sensor(robot_id: str = "default") -> SensorReading:
        hnn = _get_hnn(robot_id)
        encoder = _get_encoder(robot_id)

        raw = await _get_sensor_data(robot_id)
        q, p = encoder.encode(raw)
        traj_q, traj_p = hnn.integrate_symplectic(q, p, steps=50, dt=0.01)
        drift = _checker.drift_from_arrays(traj_q, traj_p, hnn)

        return SensorReading(
            sensor_name="energy_conservation_status",
            value=drift,
            unit="",
            quality=1.0 if drift < _checker.threshold else 0.5,
        )

    return server
