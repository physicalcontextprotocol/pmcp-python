"""
Hamiltonian Violation Detector
==============================
Continuous monitoring for energy conservation violations.
When drift exceeds threshold → triggers HAMILTONIAN_VIOLATION event.

Plugs into your existing P-MCP safety FSM via trigger_event().
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Optional

import numpy as np


class HamiltonianViolationDetector:
    """
    Real-time violation detector for Hamiltonian conservation.

    Runs continuously alongside the safety FSM.
    When conservation breaks → raises HAMILTONIAN_VIOLATION event.
    """

    VIOLATION_THRESHOLD = 0.05

    def __init__(
        self,
        threshold: float = VIOLATION_THRESHOLD,
        check_interval_sec: float = 0.1,
    ):
        self.threshold = threshold
        self.check_interval = check_interval_sec
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def run_continuous(
        self,
        robot_id: str,
        sensor_fn: Callable[[], Awaitable[dict]],
        hnn_model,
        encoder,
        safety_fsm,
    ):
        """
        Start continuous violation monitoring.

        Args:
            robot_id: Robot identifier
            sensor_fn: Async function returning sensor data dict
            hnn_model: HamiltonianNN instance
            encoder: PhaseSpaceEncoder instance
            safety_fsm: SafetyMiddleware with trigger_event method
        """
        self._running = True

        while self._running:
            try:
                sensor_data = await sensor_fn()
                q, p = encoder.encode(sensor_data)

                traj_q, traj_p = hnn_model.integrate_symplectic(q, p, steps=50, dt=0.01)

                energies = np.array(
                    [hnn_model.hamiltonian(traj_q[i], traj_p[i]) for i in range(len(traj_q))]
                )
                H0 = energies[0]
                H_final = energies[-1]

                drift = abs(H_final - H0) / abs(H0) if abs(H0) > 1e-8 else 0.0

                if drift > self.threshold:
                    severity = self._severity_level(drift)
                    await safety_fsm.trigger_event(
                        robot_id,
                        event="HAMILTONIAN_VIOLATION",
                        metadata={
                            "drift": float(drift),
                            "threshold": self.threshold,
                            "severity": severity,
                            "H0": float(H0),
                            "H_final": float(H_final),
                        },
                    )

            except Exception:
                pass

            await asyncio.sleep(self.check_interval)

    def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()

    def _severity_level(self, drift: float) -> str:
        ratio = drift / self.threshold
        if ratio < 2:
            return "minor"
        elif ratio < 5:
            return "moderate"
        elif ratio < 10:
            return "major"
        else:
            return "critical"

    async def check_once(
        self,
        sensor_data: dict,
        hnn_model,
        encoder,
    ) -> dict:
        """
        Single violation check (non-continuous).

        Returns:
            Dict with drift, is_violation, severity
        """
        q, p = encoder.encode(sensor_data)
        traj_q, traj_p = hnn_model.integrate_symplectic(q, p, steps=50, dt=0.01)

        energies = np.array(
            [hnn_model.hamiltonian(traj_q[i], traj_p[i]) for i in range(len(traj_q))]
        )
        H0 = energies[0]
        H_final = energies[-1]

        drift = abs(H_final - H0) / abs(H0) if abs(H0) > 1e-8 else 0.0
        is_violation = drift > self.threshold
        severity = self._severity_level(drift) if is_violation else "none"

        return {
            "drift": float(drift),
            "is_violation": is_violation,
            "severity": severity,
            "H0": float(H0),
            "H_final": float(H_final),
            "threshold": self.threshold,
        }
