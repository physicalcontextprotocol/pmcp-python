"""
Twin Synchronizer
=================
Bidirectional sync between real robot sensor data and the Hamiltonian digital twin.

Data flow:
    Real sensors → PhaseSpaceEncoder → (q, p) → HNN → H_predicted
    H_predicted vs H_measured → drift → violation detection
    Twin state → CRDT sync → fleet-wide state
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class TwinState:
    robot_id: str
    q_real: np.ndarray
    p_real: np.ndarray
    q_twin: np.ndarray
    p_twin: np.ndarray
    H_real: float
    H_twin: float
    deviation: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "robot_id": self.robot_id,
            "q_real": self.q_real.tolist(),
            "p_real": self.p_real.tolist(),
            "q_twin": self.q_twin.tolist(),
            "p_twin": self.p_twin.tolist(),
            "H_real": self.H_real,
            "H_twin": self.H_twin,
            "deviation": self.deviation,
            "timestamp": self.timestamp,
        }


class TwinSynchronizer:
    """
    Keeps the Hamiltonian digital twin synchronized with real sensor data.

    Key operations:
        - Push real sensors → twin (model-based prediction)
        - Pull twin state → real world (goal trajectory)
        - Detect deviations (twin drift detection)
        - Recalibrate when drift exceeds threshold
    """

    def __init__(
        self,
        robot_id: str,
        hnn_model,
        encoder,
        deviation_threshold: float = 0.05,
        sync_interval_ms: float = 100.0,
    ):
        self.robot_id = robot_id
        self.hnn = hnn_model
        self.encoder = encoder
        self.deviation_threshold = deviation_threshold
        self.sync_interval = sync_interval_ms / 1000.0

        self._q_twin = None
        self._p_twin = None
        self._history: List[TwinState] = []

    async def sync_from_sensors(self, sensor_data: Dict) -> TwinState:
        q_real, p_real = self.encoder.encode(sensor_data)
        H_real = self.hnn.hamiltonian(q_real, p_real)

        if self._q_twin is None:
            self._q_twin = q_real.copy()
            self._p_twin = p_real.copy()

        self._q_twin, self._p_twin = self.hnn.integrate_symplectic(
            self._q_twin, self._p_twin, steps=1, dt=self.sync_interval
        )

        H_twin = self.hnn.hamiltonian(self._q_twin, self._p_twin)

        deviation = float(np.linalg.norm(q_real - self._q_twin))

        state = TwinState(
            robot_id=self.robot_id,
            q_real=q_real,
            p_real=p_real,
            q_twin=self._q_twin,
            p_twin=self._p_twin,
            H_real=float(H_real),
            H_twin=float(H_twin),
            deviation=deviation,
        )

        self._history.append(state)
        return state

    async def recalibrate(self, sensor_data: Dict):
        q_real, p_real = self.encoder.encode(sensor_data)
        self._q_twin = q_real.copy()
        self._p_twin = p_real.copy()

    def get_history(self, last_n: Optional[int] = None) -> List[TwinState]:
        if last_n is None:
            return self._history.copy()
        return self._history[-last_n:]

    @property
    def is_synchronized(self) -> bool:
        if not self._history:
            return True
        return self._history[-1].deviation < self.deviation_threshold

    @property
    def last_state(self) -> Optional[TwinState]:
        return self._history[-1] if self._history else None
