"""
Phase 4 — Learning Tools
=========================
PCP actuations for online HNN training, anomaly detection, skill library.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from pcp.learning.anomaly_detector import AnomalyDetector
from pcp.learning.online_hnn import OnlineHNNTrainer
from pcp.server import PCPServer
from pcp.tools.hamiltonian_tools import _get_hnn
from pcp.types import ActuationResult, SensorReading, SensorType

_trainer_registry: Dict[str, OnlineHNNTrainer] = {}
_detector = AnomalyDetector(window_size=50)


def _get_trainer(robot_id: str) -> OnlineHNNTrainer:
    if robot_id not in _trainer_registry:
        _trainer_registry[robot_id] = OnlineHNNTrainer(
            hnn_model=_get_hnn(robot_id),
            batch_size=32,
            learning_rate=1e-3,
        )
    return _trainer_registry[robot_id]


def register_learning_tools(server: PCPServer):

    @server.actuation(
        "train_hnn_online",
        description="Run one step of online HNN training on recorded trajectory data",
        max_speed_m_s=None,
        max_energy_j=None,
    )
    async def train_hnn_online(
        robot_id: str, q_traj: List[List[float]], p_traj: List[List[float]]
    ) -> ActuationResult:
        trainer = _get_trainer(robot_id)
        trainer.add_episode(robot_id, np.array(q_traj), np.array(p_traj))
        loss = trainer.train_step()
        stats = trainer.get_stats()

        return ActuationResult(
            success=True,
            metadata={
                "loss": loss,
                **stats,
            },
        )

    @server.actuation(
        "detect_anomaly",
        description="Detect physics anomalies from energy conservation violations",
        max_speed_m_s=None,
        max_energy_j=None,
    )
    async def detect_anomaly(
        robot_id: str, drift: float, H_predicted: float, H_measured: float
    ) -> ActuationResult:
        event = _detector.detect(robot_id, drift, H_predicted, H_measured)
        trend = _detector.get_trend(robot_id)

        metadata = {
            "anomaly_detected": event is not None,
            "trend": trend,
        }
        if event:
            metadata.update(event.to_dict())

        return ActuationResult(success=True, metadata=metadata)

    @server.actuation(
        "get_training_stats",
        description="Get online training statistics for a robot's HNN",
        max_speed_m_s=None,
        max_energy_j=None,
    )
    async def get_training_stats(robot_id: str) -> ActuationResult:
        trainer = _get_trainer(robot_id)
        stats = trainer.get_stats()

        return ActuationResult(success=True, metadata=stats)

    @server.actuation(
        "clear_training_memory",
        description="Clear the online training episode memory for a robot",
        max_speed_m_s=None,
        max_energy_j=None,
    )
    async def clear_training_memory(robot_id: str) -> ActuationResult:
        trainer = _get_trainer(robot_id)
        trainer.clear_memory()

        return ActuationResult(
            success=True, metadata={"robot_id": robot_id, "memory_cleared": True}
        )

    @server.sensor(
        "anomaly_trend",
        description="Anomaly detection trend for a robot (stable/increasing/decreasing)",
        sensor_type=SensorType.CUSTOM,
        unit="",
        sample_rate_hz=1.0,
    )
    async def anomaly_trend_sensor(robot_id: str = "default") -> SensorReading:
        trend = _detector.get_trend(robot_id)
        return SensorReading(
            sensor_name="anomaly_trend",
            value=trend,
            unit="",
            quality=1.0,
        )

    return server
