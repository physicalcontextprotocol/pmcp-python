"""
Anomaly Detector
================
Detects physics anomalies from energy conservation violations.
When HNN predicts drift without physical cause → anomaly detected.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class AnomalyEvent:
    timestamp: float
    robot_id: str
    anomaly_type: str
    drift: float
    expected: float
    actual: float
    severity: str

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "robot_id": self.robot_id,
            "anomaly_type": self.anomaly_type,
            "drift": self.drift,
            "expected": self.expected,
            "actual": self.actual,
            "severity": self.severity,
        }


class AnomalyDetector:
    """
    Detects when robot physics deviates from the learned Hamiltonian model.

    Anomaly types:
        - UNEXPECTED_FORCE: external force detected (collision, payload change)
        - MODEL_DRIFT: HNN predictions becoming less accurate
        - PHYSICS_CHANGE: system dynamics have changed (broken joint, new crop)
        - SENSOR_NOISE: excessive sensor noise
    """

    ANOMALY_THRESHOLDS = {
        "UNEXPECTED_FORCE": 0.20,
        "MODEL_DRIFT": 0.10,
        "PHYSICS_CHANGE": 0.30,
        "SENSOR_NOISE": 0.05,
    }

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self._drift_history: Dict[str, List[float]] = {}

    def detect(
        self, robot_id: str, drift: float, H_predicted: float, H_measured: float
    ) -> Optional[AnomalyEvent]:
        if robot_id not in self._drift_history:
            self._drift_history[robot_id] = []

        self._drift_history[robot_id].append(drift)
        if len(self._drift_history[robot_id]) > self.window_size:
            self._drift_history[robot_id] = self._drift_history[robot_id][-self.window_size :]

        if drift > self.ANOMALY_THRESHOLDS["PHYSICS_CHANGE"]:
            return AnomalyEvent(
                timestamp=time.time(),
                robot_id=robot_id,
                anomaly_type="PHYSICS_CHANGE",
                drift=drift,
                expected=H_predicted,
                actual=H_measured,
                severity="critical",
            )
        elif drift > self.ANOMALY_THRESHOLDS["MODEL_DRIFT"]:
            return AnomalyEvent(
                timestamp=time.time(),
                robot_id=robot_id,
                anomaly_type="MODEL_DRIFT",
                drift=drift,
                expected=H_predicted,
                actual=H_measured,
                severity="warning",
            )
        elif drift > self.ANOMALY_THRESHOLDS["SENSOR_NOISE"]:
            return AnomalyEvent(
                timestamp=time.time(),
                robot_id=robot_id,
                anomaly_type="SENSOR_NOISE",
                drift=drift,
                expected=H_predicted,
                actual=H_measured,
                severity="minor",
            )

        return None

    def get_trend(self, robot_id: str) -> str:
        if robot_id not in self._drift_history or len(self._drift_history[robot_id]) < 5:
            return "unknown"

        recent = np.array(self._drift_history[robot_id][-10:])
        if len(recent) < 2:
            return "unknown"

        slope = np.polyfit(range(len(recent)), recent, 1)[0]

        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
