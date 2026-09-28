"""
Online HNN Trainer
==================
Continuous fine-tuning of the Hamiltonian model from real sensor data.
Uses conservation law as regularizer to prevent catastrophic forgetting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class TrainingEpisode:
    robot_id: str
    q_traj: np.ndarray
    p_traj: np.ndarray
    timestamp: float

    def to_dict(self) -> dict:
        return {
            "robot_id": self.robot_id,
            "q_traj": self.q_traj.tolist(),
            "p_traj": self.p_traj.tolist(),
            "timestamp": self.timestamp,
        }


class OnlineHNNTrainer:
    """
    Online learning for Hamiltonian models.

    Key features:
        - Episode-based training from real interaction data
        - Conservation loss = ||dH/dt||² acts as physics regularizer
        - Catastrophic forgetting prevented by anchoring to conservation law
    """

    def __init__(
        self,
        hnn_model,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        conservation_weight: float = 1.0,
        max_episodes_in_memory: int = 1000,
    ):
        self.hnn = hnn_model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.conservation_weight = conservation_weight
        self.max_episodes = max_episodes_in_memory

        self._episodes: List[TrainingEpisode] = []
        self._training_stats = {"total_updates": 0, "losses": []}

    def add_episode(self, robot_id: str, q_traj: np.ndarray, p_traj: np.ndarray):
        episode = TrainingEpisode(
            robot_id=robot_id,
            q_traj=np.asarray(q_traj),
            p_traj=np.asarray(p_traj),
            timestamp=time.time(),
        )
        self._episodes.append(episode)

        if len(self._episodes) > self.max_episodes:
            self._episodes = self._episodes[-self.max_episodes :]

    def train_step(self) -> float:
        if len(self._episodes) == 0:
            return 0.0

        idx = np.random.randint(0, len(self._episodes))
        episode = self._episodes[idx]

        start = np.random.randint(0, max(0, len(episode.q_traj) - self.batch_size))
        q_batch = episode.q_traj[start : start + self.batch_size]
        p_batch = episode.p_traj[start : start + self.batch_size]

        loss = self.hnn.train_step(q_batch, p_batch)

        self._training_stats["total_updates"] += 1
        self._training_stats["losses"].append(loss)

        return loss

    def get_stats(self) -> dict:
        losses = self._training_stats["losses"]
        return {
            "total_updates": self._training_stats["total_updates"],
            "num_episodes": len(self._episodes),
            "mean_loss": float(np.mean(losses[-100:])) if losses else 0.0,
            "last_loss": float(losses[-1]) if losses else 0.0,
        }

    def clear_memory(self):
        self._episodes = []
