"""
System Hamiltonian
==================
H_sys = Σ H_i + Σ V_ij for the multi-robot fleet.

V_ij = inter-robot interaction potential (repulsion when close, coupling when collaborating)
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


class SystemHamiltonian:
    """
    Fleet-wide Hamiltonian: H_sys = Σ H_i + Σ V_ij

    Robots are coupled subsystems of one physical system.
    Individual HNNs from Phase 1 are reused; fleet-level interactions are modeled here.
    """

    def __init__(
        self,
        hnn_per_robot: Dict[str, Any],
        safety_radius_m: float = 0.8,
        repulsion_strength: float = 0.1,
    ):
        self.hnn_per_robot = hnn_per_robot
        self.safety_radius = safety_radius_m
        self.repulsion_strength = repulsion_strength

    def compute_system_h(self, fleet_state: Dict[str, Dict]) -> float:
        """
        Compute H_sys for all robots in the fleet.

        Args:
            fleet_state: Dict mapping robot_id → {"q": np.ndarray, "p": np.ndarray}

        Returns:
            System Hamiltonian energy in Joules
        """
        H_individual = sum(
            self.hnn_per_robot[rid].hamiltonian(s["q"], s["p"])
            for rid, s in fleet_state.items()
            if rid in self.hnn_per_robot
        )

        H_interaction = self._compute_interaction_potential(fleet_state)

        return H_individual + H_interaction

    def _compute_interaction_potential(self, fleet_state: Dict[str, Dict]) -> float:
        V = 0.0
        robot_ids = list(fleet_state.keys())

        for i in range(len(robot_ids)):
            for j in range(i + 1, len(robot_ids)):
                rid_i = robot_ids[i]
                rid_j = robot_ids[j]

                if rid_i not in fleet_state or rid_j not in fleet_state:
                    continue

                q_i = fleet_state[rid_i]["q"]
                q_j = fleet_state[rid_j]["q"]

                pos_i = q_i[:3] if len(q_i) >= 3 else q_i
                pos_j = q_j[:3] if len(q_j) >= 3 else q_j

                distance = float(np.linalg.norm(np.array(pos_i) - np.array(pos_j)))

                if distance < self.safety_radius:
                    V += self.repulsion_strength / (distance**2 + 1e-6)

        return V

    def compute_gradient(self, fleet_state: Dict[str, Dict]) -> Dict[str, np.ndarray]:
        """
        Compute dH_sys/dq_i for each robot (for optimal control).
        """
        gradients = {}

        for rid, state in fleet_state.items():
            q = state["q"]
            p = state["p"]

            grad_q = np.zeros_like(q)

            if rid in self.hnn_per_robot:
                import torch

                q_t = torch.as_tensor(q, dtype=torch.float32, requires_grad=True)
                p_t = torch.as_tensor(p, dtype=torch.float32, requires_grad=True)

                H = self.hnn_per_robot[rid].forward(q_t, p_t)
                dH_dq = torch.autograd.grad(H.sum(), q_t)[0]
                grad_q = dH_dq.detach().numpy()

            gradients[rid] = grad_q

        return gradients

    def get_energy_budget(
        self, total_joules: float, fleet_state: Dict[str, Dict]
    ) -> Dict[str, float]:
        """
        Allocate energy budget across robots proportionally to their current H.
        """
        H_values = {
            rid: self.hnn_per_robot[rid].hamiltonian(s["q"], s["p"])
            for rid, s in fleet_state.items()
            if rid in self.hnn_per_robot
        }

        H_total = sum(H_values.values())
        if H_total == 0:
            budget_per_robot = total_joules / len(H_values) if H_values else 0
            return {rid: budget_per_robot for rid in H_values}

        return {rid: total_joules * (H_val / H_total) for rid, H_val in H_values.items()}
