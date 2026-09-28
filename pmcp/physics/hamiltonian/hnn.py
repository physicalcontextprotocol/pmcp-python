"""
Hamiltonian Neural Network (HNN)
================================
Neural network that learns the Hamiltonian H(q,p) from trajectory data.
Instead of learning next_state = f(state), we learn the energy surface H(q,p)
and derive dynamics via symplectic gradient: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q.

Training loss: L = ||dH/dt||² (Hamiltonian should be constant along trajectories)

Key properties:
    1. Conservation guaranteed — HNN respects energy conservation by construction
    2. Physical impossibility → mathematically impossible (cannot predict H increasing)
    3. Transferable — same architecture works for any robotic system
    4. Interpretable — H value is a physical quantity (energy in Joules)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class HamiltonianNN(nn.Module):
    """
    Learns H(q, p) from trajectory data.
    Uses a simple MLP that outputs a scalar H given concatenated (q, p).
    """

    def __init__(
        self,
        q_dim: int = 6,
        p_dim: int = 6,
        hidden_dims: list = None,
        activation: str = "tanh",
    ):
        super().__init__()
        self.q_dim = q_dim
        self.p_dim = p_dim
        self.input_dim = q_dim + p_dim

        if hidden_dims is None:
            hidden_dims = [64, 64, 32]

        layers = []
        prev_dim = self.input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "silu":
                layers.append(nn.SiLU())
            else:
                layers.append(nn.Tanh())
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

        self._h_nn = self.net
        self._optimizer: Optional[torch.optim.Adam] = None
        self._device = torch.device("cpu")

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        x = torch.cat([q, p], dim=-1)
        return self._h_nn(x).squeeze(-1)

    def hamiltonian(self, q: np.ndarray, p: np.ndarray) -> float:
        q_t = torch.as_tensor(q, dtype=torch.float32, device=self._device)
        p_t = torch.as_tensor(p, dtype=torch.float32, device=self._device)
        self.eval()
        with torch.no_grad():
            H = self.forward(q_t, p_t).item()
        return H

    def hamiltonians(self, q: np.ndarray, p: np.ndarray) -> np.ndarray:
        q_t = torch.as_tensor(q, dtype=torch.float32, device=self._device)
        p_t = torch.as_tensor(p, dtype=torch.float32, device=self._device)
        self.eval()
        with torch.no_grad():
            H = self.forward(q_t, p_t).numpy()
        return H

    def compute_dynamics(
        self, q: torch.Tensor, p: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q.requires_grad_(True)
        p.requires_grad_(True)
        H = self.forward(q, p)
        dH_dq = torch.autograd.grad(H.sum(), q, create_graph=True)[0]
        dH_dp = torch.autograd.grad(H.sum(), p, create_graph=True)[0]
        dq_dt = dH_dp
        dp_dt = -dH_dq
        return dq_dt, dp_dt

    def integrate_symplectic(
        self,
        q0: np.ndarray,
        p0: np.ndarray,
        steps: int = 100,
        dt: float = 0.01,
    ) -> Tuple[np.ndarray, np.ndarray]:
        from pmcp.physics.hamiltonian.integrators import StormerVerlet

        integrator = StormerVerlet(self)
        traj_q, traj_p = integrator.integrate(q0, p0, steps=steps, dt=dt)
        return traj_q, traj_p

    def setup_optimizer(self, lr: float = 1e-3, weight_decay: float = 1e-5):
        self._optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self._device = next(self.parameters()).device

    def to(self, device: torch.device):
        super().to(device)
        self._device = device
        return self

    def train_step(self, q: np.ndarray, p: np.ndarray, dt: float = 0.01) -> float:
        if self._optimizer is None:
            self.setup_optimizer()

        self.train()
        q_t = torch.as_tensor(q, dtype=torch.float32, device=self._device)
        p_t = torch.as_tensor(p, dtype=torch.float32, device=self._device)

        dH_dt = self._compute_energy_derivative(q_t, p_t, dt)
        loss = (dH_dt**2).mean()

        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self._optimizer.step()

        return loss.item()

    def _compute_energy_derivative(
        self, q: torch.Tensor, p: torch.Tensor, dt: float
    ) -> torch.Tensor:
        q = q.clone().detach().requires_grad_(True)
        p = p.clone().detach().requires_grad_(True)

        H0 = self.forward(q, p)
        dq_dt, dp_dt = self.compute_dynamics(q, p)

        q_next = q + dt * dq_dt
        p_next = p + dt * dp_dt
        H1 = self.forward(q_next, p_next)

        dH_dt = (H1 - H0) / dt
        return dH_dt

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "HamiltonianNN":
        model = cls(**kwargs)
        # weights_only=True: torch.load's default pickle-based deserialization
        # can execute arbitrary code from a malicious checkpoint file. Since
        # we only ever load a plain state_dict here (tensors, not arbitrary
        # objects), weights_only=True is both safe and sufficient.
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        state = torch.load(path, map_location=self._device, weights_only=True)
        self.load_state_dict(state)

    def get_energy(self, q: np.ndarray, p: np.ndarray) -> float:
        return self.hamiltonian(q, p)
