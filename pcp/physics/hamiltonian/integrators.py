"""
Symplectic Integrators
=======================
Energy-conserving numerical integration for Hamiltonian systems.

Key insight: symplectic methods preserve the symplectic structure of
Hamiltonian flows, which guarantees approximate energy conservation
even over long integration horizons.

Available methods:
    - StormerVerlet: 2nd order Störmer-Verlet (leapfrog)
    - Ruth3: 3rd order symplectic method (Ruth)
    - Yoshida4: 4th order method (Yoshida)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class SymplecticIntegrator(ABC):
    """Base class for symplectic numerical integrators."""

    def __init__(self, hnn_model):
        self.hnn = hnn_model

    @abstractmethod
    def step(self, q: np.ndarray, p: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Single integration step. Override in subclasses."""
        pass

    def integrate(
        self,
        q0: np.ndarray,
        p0: np.ndarray,
        steps: int,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate Hamiltonian trajectory for `steps` timesteps.

        Returns:
            (traj_q, traj_p) arrays of shape (steps+1, dim)
        """
        q = np.array(q0, dtype=np.float32)
        p = np.array(p0, dtype=np.float32)

        traj_q = np.zeros((steps + 1, q.shape[0]), dtype=np.float32)
        traj_p = np.zeros((steps + 1, p.shape[0]), dtype=np.float32)
        traj_q[0] = q
        traj_p[0] = p

        for i in range(steps):
            q, p = self.step(q, p, dt)
            traj_q[i + 1] = q
            traj_p[i + 1] = p

        return traj_q, traj_p


class StormerVerlet(SymplecticIntegrator):
    """
    Störmer-Verlet (leapfrog) integrator — 2nd order symplectic.

    Equations:
        p_{n+1/2} = p_n + (dt/2) * dp/dt(q_n, p_n)
        q_{n+1}   = q_n + dt * dq/dt(q_n, p_{n+1/2})
        p_{n+1}   = p_{n+1/2} + (dt/2) * dp/dt(q_{n+1}, p_{n+1/2})

    Properties:
        - Symplectic (preserves phase space volume)
        - Time-reversible
        - 2nd order accurate
        - No explicit force evaluation needed (uses H gradient)
    """

    def step(self, q: np.ndarray, p: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        import torch

        q_t = torch.tensor(q, dtype=torch.float32, requires_grad=True)
        p_t = torch.tensor(p, dtype=torch.float32, requires_grad=True)

        H = self.hnn.forward(q_t, p_t)

        dH_dq = torch.autograd.grad(H.sum(), q_t, create_graph=True)[0]
        dH_dp = torch.autograd.grad(H.sum(), p_t)[0]

        p_mid = p + (-dH_dq.detach().numpy()) * (dt / 2)
        q_new = q + dH_dp.detach().numpy() * dt

        q_new_t = torch.tensor(q_new, dtype=torch.float32, requires_grad=True)
        p_new_t = torch.tensor(p_mid, dtype=torch.float32, requires_grad=True)

        H_new = self.hnn.forward(q_new_t, p_new_t)
        dH_dq_new = torch.autograd.grad(H_new.sum(), p_new_t)[0]
        p_new = p_mid + (-dH_dq_new.detach().numpy()) * (dt / 2)

        return q_new, p_new


class Ruth3(SymplecticIntegrator):
    """
    3rd order Ruth symplectic method.

    Coefficients from Ruth (1990).
    More accurate than Verlet but requires more function evaluations.
    """

    c = [
        1 / (2 - 2 ** (1 / 3)),
        (1 - 2 ** (1 / 3)) / 2 / (2 - 2 ** (1 / 3)),
        1 / (2 - 2 ** (1 / 3)),
    ]
    d = [1, -(2 ** (1 / 3)) / (2 - 2 ** (1 / 3)), 1 / (2 - 2 ** (1 / 3))]

    def step(self, q: np.ndarray, p: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        import torch

        q_t = torch.tensor(q, dtype=torch.float32, requires_grad=True)
        p_t = torch.tensor(p, dtype=torch.float32, requires_grad=True)

        for i in range(3):
            H = self.hnn.forward(q_t, p_t)
            dH_dq = torch.autograd.grad(H.sum(), q_t)[0]
            dH_dp = torch.autograd.grad(H.sum(), p_t)[0]

            p_t = p_t - self.d[i] * dt * dH_dq
            q_t = q_t + self.c[i] * dt * dH_dp

        return q_t.detach().numpy(), p_t.detach().numpy()


class Yoshida4(SymplecticIntegrator):
    """
    4th order Yoshida symplectic method.

    Coefficients from Yoshida (1990).
    Highest accuracy among standard symplectic methods.
    """

    w = [0, -(2 ** (1 / 3)) / (2 - 2 ** (1 / 3)), 2 ** (1 / 3) / (2 - 2 ** (1 / 3)), -1]
    c = [1 / (2 - 2 ** (1 / 3)), (1 - 2 ** (1 / 3)) / (2 - 2 ** (1 / 3)), 1 / (2 - 2 ** (1 / 3)), 0]
    d = [
        0,
        -(2 ** (1 / 3)) / (2 - 2 ** (1 / 3)),
        2 ** (1 / 3) / (2 - 2 ** (1 / 3)),
        -1 / (2 - 2 ** (1 / 3)),
    ]

    def step(self, q: np.ndarray, p: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        import torch

        q_t = torch.tensor(q, dtype=torch.float32, requires_grad=True)
        p_t = torch.tensor(p, dtype=torch.float32, requires_grad=True)

        for i in range(4):
            H = self.hnn.forward(q_t, p_t)
            dH_dq = torch.autograd.grad(H.sum(), q_t)[0]
            dH_dp = torch.autograd.grad(H.sum(), p_t)[0]

            if self.d[i] != 0:
                p_t = p_t - self.d[i] * dt * dH_dq
            if self.c[i] != 0:
                q_t = q_t + self.c[i] * dt * dH_dp

        return q_t.detach().numpy(), p_t.detach().numpy()


def create_integrator(
    hnn_model,
    method: str = "stormer_verlet",
) -> SymplecticIntegrator:
    """Factory function to create an integrator by name."""
    integrators = {
        "stormer_verlet": StormerVerlet,
        "verlet": StormerVerlet,
        "leapfrog": StormerVerlet,
        "ruth3": Ruth3,
        "ruth_3": Ruth3,
        "yoshida4": Yoshida4,
        "yoshida_4": Yoshida4,
    }

    integrator_cls = integrators.get(method.lower())
    if integrator_cls is None:
        raise ValueError(f"Unknown integrator: {method}. Available: {list(integrators.keys())}")

    return integrator_cls(hnn_model)
