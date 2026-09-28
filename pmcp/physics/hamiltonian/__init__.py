"""
Phase 1 — Hamiltonian Physics Core
===================================
hamilcore: the mathematical foundation for energy-conserving neural physics.

Architecture:
    Sensor data → PhaseSpaceEncoder → (q, p) → HNN.hamiltonian(q,p) → H value
                                            ↓
                                    SymplecticIntegrator → trajectory

Conservation law enforced: dH/dt ≈ 0 (energy constant along Hamiltonian flow)
"""

from pmcp.physics.hamiltonian.conservation import ConservationChecker
from pmcp.physics.hamiltonian.encoder import PhaseSpaceEncoder
from pmcp.physics.hamiltonian.hnn import HamiltonianNN
from pmcp.physics.hamiltonian.integrators import StormerVerlet, SymplecticIntegrator

__all__ = [
    "HamiltonianNN",
    "PhaseSpaceEncoder",
    "SymplecticIntegrator",
    "StormerVerlet",
    "ConservationChecker",
]
