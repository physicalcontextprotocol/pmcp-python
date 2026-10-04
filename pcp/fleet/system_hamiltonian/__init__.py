"""
Phase 3 — System Hamiltonian
===========================
Fleet-wide energy computation: H_sys = Σ H_i + Σ V_ij
Coordinates via HJB optimal control.
"""

from pcp.fleet.system_hamiltonian.system_h import SystemHamiltonian

__all__ = ["SystemHamiltonian"]
