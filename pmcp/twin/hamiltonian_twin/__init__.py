"""
Phase 2 — Hamiltonian Digital Twin
==================================
Bidirectional sync between real robot sensors and the HNN-based digital twin.
Detects conservation violations in real-time and triggers safety events.
"""

from pmcp.twin.hamiltonian_twin.sync import TwinSynchronizer
from pmcp.twin.hamiltonian_twin.violation_detector import HamiltonianViolationDetector

__all__ = ["TwinSynchronizer", "HamiltonianViolationDetector"]
