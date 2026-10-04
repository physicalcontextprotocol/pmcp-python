"""
Conservation Law Checker
=========================
Detects Hamiltonian conservation violations — when predicted energy drifts
from measured energy by more than a threshold.

This is the primary safety signal: a conservation breach means either:
    1. Physical anomaly (broken joint, external force, etc.)
    2. Model error (HNN needs retraining)
    3. Unmodeled dynamics (new payload, different crop weight, etc.)

In all cases, this should raise a safety flag → ISO 10218 FSM event.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


class ConservationChecker:
    """
    Checks energy conservation along Hamiltonian trajectories.

    For a closed system, H should be constant. Any drift indicates
    non-conservative forces or model error.
    """

    DEFAULT_DRIFT_THRESHOLD = 0.05

    def __init__(
        self,
        threshold: float = DEFAULT_DRIFT_THRESHOLD,
        relative: bool = True,
    ):
        self.threshold = threshold
        self.relative = relative

    def drift(
        self,
        trajectory: Dict,
        H0: Optional[float] = None,
    ) -> float:
        """
        Compute energy drift from a trajectory dict.

        Args:
            trajectory: Dict with keys "traj_q", "traj_p", "energies"
            H0: Initial energy (if not in trajectory)

        Returns:
            Drift as absolute value. If relative=True, returns fraction of H0.
        """
        if "energies" in trajectory and len(trajectory["energies"]) > 0:
            energies = np.array(trajectory["energies"])
        else:
            return 0.0

        if H0 is None and len(energies) > 0:
            H0 = energies[0]

        if H0 is None or H0 == 0:
            return float(np.max(np.abs(energies - energies[0])))

        H_final = energies[-1]
        if self.relative:
            return float(abs(H_final - H0) / abs(H0))
        else:
            return float(abs(H_final - H0))

    def drift_from_arrays(
        self,
        traj_q: np.ndarray,
        traj_p: np.ndarray,
        hnn_model,
        relative: bool = True,
    ) -> float:
        """Compute energy drift from raw (q,p) trajectory arrays."""
        energies = [hnn_model.hamiltonian(traj_q[i], traj_p[i]) for i in range(len(traj_q))]
        energies = np.array(energies)

        H0 = energies[0]
        H_final = energies[-1]
        H_max = np.max(energies)
        H_min = np.min(energies)

        if relative:
            if abs(H0) < 1e-8:
                return float(H_max - H_min)
            return float(max(abs(H_final - H0), (H_max - H_min)) / abs(H0))
        else:
            return float(max(abs(H_final - H0), H_max - H_min))

    def check_violation(
        self,
        drift: float,
        threshold: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Determine if a drift value constitutes a violation.

        Args:
            drift: Energy drift value
            threshold: Override default threshold

        Returns:
            (is_violation, reason)
        """
        thresh = threshold if threshold is not None else self.threshold

        if drift > thresh:
            severity = self._severity_level(drift, thresh)
            return True, f"Energy drift {drift:.4%} exceeds threshold {thresh:.4%} ({severity})"
        else:
            return False, "OK"

    def _severity_level(self, drift: float, threshold: float) -> str:
        ratio = drift / threshold
        if ratio < 2:
            return "minor"
        elif ratio < 5:
            return "moderate"
        elif ratio <= 10:
            return "major"
        else:
            return "critical"

    def analyze_trajectory(
        self,
        traj_q: np.ndarray,
        traj_p: np.ndarray,
        hnn_model,
        dt: float = 0.01,
    ) -> Dict:
        """
        Full analysis of a trajectory's energy conservation properties.

        Returns:
            Dict with drift metrics, violation status, and severity.
        """
        energies = np.array(
            [hnn_model.hamiltonian(traj_q[i], traj_p[i]) for i in range(len(traj_q))]
        )

        H0 = energies[0]
        H_final = energies[-1]
        H_max = energies.max()
        H_min = energies.min()

        relative_drift = abs(H_final - H0) / abs(H0) if abs(H0) > 1e-8 else 0.0
        energy_range = H_max - H_min
        relative_range = energy_range / abs(H0) if abs(H0) > 1e-8 else 0.0

        is_violation, reason = self.check_violation(relative_drift)

        num_steps = len(traj_q)
        total_time = num_steps * dt

        return {
            "H0": float(H0),
            "H_final": float(H_final),
            "H_max": float(H_max),
            "H_min": float(H_min),
            "drift": float(relative_drift),
            "energy_range": float(energy_range),
            "relative_range": float(relative_range),
            "is_violation": is_violation,
            "reason": reason,
            "num_steps": num_steps,
            "total_time_s": float(total_time),
        }

    def compute_conservation_error(
        self,
        traj_q: np.ndarray,
        traj_p: np.ndarray,
        hnn_model,
    ) -> float:
        """
        Compute mean squared conservation error across trajectory.

        L_conservation = mean(||dH/dt||²)
        """
        energies = np.array(
            [hnn_model.hamiltonian(traj_q[i], traj_p[i]) for i in range(len(traj_q))]
        )

        if len(energies) < 2:
            return 0.0

        dH = np.diff(energies)
        mse = float(np.mean(dH**2))
        return mse

    def get_safety_status(self, drift: float) -> str:
        """Return a human-readable safety status string."""
        if drift < self.threshold * 0.5:
            return "NOMINAL"
        elif drift < self.threshold:
            return "CAUTION"
        elif drift < self.threshold * 2:
            return "WARNING"
        elif drift < self.threshold * 5:
            return "ALERT"
        else:
            return "CRITICAL"
