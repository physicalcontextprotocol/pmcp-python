"""
Phase Space Encoder
==================
Converts raw robot sensor data → (q, p) phase space coordinates.

q = generalized positions (joint angles, xyz coordinates, etc.)
p = generalized momenta (joint velocities × inertia masses)

For a serial chain robot:
    q = [theta_1, theta_2, ..., theta_n, x, y, z]  (6-DOF typically)
    p = [L_1, L_2, ..., L_n, p_x, p_y, p_z]

Conversion formula:
    p = M * q_dot  (mass matrix × generalized velocities)

Expected sensor schema:
    {
        "joint_angles": [float, ...],       # rad
        "joint_velocities": [float, ...],   # rad/s
        "end_effector_pose": {"x":, "y":, "z":, ...},
        "mass_matrix": [[float, ...], ...], # optional, for accurate momentum
    }
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


class PhaseSpaceEncoder:
    """
    Encodes raw sensor readings into phase space coordinates (q, p).

    Canonical momentum: p_i = L_i = sum_j M_ij * q_dot_j
    where M_ij is the mass/inertia matrix.
    """

    def __init__(
        self,
        n_joints: int = 6,
        default_mass: float = 1.0,
        gripper_mass: float = 0.5,
    ):
        self.n_joints = n_joints
        self.default_mass = default_mass
        self.gripper_mass = gripper_mass

        self._mass_matrix: Optional[np.ndarray] = None
        self._calibrated = False

    def encode(self, sensor_data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode sensor data into (q, p) phase space.

        Args:
            sensor_data: Dict with keys like "joint_angles", "joint_velocities",
                        "end_effector_pose", "mass_matrix"

        Returns:
            (q, p) tuple of numpy arrays
        """
        q = self._extract_positions(sensor_data)
        p = self._extract_momenta(sensor_data, q)
        return q, p

    def _extract_positions(self, data: Dict) -> np.ndarray:
        q = []

        if "joint_angles" in data:
            joint_angles = np.asarray(data["joint_angles"])
            q.extend(joint_angles.tolist())
        elif "q" in data:
            q.extend(np.asarray(data["q"]).tolist())

        if "end_effector_pose" in data:
            pose = data["end_effector_pose"]
            q.extend([pose.get("x", 0), pose.get("y", 0), pose.get("z", 0)])
        elif "xyz" in data:
            xyz = np.asarray(data["xyz"])
            q.extend(xyz.tolist()[:3])

        if "orientation" in data:
            ori = data["orientation"]
            if isinstance(ori, dict):
                q.extend([ori.get("roll", 0), ori.get("pitch", 0), ori.get("yaw", 0)])

        return np.array(q, dtype=np.float32)

    def _extract_momenta(self, data: Dict, q: np.ndarray) -> np.ndarray:
        p = []

        if "joint_velocities" in data:
            q_dot = np.asarray(data["joint_velocities"])

            if "mass_matrix" in data:
                M = np.asarray(data["mass_matrix"])
                if M.shape == (len(q_dot), len(q_dot)):
                    p_extended = M @ q_dot
                    p.extend(p_extended.tolist())
                else:
                    p.extend(q_dot.tolist())
            elif "inertia" in data:
                inertia = np.asarray(data["inertia"])
                if len(inertia) == len(q_dot):
                    p_extended = inertia * q_dot
                    p.extend(p_extended.tolist())
                else:
                    p.extend(q_dot.tolist())
            else:
                masses = self._get_default_masses(len(q_dot))
                p_extended = masses * q_dot
                p.extend(p_extended.tolist())

        elif "p" in data:
            p.extend(np.asarray(data["p"]).tolist())

        elif "momentum" in data:
            p.extend(np.asarray(data["momentum"]).tolist())

        else:
            n_missing = len(q)
            p.extend(np.zeros(n_missing).tolist())

        if "linear_velocity" in data:
            lv = data["linear_velocity"]
            if isinstance(lv, dict):
                mass = self.gripper_mass
                p.extend([lv.get("x", 0) * mass, lv.get("y", 0) * mass, lv.get("z", 0) * mass])
            else:
                lv_arr = np.asarray(lv)
                mass = self.gripper_mass
                p.extend((lv_arr * mass).tolist())

        return np.array(p, dtype=np.float32)

    def _get_default_masses(self, n: int) -> np.ndarray:
        masses = np.ones(n) * self.default_mass
        if n > 0:
            masses[-1] = self.gripper_mass
        return masses

    def calibrate(self, sample_trajectories: List[Dict[str, np.ndarray]]):
        """
        Calibrate mass matrix from recorded trajectories.
        Uses linear regression to estimate M from (q_dot, p) pairs.
        """
        q_dots = []
        momenta = []

        for traj in sample_trajectories:
            if "q_dot" in traj and "p" in traj:
                q_dots.append(traj["q_dot"])
                momenta.append(traj["p"])

        if len(q_dots) < 10:
            self._calibrated = False
            return

        Q = np.vstack(q_dots)
        P = np.vstack(momenta)

        try:
            self._mass_matrix, _, _, _ = np.linalg.lstsq(Q, P, rcond=None)
            self._calibrated = True
        except np.linalg.LinAlgError:
            self._calibrated = False

    def set_mass_matrix(self, M: np.ndarray):
        self._mass_matrix = np.asarray(M, dtype=np.float32)
        self._calibrated = True

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def get_q_dim(self, sensor_data: Dict) -> int:
        return len(self._extract_positions(sensor_data))

    def get_p_dim(self, sensor_data: Dict) -> int:
        return len(self._extract_momenta(sensor_data, self._extract_positions(sensor_data)))

    def decode(self, q: np.ndarray, p: np.ndarray) -> Dict:
        """
        Convert (q, p) back to sensor-like format for downstream compatibility.
        """
        result = {}
        n_joints = self.n_joints

        if len(q) >= n_joints:
            result["joint_angles"] = q[:n_joints].tolist()

        if len(q) > n_joints:
            result["end_effector_pose"] = {
                "x": float(q[n_joints]),
                "y": float(q[n_joints + 1]) if len(q) > n_joints + 1 else 0,
                "z": float(q[n_joints + 2]) if len(q) > n_joints + 2 else 0,
            }

        result["joint_momenta"] = p[:n_joints].tolist() if len(p) >= n_joints else []

        return result

    def validate(self, q: np.ndarray, p: np.ndarray) -> Tuple[bool, str]:
        if len(q) == 0 or len(p) == 0:
            return False, "Empty phase space vector"

        if len(q) != len(p):
            return False, f"Dimension mismatch: q={len(q)}, p={len(p)}"

        if np.any(np.isnan(q)) or np.any(np.isnan(p)):
            return False, "NaN detected in phase space"

        if np.any(np.isinf(q)) or np.any(np.isinf(p)):
            return False, "Inf detected in phase space"

        return True, "Valid"


def create_encoder_from_robot_config(robot_config: Dict) -> PhaseSpaceEncoder:
    n_joints = robot_config.get("n_joints", 6)
    masses = robot_config.get("link_masses", [1.0] * n_joints)

    encoder = PhaseSpaceEncoder(n_joints=n_joints)
    if len(masses) == n_joints:
        M = np.diag(masses)
        encoder.set_mass_matrix(M)

    return encoder
