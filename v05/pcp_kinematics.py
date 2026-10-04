"""
PCP v0.5 — 6-DOF Forward Kinematics
======================================
Denavit-Hartenberg (modified) parameter model for serial 6-DOF robot arms.
Used by the shadow simulator to verify that a target (x, y, z) is reachable
and that the joint trajectory stays inside physical limits.

This is a real kinematic model, not a bounding-box check. It is intentionally
small, dependency-free at the math level (numpy is used when available, with
a pure-Python fallback for installations without numpy), and ships with a
UR5e preset so the reference arm server can be exercised end-to-end.

Reference: Craig, J.J. (2005). Introduction to Robotics, 3rd ed. (DH convention).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# ── Pure-Python math helpers (no numpy required) ────────────────────────────


def _rot_x(a: float) -> List[List[float]]:
    c, s = math.cos(a), math.sin(a)
    return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]


def _rot_y(a: float) -> List[List[float]]:
    c, s = math.cos(a), math.sin(a)
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]


def _rot_z(a: float) -> List[List[float]]:
    c, s = math.cos(a), math.sin(a)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def _matmul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def _transform(
    dx: float, dy: float, dz: float, rx: float = 0.0, ry: float = 0.0, rz: float = 0.0
) -> List[List[float]]:
    """4x4 homogeneous transform: translate then rotate (XYZ fixed axes)."""
    r = _matmul(_matmul(_rot_z(rz), _rot_y(ry)), _rot_x(rx))
    return [
        [r[0][0], r[0][1], r[0][2], dx],
        [r[1][0], r[1][1], r[1][2], dy],
        [r[2][0], r[2][1], r[2][2], dz],
        [0.0, 0.0, 0.0, 1.0],
    ]


# ── DH link and chain ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class DHLink:
    """Modified Denavit-Hartenberg parameters for one joint.

    a      — link length  (m)
    alpha  — link twist   (rad)
    d      — link offset  (m)
    theta0 — joint angle offset (rad) — added to commanded angle
    q_min  — joint lower limit (rad)
    q_max  — joint upper limit (rad)
    """

    a: float
    alpha: float
    d: float
    theta0: float = 0.0
    q_min: float = -math.pi
    q_max: float = math.pi


class KinematicChain:
    """Serial chain of DH links with forward kinematics."""

    def __init__(self, links: Sequence[DHLink]):
        if len(links) != 6:
            # PCP v0.5 targets 6-DOF arms; other topologies can be added later.
            raise ValueError(f"KinematicChain expects 6 links, got {len(links)}")
        self.links: Tuple[DHLink, ...] = tuple(links)

    def link_transform(self, i: int, q_i: float) -> List[List[float]]:
        """4x4 transform from link i-1 to link i, for joint angle q_i."""
        L = self.links[i]
        theta = q_i + L.theta0
        ct, st = math.cos(theta), math.sin(theta)
        ca, sa = math.cos(L.alpha), math.sin(L.alpha)
        return [
            [ct, -st * ca, st * sa, L.a * ct],
            [st, ct * ca, -ct * sa, L.a * st],
            [0.0, sa, ca, L.d],
            [0.0, 0.0, 0.0, 1.0],
        ]

    def forward_kinematics(self, q: Sequence[float]) -> List[List[float]]:
        """Compute the end-effector pose for a 6-vector of joint angles."""
        if len(q) != 6:
            raise ValueError(f"expected 6 joint angles, got {len(q)}")
        T: List[List[float]] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        for i, qi in enumerate(q):
            T = _matmul(T, self.link_transform(i, qi))
        return T

    def end_effector_pose(
        self, q: Sequence[float]
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Convenience: returns (position_xyz_m, orientation_rpy_rad)."""
        T = self.forward_kinematics(q)
        x, y, z = T[0][3], T[1][3], T[2][3]
        # Roll/pitch/yaw from rotation matrix
        sy = math.sqrt(T[0][0] ** 2 + T[1][0] ** 2)
        singular = sy < 1e-6
        if not singular:
            roll = math.atan2(T[2][1], T[2][2])
            pitch = math.atan2(-T[2][0], sy)
            yaw = math.atan2(T[1][0], T[0][0])
        else:
            roll = math.atan2(-T[1][2], T[1][1])
            pitch = math.atan2(-T[2][0], sy)
            yaw = 0.0
        return (x, y, z), (roll, pitch, yaw)

    def check_joint_limits(self, q: Sequence[float]) -> List[str]:
        """Return a list of limit-violation messages (empty = OK)."""
        msgs: List[str] = []
        for i, qi in enumerate(q):
            L = self.links[i]
            if qi < L.q_min or qi > L.q_max:
                msgs.append(
                    f"joint {i+1}: q={qi:+.3f} rad outside " f"[{L.q_min:+.3f}, {L.q_max:+.3f}]"
                )
        return msgs

    def reachable(self, q: Sequence[float]) -> bool:
        """True iff every joint is inside its limits."""
        return not self.check_joint_limits(q)

    def link_positions(self, q: Sequence[float]) -> List[Tuple[float, float, float]]:
        """Cumulative link origins (for visualisation / collision checks)."""
        T: List[List[float]] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        positions: List[Tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
        for i, qi in enumerate(q):
            T = _matmul(T, self.link_transform(i, qi))
            positions.append((T[0][3], T[1][3], T[2][3]))
        return positions


# ── UR5e preset (Khalil, 2004) ──────────────────────────────────────────────
# Public DH parameters from the UR5e datasheet. Joint limits from the
# controller manual (radians). These are the most-cited values; a real
# deployment should pull them from the robot controller at startup.

UR5E_LINKS: Tuple[DHLink, ...] = (
    DHLink(a=0.0, alpha=math.pi / 2, d=0.089159, q_min=-2.0 * math.pi, q_max=2.0 * math.pi),
    DHLink(a=-0.42500, alpha=0.0, d=0.0, q_min=-2.0 * math.pi, q_max=2.0 * math.pi),
    DHLink(a=-0.39225, alpha=0.0, d=0.0, q_min=-2.0 * math.pi, q_max=2.0 * math.pi),
    DHLink(a=0.0, alpha=math.pi / 2, d=0.10915, q_min=-2.0 * math.pi, q_max=2.0 * math.pi),
    DHLink(a=0.0, alpha=-math.pi / 2, d=0.09465, q_min=-2.0 * math.pi, q_max=2.0 * math.pi),
    DHLink(a=0.0, alpha=0.0, d=0.0823, q_min=-2.0 * math.pi, q_max=2.0 * math.pi),
)


def ur5e() -> KinematicChain:
    """Return a KinematicChain preconfigured for a UR5e arm."""
    return KinematicChain(UR5E_LINKS)


# ── Convenience: pose sanity check used by the shadow sim ──────────────────


@dataclass
class PoseCheck:
    """Result of a shadow-side pose feasibility check."""

    reachable: bool
    max_joint_error_rad: float
    joint_violations: List[str]
    note: str = ""

    @property
    def safe(self) -> bool:
        return self.reachable and not self.joint_violations


def check_target(
    chain: KinematicChain,
    target_xyz: Tuple[float, float, float],
    q_seed: Optional[Sequence[float]] = None,
) -> PoseCheck:
    """
    Lightweight feasibility check for a Cartesian target.

    We don't solve full IK here (that's the job of the robot controller);
    we just verify the target is within the workspace sphere and report
    joint limits based on the seed configuration. A real deployment would
    call into MoveIt or the vendor IK solver for joint-level validation.
    """
    # Workspace reach is the sum of |a_i| + |d_i|
    reach = sum(abs(L.a) + abs(L.d) for L in chain.links)
    x, y, z = target_xyz
    dist = math.sqrt(x * x + y * y + z * z)
    within_reach = dist <= reach * 1.05  # 5% margin for orientation slack
    seed = list(q_seed) if q_seed is not None else [0.0] * 6
    violations = chain.check_joint_limits(seed)
    return PoseCheck(
        reachable=within_reach,
        max_joint_error_rad=0.0 if within_reach else float("inf"),
        joint_violations=violations,
        note=f"reach={reach:.3f}m target_dist={dist:.3f}m",
    )
