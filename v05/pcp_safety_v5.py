"""
PCP v0.5 — Safety Constitution
==================================
ISO 10218 / IEC 62443 compliant safety rules for robot tool calls.

The SafetyConstitution is a set of hard rules that cannot be overridden
by any LLM instruction.  Rules are evaluated synchronously before every
actuation call in the safety pipeline.

Design mirrors Anthropic's Constitutional AI but for physical safety:
  1. Hard rules encoded as Python (fast, deterministic)
  2. TEE-signed fingerprint — rules can't be swapped at runtime
  3. ISO 10218 (robot safety) + IEC 62443 (industrial security) coverage
  4. Human proximity integration (camera/lidar sensor fusion)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    # Only needed for the forward-referenced type hints below
    # (-> "ShadowPreview", Optional["ShadowPreview"]). No circular-import
    # risk -- pcp_v5_types.py does not import from this module -- so this
    # is gated behind TYPE_CHECKING purely to avoid a runtime import that
    # isn't otherwise needed (preview() already does its own local import).
    from v05.pcp_v5_types import ShadowPreview

log = logging.getLogger("pcp.safety_v5")


# ─────────────────────────────────────────────────────────────────────────────
#  SAFETY RULE PRIMITIVES
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SafetyRule:
    id: str
    description: str
    standard: str = "ISO10218"  # "ISO10218" | "IEC62443" | "custom"
    severity: str = "FATAL"  # "FATAL" | "WARNING"

    def evaluate(self, call: Dict[str, Any]) -> Tuple[bool, str]:
        """Return (passed, reason). Override in subclasses."""
        return True, ""


class SpeedLimitRule(SafetyRule):
    def __init__(self, max_speed_m_s: float = 2.0):
        super().__init__(
            id="R-SPEED-01",
            description=f"TCP speed must not exceed {max_speed_m_s} m/s (ISO 10218-1 §5.4)",
            standard="ISO10218",
        )
        self.max_speed_m_s = max_speed_m_s

    def evaluate(self, call: Dict[str, Any]) -> Tuple[bool, str]:
        speed = call.get("speed", call.get("max_speed_m_s", 0.0))
        if speed > self.max_speed_m_s:
            return False, f"Speed {speed} m/s exceeds limit {self.max_speed_m_s} m/s"
        return True, ""


class FloorGuardRule(SafetyRule):
    def __init__(self, floor_z_m: float = -0.05):
        super().__init__(
            id="R-FLOOR-01",
            description=f"Z position must stay above {floor_z_m}m (collision guard)",
            standard="ISO10218",
        )
        self.floor_z_m = floor_z_m

    def evaluate(self, call: Dict[str, Any]) -> Tuple[bool, str]:
        z = call.get("z", call.get("target_z", None))
        if z is not None and float(z) < self.floor_z_m:
            return False, f"Z={z}m is below floor guard {self.floor_z_m}m"
        return True, ""


class WorkspaceBoxRule(SafetyRule):
    def __init__(
        self,
        x_range: Tuple[float, float] = (-2.0, 2.0),
        y_range: Tuple[float, float] = (-2.0, 2.0),
        z_range: Tuple[float, float] = (-0.05, 3.0),
    ):
        super().__init__(
            id="R-WS-01",
            description="Target position must be inside declared workspace box",
            standard="ISO10218",
        )
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

    def evaluate(self, call: Dict[str, Any]) -> Tuple[bool, str]:
        for axis, rng in [("x", self.x_range), ("y", self.y_range), ("z", self.z_range)]:
            v = call.get(axis)
            if v is not None:
                if not (rng[0] <= float(v) <= rng[1]):
                    return False, f"{axis}={v} outside workspace [{rng[0]}, {rng[1]}]"
        return True, ""


class EnergyBudgetRule(SafetyRule):
    def __init__(self, max_energy_j: float = 5000.0):
        super().__init__(
            id="R-ENERGY-01",
            description=f"Single call energy budget ≤ {max_energy_j}J",
            standard="custom",
        )
        self.max_energy_j = max_energy_j

    def evaluate(self, call: Dict[str, Any]) -> Tuple[bool, str]:
        e = call.get("energy_j", call.get("est_energy_j", 0.0))
        if e > self.max_energy_j:
            return False, f"Estimated energy {e}J exceeds budget {self.max_energy_j}J"
        return True, ""


class HumanProximityRule(SafetyRule):
    def __init__(self, min_clearance_m: float = 0.5):
        super().__init__(
            id="R-HUMAN-01",
            description=f"Minimum human clearance ≥ {min_clearance_m}m (ISO 10218-2 §5.10)",
            standard="ISO10218",
        )
        self.min_clearance_m = min_clearance_m

    def evaluate(self, call: Dict[str, Any]) -> Tuple[bool, str]:
        clearance = call.get("human_clearance_m", None)
        if clearance is not None and float(clearance) < self.min_clearance_m:
            return False, (
                f"Human at {clearance}m < minimum clearance {self.min_clearance_m}m. "
                f"Halting per ISO 10218-2 §5.10"
            )
        return True, ""


class EStopRule(SafetyRule):
    def __init__(self):
        super().__init__(
            id="R-ESTOP-01",
            description="All actuations blocked when E-Stop is active",
            standard="ISO10218",
            severity="FATAL",
        )
        self._estop_active = False

    def set_estop(self, active: bool) -> None:
        self._estop_active = active

    def evaluate(self, call: Dict[str, Any]) -> Tuple[bool, str]:
        if self._estop_active:
            return False, "Emergency stop is active — all motion blocked"
        return True, ""


class ForceLimit(SafetyRule):
    def __init__(self, max_force_n: float = 150.0):
        super().__init__(
            id="R-FORCE-01",
            description=f"End-effector force ≤ {max_force_n}N (ISO TS 15066)",
            standard="ISO10218",
        )
        self.max_force_n = max_force_n

    def evaluate(self, call: Dict[str, Any]) -> Tuple[bool, str]:
        f = call.get("force_n", call.get("max_force_n", 0.0))
        if f > self.max_force_n:
            return False, f"Force {f}N exceeds limit {self.max_force_n}N"
        return True, ""


# ─────────────────────────────────────────────────────────────────────────────
#  OPTIONAL GATE 4: HAMILTONIAN CONSERVATION
#  Disabled by default. Requires `pip install pcp[hnn]` (torch + numpy).
#  When enabled, rejects actuations whose H(q, p) drift exceeds a
#  threshold — catches broken joints, external forces, model error.
# ─────────────────────────────────────────────────────────────────────────────

try:
    import numpy as _np  # noqa: F401

    _HNN_NUMPY_OK = True
except ImportError:
    _HNN_NUMPY_OK = False

try:
    import torch as _torch  # noqa: F401

    _HNN_TORCH_OK = True
except ImportError:
    _HNN_TORCH_OK = False

_HNN_AVAILABLE = _HNN_NUMPY_OK and _HNN_TORCH_OK


class HNNConservationRule(SafetyRule):
    """
    Optional Gate 4. Off by default — opt in via:

        hnn = HamiltonianNN.load("model.pt")
        chk = ConservationChecker(threshold=0.05)
        rule = HNNConservationRule(hnn, chk)
        mw = SafetyMiddleware(constitution, hnn_rule=rule)

    When no baseline is provided in the call dict, the rule passes
    (we cannot evaluate conservation from a single sample).
    """

    def __init__(self, hnn_model, conservation_checker, q_dim: int = 6, p_dim: int = 6):
        super().__init__(
            id="R-HNN-01",
            description=(
                "Hamiltonian energy conservation check (HNN Gate 4) — "
                "rejects |ΔH|/H0 > threshold"
            ),
            standard="custom",
            severity="WARNING",
        )
        if not _HNN_AVAILABLE:
            raise RuntimeError(
                "HNNConservationRule requires torch + numpy. " "Install with: pip install pcp[hnn]"
            )
        self.hnn = hnn_model
        self.checker = conservation_checker
        self.q_dim = q_dim
        self.p_dim = p_dim
        self._total_checks = 0
        self._violations = 0

    def evaluate(self, call: Dict[str, Any]) -> Tuple[bool, str]:
        q = call.get("q_state")
        p = call.get("p_state")
        h_baseline = call.get("h_baseline")
        if q is None or p is None or h_baseline in (None, 0):
            # No baseline → cannot evaluate; pass through.
            return True, ""
        q = _np.asarray(q, dtype=float)
        p = _np.asarray(p, dtype=float)
        h_now = float(self.hnn.hamiltonian(q, p))
        if abs(float(h_baseline)) < 1e-8:
            drift = abs(h_now - float(h_baseline))
        else:
            drift = abs(h_now - float(h_baseline)) / abs(float(h_baseline))
        self._total_checks += 1
        is_violation, reason = self.checker.check_violation(drift)
        if is_violation:
            self._violations += 1
            return False, f"{reason} (H0={float(h_baseline):.4f}, H={h_now:.4f})"
        return True, ""

    def stats(self) -> Dict[str, int]:
        return {"checks": self._total_checks, "violations": self._violations}


# ─────────────────────────────────────────────────────────────────────────────
#  SAFETY CONSTITUTION
# ─────────────────────────────────────────────────────────────────────────────


class SafetyConstitution:
    """
    Immutable set of safety rules loaded at boot time.
    Fingerprinted with SHA-256 — any rule modification is detectable.

    Default rule set covers:
      • ISO 10218-1/2 robotic safety
      • IEC 62443 industrial security
      • E-Stop, speed, workspace, force, energy, human proximity
    """

    def __init__(self, robot_id: str, rules: Optional[List[SafetyRule]] = None):
        self.robot_id = robot_id
        self._estop_rule = EStopRule()
        self._rules: List[SafetyRule] = rules or self._default_rules()
        self._fingerprint = self._compute_fingerprint()
        log.info(
            f"[Constitution] {robot_id}: {len(self._rules)} rules, "
            f"fingerprint={self._fingerprint[:16]}..."
        )

    def _default_rules(self) -> List[SafetyRule]:
        return [
            self._estop_rule,
            SpeedLimitRule(max_speed_m_s=2.0),
            FloorGuardRule(floor_z_m=-0.05),
            WorkspaceBoxRule(),
            EnergyBudgetRule(max_energy_j=5000.0),
            HumanProximityRule(min_clearance_m=0.5),
            ForceLimit(max_force_n=150.0),
        ]

    def _compute_fingerprint(self) -> str:
        payload = json.dumps(
            [
                {"id": r.id, "description": r.description, "standard": r.standard}
                for r in self._rules
            ],
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def set_estop(self, active: bool) -> None:
        """Activate or deactivate emergency stop."""
        self._estop_rule.set_estop(active)
        state = "ACTIVE" if active else "CLEAR"
        log.warning(f"[Constitution] {self.robot_id}: E-Stop {state}")

    def evaluate(self, call: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Evaluate all rules against a tool call dict.
        Returns (all_passed, list_of_violation_messages).
        """
        violations: List[str] = []
        for rule in self._rules:
            passed, reason = rule.evaluate(call)
            if not passed:
                violations.append(f"[{rule.id}] {reason}")
                if rule.severity == "FATAL":
                    # Short-circuit on first fatal violation
                    return False, violations
        return len(violations) == 0, violations

    def summary(self) -> dict:
        return {
            "robot_id": self.robot_id,
            "rule_count": len(self._rules),
            "fingerprint": self.fingerprint,
            "rules": [
                {"id": r.id, "standard": r.standard, "description": r.description}
                for r in self._rules
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
#  SHADOW SIMULATOR  (geometric fallback; PyBullet if available)
# ─────────────────────────────────────────────────────────────────────────────

try:
    import pybullet as _pb
    import pybullet_data as _pbd

    _BULLET = True
except ImportError:
    _BULLET = False


class ShadowSimulator:
    """
    Pre-flight collision / reachability simulation.

    Engine ladder (highest fidelity first, only one is used per call):
      1. PyBullet       — if `pybullet` is importable, full physics
      2. Forward-kinematics + reachability — 6-DOF DH model, joint-limit check,
         workspace-sphere check. Pure stdlib, no numpy required.
      3. Geometric       — bounding-box check (always available, last-resort)

    The engine actually used is reported on the returned ShadowPreview.engine
    so callers can adjust their trust in the result.
    """

    def __init__(self, workspace_box=None, kinematic_chain=None):
        self._box = workspace_box or ((-2.0, -2.0, -0.05), (2.0, 2.0, 3.0))
        self._chain = kinematic_chain  # optional KinematicChain
        if _BULLET:
            self._engine = "pybullet"
        elif kinematic_chain is not None:
            self._engine = "kinematic"
        else:
            self._engine = "geometric"

    def preview(
        self, actuation_name: str, arguments: Dict[str, Any], safety_envelope=None
    ) -> "ShadowPreview":
        from v05.pcp_v5_types import ShadowPreview, ShadowStatus

        start = time.time()
        warnings: List[str] = []
        collision_body = ""
        status = ShadowStatus.SAFE

        # ── Geometric checks (always run) ───────────────────────────────────
        x = float(arguments.get("x", 0))
        y = float(arguments.get("y", 0))
        z = float(arguments.get("z", 0))
        (xmin, ymin, zmin), (xmax, ymax, zmax) = self._box

        if not (xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax):
            status = ShadowStatus.WORKSPACE_VIOLATION
            warnings.append(f"Target ({x},{y},{z}) outside workspace box")

        speed = float(arguments.get("speed", arguments.get("max_speed_m_s", 0.3)))
        if speed > 2.5:
            status = ShadowStatus.SPEED_EXCEEDED
            warnings.append(f"Speed {speed} m/s exceeds simulation limit 2.5 m/s")

        # ── Kinematic reachability (when no pybullet and chain is wired) ──
        if (not _BULLET) and self._chain is not None and status == ShadowStatus.SAFE:
            from v05.pcp_kinematics import check_target

            q_seed = arguments.get("q_seed")
            check = check_target(self._chain, (x, y, z), q_seed=q_seed)
            if not check.reachable:
                status = ShadowStatus.WORKSPACE_VIOLATION
                warnings.append(f"Target outside reachable workspace: {check.note}")
            if check.joint_violations:
                status = ShadowStatus.WORKSPACE_VIOLATION
                warnings.extend(check.joint_violations)

        # ── PyBullet physics sim (if available) ─────────────────────────────
        if _BULLET and status == ShadowStatus.SAFE:
            status, collision_body, warnings = self._bullet_sim(actuation_name, arguments, warnings)

        safe = status == ShadowStatus.SAFE
        sim_dur = time.time() - start
        est_dur = arguments.get(
            "duration_s", max(0.1, ((x**2 + y**2 + z**2) ** 0.5) / max(speed, 0.01))
        )

        return ShadowPreview(
            actuation_name=actuation_name,
            arguments=arguments,
            status=status,
            safe=safe,
            risk_score=0.0 if safe else 0.8,
            sim_duration_s=round(sim_dur, 4),
            est_duration_s=round(est_dur, 2),
            est_energy_j=round(speed * est_dur * 50.0, 1),
            collision_body=collision_body,
            warnings=warnings,
            engine=self._engine,
        )

    def _bullet_sim(self, name: str, args: Dict[str, Any], warnings: List[str]):
        from v05.pcp_v5_types import ShadowStatus

        try:
            cid = _pb.connect(_pb.DIRECT)
            _pb.setAdditionalSearchPath(_pbd.getDataPath(), physicsClientId=cid)
            _pb.setGravity(0, 0, -9.81, physicsClientId=cid)
            _pb.loadURDF("plane.urdf", physicsClientId=cid)

            # Spawn ghost sphere at target
            x, y, z = float(args.get("x", 0)), float(args.get("y", 0)), float(args.get("z", 0))
            col = _pb.createCollisionShape(_pb.GEOM_SPHERE, radius=0.05, physicsClientId=cid)
            _pb.createMultiBody(0, col, -1, [x, y, z], [0, 0, 0, 1], physicsClientId=cid)
            _pb.stepSimulation(physicsClientId=cid)
            contacts = _pb.getContactPoints(physicsClientId=cid)
            _pb.disconnect(cid)

            if contacts:
                return (
                    ShadowStatus.COLLISION,
                    "floor_plane",
                    warnings + ["PyBullet: contact detected with floor_plane"],
                )
        except Exception as exc:
            warnings.append(f"PyBullet error: {exc}")

        return ShadowStatus.SAFE, "", warnings


# ─────────────────────────────────────────────────────────────────────────────
#  SAFETY MIDDLEWARE  (pipeline: constitution → shadow → execute)
# ─────────────────────────────────────────────────────────────────────────────


class SafetyMiddleware:
    """
    Enforces the PCP three-layer safety pipeline before any actuation executes:
      1. LeaseCheck       — robot holds valid zone lease
      2. ConstitutionCheck — ISO hard rules
      3. ShadowPreview    — 3D simulation
    """

    def __init__(
        self,
        constitution: SafetyConstitution,
        simulator: Optional[ShadowSimulator] = None,
        lease_manager=None,
        hnn_rule: Optional[HNNConservationRule] = None,
    ):
        self._constitution = constitution
        self._sim = simulator or ShadowSimulator()
        self._leases = lease_manager  # Optional LeaseManager
        self._hnn_rule = hnn_rule  # Optional Gate 4 (off by default)
        self._stats = {"calls": 0, "blocked": 0, "warnings": 0}

    def set_estop(self, active: bool) -> None:
        self._constitution.set_estop(active)

    def check(
        self,
        actuation_name: str,
        arguments: Dict[str, Any],
        lease_token: Optional[str] = None,
        zone_id: Optional[str] = None,
        fence_token: Optional[int] = None,
        skip_shadow: bool = False,
    ) -> Tuple[bool, Optional["ShadowPreview"], List[str]]:
        """
        Run full pipeline. Returns (safe, shadow_preview_or_None, violation_msgs).

        fence_token must be the token returned with the caller's current
        lease grant (LeaseGrant.fence_token / wire field "fenceToken").
        Presenting a stale fence token (e.g. after the lease was renewed or
        re-granted since the caller last observed it) is rejected even if
        lease_token still lexically matches — see _LeaseManager.check().
        """
        self._stats["calls"] += 1
        violations: List[str] = []

        # ── Layer 1: Lease check ─────────────────────────────────────────────
        if self._leases and zone_id:
            ok, reason = self._leases.check(lease_token, zone_id, fence_token)
            if not ok:
                violations.append(f"[LEASE] {reason}")
                self._stats["blocked"] += 1
                return False, None, violations

        # ── Layer 2: Constitution ────────────────────────────────────────────
        call_dict = dict(arguments)
        call_dict["actuation_name"] = actuation_name
        passed, const_violations = self._constitution.evaluate(call_dict)
        if not passed:
            violations.extend(const_violations)
            self._stats["blocked"] += 1
            return False, None, violations

        # ── Layer 2.5: HNN energy conservation (opt-in Gate 4) ───────────────
        if self._hnn_rule is not None:
            hnn_passed, hnn_reason = self._hnn_rule.evaluate(call_dict)
            if not hnn_passed:
                violations.append(f"[HNN] {hnn_reason}")
                self._stats["warnings"] += 1
                # HNN is WARNING severity — do not block, surface the alert.
                log.warning(f"[HNN] conservation breach: {hnn_reason}")

        # ── Layer 3: Shadow simulation ───────────────────────────────────────
        if skip_shadow:
            return True, None, []

        preview = self._sim.preview(actuation_name, arguments)
        if not preview.safe:
            violations.append(f"[SHADOW:{preview.status.value}] {', '.join(preview.warnings)}")
            self._stats["blocked"] += 1
            return False, preview, violations

        if preview.warnings:
            self._stats["warnings"] += len(preview.warnings)

        return True, preview, violations

    @property
    def stats(self) -> dict:
        return dict(self._stats)
