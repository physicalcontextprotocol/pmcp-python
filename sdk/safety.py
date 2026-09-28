"""
P-MCP SDK — Safety Middleware
==============================
The safety pipeline that runs between "actuation received" and "hardware moves".

This is P-MCP's killer feature vs plain MCP — three automatic safety layers:
  1. ConstitutionGuard — TEE-signed hard rules (ISO 10218 / IEC 62443)
  2. ShadowValidator   — ghost simulation before any hardware movement
  3. LeaseGuard        — temporal zone ownership to prevent collision

Usage:

    from pmcp.safety import SafetyMiddleware
    from pmcp.server import PMCPServer

    # Build middleware with constitution rules
    safety = SafetyMiddleware.default(robot_id="arm-01")

    server = PMCPServer("ur5-server", safety_middleware=safety)

    @server.actuation("move_to")
    async def move_to(x, y, z, speed=0.3):
        ...

You can also plug in your own shadow simulator:

    async def my_shadow(name, robot_id, args):
        # e.g. call PyBullet / MuJoCo / Isaac Sim
        safe = run_physics_simulation(name, args)
        return ShadowPreview(..., safe=safe)

    safety = SafetyMiddleware(shadow_fn=my_shadow)
"""
from __future__ import annotations

import hashlib
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from pmcp.types import (
    ConstitutionCheck, LeaseGrant, LeaseRequest,
    ShadowPreview, ShadowStatus,
)


# ─────────────────────────────────────────────────────────────────────────────
#  BUILT-IN CONSTITUTION RULES
# ─────────────────────────────────────────────────────────────────────────────

# Each rule: (id, description, check_fn(payload) -> Optional[str violation])
_BUILTIN_RULES: List[tuple] = [
    ("CONST-01", "Speed must not exceed 2.0 m/s",
     lambda p: f"Speed {p.get('speed',0):.2f} m/s > 2.0 limit"
               if float(p.get("speed", 0)) > 2.0 else None),

    ("CONST-02", "Target Z must be above floor (z ≥ 0.0 m)",
     lambda p: f"Z={p.get('z',1):.3f} below floor (0.0 m)"
               if float(p.get("z", 1)) < 0.0 else None),

    ("CONST-03", "Estimated energy must be ≤ 50,000 J",
     lambda p: f"energy_j={p.get('energy_j',0)} exceeds 50000 J budget"
               if float(p.get("energy_j", 0)) > 50_000 else None),

    ("CONST-04", "Shadow validation must be recent (< 2 s)",
     lambda p: "Shadow timestamp missing or stale (> 2 s old)"
               if (time.time() - float(p.get("_shadow_ts", 0))) > 2.0 else None),

    ("CONST-05", "Force must not exceed 500 N",
     lambda p: f"force_n={p.get('force_n',0)} exceeds 500 N limit"
               if float(p.get("force_n", 0)) > 500 else None),

    ("CONST-06", "Emergency stop bit must be clear",
     lambda p: "ESTOP active — all motion blocked"
               if bool(p.get("estop", False)) else None),

    ("CONST-07", "Human proximity: minimum 0.5 m clearance required",
     lambda p: f"Human at {p.get('human_dist_m', 99):.2f} m — too close (< 0.5 m)"
               if float(p.get("human_dist_m", 99)) < 0.5 else None),

    ("CONST-08", "Call ID must be present and ≥ 8 characters",
     lambda p: f"call_id '{p.get('call_id','')}' too short (need ≥ 8 chars)"
               if len(str(p.get("call_id", ""))) < 8 else None),
]


# ─────────────────────────────────────────────────────────────────────────────
#  SAFETY MIDDLEWARE
# ─────────────────────────────────────────────────────────────────────────────

class SafetyMiddleware:
    """
    Composable safety pipeline for P-MCP servers.

    Layers (in order):
      1. Constitution rules  — hard-coded or custom ruleset
      2. Shadow preview      — physics simulation of trajectory
      3. Lease guard         — temporal zone ownership

    All three must pass before hardware moves.
    """

    def __init__(
        self,
        rules:          Optional[List[tuple]] = None,
        shadow_fn:      Optional[Callable]    = None,
        lease_manager:  Optional[Any]         = None,
        robot_id:       str                   = "robot-01",
    ):
        self._rules        = rules if rules is not None else list(_BUILTIN_RULES)
        self._shadow_fn    = shadow_fn    # async (name, robot_id, args) -> ShadowPreview
        self._lease_mgr    = lease_manager
        self._robot_id     = robot_id
        self._rule_ids     = [r[0] for r in self._rules]
        self._fingerprint  = self._compute_fingerprint()

    @classmethod
    def default(cls, robot_id: str = "robot-01") -> "SafetyMiddleware":
        """Create a SafetyMiddleware with all built-in rules loaded."""
        return cls(rules=list(_BUILTIN_RULES), robot_id=robot_id)

    @classmethod
    def minimal(cls, robot_id: str = "robot-01") -> "SafetyMiddleware":
        """Only floor guard and speed limit — for constrained environments."""
        rules = [r for r in _BUILTIN_RULES if r[0] in ("CONST-01", "CONST-02", "CONST-06")]
        return cls(rules=rules, robot_id=robot_id)

    def add_rule(self, rule_id: str, description: str, check_fn: Callable) -> "SafetyMiddleware":
        """Add a custom rule to the constitution."""
        self._rules.append((rule_id, description, check_fn))
        self._rule_ids.append(rule_id)
        self._fingerprint = self._compute_fingerprint()
        return self

    # ── Constitution ──────────────────────────────────────────────────────────

    def check_constitution(self, payload: dict) -> Tuple[bool, List[str]]:
        """
        Evaluate all constitution rules against the payload.
        Returns (cleared: bool, violations: List[str]).
        """
        violations = []
        for rule_id, _desc, check_fn in self._rules:
            try:
                violation = check_fn(payload)
                if violation:
                    violations.append(f"[{rule_id}] {violation}")
            except Exception as e:
                violations.append(f"[{rule_id}] Rule evaluation error: {e}")
        return len(violations) == 0, violations

    def check_report(self, payload: dict) -> ConstitutionCheck:
        """Full ConstitutionCheck report."""
        cleared, violations = self.check_constitution(payload)
        return ConstitutionCheck(
            cleared     = cleared,
            violations  = violations,
            rule_ids    = self._rule_ids,
            fingerprint = self._fingerprint,
        )

    # ── Shadow ────────────────────────────────────────────────────────────────

    async def run_shadow(self, actuation_name: str, robot_id: str, args: dict) -> ShadowPreview:
        """Run shadow preview simulation."""
        if self._shadow_fn:
            return await self._shadow_fn(actuation_name, robot_id, args)
        return _builtin_shadow(actuation_name, robot_id, args)

    def set_shadow_simulator(self, fn: Callable) -> "SafetyMiddleware":
        """
        Plug in a custom shadow simulator (e.g. PyBullet, MuJoCo, Isaac Sim).

            async def pybullet_shadow(name, robot_id, args):
                safe, collisions = simulate_in_pybullet(name, args)
                return ShadowPreview(safe=safe, collisions=collisions, ...)

            safety.set_shadow_simulator(pybullet_shadow)
        """
        self._shadow_fn = fn
        return self

    # ── Lease ─────────────────────────────────────────────────────────────────

    async def request_lease(self, req: LeaseRequest) -> LeaseGrant:
        """Request a temporal zone lease via the attached lease manager."""
        if self._lease_mgr:
            # Use v04 LeaseManager if available
            ok = await self._lease_mgr.request_lease(
                req.robot_id, req.zone_id, req.duration_ms, req.bid_energy_j, max_tool_calls=100)
            return LeaseGrant(
                lease_id   = f"{req.zone_id}-{uuid.uuid4().hex[:8]}",
                robot_id   = req.robot_id,
                zone_id    = req.zone_id,
                granted    = ok,
                expires_at = time.time() + req.duration_ms / 1000.0,
                deny_reason = "" if ok else "Auction lost or zone occupied",
            )
        # Default: always grant
        return LeaseGrant(
            lease_id   = f"{req.zone_id}-{uuid.uuid4().hex[:8]}",
            robot_id   = req.robot_id,
            zone_id    = req.zone_id,
            granted    = True,
            expires_at = time.time() + req.duration_ms / 1000.0,
        )

    def release_lease(self, lease_id: str) -> bool:
        if self._lease_mgr and hasattr(self._lease_mgr, "release_lease"):
            return self._lease_mgr.release_lease(lease_id)
        return True

    # ── Fingerprint ───────────────────────────────────────────────────────────

    def _compute_fingerprint(self) -> str:
        rule_str = "|".join(r[0] + r[1] for r in self._rules)
        return hashlib.sha256(rule_str.encode()).hexdigest()[:16]

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def rule_count(self) -> int:
        return len(self._rules)

    def describe(self) -> dict:
        return {
            "ruleCount":   len(self._rules),
            "ruleIds":     self._rule_ids,
            "fingerprint": self._fingerprint,
            "robotId":     self._robot_id,
            "hasShadow":   self._shadow_fn is not None,
            "hasLeases":   self._lease_mgr is not None,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  BUILT-IN SHADOW (simple kinematic validator, no physics engine required)
# ─────────────────────────────────────────────────────────────────────────────

def _builtin_shadow(actuation_name: str, robot_id: str, args: dict) -> ShadowPreview:
    """
    Lightweight built-in shadow validator.
    Checks floor, speed, and simple joint reachability without a physics engine.
    For production, replace with PyBullet / MuJoCo / Isaac Sim via set_shadow_simulator().
    """
    violations: List[str] = []
    collisions: List[str] = []

    # Floor guard
    z = float(args.get("z", 1.0))
    if z < 0.0:
        violations.append(f"Z={z:.3f} below floor plane (z=0)")
        collisions.append("floor")

    # Speed guard
    speed = float(args.get("speed", 0.3))
    if speed > 1.5:
        violations.append(f"Speed={speed:.2f} m/s exceeds soft limit (1.5 m/s)")

    # Simple workspace check (1 m cube centred at origin, configurable)
    for ax, val, lo, hi in [
        ("x", float(args.get("x", 0)), -1.0, 1.0),
        ("y", float(args.get("y", 0)), -1.0, 1.0),
        ("z", float(args.get("z", 1)), 0.0,  1.2),
    ]:
        if not (lo <= val <= hi):
            violations.append(f"{ax}={val:.3f} outside workspace [{lo}, {hi}] m")

    safe = len(violations) == 0 and len(collisions) == 0

    predicted_pose = {k: float(args[k]) for k in ("x", "y", "z") if k in args}
    if "roll" in args:   predicted_pose["roll"]  = float(args["roll"])
    if "pitch" in args:  predicted_pose["pitch"] = float(args["pitch"])
    if "yaw" in args:    predicted_pose["yaw"]   = float(args["yaw"])

    return ShadowPreview(
        actuation_name = actuation_name,
        robot_id       = robot_id,
        status         = ShadowStatus.SAFE if safe else (
            ShadowStatus.COLLISION if collisions else ShadowStatus.UNSAFE),
        safe           = safe,
        predicted_pose = predicted_pose,
        duration_s     = 0.001,   # Built-in is near-instant
        collisions     = collisions,
        violations     = violations,
    )
