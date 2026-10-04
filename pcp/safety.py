"""
P-MCP SDK — Safety Middleware
==============================
The safety pipeline that runs between "actuation received" and "hardware moves".

This is P-MCP's killer feature vs plain MCP — three automatic safety layers:
  1. ConstitutionGuard — TEE-signed hard rules (ISO 10218 / IEC 62443)
  2. ShadowValidator   — ghost simulation before any hardware movement
  3. LeaseGuard        — temporal zone ownership to prevent collision
  4. RateLimiter       — per-robot actuation frequency cap (v0.5)

Usage:

    from pcp.safety import SafetyMiddleware
    from pcp.server import PCPServer

    # Build middleware with constitution rules
    safety = SafetyMiddleware.default(robot_id="arm-01")

    # Or use a robot-class profile:
    safety = SafetyMiddleware.for_arm(robot_id="ur5-01")
    safety = SafetyMiddleware.for_mobile(robot_id="tb3-01")

    server = PCPServer("ur5-server", safety_middleware=safety)

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
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from pcp.types import (
    ConstitutionCheck,
    LeaseGrant,
    LeaseRequest,
    LeaseState,
    ShadowPreview,
    ShadowStatus,
)

# ─────────────────────────────────────────────────────────────────────────────
#  BUILT-IN CONSTITUTION RULES
# ─────────────────────────────────────────────────────────────────────────────

# Each rule: (id, description, check_fn(payload) -> Optional[str violation])
_BUILTIN_RULES: List[tuple] = [
    (
        "CONST-01",
        "Speed must not exceed 2.0 m/s",
        lambda p: (
            f"Speed {p.get('speed',0):.2f} m/s > 2.0 limit"
            if float(p.get("speed", 0)) > 2.0
            else None
        ),
    ),
    (
        "CONST-02",
        "Target Z must be above floor (z ≥ 0.0 m)",
        lambda p: (
            f"Z={p.get('z',1):.3f} below floor (0.0 m)" if float(p.get("z", 1)) < 0.0 else None
        ),
    ),
    (
        "CONST-03",
        "Estimated energy must be ≤ 50,000 J",
        lambda p: (
            f"energy_j={p.get('energy_j',0)} exceeds 50000 J budget"
            if float(p.get("energy_j", 0)) > 50_000
            else None
        ),
    ),
    (
        "CONST-04",
        "Shadow validation must be recent (< 2 s)",
        lambda p: (
            "Shadow timestamp missing or stale (> 2 s old)"
            if (time.time() - float(p.get("_shadow_ts", 0))) > 2.0
            else None
        ),
    ),
    (
        "CONST-05",
        "Force must not exceed 500 N",
        lambda p: (
            f"force_n={p.get('force_n',0)} exceeds 500 N limit"
            if float(p.get("force_n", 0)) > 500
            else None
        ),
    ),
    (
        "CONST-06",
        "Emergency stop bit must be clear",
        lambda p: "ESTOP active — all motion blocked" if bool(p.get("estop", False)) else None,
    ),
    (
        "CONST-07",
        "Human proximity: minimum 0.5 m clearance required",
        lambda p: (
            f"Human at {p.get('human_dist_m', 99):.2f} m — too close (< 0.5 m)"
            if float(p.get("human_dist_m", 99)) < 0.5
            else None
        ),
    ),
    (
        "CONST-08",
        "Call ID must be present and ≥ 8 characters",
        lambda p: (
            f"call_id '{p.get('call_id','')}' too short (need ≥ 8 chars)"
            if len(str(p.get("call_id", ""))) < 8
            else None
        ),
    ),
    (
        "CONST-09",
        "Joint angles must be within ±π rad",
        lambda p: (
            f"Joint angle(s) exceed ±π rad: "
            f"{[round(j, 3) for j in p.get('joint_angles', []) if abs(float(j)) > 3.14159]}"
            if any(abs(float(j)) > 3.14159 for j in p.get("joint_angles", []))
            else None
        ),
    ),
    (
        "CONST-10",
        "Workspace envelope: target must be within ±2.0 m on X and Y axes",
        lambda p: (
            "; ".join(
                f"{ax}={float(p.get(ax, 0)):.3f} outside [-2.0, 2.0] m"
                for ax in ("x", "y")
                if abs(float(p.get(ax, 0))) > 2.0
            )
            or None
        ),
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
#  RATE LIMITER
# ─────────────────────────────────────────────────────────────────────────────


class _RateLimiter:
    """
    Sliding-window rate limiter per (robot_id, actuation_name) pair.

    Prevents actuation storms that could damage hardware or exhaust energy budgets.
    Uses a 1-second sliding window by default.
    """

    def __init__(self, max_calls_per_second: float = 10.0, window_s: float = 1.0):
        self._max_cps = max_calls_per_second
        self._window = window_s
        self._history: Dict[str, List[float]] = defaultdict(list)

    def check(self, robot_id: str, actuation_name: str) -> Optional[str]:
        """
        Record a call attempt and return a violation string if rate is exceeded,
        or None if the call is within limits.
        """
        key = f"{robot_id}:{actuation_name}"
        now = time.time()

        # Purge timestamps outside the sliding window
        self._history[key] = [t for t in self._history[key] if now - t < self._window]

        if len(self._history[key]) >= self._max_cps:
            return (
                f"Rate limit exceeded for '{actuation_name}': "
                f"{len(self._history[key])} calls in last {self._window:.1f}s "
                f"(limit: {self._max_cps:.0f}/s)"
            )

        self._history[key].append(now)
        return None

    def reset(self, robot_id: str = "", actuation_name: str = ""):
        """Reset counters (for testing or after an e-stop clear)."""
        if robot_id and actuation_name:
            self._history.pop(f"{robot_id}:{actuation_name}", None)
        else:
            self._history.clear()

    def describe(self) -> dict:
        return {
            "maxCallsPerSecond": self._max_cps,
            "windowS": self._window,
            "trackedPairs": len(self._history),
        }


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
      4. Rate limiter        — per-robot actuation frequency cap

    All layers must pass before hardware moves.
    """

    def __init__(
        self,
        rules: Optional[List[tuple]] = None,
        shadow_fn: Optional[Callable] = None,
        lease_manager: Optional[Any] = None,
        robot_id: str = "robot-01",
        max_calls_per_second: float = 10.0,
        rate_limit_enabled: bool = True,
    ):
        self._rules = rules if rules is not None else list(_BUILTIN_RULES)
        self._shadow_fn = shadow_fn  # async (name, robot_id, args) -> ShadowPreview
        self._lease_mgr = lease_manager
        self._robot_id = robot_id
        self._rule_ids = [r[0] for r in self._rules]
        self._fingerprint = self._compute_fingerprint()
        self._rate_limiter: Optional[_RateLimiter] = (
            _RateLimiter(max_calls_per_second) if rate_limit_enabled else None
        )
        # Fencing tokens (Kleppmann 2016) -- monotonic per-zone counter,
        # bumped on every grant/renewal. See LeaseGrant.fence_token.
        self._fence_counters: Dict[str, int] = defaultdict(int)

    def _next_fence_token(self, zone_id: str) -> int:
        self._fence_counters[zone_id] += 1
        return self._fence_counters[zone_id]

    # ── Factory class methods ─────────────────────────────────────────────────

    @classmethod
    def default(cls, robot_id: str = "robot-01") -> "SafetyMiddleware":
        """Create a SafetyMiddleware with all built-in rules loaded."""
        return cls(rules=list(_BUILTIN_RULES), robot_id=robot_id)

    @classmethod
    def minimal(cls, robot_id: str = "robot-01") -> "SafetyMiddleware":
        """Only floor guard and speed limit — for constrained environments."""
        rules = [r for r in _BUILTIN_RULES if r[0] in ("CONST-01", "CONST-02", "CONST-06")]
        return cls(rules=rules, robot_id=robot_id)

    @classmethod
    def for_arm(
        cls,
        robot_id: str = "robot-01",
        max_speed_m_s: float = 1.5,
        max_force_n: float = 300.0,
        workspace_m: float = 1.0,
        max_calls_per_s: float = 5.0,
    ) -> "SafetyMiddleware":
        """
        Pre-configured profile for industrial robot arms (UR5, Kuka, ABB).

        Tighter speed and force limits than the generic default, plus a
        configurable workspace radius and stricter rate limiting.
        """
        base_rules = list(_BUILTIN_RULES)
        custom_rules = [
            (
                "ARM-01",
                f"Arm speed must not exceed {max_speed_m_s} m/s",
                lambda p, lim=max_speed_m_s: (
                    f"Speed {float(p.get('speed', 0)):.2f} m/s > arm limit {lim} m/s"
                    if float(p.get("speed", 0)) > lim
                    else None
                ),
            ),
            (
                "ARM-02",
                f"Arm force must not exceed {max_force_n} N",
                lambda p, lim=max_force_n: (
                    f"force_n={float(p.get('force_n', 0)):.1f} > arm limit {lim} N"
                    if float(p.get("force_n", 0)) > lim
                    else None
                ),
            ),
            (
                "ARM-03",
                f"Arm workspace radius must not exceed {workspace_m} m",
                lambda p, lim=workspace_m: (
                    f"Reach {(float(p.get('x',0))**2 + float(p.get('y',0))**2 + float(p.get('z',0))**2)**0.5:.3f} m "
                    f"> arm workspace {lim} m"
                    if (
                        float(p.get("x", 0)) ** 2
                        + float(p.get("y", 0)) ** 2
                        + float(p.get("z", 0)) ** 2
                    )
                    ** 0.5
                    > lim
                    else None
                ),
            ),
        ]
        return cls(
            rules=base_rules + custom_rules,
            robot_id=robot_id,
            max_calls_per_second=max_calls_per_s,
        )

    @classmethod
    def for_mobile(
        cls,
        robot_id: str = "robot-01",
        max_speed_m_s: float = 0.8,
        min_obstacle_m: float = 0.3,
        max_calls_per_s: float = 20.0,
    ) -> "SafetyMiddleware":
        """
        Pre-configured profile for mobile robots (TurtleBot, AMR, AGV).

        Lower speed limits, obstacle clearance rule, and higher call rate for
        continuous navigation commands.
        """
        base_rules = list(_BUILTIN_RULES)
        custom_rules = [
            (
                "MOB-01",
                f"Mobile robot speed must not exceed {max_speed_m_s} m/s",
                lambda p, lim=max_speed_m_s: (
                    f"Speed {float(p.get('linear_x', p.get('speed', 0))):.2f} m/s > mobile limit {lim} m/s"
                    if float(p.get("linear_x", p.get("speed", 0))) > lim
                    else None
                ),
            ),
            (
                "MOB-02",
                f"Minimum obstacle clearance {min_obstacle_m} m required",
                lambda p, lim=min_obstacle_m: (
                    f"Nearest obstacle at {float(p.get('obstacle_dist_m', 99)):.2f} m "
                    f"— minimum clearance {lim} m required"
                    if float(p.get("obstacle_dist_m", 99)) < lim
                    else None
                ),
            ),
        ]
        return cls(
            rules=base_rules + custom_rules,
            robot_id=robot_id,
            max_calls_per_second=max_calls_per_s,
        )

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
            cleared=cleared,
            violations=violations,
            rule_ids=self._rule_ids,
            fingerprint=self._fingerprint,
        )

    def check_rate_limit(self, robot_id: str, actuation_name: str) -> Optional[str]:
        """
        Check rate limit for a (robot_id, actuation_name) pair.
        Returns a violation string if rate is exceeded, or None if within limits.
        """
        if self._rate_limiter is None:
            return None
        return self._rate_limiter.check(robot_id, actuation_name)

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
                req.robot_id, req.zone_id, req.duration_ms, req.bid_energy_j, max_tool_calls=100
            )
            return LeaseGrant(
                lease_id=f"{req.zone_id}-{uuid.uuid4().hex[:8]}",
                robot_id=req.robot_id,
                zone_id=req.zone_id,
                state=LeaseState.ACTIVE if ok else LeaseState.DENIED,
                expires_at=time.time() + req.duration_ms / 1000.0 if ok else 0.0,
                deny_reason="" if ok else "Auction lost or zone occupied",
                fence_token=self._next_fence_token(req.zone_id) if ok else 0,
            )
        # Default: always grant
        return LeaseGrant(
            lease_id=f"{req.zone_id}-{uuid.uuid4().hex[:8]}",
            robot_id=req.robot_id,
            zone_id=req.zone_id,
            state=LeaseState.ACTIVE,
            expires_at=time.time() + req.duration_ms / 1000.0,
            fence_token=self._next_fence_token(req.zone_id),
        )

    def check_lease_fence(self, zone_id: str, fence_token: Optional[int]) -> Optional[str]:
        """
        Verify a presented fence token is still current for zone_id. Returns
        a violation string if stale, or None if OK / not tracked here.
        Only meaningful when using the default (no external lease_manager)
        fencing path -- an external lease_manager should perform its own
        equivalent check.
        """
        if self._lease_mgr or fence_token is None:
            return None
        current = self._fence_counters.get(zone_id, 0)
        if fence_token != current:
            return (
                f"Stale fence token for zone={zone_id}: presented "
                f"{fence_token}, current is {current}"
            )
        return None

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
            "ruleCount": len(self._rules),
            "ruleIds": self._rule_ids,
            "fingerprint": self._fingerprint,
            "robotId": self._robot_id,
            "hasShadow": self._shadow_fn is not None,
            "hasLeases": self._lease_mgr is not None,
            "rateLimit": self._rate_limiter.describe() if self._rate_limiter else None,
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
        ("z", float(args.get("z", 1)), 0.0, 1.2),
    ]:
        if not (lo <= val <= hi):
            violations.append(f"{ax}={val:.3f} outside workspace [{lo}, {hi}] m")

    safe = len(violations) == 0 and len(collisions) == 0

    predicted_pose = {k: float(args[k]) for k in ("x", "y", "z") if k in args}
    if "roll" in args:
        predicted_pose["roll"] = float(args["roll"])
    if "pitch" in args:
        predicted_pose["pitch"] = float(args["pitch"])
    if "yaw" in args:
        predicted_pose["yaw"] = float(args["yaw"])

    return ShadowPreview(
        actuation_name=actuation_name,
        robot_id=robot_id,
        status=(
            ShadowStatus.SAFE
            if safe
            else (ShadowStatus.COLLISION if collisions else ShadowStatus.UNSAFE)
        ),
        safe=safe,
        predicted_pose=predicted_pose,
        duration_s=0.001,  # Built-in is near-instant
        collisions=collisions,
        violations=violations,
    )
