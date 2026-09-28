"""
P-MCP v0.5 — Recorded-Trajectory Replay Harness
================================================
Run a recorded joint-space or Cartesian trajectory through the full safety
pipeline (lease → constitution → shadow → execute) without any hardware.

This is the v0.5 substitute for "tested against a real robot": a recorded
trajectory contains the *exact* joint angles / TCP positions a real UR5e
would command during a mission. If the safety pipeline flags any of them
as unsafe, the robot would have crashed. If it accepts all of them, the
trajectory is at least pipeline-validated.

Format (JSON, line-delimited):
    {"t": 0.0, "actuation": "move_to", "args": {"x": 0.4, "y": 0.0, "z": 0.3, "speed": 0.3}}
    {"t": 0.5, "actuation": "move_to", "args": {"x": 0.5, "y": 0.1, "z": 0.3, "speed": 0.3}}
    ...
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from v05.pmcp_kinematics import ur5e
from v05.pmcp_safety_v5 import SafetyConstitution, SafetyMiddleware, ShadowSimulator


@dataclass
class TrajectoryStep:
    t: float
    actuation: str
    args: Dict[str, Any]
    note: str = ""


@dataclass
class ReplayResult:
    steps_total: int
    steps_passed: int
    steps_blocked: int
    first_blocked_step: Optional[int] = None
    block_reasons: List[str] = field(default_factory=list)
    wall_clock_s: float = 0.0

    @property
    def safe(self) -> bool:
        return self.steps_blocked == 0

    def summary(self) -> str:
        ok = "✅" if self.safe else "❌"
        return (
            f"{ok} Replay: {self.steps_passed}/{self.steps_total} steps passed, "
            f"{self.steps_blocked} blocked "
            f"in {self.wall_clock_s:.2f}s"
        )


def load_trajectory(path: str | Path) -> List[TrajectoryStep]:
    """Load a JSONL trajectory file."""
    steps: List[TrajectoryStep] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            steps.append(
                TrajectoryStep(
                    t=float(obj.get("t", 0.0)),
                    actuation=obj["actuation"],
                    args=obj.get("args", {}),
                    note=obj.get("note", ""),
                )
            )
    return steps


def save_trajectory(path: str | Path, steps: Iterable[TrajectoryStep]) -> None:
    """Write a trajectory to JSONL."""
    with open(path, "w") as f:
        for s in steps:
            f.write(
                json.dumps(
                    {
                        "t": s.t,
                        "actuation": s.actuation,
                        "args": s.args,
                        "note": s.note,
                    }
                )
                + "\n"
            )


# ── Built-in reference trajectories ────────────────────────────────────────


def reference_pick_and_place() -> List[TrajectoryStep]:
    """
    A canonical UR5e pick-and-place trajectory (Cartesian-space).
    Roughly 6 seconds, 11 waypoints. All positions inside the UR5e
    workspace sphere (~0.85 m reach from base).
    """
    pts = [
        (0.40, 0.10, 0.30, "approach pre-pick"),
        (0.40, 0.10, 0.20, "descend to pick"),
        (0.40, 0.10, 0.10, "grasp"),
        (0.40, 0.10, 0.30, "lift"),
        (0.30, 0.30, 0.40, "transport midpoint"),
        (0.20, 0.50, 0.30, "approach pre-place"),
        (0.20, 0.50, 0.20, "descend to place"),
        (0.20, 0.50, 0.10, "release"),
        (0.20, 0.50, 0.30, "retract"),
        (0.40, 0.10, 0.30, "return home"),
        (0.40, 0.10, 0.50, "safe pose"),
    ]
    return [
        TrajectoryStep(
            t=i * 0.5, actuation="move_to", args={"x": x, "y": y, "z": z, "speed": 0.3}, note=note
        )
        for i, (x, y, z, note) in enumerate(pts)
    ]


def reference_pour_cycle() -> List[TrajectoryStep]:
    """A 5-step pour cycle — all inside workspace, all slow."""
    return [
        TrajectoryStep(
            0.0, "move_to", {"x": 0.30, "y": -0.20, "z": 0.25, "speed": 0.2}, "grip cup"
        ),
        TrajectoryStep(1.0, "move_to", {"x": 0.30, "y": -0.20, "z": 0.40, "speed": 0.2}, "lift"),
        TrajectoryStep(
            2.0, "move_to", {"x": 0.10, "y": -0.30, "z": 0.35, "speed": 0.15}, "transport"
        ),
        TrajectoryStep(
            3.0, "move_to", {"x": 0.10, "y": -0.30, "z": 0.15, "speed": 0.15}, "lower to target"
        ),
        TrajectoryStep(
            4.0, "move_to", {"x": 0.10, "y": -0.30, "z": 0.30, "speed": 0.15}, "retract"
        ),
    ]


# ── Replay driver ──────────────────────────────────────────────────────────


def replay(
    steps: Sequence[TrajectoryStep],
    middleware: Optional[SafetyMiddleware] = None,
    verbose: bool = False,
) -> ReplayResult:
    """
    Run `steps` through the safety pipeline.

    The middleware is created with a UR5e kinematic chain by default
    (pure-stdlib, no numpy required). Pass your own middleware to
    override.
    """
    if middleware is None:
        mw = SafetyMiddleware(
            SafetyConstitution("replay"),
            ShadowSimulator(kinematic_chain=ur5e()),
        )
    else:
        mw = middleware

    t0 = time.time()
    passed = 0
    blocked = 0
    first_block = None
    reasons: List[str] = []

    for i, step in enumerate(steps):
        safe, _preview, violations = mw.check(step.actuation, step.args)
        if safe:
            passed += 1
            if verbose:
                print(f"  [{i:3d} t={step.t:.2f}] OK  {step.actuation}{step.args}  {step.note}")
        else:
            blocked += 1
            if first_block is None:
                first_block = i
            for v in violations:
                reasons.append(f"step {i}: {v}")
            if verbose:
                print(f"  [{i:3d} t={step.t:.2f}] ❌  {step.actuation}{step.args}  -> {violations}")

    return ReplayResult(
        steps_total=len(steps),
        steps_passed=passed,
        steps_blocked=blocked,
        first_blocked_step=first_block,
        block_reasons=reasons,
        wall_clock_s=time.time() - t0,
    )


def replay_pick_and_place(verbose: bool = True) -> ReplayResult:
    """Convenience: replay the canonical pick-and-place trajectory."""
    return replay(reference_pick_and_place(), verbose=verbose)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "pour":
        result = replay(reference_pour_cycle(), verbose=True)
    else:
        result = replay_pick_and_place(verbose=True)
    print()
    print(result.summary())
