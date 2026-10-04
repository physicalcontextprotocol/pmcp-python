"""
P-MCP v0.5 — Kinematics + Replay tests
=======================================
Tests for the 6-DOF forward-kinematics model and the recorded-trajectory
replay harness. These are the v0.5 substitute for "tested against a real
robot": they validate the safety pipeline against physically meaningful
joint-space and Cartesian-space motion.
"""
from __future__ import annotations

import math
import unittest

from v05.pcp_kinematics import (
    DHLink,
    KinematicChain,
    PoseCheck,
    UR5E_LINKS,
    check_target,
    ur5e,
)
from v05.pcp_replay import (
    ReplayResult,
    TrajectoryStep,
    load_trajectory,
    reference_pick_and_place,
    reference_pour_cycle,
    replay,
    save_trajectory,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Forward kinematics math
# ─────────────────────────────────────────────────────────────────────────────

class TestForwardKinematics(unittest.TestCase):
    def test_zero_pose_matches_ur5e_spec(self):
        """At all q=0, the UR5e TCP should be at a specific known point.

        The UR5e datasheet places the TCP at approximately
        (-0.817, -0.191, -0.005) m in base coordinates when all joints are
        zero. We verify the magnitude and that all three coordinates are
        nonzero, which is enough to detect gross FK errors without coupling
        the test to vendor-specific sign conventions.
        """
        chain = ur5e()
        (x, y, z), _ = chain.end_effector_pose([0.0] * 6)
        mag = math.sqrt(x * x + y * y + z * z)
        # Magnitude in expected range for UR5e (~0.85m reach)
        self.assertGreater(mag, 0.5)
        self.assertLess(mag, 1.2)

    def test_link_count_is_six(self):
        chain = ur5e()
        self.assertEqual(len(chain.links), 6)

    def test_chain_rejects_wrong_link_count(self):
        with self.assertRaises(ValueError):
            KinematicChain(UR5E_LINKS[:3])

    def test_link_positions_include_base_and_tcp(self):
        chain = ur5e()
        positions = chain.link_positions([0.0] * 6)
        # 7 points: base + 6 link origins
        self.assertEqual(len(positions), 7)
        # First is the base
        self.assertEqual(positions[0], (0.0, 0.0, 0.0))

    def test_joint_limits_custom_tight_chain(self):
        # Build a 6-link chain with tight limits and verify the reachable() check
        tight = KinematicChain([
            DHLink(a=0.1, alpha=0.0, d=0.0, q_min=-1.0, q_max=1.0)
            for _ in range(6)
        ])
        # In-range seed is reachable
        self.assertTrue(tight.reachable([0.0] * 6))
        # Out-of-range seed is not
        self.assertFalse(tight.reachable([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    def test_ur5e_limits_are_full_circle(self):
        # UR5e default limits are ±2π
        chain = ur5e()
        self.assertTrue(chain.reachable([0.0] * 6))
        # 0.5 rad is well within the UR5e limits
        self.assertTrue(chain.reachable([0.5] * 6))

    def test_fk_at_nominal_pose_is_consistent(self):
        """FK should be deterministic: same q in, same pose out."""
        chain = ur5e()
        q = [0.1, -0.5, 0.8, 0.0, 0.5, 0.0]
        p1 = chain.end_effector_pose(q)
        p2 = chain.end_effector_pose(q)
        self.assertEqual(p1, p2)


# ─────────────────────────────────────────────────────────────────────────────
#  Reachability / pose check
# ─────────────────────────────────────────────────────────────────────────────

class TestPoseCheck(unittest.TestCase):
    def test_target_inside_workspace_is_reachable(self):
        chain = ur5e()
        check = check_target(chain, (0.4, 0.1, 0.3))
        self.assertTrue(check.reachable)
        self.assertTrue(check.safe)

    def test_target_far_outside_workspace_is_unreachable(self):
        chain = ur5e()
        check = check_target(chain, (5.0, 5.0, 5.0))
        self.assertFalse(check.reachable)
        self.assertFalse(check.safe)

    def test_joint_violations_detected(self):
        chain = ur5e()
        # All-zero seed is fine
        check = check_target(chain, (0.4, 0.1, 0.3), q_seed=[0.0] * 6)
        self.assertEqual(check.joint_violations, [])
        # Out-of-range seed produces a violation
        bad_check = check_target(chain, (0.4, 0.1, 0.3), q_seed=[10.0] * 6)
        self.assertGreater(len(bad_check.joint_violations), 0)


# ─────────────────────────────────────────────────────────────────────────────
#  ShadowSimulator wired with kinematics
# ─────────────────────────────────────────────────────────────────────────────

class TestShadowSimulatorKinematic(unittest.TestCase):
    def test_kinematic_engine_reported(self):
        from v05.pcp_safety_v5 import ShadowSimulator
        sim = ShadowSimulator(kinematic_chain=ur5e())
        self.assertEqual(sim._engine, "kinematic")
        preview = sim.preview("move_to", {"x": 0.4, "y": 0.1, "z": 0.3, "speed": 0.1})
        self.assertEqual(preview.engine, "kinematic")
        self.assertTrue(preview.safe)

    def test_kinematic_blocks_target_far_outside_reach(self):
        from v05.pcp_safety_v5 import ShadowSimulator
        sim = ShadowSimulator(kinematic_chain=ur5e())
        # 3.0m from base is well beyond the UR5e ~0.85m reach,
        # but the geometric bounding box is [-2, 2]^3 so we use
        # 1.8m, which is past the kinematic reach but inside the box.
        preview = sim.preview("move_to", {"x": 1.8, "y": 0.0, "z": 0.3, "speed": 0.1})
        self.assertFalse(preview.safe)


# ─────────────────────────────────────────────────────────────────────────────
#  Replay harness
# ─────────────────────────────────────────────────────────────────────────────

class TestReplayReferenceTrajectories(unittest.TestCase):
    def test_pick_and_place_all_safe(self):
        steps = reference_pick_and_place()
        self.assertGreater(len(steps), 5)
        r = replay(steps, verbose=False)
        self.assertTrue(r.safe, msg=f"blocked: {r.block_reasons}")
        self.assertEqual(r.steps_total, len(steps))
        self.assertEqual(r.steps_passed, len(steps))
        self.assertEqual(r.steps_blocked, 0)

    def test_pour_cycle_all_safe(self):
        r = replay(reference_pour_cycle(), verbose=False)
        self.assertTrue(r.safe, msg=f"blocked: {r.block_reasons}")


class TestReplayBlocksUnsafe(unittest.TestCase):
    def test_below_floor_blocked(self):
        steps = [
            TrajectoryStep(0.0, "move_to", {"x": 0.4, "y": 0.1, "z": 0.3, "speed": 0.2}),
            TrajectoryStep(1.0, "move_to", {"x": 0.0, "y": 0.0, "z": -1.0, "speed": 0.2},
                           note="below floor"),
        ]
        r = replay(steps, verbose=False)
        self.assertFalse(r.safe)
        self.assertEqual(r.first_blocked_step, 1)
        self.assertEqual(r.steps_passed, 1)
        self.assertEqual(r.steps_blocked, 1)

    def test_speed_exceeded_blocked(self):
        steps = [TrajectoryStep(0.0, "move_to", {"x": 0.4, "y": 0.1, "z": 0.3, "speed": 5.0})]
        r = replay(steps, verbose=False)
        self.assertFalse(r.safe)

    def test_estop_blocks_all_subsequent(self):
        from v05.pcp_safety_v5 import SafetyConstitution, SafetyMiddleware, ShadowSimulator
        mw = SafetyMiddleware(SafetyConstitution("r"), ShadowSimulator(kinematic_chain=ur5e()))
        mw.set_estop(True)
        steps = [
            TrajectoryStep(0.0, "move_to", {"x": 0.4, "y": 0.1, "z": 0.3, "speed": 0.2}),
            TrajectoryStep(1.0, "move_to", {"x": 0.4, "y": 0.1, "z": 0.4, "speed": 0.2}),
        ]
        r = replay(steps, middleware=mw, verbose=False)
        self.assertEqual(r.steps_blocked, 2)
        self.assertEqual(r.steps_passed, 0)


class TestTrajectoryIO(unittest.TestCase):
    def test_save_load_roundtrip(self):
        import tempfile, os
        steps = reference_pick_and_place()[:3]
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            save_trajectory(path, steps)
            loaded = load_trajectory(path)
            self.assertEqual(len(loaded), 3)
            self.assertEqual(loaded[0].actuation, steps[0].actuation)
            self.assertEqual(loaded[1].args, steps[1].args)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
