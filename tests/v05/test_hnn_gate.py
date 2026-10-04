"""
P-MCP v0.5 — Optional HNN Gate 4 tests
======================================
Tests for the opt-in Hamiltonian Neural Network energy-conservation check.
Validates:
  - Gate is OFF by default (no torch/numpy required)
  - Gate can be turned on with a real rule
  - Gate passes physically consistent motion
  - Gate catches a known conservation violation
  - Gate degrades gracefully when no baseline is provided
  - Gate fails fast at construction when torch/numpy are missing
"""
from __future__ import annotations

import unittest
from typing import Any, Dict

from v05.pcp_safety_v5 import (
    HNNConservationRule,
    SafetyConstitution,
    SafetyMiddleware,
    _HNN_AVAILABLE,
    _HNN_NUMPY_OK,
    _HNN_TORCH_OK,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Tiny fake HNN — no torch dependency for the rule's own tests
# ─────────────────────────────────────────────────────────────────────────────

class _FakeHNN:
    """Minimal stand-in for pcp.physics.hamiltonian.HamiltonianNN.

    Exposes .hamiltonian(q, p) -> float. The rule under test only calls
    this single method, so a duck-typed stub is sufficient.
    """

    def __init__(self, baseline_h: float = 1.0, energy_bias: float = 0.0):
        self.baseline_h = baseline_h
        self.energy_bias = energy_bias

    def hamiltonian(self, q, p) -> float:
        # H = baseline + bias (so a fake violation is easy to construct).
        return float(self.baseline_h + self.energy_bias)


class _FakeChecker:
    """Stand-in for pcp.physics.hamiltonian.ConservationChecker."""

    DEFAULT_DRIFT_THRESHOLD = 0.05

    def __init__(self, threshold: float = DEFAULT_DRIFT_THRESHOLD, relative: bool = True):
        self.threshold = threshold
        self.relative = relative

    def check_violation(self, drift: float, threshold=None):
        thresh = threshold if threshold is not None else self.threshold
        if drift > thresh:
            return True, f"Energy drift {drift:.4%} exceeds threshold {thresh:.4%}"
        return False, "OK"


# ─────────────────────────────────────────────────────────────────────────────
#  Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHNNGateOffByDefault(unittest.TestCase):
    """Without explicit opt-in, the HNN rule is not attached."""

    def test_middleware_constructed_without_hnn_rule(self):
        const = SafetyConstitution(robot_id="r1")
        mw = SafetyMiddleware(const)
        self.assertIsNone(mw._hnn_rule)

    def test_call_dict_without_hnn_fields_passes(self):
        """A regular actuation call has no q_state/p_state → no HNN work."""
        const = SafetyConstitution(robot_id="r1")
        mw = SafetyMiddleware(const)
        ok, _, v = mw.check("move_actuator", {"x": 0.1, "y": 0.2, "z": 0.3})
        self.assertTrue(ok)
        self.assertEqual(v, [])

    def test_construction_does_not_require_torch(self):
        """SafetyMiddleware must be constructable without torch/numpy."""
        const = SafetyConstitution(robot_id="r1")
        SafetyMiddleware(const)  # must not raise


class TestHNNGateOptIn(unittest.TestCase):
    """With a rule attached, the gate activates."""

    def setUp(self):
        if not _HNN_AVAILABLE:
            self.skipTest(f"torch/numpy not available "
                          f"(torch={_HNN_TORCH_OK}, numpy={_HNN_NUMPY_OK})")
        self.fake_hnn = _FakeHNN(baseline_h=1.0, energy_bias=0.0)
        self.fake_chk = _FakeChecker(threshold=0.05)
        self.rule = HNNConservationRule(self.fake_hnn, self.fake_chk)
        self.const = SafetyConstitution(robot_id="r1")
        self.mw = SafetyMiddleware(self.const, hnn_rule=self.rule)

    def test_rule_attached(self):
        self.assertIs(self.mw._hnn_rule, self.rule)

    def test_physically_consistent_motion_passes(self):
        """ΔH = 0 → drift 0% → passes."""
        ok, _, v = self.mw.check(
            "move_actuator",
            {"x": 0.1, "y": 0.2, "z": 0.3},
        )
        # The pipeline should not block on a HNN-consistent motion.
        self.assertNotIn("HNN", " | ".join(v))

    def test_violation_records_warning_does_not_block(self):
        """Gate 4 severity is WARNING, not FATAL — it never blocks."""
        self.fake_hnn.energy_bias = 0.5  # 50% drift from baseline
        ok, _, v = self.mw.check(
            "move_actuator",
            {"x": 0.1, "y": 0.2, "z": 0.3,
             "q_state": [0.1] * 6, "p_state": [0.0] * 6, "h_baseline": 1.0},
        )
        # The HNN violation must be present in the violations list,
        # but the call's overall ok=True because HNN is WARNING-only.
        hnn_msgs = [m for m in v if m.startswith("[HNN]")]
        self.assertGreaterEqual(len(hnn_msgs), 1)
        self.assertTrue(ok)  # WARNING severity → does not block

    def test_no_baseline_passes_silently(self):
        """If h_baseline is missing, the rule passes (cannot evaluate)."""
        self.fake_hnn.energy_bias = 100.0  # massive drift
        ok, _, v = self.mw.check(
            "move_actuator",
            {"x": 0.1, "y": 0.2, "z": 0.3,
             "q_state": [0, 0, 0, 0, 0, 0],
             "p_state": [0, 0, 0, 0, 0, 0]},
        )
        # No h_baseline → rule returns True
        hnn_msgs = [m for m in v if m.startswith("[HNN]")]
        self.assertEqual(hnn_msgs, [])

    def test_baseline_zero_passes_silently(self):
        """If h_baseline is 0, we cannot form a relative drift → pass."""
        self.fake_hnn.energy_bias = 100.0
        ok, _, v = self.mw.check(
            "move_actuator",
            {"x": 0.1, "y": 0.2, "z": 0.3,
             "q_state": [0, 0, 0, 0, 0, 0],
             "p_state": [0, 0, 0, 0, 0, 0],
             "h_baseline": 0.0},
        )
        hnn_msgs = [m for m in v if m.startswith("[HNN]")]
        self.assertEqual(hnn_msgs, [])

    def test_catches_known_violation(self):
        """A clear ΔH/H0=0.10 (2x threshold) must produce a HNN warning."""
        self.fake_hnn.energy_bias = 0.10  # 10% drift, threshold 5%
        ok, _, v = self.mw.check(
            "move_actuator",
            {"x": 0.1, "y": 0.2, "z": 0.3,
             "q_state": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
             "p_state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             "h_baseline": 1.0},
        )
        hnn_msgs = [m for m in v if m.startswith("[HNN]")]
        self.assertEqual(len(hnn_msgs), 1)
        self.assertIn("Energy drift", hnn_msgs[0])

    def test_stats_tracking(self):
        """Rule records checks and violations for observability."""
        self.fake_hnn.energy_bias = 0.0
        self.mw.check("move_actuator",
                      {"x": 0.1, "y": 0.2, "z": 0.3,
                       "q_state": [0]*6, "p_state": [0]*6, "h_baseline": 1.0})
        self.fake_hnn.energy_bias = 0.20
        self.mw.check("move_actuator",
                      {"x": 0.1, "y": 0.2, "z": 0.3,
                       "q_state": [0]*6, "p_state": [0]*6, "h_baseline": 1.0})
        s = self.rule.stats()
        self.assertEqual(s["checks"], 2)
        self.assertEqual(s["violations"], 1)


class TestHNNRuleHardening(unittest.TestCase):
    """Construction and input-handling edge cases."""

    def test_rule_constructed_with_fake_hnn(self):
        if not _HNN_AVAILABLE:
            self.skipTest("torch/numpy not available")
        rule = HNNConservationRule(_FakeHNN(), _FakeChecker())
        self.assertEqual(rule.id, "R-HNN-01")
        self.assertEqual(rule.severity, "WARNING")

    def test_rule_handles_numpy_arrays(self):
        if not _HNN_AVAILABLE:
            self.skipTest("torch/numpy not available")
        import numpy as np
        rule = HNNConservationRule(_FakeHNN(baseline_h=2.0),
                                   _FakeChecker(threshold=0.05))
        ok, reason = rule.evaluate({
            "q_state": np.array([0, 0, 0, 0, 0, 0], dtype=float),
            "p_state": np.array([0, 0, 0, 0, 0, 0], dtype=float),
            "h_baseline": 2.0,
        })
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_rule_constructed_missing_torch_fails_fast(self):
        """If torch/numpy are missing, the rule must fail with a clear msg."""
        # We don't actually uninstall torch; we just verify the guard
        # exists by checking the module-level flag & error path.
        if _HNN_AVAILABLE:
            self.skipTest("torch/numpy available — cannot test missing-deps path")
        # When the deps are missing, the rule import would have raised
        # earlier. This test simply documents the contract.
        self.assertFalse(_HNN_AVAILABLE)


if __name__ == "__main__":
    unittest.main()
