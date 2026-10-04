"""
PCP v0.5 SDK — Unit Tests
============================
Tests for the shipping v0.5 surface:
  - PCPServer construction and decorator API
  - SafetyConstitution (all 6 default rules)
  - ShadowSimulator (geometric fallback)
  - SafetyMiddleware.check() (constitution → shadow → execute pipeline)
  - PCPRegistry
  - Public types (ActuationResult, SensorReading, etc.)
"""
from __future__ import annotations

import asyncio
import unittest

from v05.pcp_v5_server import PCPServer, _LeaseManager
from v05.pcp_v5_types import (
    ActuationResult,
    Capabilities,
    LeaseRequest,
    LeaseState,
    MCP_VERSION,
    PCP_VERSION,
    SensorReading,
    SensorType,
    ShadowPreview,
    ShadowStatus,
)
from v05.pcp_safety_v5 import (
    SafetyConstitution,
    SafetyMiddleware,
    ShadowSimulator,
)
from v05.pcp_registry import PCPRegistry, RegistryEntry


# ─────────────────────────────────────────────────────────────────────────────
#  Types
# ─────────────────────────────────────────────────────────────────────────────

class TestTypes(unittest.TestCase):
    def test_actuation_result_minimal(self):
        r = ActuationResult(success=True)
        self.assertTrue(r.success)
        self.assertEqual(r.output, {})
        self.assertEqual(r.error_message, "")

    def test_actuation_result_with_output(self):
        r = ActuationResult(success=True, output={"x": 1.0, "y": 2.0})
        self.assertEqual(r.output["x"], 1.0)

    def test_sensor_reading_validates(self):
        s = SensorReading(
            sensor_name="joint_angles",
            robot_id="ur5-arm-01",
            value=[0.0, -1.57, 0.0],
            unit="rad",
        )
        self.assertEqual(s.sensor_name, "joint_angles")
        self.assertEqual(s.unit, "rad")
        self.assertEqual(len(s.value), 3)

    def test_version_constants_present(self):
        self.assertTrue(PCP_VERSION.startswith("0.5"))
        self.assertTrue(MCP_VERSION)


# ─────────────────────────────────────────────────────────────────────────────
#  Safety Constitution
# ─────────────────────────────────────────────────────────────────────────────

class TestSafetyConstitution(unittest.TestCase):
    def test_default_rules_loaded(self):
        c = SafetyConstitution("robot-1")
        summary = c.summary()
        self.assertGreaterEqual(summary["rule_count"], 6)
        self.assertIn("fingerprint", summary)
        self.assertEqual(summary["robot_id"], "robot-1")

    def test_fingerprint_stable(self):
        a = SafetyConstitution("robot-1")
        b = SafetyConstitution("robot-1")
        self.assertEqual(a.fingerprint, b.fingerprint)

    def test_evaluate_safe_call(self):
        c = SafetyConstitution("robot-1")
        ok, violations = c.evaluate({
            "actuation_name": "move_to",
            "x": 0.4, "y": 0.0, "z": 0.3,
            "max_speed": 0.3,
            "max_force": 50.0,
            "max_energy": 100.0,
        })
        self.assertTrue(ok, msg=f"violations: {violations}")
        self.assertEqual(violations, [])

    def test_evaluate_returns_violation_messages(self):
        c = SafetyConstitution("robot-1")
        ok, violations = c.evaluate({
            "actuation_name": "move_to",
            "x": 0.0, "y": 0.0, "z": -10.0,  # below floor
        })
        self.assertFalse(ok)
        self.assertGreater(len(violations), 0)

    def test_estop_blocks_via_set_estop(self):
        c = SafetyConstitution("robot-1")
        c.set_estop(True)
        ok, violations = c.evaluate({"actuation_name": "move_to"})
        self.assertFalse(ok)
        self.assertTrue(any("ESTOP" in v or "stop" in v.lower() for v in violations))


# ─────────────────────────────────────────────────────────────────────────────
#  Shadow Simulator
# ─────────────────────────────────────────────────────────────────────────────

class TestShadowSimulator(unittest.TestCase):
    def test_safe_position_is_safe(self):
        sim = ShadowSimulator()
        preview = sim.preview("move_to", {"x": 0.4, "y": 0.0, "z": 0.3, "speed": 0.1})
        self.assertIsInstance(preview, ShadowPreview)
        self.assertTrue(preview.safe)
        self.assertGreaterEqual(preview.est_duration_s, 0.0)

    def test_unsafe_position_below_floor(self):
        sim = ShadowSimulator()
        preview = sim.preview("move_to", {"x": 0.0, "y": 0.0, "z": -1.0})
        self.assertFalse(preview.safe)


# ─────────────────────────────────────────────────────────────────────────────
#  Safety Middleware
# ─────────────────────────────────────────────────────────────────────────────

class TestSafetyMiddleware(unittest.TestCase):
    def test_check_passes_through_safe_actuation(self):
        mw = SafetyMiddleware(SafetyConstitution("r1"), ShadowSimulator())
        safe, preview, violations = mw.check(
            "move_to", {"x": 0.4, "y": 0.0, "z": 0.3, "speed": 0.1})
        self.assertTrue(safe, msg=f"violations: {violations}")
        self.assertIsNotNone(preview)

    def test_check_blocks_unsafe_actuation(self):
        mw = SafetyMiddleware(SafetyConstitution("r1"), ShadowSimulator())
        safe, preview, violations = mw.check(
            "move_to", {"x": 0.0, "y": 0.0, "z": -10.0})
        self.assertFalse(safe)
        self.assertGreater(len(violations), 0)

    def test_estop_blocks_all_calls(self):
        mw = SafetyMiddleware(SafetyConstitution("r1"), ShadowSimulator())
        mw.set_estop(True)
        safe, _, violations = mw.check(
            "move_to", {"x": 0.4, "y": 0.0, "z": 0.3})
        self.assertFalse(safe)
        self.assertTrue(any("stop" in v.lower() for v in violations))

    def test_stats_increment(self):
        mw = SafetyMiddleware(SafetyConstitution("r1"), ShadowSimulator())
        mw.check("move_to", {"x": 0.4, "y": 0.0, "z": 0.3})
        mw.check("move_to", {"x": 0.0, "y": 0.0, "z": -10.0})
        self.assertEqual(mw.stats["calls"], 2)
        self.assertEqual(mw.stats["blocked"], 1)


class TestLeaseManager(unittest.TestCase):
    def test_grant_first_request(self):
        mgr = _LeaseManager()
        req = LeaseRequest(robot_id="r1", zone_id="z1", duration_ms=5000, bid_energy_j=10.0)
        grant = mgr.request(req)
        self.assertEqual(grant.state, LeaseState.ACTIVE)
        self.assertEqual(grant.robot_id, "r1")

    def test_lower_bid_denied(self):
        mgr = _LeaseManager()
        mgr.request(LeaseRequest(robot_id="r1", zone_id="z1", duration_ms=5000, bid_energy_j=100.0))
        grant2 = mgr.request(
            LeaseRequest(robot_id="r2", zone_id="z1", duration_ms=5000, bid_energy_j=10.0))
        self.assertEqual(grant2.state, LeaseState.DENIED)

    def test_higher_bid_evicts(self):
        mgr = _LeaseManager()
        mgr.request(LeaseRequest(robot_id="r1", zone_id="z1", duration_ms=5000, bid_energy_j=10.0))
        grant2 = mgr.request(
            LeaseRequest(robot_id="r2", zone_id="z1", duration_ms=5000, bid_energy_j=100.0))
        self.assertEqual(grant2.state, LeaseState.ACTIVE)

    def test_release(self):
        mgr = _LeaseManager()
        grant = mgr.request(
            LeaseRequest(robot_id="r1", zone_id="z1", duration_ms=5000, bid_energy_j=10.0))
        self.assertTrue(mgr.release(grant.lease_id))

    def test_fence_token_issued_and_monotonic(self):
        # Fence tokens are scoped per-zone (the guarantee is "this is the
        # current view of THIS zone's ownership", not a global ordering
        # across unrelated zones), so monotonicity is only meaningful
        # within the same zone.
        mgr = _LeaseManager()
        g1 = mgr.request(LeaseRequest(robot_id="r1", zone_id="z1", duration_ms=5000, bid_energy_j=10.0))
        self.assertGreater(g1.fence_token, 0)
        # Different robot outbids for the SAME zone -> new grant, same
        # zone's counter continues incrementing.
        g2 = mgr.request(LeaseRequest(robot_id="r2", zone_id="z1", duration_ms=5000, bid_energy_j=20.0))
        self.assertGreater(g2.fence_token, g1.fence_token)
        # A different, unrelated zone independently starts its own
        # per-zone counter at 1 -- this is correct, not a bug.
        g3 = mgr.request(LeaseRequest(robot_id="r3", zone_id="z2", duration_ms=5000, bid_energy_j=10.0))
        self.assertGreater(g3.fence_token, 0)

    def test_fence_token_bumped_on_renewal(self):
        # Kleppmann 2016: a renewal must invalidate a caller's stale view of
        # ownership, even though lease_id (and lexical validity) don't change.
        mgr = _LeaseManager()
        req = LeaseRequest(robot_id="r1", zone_id="z1", duration_ms=5000, bid_energy_j=10.0)
        g1 = mgr.request(req)
        stale_fence = g1.fence_token
        g2 = mgr.request(req)  # same robot, same zone -> renewal path
        self.assertEqual(g1.lease_id, g2.lease_id)       # lease_id unchanged
        self.assertGreater(g2.fence_token, stale_fence)  # fence token bumped

        ok, _ = mgr.check(g2.lease_id, "z1", fence_token=g2.fence_token)
        self.assertTrue(ok)
        ok, reason = mgr.check(g2.lease_id, "z1", fence_token=stale_fence)
        self.assertFalse(ok)
        self.assertIn("Stale fence token", reason)

    def test_fence_token_rejects_stale_after_re_grant(self):
        mgr = _LeaseManager()
        g1 = mgr.request(LeaseRequest(robot_id="r1", zone_id="z1", duration_ms=5000, bid_energy_j=10.0))
        mgr.release(g1.lease_id)
        g2 = mgr.request(LeaseRequest(robot_id="r2", zone_id="z1", duration_ms=5000, bid_energy_j=10.0))
        # r1 presents its old lease_id + fence_token after r2 now holds the zone.
        ok, reason = mgr.check(g1.lease_id, "z1", fence_token=g1.fence_token)
        self.assertFalse(ok)  # lease_id mismatch already catches this
        # Even if a caller somehow held a matching lease_id (shouldn't happen
        # since lease_id is regenerated per grant, but fence check is
        # defense-in-depth), a stale fence_token alone is also rejected:
        ok2, reason2 = mgr.check(g2.lease_id, "z1", fence_token=g1.fence_token)
        self.assertFalse(ok2)
        self.assertIn("Stale fence token", reason2)

    def test_check_without_lease_denies(self):
        mgr = _LeaseManager()
        ok, reason = mgr.check(lease_token=None, zone_id="z1")
        self.assertFalse(ok)
        self.assertIn("No lease", reason)


# ─────────────────────────────────────────────────────────────────────────────
#  PCPServer
# ─────────────────────────────────────────────────────────────────────────────

class TestPCPServer(unittest.TestCase):
    def test_constructs_with_defaults(self):
        s = PCPServer("ur5-arm-01")
        self.assertEqual(s.name, "ur5-arm-01")
        self.assertEqual(s.version, "1.0.0")
        self.assertEqual(s.robot_id, "ur5-arm-01")

    def test_identity_generated(self):
        s = PCPServer("ur5-arm-01", robot_class="arm", model="UR5e", serial="ABC123")
        self.assertEqual(s.identity.robot_class, "arm")
        self.assertEqual(s.identity.model, "UR5e")

    def test_actuation_decorator_registers(self):
        s = PCPServer("ur5-arm-01")

        @s.actuation("move_to", description="Move TCP to XYZ")
        async def move_to(x: float, y: float, z: float):
            return ActuationResult(success=True, output={"x": x, "y": y, "z": z})

        self.assertIn("move_to", s._actuations)
        spec = s._actuations["move_to"].spec
        self.assertEqual(spec.name, "move_to")
        self.assertEqual(spec.description, "Move TCP to XYZ")

    def test_sensor_decorator_registers(self):
        s = PCPServer("ur5-arm-01")

        @s.sensor("joint_angles", description="Joint angles",
                  sensor_type=SensorType.JOINT_STATES, unit="rad")
        async def joints():
            return SensorReading(
                sensor_name="joint_angles",
                robot_id=s.robot_id,
                value=[0.0, -1.57, 0.0],
                unit="rad",
            )

        self.assertIn("joint_angles", s._sensors)

    def test_mission_decorator_registers(self):
        s = PCPServer("ur5-arm-01")

        @s.mission("pick_and_place", description="Pick and place")
        async def mission():
            return "Mission template"

        self.assertIn("pick_and_place", s._missions)

    def test_initialize_handler_returns_protocol_metadata(self):
        s = PCPServer("ur5-arm-01")

        async def run():
            return await s._h_initialize({
                "protocolVersion": MCP_VERSION,
                "clientInfo": {"name": "test", "version": "1.0"},
                "capabilities": {},
            })

        resp = asyncio.run(run())
        self.assertEqual(resp["protocolVersion"], MCP_VERSION)
        self.assertIn("serverInfo", resp)
        self.assertIn("capabilities", resp)
        self.assertIn("pcp", resp)
        self.assertEqual(resp["pcp"]["version"], PCP_VERSION)
        self.assertEqual(resp["pcp"]["robotId"], "ur5-arm-01")

    def test_tools_list_returns_registered_actuations(self):
        s = PCPServer("ur5-arm-01")

        @s.actuation("move_to", description="Move")
        async def move_to(x: float):
            return ActuationResult(success=True)

        async def run():
            return await s._h_tools_list({})

        resp = asyncio.run(run())
        tool_names = [t["name"] for t in resp["tools"]]
        self.assertIn("move_to", tool_names)


# ─────────────────────────────────────────────────────────────────────────────
#  Registry
# ─────────────────────────────────────────────────────────────────────────────

class TestRegistry(unittest.TestCase):
    def _entry(self, rid, cls="arm"):
        return RegistryEntry(
            robot_id=rid, name=rid, robot_class=cls, model="m", location="lab",
            transport="stdio", endpoint=f"stdio://{rid}",
            actuations=["move_to"], sensors=["joint_angles"],
        )

    def test_register_and_lookup(self):
        reg = PCPRegistry()
        reg.register(self._entry("ur5-arm-01"))
        found = reg.get("ur5-arm-01")
        self.assertIsNotNone(found)
        self.assertEqual(found.robot_id, "ur5-arm-01")

    def test_list_by_class(self):
        reg = PCPRegistry()
        reg.register(self._entry("arm-1", cls="arm"))
        reg.register(self._entry("mobile-1", cls="mobile"))
        arms = reg.find(robot_class="arm")
        self.assertEqual(len(arms), 1)
        self.assertEqual(arms[0].robot_id, "arm-1")

    def test_deregister(self):
        reg = PCPRegistry()
        reg.register(self._entry("x"))
        reg.deregister("x")
        self.assertIsNone(reg.get("x"))

    def test_to_dict_roundtrip(self):
        reg = PCPRegistry()
        reg.register(self._entry("a"))
        d = reg.get("a").to_dict()
        self.assertEqual(d["id"], "a")
        self.assertEqual(d["class"], "arm")
        self.assertIn("actuations", d)


if __name__ == "__main__":
    unittest.main()
