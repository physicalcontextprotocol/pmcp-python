"""
PCP v0.5 — Coverage tests for the demo runner, registry, server handlers,
and client transport.
"""
from __future__ import annotations

import asyncio
import unittest


# ─────────────────────────────────────────────────────────────────────────────
#  Demo runner (main_pcp_v5.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestDemoRunner(unittest.TestCase):
    def test_main_function_runs_all_demos(self):
        from v05 import main_pcp_v5
        asyncio.run(main_pcp_v5.main("all"))

    def test_main_function_runs_arm_only(self):
        from v05 import main_pcp_v5
        asyncio.run(main_pcp_v5.main("arm"))

    def test_main_function_runs_mobile(self):
        from v05 import main_pcp_v5
        asyncio.run(main_pcp_v5.main("mobile"))

    def test_main_function_runs_agri(self):
        from v05 import main_pcp_v5
        asyncio.run(main_pcp_v5.main("agri"))

    def test_main_function_runs_multi(self):
        from v05 import main_pcp_v5
        asyncio.run(main_pcp_v5.main("multi"))


# ─────────────────────────────────────────────────────────────────────────────
#  PCPRegistry: heartbeat, summary, find filters, MCP format
# ─────────────────────────────────────────────────────────────────────────────

from v05.pcp_registry import PCPRegistry, RegistryEntry


def _entry(rid, cls="arm", model="m", location="lab"):
    return RegistryEntry(
        robot_id=rid, name=rid, robot_class=cls, model=model, location=location,
        transport="stdio", endpoint=f"stdio://{rid}",
        actuations=["move_to"], sensors=["joint_angles"],
    )


class TestRegistryCoverage(unittest.TestCase):
    def setUp(self):
        self.reg = PCPRegistry("test-registry")
        self.reg.register(_entry("arm-1", cls="arm", location="warehouse-a"))
        self.reg.register(_entry("arm-2", cls="arm", location="warehouse-b"))
        self.reg.register(_entry("mobile-1", cls="mobile", location="warehouse-a"))

    def test_summary(self):
        s = self.reg.summary()
        self.assertEqual(s["registry"], "test-registry")
        self.assertEqual(s["total"], 3)
        self.assertIn("by_class", s)

    def test_heartbeat_updates_last_ping(self):
        ok = self.reg.heartbeat("arm-1")
        self.assertTrue(ok)
        # Unknown id
        self.assertFalse(self.reg.heartbeat("nope"))

    def test_find_by_location(self):
        results = self.reg.find(location="warehouse-a")
        self.assertEqual(len(results), 2)

    def test_find_by_actuation(self):
        results = self.reg.find(actuation="move_to")
        self.assertEqual(len(results), 3)

    def test_find_by_tag(self):
        self.reg.get("arm-1").tags.append("critical")
        results = self.reg.find(tag="critical")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].robot_id, "arm-1")

    def test_list_all(self):
        self.assertEqual(len(self.reg.list_all()), 3)

    def test_to_mcp_format(self):
        f = self.reg.to_mcp_format()
        self.assertEqual(f["total"], 3)
        self.assertIn("servers", f)
        self.assertEqual(len(f["servers"]), 3)

    def test_to_mcp_server_entry(self):
        e = _entry("x", cls="arm", model="UR5e", location="lab")
        d = e.to_mcp_server_entry()
        self.assertEqual(d["id"], "x")
        self.assertIn("versionDetail", d)
        self.assertIn("packages", d)


# ─────────────────────────────────────────────────────────────────────────────
#  PCPServer: shadow/lease/estop/status/identity/constitution handlers
# ─────────────────────────────────────────────────────────────────────────────

from v05.pcp_v5_server import PCPServer
from v05.pcp_v5_types import ActuationResult, SensorReading, SensorType


def _server():
    s = PCPServer("test-server", robot_class="arm", model="UR5e")

    @s.actuation("move_to", description="Move TCP to XYZ")
    async def move_to(x: float, y: float, z: float, speed: float = 0.3):
        return ActuationResult(success=True, output={"x": x, "y": y, "z": z})

    @s.sensor("joint_angles", sensor_type=SensorType.JOINT_STATES, unit="rad")
    async def joints():
        return SensorReading(
            sensor_name="joint_angles", robot_id=s.robot_id,
            value=[0.0] * 6, unit="rad",
        )

    @s.mission("pick_and_place", description="Pick and place")
    async def mission():
        return "Mission template"

    return s


class TestServerHandlers(unittest.TestCase):
    def test_ping(self):
        s = _server()
        async def run():
            return await s._h_ping({})
        r = asyncio.run(run())
        self.assertTrue(r["pong"])

    def test_shadow_preview(self):
        s = _server()
        async def run():
            return await s._h_shadow_preview({
                "name": "move_to",
                "arguments": {"x": 0.4, "y": 0.0, "z": 0.3, "speed": 0.1},
            })
        r = asyncio.run(run())
        self.assertIn("preview", r)

    def test_lease_request(self):
        s = _server()
        async def run():
            return await s._h_lease_request({
                "robotId": "test-server",
                "zoneId": "z1",
                "durationMs": 5000,
                "bidEnergyJ": 10.0,
            })
        r = asyncio.run(run())
        self.assertIn("lease", r)
        self.assertEqual(r["lease"]["zoneId"], "z1")

    def test_lease_release(self):
        s = _server()
        async def run():
            return await s._h_lease_release({"leaseId": "nonexistent"})
        r = asyncio.run(run())
        self.assertIn("released", r)

    def test_estop_engage_and_disengage(self):
        s = _server()
        async def run():
            engage = await s._h_estop({"active": True})
            disengage = await s._h_estop({"active": False})
            return engage, disengage
        e, d = asyncio.run(run())
        self.assertTrue(e["estop"])
        self.assertFalse(d["estop"])

    def test_status(self):
        s = _server()
        async def run():
            return await s._h_status({})
        r = asyncio.run(run())
        self.assertIn("robot_id", r)

    def test_identity(self):
        s = _server()
        async def run():
            return await s._h_identity({})
        r = asyncio.run(run())
        # identity.to_dict() returns class/model/serial/location etc.
        self.assertIn("class", r)
        self.assertIn("did", r)
        self.assertEqual(r["class"], "arm")

    def test_constitution(self):
        s = _server()
        async def run():
            return await s._h_constitution({})
        r = asyncio.run(run())
        self.assertIn("robot_id", r)
        self.assertGreaterEqual(r["rule_count"], 6)

    def test_resources_list(self):
        s = _server()
        async def run():
            return await s._h_resources_list({})
        r = asyncio.run(run())
        self.assertIn("resources", r)

    def test_resources_read(self):
        s = _server()
        async def run():
            return await s._h_resources_read({"uri": "pcp://test-server/sensors/joint_angles"})
        r = asyncio.run(run())
        self.assertIn("contents", r)

    def test_prompts_list(self):
        s = _server()
        async def run():
            return await s._h_prompts_list({})
        r = asyncio.run(run())
        self.assertIn("prompts", r)

    def test_prompts_get(self):
        s = _server()
        async def run():
            return await s._h_prompts_get({
                "name": "pick_and_place",
                "arguments": {},
            })
        r = asyncio.run(run())
        self.assertIn("messages", r)

    def test_tools_call_safe(self):
        s = _server()
        async def run():
            return await s._h_tools_call({
                "name": "move_to",
                "arguments": {"x": 0.4, "y": 0.0, "z": 0.3, "speed": 0.1},
            })
        r = asyncio.run(run())
        self.assertIn("content", r)

    def test_tools_call_unknown_actuation(self):
        from v05.pcp_v5_types import PCPError
        s = _server()
        async def run():
            return await s._h_tools_call({"name": "nope", "arguments": {}})
        with self.assertRaises(PCPError):
            asyncio.run(run())

    def test_logging_set_level(self):
        s = _server()
        async def run():
            return await s._h_logging_set_level({"level": "debug"})
        r = asyncio.run(run())
        # Handler returns empty dict (side-effect is on the logger)
        self.assertIsInstance(r, dict)


# ─────────────────────────────────────────────────────────────────────────────
#  PCPClient: list_sensors, read_sensor, list_missions, get_mission, ping
# ─────────────────────────────────────────────────────────────────────────────

from v05.pcp_v5_client import PCPClient


class TestClientCoverage(unittest.TestCase):
    def test_list_sensors(self):
        server = _server()
        async def run():
            client = PCPClient()
            await client.connect_server(server)
            try:
                return await client.list_sensors()
            finally:
                await client.close()
        sensors = asyncio.run(run())
        self.assertGreater(len(sensors), 0)
        self.assertIn("joint_angles", [s["name"] for s in sensors])

    def test_read_sensor(self):
        server = _server()
        async def run():
            client = PCPClient()
            await client.connect_server(server)
            try:
                return await client.read_sensor("joint_angles")
            finally:
                await client.close()
        r = asyncio.run(run())
        self.assertIn("contents", r)

    def test_list_missions(self):
        server = _server()
        async def run():
            client = PCPClient()
            await client.connect_server(server)
            try:
                return await client.list_missions()
            finally:
                await client.close()
        missions = asyncio.run(run())
        self.assertGreater(len(missions), 0)

    def test_get_mission(self):
        server = _server()
        async def run():
            client = PCPClient()
            await client.connect_server(server)
            try:
                return await client.get_mission("pick_and_place", arguments={})
            finally:
                await client.close()
        r = asyncio.run(run())
        self.assertIn("messages", r)

    def test_ping(self):
        server = _server()
        async def run():
            client = PCPClient()
            await client.connect_server(server)
            try:
                return await client.ping()
            finally:
                await client.close()
        r = asyncio.run(run())
        self.assertTrue(r["pong"])


if __name__ == "__main__":
    unittest.main()
