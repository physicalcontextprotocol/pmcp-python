"""
P-MCP v0.5 — End-to-end integration test
=========================================
Drives a real PMCPServer through a real PMCPClient over the in-process
transport, verifying the full MCP handshake: initialize → tools/list →
tools/call, including the safety pipeline blocking an unsafe call.
"""
from __future__ import annotations

import asyncio
import unittest

from v05.pmcp_v5_server import PMCPServer
from v05.pmcp_v5_types import ActuationResult, SensorReading, SensorType
from v05.pmcp_v5_client import PMCPClient


def _make_server() -> PMCPServer:
    server = PMCPServer("test-robot-ip", robot_class="arm", model="MockArm")

    @server.actuation("move_to", description="Move TCP to XYZ")
    async def move_to(x: float, y: float, z: float, speed: float = 0.3):
        return ActuationResult(
            success=True,
            output={"x": x, "y": y, "z": z},
            energy_consumed_j=1.5,
            duration_s=0.1,
        )

    @server.sensor("joint_angles", sensor_type=SensorType.JOINT_STATES, unit="rad")
    async def joints():
        return SensorReading(
            sensor_name="joint_angles",
            robot_id=server.robot_id,
            value=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            unit="rad",
        )

    return server


class TestInProcessRoundtrip(unittest.TestCase):
    """Full MCP handshake over in-process transport (no subprocess)."""

    def test_initialize_tools_list_tools_call(self):
        server = _make_server()

        async def run():
            client = PMCPClient(client_name="test", client_version="0.0.1")
            await client.connect_server(server)
            try:
                # 1) initialize
                self.assertTrue(client._initialized)
                self.assertIsNotNone(client._server_info)
                self.assertEqual(client._server_info["name"], "test-robot-ip")

                # 2) tools/list
                tools = await client.list_tools()
                names = [t["name"] for t in tools]
                self.assertIn("move_to", names)

                # 3) tools/call (safe position → must succeed)
                result = await client.call_tool(
                    "move_to", {"x": 0.4, "y": 0.0, "z": 0.3, "speed": 0.1})
                # In-process transport returns the raw handler dict
                self.assertIn("content", result)
            finally:
                await client.close()

        asyncio.run(run())

    def test_shadow_blocks_unsafe_call(self):
        server = _make_server()

        async def run():
            client = PMCPClient()
            await client.connect_server(server)
            try:
                await client.initialize() if hasattr(client, "initialize") else None
                # z = -2.0 is below the floor guard — safety must block
                with self.assertRaises(Exception):
                    # Either an error-code exception or a blocked ActuationResult
                    await client.call_tool(
                        "move_to", {"x": 0.0, "y": 0.0, "z": -2.0, "speed": 0.1})
            finally:
                await client.close()

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
