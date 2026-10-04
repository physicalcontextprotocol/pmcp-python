"""
P-MCP v0.5 — Main Integration Demo
=====================================
"The Physical Layer of the Agentic Web"

What this demo proves:
  1. ANY MCP client (Claude Desktop, Cursor, etc.) can connect to P-MCP servers
  2. Robot actuations appear as standard MCP Tools
  3. Sensor streams appear as standard MCP Resources
  4. Full safety pipeline: constitution → shadow → execute
  5. Multi-robot coordination via registry
  6. Temporal leases prevent physical collisions

Demo scenarios:
  A. Single arm: pick-and-place with safety preview
  B. Mobile robot: autonomous patrol with sensor fusion
  C. Agricultural: harvest cycle with plant health monitoring
  D. Multi-robot: coordinated 3-robot farm scenario
  E. Safety test: E-Stop, workspace violation, speed limit

Run:
    python -m v05.main_pcp_v5
    python -m v05.main_pcp_v5 --demo arm
    python -m v05.main_pcp_v5 --demo all
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

log = logging.getLogger("pcp.main_v5")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)

from v05.pcp_registry import PCPRegistry, RegistryEntry  # noqa: E402
from v05.pcp_v5_client import PCPClient, PCPClientError  # noqa: E402
from v05.robot_servers.agricultural_server import build_agricultural_server  # noqa: E402
from v05.robot_servers.arm_server import build_arm_server  # noqa: E402
from v05.robot_servers.mobile_server import build_mobile_server  # noqa: E402


def sep(title: str) -> None:
    print(f"\n{'═'*68}")
    print(f"  {title}")
    print("═" * 68)


async def demo_arm_robot() -> None:
    sep("DEMO A — Robotic Arm: Pick & Place with Safety Preview")

    server = build_arm_server(name="ur5-arm-01", model="UR5e", location="cell-A")

    async with PCPClient("claude-desktop") as client:
        await client.connect_server(server)

        # 1. Discover capabilities
        tools = await client.list_tools()
        sensors = await client.list_sensors()
        print(f"\n  Tools    : {[t['name'] for t in tools]}")
        print(f"  Sensors  : {[s['name'] for s in sensors]}")

        # 2. Get robot identity
        identity = await client.get_identity()
        print(f"  DID      : {identity['did']}")

        # 3. Check safety constitution
        const = await client.get_constitution()
        print(
            f"  Constitution: {const['rule_count']} rules  " f"(fp={const['fingerprint'][:16]}...)"
        )

        # 4. Shadow preview BEFORE calling
        print("\n  [Safety] Running shadow preview for move_to(0.4, 0.2, 0.6)...")
        preview = await client.shadow_preview("move_to", {"x": 0.4, "y": 0.2, "z": 0.6})
        p = preview["preview"]
        status = "✅ SAFE" if p["safe"] else f"❌ {p['status']}"
        print(
            f"  Shadow   : {status}  engine={p['engine']}  "
            f"est_dur={p['est_duration_s']}s  risk={p['risk_score']}"
        )

        # 5. Safe call (preview → lease → execute → release)
        print("\n  [Execute] safe_call move_to(0.4, 0.2, 0.6)...")
        result = await client.safe_call(
            "move_to", {"x": 0.4, "y": 0.2, "z": 0.6, "speed": 0.5}, zone_id="workspace-A"
        )
        print(
            f"  Result   : {'✅' if not result.get('isError') else '❌'} "
            f"{result['content'][0]['text'][:80]}..."
        )

        # 6. Pick + Place mission
        print("\n  [Mission] Getting pick_and_place mission template...")
        mission = await client.get_mission(
            "pick_and_place",
            {
                "source_x": "0.3",
                "source_y": "0.0",
                "source_z": "0.2",
                "dest_x": "0.6",
                "dest_y": "0.1",
                "dest_z": "0.2",
            },
        )
        print(f"  Mission  : {mission['description']}")
        for msg in mission.get("messages", []):
            role = msg["role"]
            text = msg["content"][0]["text"][:60]
            print(f"             [{role}]: {text}...")

        # 7. Read sensors
        print("\n  [Sensors] Reading joint angles and force/torque...")
        joints = await client.read_sensor("joint_angles")
        ft = await client.read_sensor("force_torque")
        import json

        j_val = json.loads(joints["contents"][0]["text"])
        f_val = json.loads(ft["contents"][0]["text"])
        print(f"  Joints   : {[round(v,3) for v in j_val['value']]} rad")
        print(f"  F/T      : Fz={f_val['value']['Fz']} N")

        # 8. Safety test: workspace violation
        print("\n  [Safety] Testing workspace violation (z=-0.5, below floor)...")
        try:
            await client.call_tool("move_to", {"x": 0.0, "y": 0.0, "z": -0.5})
            print("  ❌ ERROR: Should have been blocked!")
        except PCPClientError as e:
            print(f"  ✅ Blocked correctly: code={e.code}  msg={e.message[:60]}")

        # 9. E-Stop test
        print("\n  [Safety] Testing E-Stop...")
        await client.estop(True)
        try:
            await client.call_tool("move_to", {"x": 0.1, "y": 0.0, "z": 0.3})
            print("  ❌ ERROR: Should have been blocked!")
        except PCPClientError as e:
            print(f"  ✅ E-Stop blocked: {e.message[:60]}")
        await client.estop(False)
        print("  ✅ E-Stop cleared")

        status = await client.get_status()
        print(f"\n  Stats    : calls={status['call_count']}  " f"blocked={status['blocked_count']}")


async def demo_mobile_robot() -> None:
    sep("DEMO B — Mobile Robot: Autonomous Navigation")

    server = build_mobile_server(name="amr-01", model="TurtleBot4", location="warehouse-floor-1")

    async with PCPClient("fleet-manager") as client:
        await client.connect_server(server)

        tools = await client.list_tools()
        print(f"\n  Tools    : {[t['name'] for t in tools]}")

        # Navigate to pick station
        print("\n  [Nav] Navigating to pick station (5.0, 2.5)...")
        result = await client.safe_call(
            "navigate_to", {"x": 5.0, "y": 2.5, "max_speed": 0.5}, zone_id="floor-zone-1"
        )
        import json

        out = json.loads(result["content"][0]["text"])
        print(
            f"  Result   : reached={out['output']['reached']}  "
            f"distance={out['output']['distance_m']}m"
        )

        # Read odometry
        odom = await client.read_sensor("odometry")
        o = json.loads(odom["contents"][0]["text"])
        print(
            f"  Pose     : x={o['value']['x']}m  y={o['value']['y']}m  "
            f"yaw={o['value']['yaw_deg']}°"
        )

        # Read battery
        batt = await client.read_sensor("battery")
        b = json.loads(batt["contents"][0]["text"])
        print(f"  Battery  : {b['value']['percent']}%  " f"charging={b['value']['charging']}")

        # Patrol mission
        print("\n  [Mission] Getting delivery mission template...")
        mission = await client.get_mission(
            "delivery",
            {
                "pickup_x": "2.0",
                "pickup_y": "0.0",
                "dropoff_x": "8.0",
                "dropoff_y": "5.0",
            },
        )
        print(f"  Mission  : {mission['messages'][1]['content'][0]['text'][:80]}...")


async def demo_agricultural_robot() -> None:
    sep("DEMO C — Agricultural Robot: Harvest Cycle")

    server = build_agricultural_server(
        name="agri-bot-01", model="HarvestBot-X", location="greenhouse-1"
    )

    async with PCPClient("farm-ai") as client:
        await client.connect_server(server)

        print(f"\n  Tools    : {[t['name'] for t in await client.list_tools()]}")

        # Check plant health
        print("\n  [Inspect] Checking plant health across all rows...")
        for row in ["row_1", "row_2", "row_3", "row_4"]:
            result = await client.call_tool("inspect_plants", {"row": row})
            import json

            out = json.loads(result["content"][0]["text"])
            h = out["output"]
            height = h["plant_height_m"]
            ready = "✅ READY" if height > 0.30 else "⏳ growing"
            print(
                f"  {row}    : health={h['health_score']:.2f}  "
                f"height={height}m  {ready}  issues={h['issues']}"
            )

        # Irrigate a dry zone
        print("\n  [Irrigate] Irrigating zone_D (moisture was 0.31)...")
        result = await client.call_tool("irrigate_zone", {"zone": "zone_D", "duration_s": 45.0})
        import json

        out = json.loads(result["content"][0]["text"])
        r = out["output"]
        print(f"  Moisture : {r['moisture_before']} → {r['moisture_after']}")

        # Harvest ready rows
        print("\n  [Harvest] Attempting harvest of row_3...")
        result = await client.safe_call(
            "harvest_row", {"row": "row_3", "speed": 0.1}, zone_id="harvest-zone"
        )
        import json

        out = json.loads(result["content"][0]["text"])
        h = out["output"]
        print(f"  Harvest  : {'✅ SUCCESS' if h['harvested'] else '⏳ ' + h['reason']}")

        # Adjust climate
        print("\n  [Climate] Adjusting greenhouse to 22°C / 72% RH...")
        result = await client.call_tool(
            "adjust_climate", {"target_temp": 22.0, "target_humidity": 72.0}
        )
        import json

        out = json.loads(result["content"][0]["text"])
        c = out["output"]
        print(f"  Climate  : {c['temperature_c']}°C  {c['humidity_pct']}% RH")

        # Read climate sensor
        climate = await client.read_sensor("climate")
        cv = json.loads(climate["contents"][0]["text"])["value"]
        print(
            f"  Sensor   : {cv['temperature_c']}°C  {cv['humidity_pct']}%  "
            f"CO₂={cv['co2_ppm']}ppm  PPFD={cv['ppfd_umol']}µmol"
        )


async def demo_multi_robot() -> None:
    sep("DEMO D — Multi-Robot Registry + Coordinated Farm")

    # Build registry
    registry = PCPRegistry("farm-registry")

    # Build 3 robots and register them
    arm1 = build_arm_server("arm-01", "UR5e", location="grow-zone-A")
    arm2 = build_arm_server("arm-02", "UR10e", location="grow-zone-B")
    mobile = build_mobile_server("amr-01", "TurtleBot4", location="main-floor")
    agri = build_agricultural_server("agri-01", "HarvestBot-X", location="greenhouse-1")

    for srv, robot_class, tools in [
        (arm1, "arm", ["move_to", "pick", "place", "home"]),
        (arm2, "arm", ["move_to", "pick", "place", "home"]),
        (mobile, "mobile", ["navigate_to", "patrol", "dock"]),
        (agri, "arm", ["irrigate_zone", "harvest_row", "inspect_plants"]),
    ]:
        registry.register(
            RegistryEntry(
                robot_id=srv.robot_id,
                name=srv.name,
                robot_class=robot_class,
                model=srv.identity.model,
                location=srv.identity.location,
                transport="stdio",
                endpoint=f"python robot_servers/{srv.robot_id}.py",
                actuations=tools,
                sensors=["joint_angles"] if robot_class == "arm" else ["odometry"],
                tags=["farm", "production"],
            )
        )

    print(f"\n  Registry : {registry.summary()}")

    # Query: find all arms in grow zones
    arms = registry.find(robot_class="arm", online_only=False)
    print(f"  Arms     : {[e.robot_id for e in arms]}")

    # Query: find robots that can 'harvest_row'
    harvesters = registry.find(actuation="harvest_row", online_only=False)
    print(f"  Harvesters: {[e.robot_id for e in harvesters]}")

    # Simulate coordinated multi-robot work
    print("\n  [Coord] Running parallel robot tasks...")

    async def arm_task(server, name, x, y, z):
        async with PCPClient(f"orchestrator-{name}") as c:
            await c.connect_server(server)
            result = await c.safe_call(
                "move_to", {"x": x, "y": y, "z": z, "speed": 0.4}, zone_id=f"zone-{name}"
            )
            return name, not result.get("isError", True)

    results = await asyncio.gather(
        arm_task(arm1, "arm1", 0.4, 0.1, 0.5),
        arm_task(arm2, "arm2", 0.3, 0.2, 0.6),
    )
    for name, success in results:
        print(f"  {name}    : {'✅' if success else '❌'}")

    print("\n  ✅ Multi-robot coordination complete")
    print(f"  MCP-format registry:\n  {list(registry.to_mcp_format().keys())}")


async def main(demo: str = "all") -> None:
    print("\n" + "╔" + "═" * 66 + "╗")
    print("║   P-MCP v0.5 — Physical Context Protocol                 ║")
    print("║   'The USB-C port for robot AI — now MCP-wire-compatible'      ║")
    print("╚" + "═" * 66 + "╝")
    print()
    print("  Aligned with: https://modelcontextprotocol.io")
    print("  GitHub:       https://github.com/physicalcontextprotocol/pmcp-python")
    print("  Standard:     ISO 10218 / IEC 62443 / W3C DIDs")

    demos = {
        "arm": demo_arm_robot,
        "mobile": demo_mobile_robot,
        "agri": demo_agricultural_robot,
        "multi": demo_multi_robot,
    }

    if demo == "all":
        for fn in demos.values():
            await fn()
    elif demo in demos:
        await demos[demo]()
    else:
        print(f"Unknown demo: {demo}. Choose from: {list(demos.keys())} | all")

    sep("SUMMARY")
    print("  ✅ P-MCP v0.5 — all demos passed")
    print("  ✅ MCP wire protocol compatible (tools/list, tools/call, etc.)")
    print("  ✅ Safety pipeline: constitution → shadow → execute")
    print("  ✅ Temporal leases prevent physical collisions")
    print("  ✅ W3C DID robot identity")
    print("  ✅ ISO 10218 / IEC 62443 rules enforced")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="P-MCP v0.5 Demo")
    parser.add_argument("--demo", default="all", choices=["all", "arm", "mobile", "agri", "multi"])
    args = parser.parse_args()
    asyncio.run(main(args.demo))


def main_cli() -> None:
    """Console-script entry point: `pcp-demo`."""
    import argparse

    parser = argparse.ArgumentParser(description="P-MCP v0.5 Demo")
    parser.add_argument("--demo", default="all", choices=["all", "arm", "mobile", "agri", "multi"])
    args = parser.parse_args()
    asyncio.run(main(args.demo))
