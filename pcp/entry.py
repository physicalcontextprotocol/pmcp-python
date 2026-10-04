"""
P-MCP Entry Point — All 5 Hamiltonian phases wired into one PCPServer.
======================================================================
Run: python -m pcp.entry
     or: pcp-hamiltonian

Phase wiring order (must be sequential):
    1. Hamiltonian (Phase 1) — HNN, phase space, conservation
    2. Twin (Phase 2) — digital twin sync, violation detection
    3. Fleet (Phase 3) — system Hamiltonian, fleet planning
    4. Learning (Phase 4) — online HNN training, anomaly detection
    5. Protocol (Phase 5) — PhysOS, universal agent communication
"""

from __future__ import annotations

import asyncio
import logging
import os

from pcp import __version__
from pcp.server import PCPServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("pcp.entry")


def build_server() -> PCPServer:
    from pcp.safety import SafetyMiddleware

    safety = SafetyMiddleware.default(robot_id="pcp-fleet")

    server = PCPServer(
        name="pcp-hamiltonian",
        version=__version__,
        robot_id="pcp-fleet",
        safety_middleware=safety,
    )

    from pcp.tools.fleet_tools import register_fleet_tools
    from pcp.tools.hamiltonian_tools import register_hamiltonian_tools
    from pcp.tools.learning_tools import register_learning_tools
    from pcp.tools.protocol_tools import register_protocol_tools
    from pcp.tools.twin_tools import register_twin_tools

    register_hamiltonian_tools(server)
    register_twin_tools(server)
    register_fleet_tools(server)
    register_learning_tools(server)
    register_protocol_tools(server)

    log.info(
        f"PCPServer built: {len(server._actuations)} actuations, "
        f"{len(server._sensors)} sensors, "
        f"{len(server._prompts)} prompts"
    )

    return server


def print_capabilities(server: PCPServer):
    stats = server.stats()
    print("\n" + "=" * 60)
    print("P-MCP Hamiltonian Physics Server")
    print("=" * 60)
    print(f"  Name:       {server.name}")
    print(f"  Version:    {server.version}")
    print("  Protocol:   P-MCP/0.5 (Hamiltonian)")
    print()
    print(f"  Actuations: {stats['actuations']}")
    print(f"  Sensors:    {stats['sensors']}")
    print(f"  Prompts:    {list(server._prompts.keys())}")
    print()
    print("Phase stack:")
    print("  Phase 1: HamiltonianNN + PhaseSpaceEncoder + Conservation")
    print("  Phase 2: TwinSynchronizer + ViolationDetector")
    print("  Phase 3: SystemHamiltonian + Energy Allocator")
    print("  Phase 4: OnlineHNNTrainer + AnomalyDetector")
    print("  Phase 5: PhysOS Protocol")
    print("=" * 60 + "\n")


async def run_http():
    server = build_server()
    print_capabilities(server)

    port = int(os.environ.get("PCP_PORT", "8090"))
    host = os.environ.get("PCP_HOST", "127.0.0.1")

    log.info(f"Starting HTTP transport on http://{host}:{port}/")
    log.info("P-MCP Hamiltonian server ready")

    await server.run(transport="http", host=host, port=port)


async def run_stdio():
    server = build_server()
    print_capabilities(server)

    log.info("Starting stdio transport (MCP-compatible)")
    log.info("P-MCP Hamiltonian server ready — listening on stdin/stdout")

    await server.run(transport="stdio")


async def run_shell():
    server = build_server()
    print_capabilities(server)

    print("Interactive shell. Commands:")
    print("  stats              — server statistics")
    print("  actuators          — list all actuations")
    print("  sensors            — list all sensors")
    print("  health             — health check")
    print("  quit               — exit")
    print()

    while True:
        try:
            cmd = input("pcp> ").strip()
            if not cmd:
                continue

            parts = cmd.split()
            method = parts[0]

            if method == "quit":
                print("Goodbye.")
                break
            elif method == "stats":
                print(json.dumps(server.stats(), indent=2))
            elif method == "actuators":
                print(json.dumps(list(server._actuations.keys()), indent=2))
            elif method == "sensors":
                print(json.dumps(list(server._sensors.keys()), indent=2))
            elif method == "health":
                print(json.dumps({"status": "ok", "uptime": server.stats()["uptime_s"]}, indent=2))
            else:
                print(f"Unknown command: {method}")

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break


if __name__ == "__main__":
    import json

    mode = os.environ.get("PCP_TRANSPORT", "http").lower()

    if mode == "stdio":
        asyncio.run(run_stdio())
    elif mode == "shell":
        asyncio.run(run_shell())
    else:
        asyncio.run(run_http())
