"""
PCP v0.5 — Robot Registry
=============================
Decentralized robot discovery — analogous to the MCP registry
(https://github.com/modelcontextprotocol/registry) but for physical robots.

A robot registers itself at boot; LLM applications query the registry
to find robots by class, capability, or location.

Compatible with the official MCP registry wire format.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger("pcp.registry")


@dataclass
class RegistryEntry:
    robot_id: str
    name: str
    robot_class: str  # arm | mobile | drone | plc | custom
    model: str
    location: str
    transport: str  # "stdio" | "http"
    endpoint: str  # command (stdio) or URL (http)
    actuations: List[str]  # tool names
    sensors: List[str]  # resource names
    pcp_version: str = "0.5"
    registered_at: float = field(default_factory=time.time)
    last_ping: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def online(self) -> bool:
        return (time.time() - self.last_ping) < 60.0  # 60s TTL

    def to_dict(self) -> dict:
        return {
            "id": self.robot_id,
            "name": self.name,
            "class": self.robot_class,
            "model": self.model,
            "location": self.location,
            "transport": self.transport,
            "endpoint": self.endpoint,
            "actuations": self.actuations,
            "sensors": self.sensors,
            "pcp_version": self.pcp_version,
            "online": self.online,
            "tags": self.tags,
            "registered_at": self.registered_at,
        }

    # MCP registry-compatible format
    def to_mcp_server_entry(self) -> dict:
        return {
            "id": self.robot_id,
            "name": self.name,
            "description": f"PCP robot: {self.model} at {self.location}",
            "repository": {"url": "https://github.com/physicalcontextprotocol/pmcp-python"},
            "versionDetail": {
                "version": self.pcp_version,
                "releaseDate": time.strftime("%Y-%m-%d", time.gmtime(self.registered_at)),
            },
            "packages": [
                {
                    "registryName": "pcp",
                    "name": self.name,
                    "version": self.pcp_version,
                    "runtimeHint": "python",
                    "environmentVariables": [],
                }
            ],
            "pcp": self.to_dict(),
        }


class PCPRegistry:
    """
    In-process robot registry.
    For production, use the HTTP-backed registry server.
    """

    def __init__(self, name: str = "pcp-registry"):
        self.name = name
        self._robots: Dict[str, RegistryEntry] = {}
        log.info(f"[Registry] {name} initialized")

    def register(self, entry: RegistryEntry) -> None:
        self._robots[entry.robot_id] = entry
        log.info(f"[Registry] Registered {entry.robot_id} ({entry.model}) at {entry.location}")

    def deregister(self, robot_id: str) -> bool:
        if robot_id in self._robots:
            del self._robots[robot_id]
            return True
        return False

    def heartbeat(self, robot_id: str) -> bool:
        if robot_id in self._robots:
            self._robots[robot_id].last_ping = time.time()
            return True
        return False

    def find(
        self,
        robot_class: Optional[str] = None,
        location: Optional[str] = None,
        actuation: Optional[str] = None,
        tag: Optional[str] = None,
        online_only: bool = True,
    ) -> List[RegistryEntry]:
        """Query registry with optional filters."""
        results = []
        for entry in self._robots.values():
            if online_only and not entry.online:
                continue
            if robot_class and entry.robot_class != robot_class:
                continue
            if location and location.lower() not in entry.location.lower():
                continue
            if actuation and actuation not in entry.actuations:
                continue
            if tag and tag not in entry.tags:
                continue
            results.append(entry)
        return results

    def get(self, robot_id: str) -> Optional[RegistryEntry]:
        return self._robots.get(robot_id)

    def list_all(self) -> List[RegistryEntry]:
        return list(self._robots.values())

    def summary(self) -> dict:
        online = sum(1 for e in self._robots.values() if e.online)
        by_class: Dict[str, int] = {}
        for e in self._robots.values():
            by_class[e.robot_class] = by_class.get(e.robot_class, 0) + 1
        return {
            "registry": self.name,
            "total": len(self._robots),
            "online": online,
            "by_class": by_class,
        }

    def to_mcp_format(self) -> dict:
        """Return registry in MCP-compatible server list format."""
        return {
            "servers": [e.to_mcp_server_entry() for e in self._robots.values()],
            "total": len(self._robots),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  HTTP REGISTRY SERVER  (optional, requires aiohttp)
# ─────────────────────────────────────────────────────────────────────────────


async def run_registry_server(
    registry: PCPRegistry, host: str = "127.0.0.1", port: int = 9090
) -> None:
    """
    Serve the registry over HTTP with MCP-compatible endpoints.

    GET  /servers            — list all robots (MCP registry format)
    GET  /servers/{id}       — get specific robot
    POST /servers/register   — register a robot
    POST /servers/{id}/ping  — heartbeat
    GET  /health             — health check
    """
    try:
        from aiohttp import web
    except ImportError:
        log.error("[Registry] aiohttp required for HTTP server: pip install aiohttp")
        return

    app = web.Application()

    async def list_servers(req: web.Request) -> web.Response:
        robot_class = req.query.get("class")
        location = req.query.get("location")
        actuation = req.query.get("actuation")
        results = registry.find(robot_class=robot_class, location=location, actuation=actuation)
        return web.json_response(
            {
                "servers": [e.to_mcp_server_entry() for e in results],
                "total": len(results),
            }
        )

    async def get_server(req: web.Request) -> web.Response:
        robot_id = req.match_info["id"]
        entry = registry.get(robot_id)
        if not entry:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response(entry.to_mcp_server_entry())

    async def register_server(req: web.Request) -> web.Response:
        data = await req.json()
        entry = RegistryEntry(
            robot_id=data.get("robotId", str(uuid.uuid4())[:8]),
            name=data["name"],
            robot_class=data["class"],
            model=data.get("model", "generic"),
            location=data.get("location", "unknown"),
            transport=data.get("transport", "http"),
            endpoint=data.get("endpoint", ""),
            actuations=data.get("actuations", []),
            sensors=data.get("sensors", []),
            tags=data.get("tags", []),
        )
        registry.register(entry)
        return web.json_response({"registered": True, "robotId": entry.robot_id})

    async def ping_server(req: web.Request) -> web.Response:
        robot_id = req.match_info["id"]
        ok = registry.heartbeat(robot_id)
        return web.json_response({"ok": ok})

    async def health(req: web.Request) -> web.Response:
        return web.json_response(registry.summary())

    app.router.add_get("/servers", list_servers)
    app.router.add_get("/servers/{id}", get_server)
    app.router.add_post("/servers/register", register_server)
    app.router.add_post("/servers/{id}/ping", ping_server)
    app.router.add_get("/health", health)
    app.router.add_get("/", health)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    log.info(f"[Registry] PCP Registry listening on http://{host}:{port}")
    await asyncio.Event().wait()


# ── Console-script entry point: `pcp-registry` ──────────────────────────────


def main() -> None:
    """CLI entry point — serve the registry over HTTP."""
    import argparse

    parser = argparse.ArgumentParser(description="PCP Registry server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9090)
    args = parser.parse_args()

    registry = PCPRegistry()
    asyncio.run(run_registry_server(registry, host=args.host, port=args.port))
