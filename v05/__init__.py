"""
PCP v0.5 — MCP-Aligned Physical Robot Protocol
=================================================
"Any MCP client can talk to any PCP robot."

v0.5 aligns PCP wire format with Anthropic's Model Context Protocol,
making robots first-class citizens of the agentic web.

Architecture:
  - Full MCP JSON-RPC 2.0 wire compatibility (Claude Desktop, Cursor, etc.)
  - Robot actuations exposed as standard MCP Tools
  - Sensor streams exposed as standard MCP Resources
  - Mission templates exposed as standard MCP Prompts
  - Physical-only extensions: shadow/*, lease/*, pcp/estop
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

try:
    __version__ = _dist_version("physicalcontextprotocol")
except PackageNotFoundError:  # running from a source tree, not installed
    __version__ = "0.0.0+unknown"

from v05.pcp_registry import PCPRegistry, RegistryEntry
from v05.pcp_safety_v5 import (
    SafetyConstitution,
    SafetyMiddleware,
    ShadowSimulator,
)
from v05.pcp_v5_client import PCPClient
from v05.pcp_v5_server import PCPServer
from v05.pcp_v5_types import (
    PCP_VERSION,
    ActuationResult,
    ActuationSpec,
    LeaseGrant,
    MissionSpec,
    SafetyEnvelope,
    SensorReading,
    SensorSpec,
    ShadowPreview,
)

__all__ = [
    "PCPServer",
    "PCPClient",
    "SafetyConstitution",
    "SafetyMiddleware",
    "ShadowSimulator",
    "PCPRegistry",
    "RegistryEntry",
    "ActuationSpec",
    "SensorSpec",
    "MissionSpec",
    "ActuationResult",
    "SensorReading",
    "SafetyEnvelope",
    "LeaseGrant",
    "ShadowPreview",
    "PCP_VERSION",
]
