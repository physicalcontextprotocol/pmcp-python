"""
P-MCP v0.5 — MCP-Aligned Physical Robot Protocol
=================================================
"Any MCP client can talk to any P-MCP robot."

v0.5 aligns P-MCP wire format with Anthropic's Model Context Protocol,
making robots first-class citizens of the agentic web.

Architecture:
  - Full MCP JSON-RPC 2.0 wire compatibility (Claude Desktop, Cursor, etc.)
  - Robot actuations exposed as standard MCP Tools
  - Sensor streams exposed as standard MCP Resources
  - Mission templates exposed as standard MCP Prompts
  - Physical-only extensions: shadow/*, lease/*, pmcp/estop
"""

from importlib.metadata import PackageNotFoundError, version as _dist_version

try:
    __version__ = _dist_version("pmcp")
except PackageNotFoundError:  # running from a source tree, not installed
    __version__ = "0.0.0+unknown"

from v05.pmcp_registry import PMCPRegistry, RegistryEntry
from v05.pmcp_safety_v5 import (
    SafetyConstitution,
    SafetyMiddleware,
    ShadowSimulator,
)
from v05.pmcp_v5_client import PMCPClient
from v05.pmcp_v5_server import PMCPServer
from v05.pmcp_v5_types import (
    PMCP_VERSION,
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
    "PMCPServer",
    "PMCPClient",
    "SafetyConstitution",
    "SafetyMiddleware",
    "ShadowSimulator",
    "PMCPRegistry",
    "RegistryEntry",
    "ActuationSpec",
    "SensorSpec",
    "MissionSpec",
    "ActuationResult",
    "SensorReading",
    "SafetyEnvelope",
    "LeaseGrant",
    "ShadowPreview",
    "PMCP_VERSION",
]
