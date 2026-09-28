"""
P-MCP — Physical Model Context Protocol  v0.4
==============================================
The Python SDK for P-MCP.  Install: pip install pmcp

    from pmcp.server import PMCPServer
    from pmcp.client import PMCPClient
    from pmcp.safety import SafetyMiddleware
    from pmcp.types  import ActuationResult, SensorReading

Protocol spec: https://github.com/physicalcontextprotocol/pmcp-spec/blob/main/docs/PROTOCOL_SPEC.md
"""
from __future__ import annotations

__version__      = "0.4.0"
__protocol__     = "P-MCP/0.4"
__spec_version__ = "2025-05-04"

from pmcp.types import (          # noqa: F401 — re-export for convenience
    ActuationSpec, ActuationResult,
    SensorSpec, SensorReading, SensorType,
    PromptSpec, PromptResult,
    ShadowPreview, ShadowStatus, ConstitutionCheck,
    LeaseRequest, LeaseGrant,
    PMCPRequest, PMCPResponse, PMCPNotification,
    ServerInfo, ClientInfo, Capabilities,
    PMCPError, PMCPErrorCode,
)
from pmcp.server import PMCPServer     # noqa: F401
from pmcp.client import PMCPClient     # noqa: F401
from pmcp.safety import SafetyMiddleware  # noqa: F401
