"""
PCP — Physical Context Protocol  v0.4
==============================================
The Python SDK for PCP.  Install: pip install pcp

    from pcp.server import PCPServer
    from pcp.client import PCPClient
    from pcp.safety import SafetyMiddleware
    from pcp.types  import ActuationResult, SensorReading

Protocol spec: https://github.com/physicalcontextprotocol/pcp-spec/blob/main/docs/PROTOCOL_SPEC.md
"""
from __future__ import annotations

__version__      = "0.4.0"
__protocol__     = "PCP/0.4"
__spec_version__ = "2025-05-04"

from pcp.types import (          # noqa: F401 — re-export for convenience
    ActuationSpec, ActuationResult,
    SensorSpec, SensorReading, SensorType,
    PromptSpec, PromptResult,
    ShadowPreview, ShadowStatus, ConstitutionCheck,
    LeaseRequest, LeaseGrant,
    PCPRequest, PCPResponse, PCPNotification,
    ServerInfo, ClientInfo, Capabilities,
    PCPError, PCPErrorCode,
)
from pcp.server import PCPServer     # noqa: F401
from pcp.client import PCPClient     # noqa: F401
from pcp.safety import SafetyMiddleware  # noqa: F401
