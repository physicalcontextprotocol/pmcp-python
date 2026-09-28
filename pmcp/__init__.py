"""
P-MCP v0.5 — Physical Model Context Protocol with Hamiltonian Physics
======================================================================
The Python SDK with energy-conserving neural physics (Phase 1-5).

Phase 1: Hamiltonian Neural Networks — energy-conserving physics
Phase 2: Digital Twin — real-time sync with violation detection
Phase 3: Fleet Coordination — system Hamiltonian + HJB optimal control
Phase 4: Adaptive Learning — online HNN training + anomaly detection
Phase 5: PhysOS Protocol — universal physical agent communication

New in v0.5:
  - PMCP_VERSION bumped to "0.5"
  - Audit log (ISO 10218 / IEC 62443 compliance)
  - Batch actuation (atomic multi-actuation execution)
  - MetricsSnapshot (server telemetry)
  - Lease enforcement in actuation pipeline
  - Rate limiter in SafetyMiddleware
  - Robot profile factories: SafetyMiddleware.for_arm(), for_mobile()
  - Two new constitution rules: CONST-09 (joint limits), CONST-10 (workspace)
  - Pure-asyncio HTTP transport (replaces broken new_event_loop approach)
  - PMCPClient: batch_execute(), get_metrics(), get_audit_log()

Usage:
    from pmcp.server import PMCPServer
    from pmcp.tools.hamiltonian_tools import register_hamiltonian_tools
    from pmcp.physics.hamiltonian import HamiltonianNN

    server = PMCPServer("pmcp-hamiltonian")
    register_hamiltonian_tools(server)
    asyncio.run(server.run())
"""

from __future__ import annotations

__version__ = "0.5.0"
__protocol__ = "P-MCP/0.5"
__spec_version__ = "2026-05-16"

from pmcp.client import PMCPClient
from pmcp.physics.hamiltonian import (
    ConservationChecker,
    HamiltonianNN,
    PhaseSpaceEncoder,
    StormerVerlet,
)
from pmcp.safety import SafetyMiddleware
from pmcp.server import PMCPServer
from pmcp.twin.hamiltonian_twin import HamiltonianViolationDetector, TwinSynchronizer
from pmcp.types import (
    ActuationResult,
    ActuationSpec,
    AuditEntry,
    AuditEventType,
    BatchActuationRequest,
    BatchActuationResult,
    Capabilities,
    ClientInfo,
    ConstitutionCheck,
    EStopMessage,
    EStopSource,
    LeaseGrant,
    LeaseRequest,
    LeaseState,
    MetricsSnapshot,
    PMCPError,
    PMCPErrorCode,
    PMCPNotification,
    PMCPRequest,
    PMCPResponse,
    PromptResult,
    PromptSpec,
    SensorReading,
    SensorSpec,
    SensorType,
    ServerInfo,
    ShadowPreview,
    ShadowStatus,
    ShadowVerdict,
    StopCategory,
)

__all__ = [
    "PMCPServer",
    "PMCPClient",
    "SafetyMiddleware",
    # Types
    "ActuationSpec",
    "ActuationResult",
    "AuditEntry",
    "AuditEventType",
    "BatchActuationRequest",
    "BatchActuationResult",
    "MetricsSnapshot",
    "SensorSpec",
    "SensorReading",
    "SensorType",
    "PromptSpec",
    "PromptResult",
    "ShadowPreview",
    "ShadowStatus",
    "ShadowVerdict",
    "ConstitutionCheck",
    "LeaseRequest",
    "LeaseGrant",
    "LeaseState",
    "EStopMessage",
    "EStopSource",
    "StopCategory",
    "PMCPError",
    "PMCPErrorCode",
    "Capabilities",
    "ClientInfo",
    "ServerInfo",
    "PMCPNotification",
    "PMCPRequest",
    "PMCPResponse",
    # Physics
    "HamiltonianNN",
    "PhaseSpaceEncoder",
    "StormerVerlet",
    "ConservationChecker",
    # Twin
    "TwinSynchronizer",
    "HamiltonianViolationDetector",
]
