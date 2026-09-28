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

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

try:
    # Read it from the installed distribution rather than repeating it
    # here. A literal in this file silently rotted at 0.5.0 through a
    # package bump to 1.0.0, and nothing caught it: the stale editable
    # install reported 0.5.0 too, so the two agreed with each other and
    # disagreed with pyproject.toml. One source of truth, taken from the
    # metadata pyproject generates, cannot drift like that.
    __version__ = _dist_version("pmcp")
except PackageNotFoundError:  # running from a source tree, not installed
    __version__ = "0.0.0+unknown"

# The wire protocol is 0.5 and the JSON Schema is 0.6.0. Those are
# different things on purpose -- see pmcp-spec/docs/PROTOCOL_SPEC.md 3.2,
# and tests/v05/test_v05_sdk.py asserts the 0.5 prefix. Do not "fix" this
# to match __version__.
__protocol__ = "P-MCP/0.5"
__spec_version__ = "2026-05-16"

from pmcp.client import PMCPClient
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

# Resolved lazily on first attribute access, because the physics
# subpackage needs numpy, and HamiltonianNN additionally needs torch --
# both optional extras. Without this, a bare `import pmcp` failed
# outright on a machine without torch, which broke the CI smoke-import
# job on all four Python versions. `from pmcp import HamiltonianNN` still
# works, and if the extras are missing the error names the extra to
# install. See pmcp/physics/hamiltonian/__init__.py.
_LAZY = {
    "HamiltonianNN": "pmcp.physics.hamiltonian.hnn",
    "PhaseSpaceEncoder": "pmcp.physics.hamiltonian.encoder",
    "StormerVerlet": "pmcp.physics.hamiltonian.integrators",
    "ConservationChecker": "pmcp.physics.hamiltonian.conservation",
}


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


def __getattr__(name):
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    try:
        mod = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            "HamiltonianNN requires torch. Install it with: pip install 'pmcp[hnn]'"
        ) from exc
    value = getattr(mod, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
