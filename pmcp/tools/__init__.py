"""
P-MCP Tools — MCP tool registrations for all phases.
"""

from pmcp.tools.fleet_tools import register_fleet_tools
from pmcp.tools.hamiltonian_tools import register_hamiltonian_tools
from pmcp.tools.learning_tools import register_learning_tools
from pmcp.tools.protocol_tools import register_protocol_tools
from pmcp.tools.twin_tools import register_twin_tools

__all__ = [
    "register_hamiltonian_tools",
    "register_twin_tools",
    "register_fleet_tools",
    "register_learning_tools",
    "register_protocol_tools",
]
