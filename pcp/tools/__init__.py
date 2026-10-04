"""
P-MCP Tools — MCP tool registrations for all phases.
"""

from pcp.tools.fleet_tools import register_fleet_tools
from pcp.tools.hamiltonian_tools import register_hamiltonian_tools
from pcp.tools.learning_tools import register_learning_tools
from pcp.tools.protocol_tools import register_protocol_tools
from pcp.tools.twin_tools import register_twin_tools

__all__ = [
    "register_hamiltonian_tools",
    "register_twin_tools",
    "register_fleet_tools",
    "register_learning_tools",
    "register_protocol_tools",
]
