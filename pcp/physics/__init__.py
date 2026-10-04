"""
Physics modules — simulation and Hamiltonian mechanics.
"""

# Resolved lazily: the Hamiltonian subpackage needs numpy (and, for
# HamiltonianNN, torch), both of which are optional extras. See
# pcp/physics/hamiltonian/__init__.py for the full explanation.
_LAZY = {
    "HamiltonianNN": "pcp.physics.hamiltonian",
    "PhaseSpaceEncoder": "pcp.physics.hamiltonian",
    "StormerVerlet": "pcp.physics.hamiltonian",
    "ConservationChecker": "pcp.physics.hamiltonian",
}


def __getattr__(name):
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


__all__ = ["HamiltonianNN", "PhaseSpaceEncoder", "StormerVerlet", "ConservationChecker"]
