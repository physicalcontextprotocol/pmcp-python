"""
Phase 1 — Hamiltonian Physics Core
===================================
hamilcore: the mathematical foundation for energy-conserving neural physics.

Architecture:
    Sensor data → PhaseSpaceEncoder → (q, p) → HNN.hamiltonian(q,p) → H value
                                            ↓
                                    SymplecticIntegrator → trajectory

Conservation law enforced: dH/dt ≈ 0 (energy constant along Hamiltonian flow)
"""

# Everything in this subpackage is resolved lazily, on first attribute
# access.
#
# These five names used to be imported at module scope. Three of the
# submodules need numpy, and `hamiltonian.hnn` needs torch as well --
# and torch is an OPTIONAL extra (`pip install pcp[hnn]`), not a base
# dependency. Because pcp/__init__.py re-exports from here, a top-level
# import made the plain `import pcp` fail on any machine without torch,
# which broke the CI smoke-import job on all four Python versions. The
# bug was invisible locally because the development machine had torch
# installed.
#
# PEP 562 module __getattr__ keeps `from pcp.physics.hamiltonian import
# X` working for every name, with no behaviour change for a caller who
# does have the extras -- the only difference is that the cost is paid
# on use rather than on import.
_LAZY = {
    "HamiltonianNN": "pcp.physics.hamiltonian.hnn",
    "PhaseSpaceEncoder": "pcp.physics.hamiltonian.encoder",
    "StormerVerlet": "pcp.physics.hamiltonian.integrators",
    "SymplecticIntegrator": "pcp.physics.hamiltonian.integrators",
    "ConservationChecker": "pcp.physics.hamiltonian.conservation",
}

_HINTS = {
    "torch": ("HamiltonianNN requires torch. Install it with: pip install 'pcp[hnn]'"),
    "numpy": (
        "The Hamiltonian subpackage requires numpy. "
        "Install it with: pip install 'pcp[numerics]'"
    ),
}


def __getattr__(name):
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    try:
        mod = importlib.import_module(module_path)
    except ImportError as exc:
        # Report the extra that would fix it, rather than a bare
        # "No module named 'torch'" from three frames down.
        for pkg in ("torch", "numpy"):
            if getattr(exc, "name", "") == pkg or pkg in str(exc):
                raise ImportError(_HINTS[pkg]) from exc
        raise
    value = getattr(mod, name)
    globals()[name] = value  # cache: later lookups skip this entirely
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "HamiltonianNN",
    "PhaseSpaceEncoder",
    "SymplecticIntegrator",
    "StormerVerlet",
    "ConservationChecker",
]
