"""
Twin layer — basic + Hamiltonian digital twin.
"""

# Resolved lazily (PEP 562). Both of these need numpy, which is the
# optional `numerics` extra. A top-level import here meant that a plain
# `pip install pmcp` produced a package that could not be imported at
# all without the extras -- reproduced as:
#   ModuleNotFoundError: No module named 'numpy'
# from `import pmcp`, because pmcp/__init__.py imports this module.
#
# Lazy resolution keeps `from pmcp.twin import TwinSynchronizer` and
# `pmcp.TwinSynchronizer` working, and reports the missing extra by name
# if it is used without it.
_LAZY = {
    "TwinSynchronizer": "pmcp.twin.hamiltonian_twin.sync",
    "HamiltonianViolationDetector": "pmcp.twin.hamiltonian_twin.violation_detector",
}

_NUMPY_HINT = "The digital twin requires numpy. Install it with: pip install 'pmcp[numerics]'"


def __getattr__(name):
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    try:
        mod = importlib.import_module(module_path)
    except ImportError as exc:
        if "numpy" in str(exc):
            raise ImportError(_NUMPY_HINT) from exc
        raise
    value = getattr(mod, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


__all__ = ["TwinSynchronizer", "HamiltonianViolationDetector"]
