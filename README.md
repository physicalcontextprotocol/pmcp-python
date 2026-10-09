# pmcp-python

Python implementation of PCP.

## Layout
- `pcp/` — the packaged library. The PyPI distribution is
  `physicalcontextprotocol` (`pip install physicalcontextprotocol`,
  version 1.0.0); the import package is still `pcp`. Includes protocol, physics
  (Hamiltonian Neural Network validation), digital twin, fleet,
  learning, and tools submodules.
- `v05/` — a parallel v0.5 implementation (server, client, types,
  kinematics, safety, registry, replay, robot_servers/). Overlaps
  significantly with `pcp/`.
- `sdk/` — a third, smaller client/server/safety/types implementation.
- `tests/` — everything except conformance (unit, integration, e2e,
  load, security, chaos, benchmark, v05, plus top-level test_types.py
  / test_compliance.py). Conformance tests now live in
  `pmcp-conformance`.

## ⚠️ Known issue: duplicate client implementations
There are at least **four** separate client implementations shipping
together and not yet consolidated:
1. `pcp/client.py`
2. `pcp/client_v2.py`
3. `sdk/client.py`
4. `v05/pcp_v5_client.py`

(plus `pcp/fleet/client.py`, which may be legitimately distinct as a
fleet-coordination client rather than a duplicate). This split
preserved all of them as-is — picking a canonical one and deprecating
the rest is flagged as follow-up work in `MIGRATION_MAP.md` and should
happen before this repo's first independent release.
