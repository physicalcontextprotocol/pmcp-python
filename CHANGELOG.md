# Changelog — pmcp-python

Changes to the Python SDK (`pip install physicalcontextprotocol`). Organization-wide policy
and the maintained list of what is *not* yet proven live in
[`pmcp-spec`](https://github.com/physicalcontextprotocol/pmcp-spec) —
see its `LIMITATIONS.md` and `SECURITY.md`.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Removed
- **62 collected tests asserted against simulators, not the SDK.** Seven files
  under `tests/` imported only the standard library, so their assertions could
  not fail because of a defect in `physicalcontextprotocol`. They have been
  moved to `simulations/` (excluded from pytest by `testpaths`), not deleted:
  `security/test_mtls.py`, `e2e/test_fleet_workflow.py`,
  `unit/test_command_validation.py`, `integration/test_integration.py`,
  `chaos/test_chaos.py`, `utils/test_helpers.py`, `load/test_load.py`.
  See `simulations/README.md`. The suite is now 159 passed / 1 skipped, and
  every remaining test exercises real SDK behaviour.
- `tests/test_compliance.py` renamed to `tests/compliance_runner.py`. It was a
  CLI conformance runner requiring a live server, not a pytest file, and it
  contributed 0 tests while emitting 3 collection warnings.

### Fixed
- **Five integration tests could not fail.** They did
  `return suite.failed_tests == 0`; pytest treats any non-`None` return as a
  pass, so a genuinely failing workflow was reported green. They now
  `assert suite.failed_tests == 0, suite.get_summary()`. pytest has warned
  that returning a value "will be an error in a future version".
- `python_classes` widened from `["Test*"]` to `["Test*", "*Test", "*TestSuite"]`
  so classes named `*TestSuite` are collectable at all. 15 further tests remain
  uncollectable because pytest refuses classes defining `__init__` and those
  classes are also instantiated directly; fixing that needs the state setup
  moved into `setup_method`.
- `SecurityVulnerabilityTestSuite::test_insecure_defaults` failed once
  collection was enabled. Its `_check_config_security` helper had no rule for
  hardcoded `admin_user`/`admin_pass` or for disabled TLS. This was a bug in a
  test-local simulation, not a product defect, and the file has been moved to
  `simulations/` rather than patched.

### Changed
- **Distribution renamed to `physicalcontextprotocol`.** The PyPI name was
  previously `pcp`, which is **already taken on PyPI** by an unrelated
  project ("PCP - Progressive MCP", github.com/Consiliency/pcp) — so that
  name was never publishable. Use `pip install physicalcontextprotocol`.
- Import namespaces are unchanged (`pcp`, `sdk`, `v05`), so
  `from pcp import PCPClient` keeps working. The parallel-implementation
  consolidation is tracked in `pmcp-spec/MIGRATION_MAP.md` and is not
  addressed by this rename.

## [1.0.0] — 2026-09-28

First tagged public release.

### Verified

- **160 tests collected. 159 pass, 1 skip** on a default
  `pip install -e ".[dev,numerics]"`.

  The single skip is `test_rule_constructed_missing_torch_fails_fast`
  in `tests/v05/test_hnn_gate.py`, which cannot run when torch *is*
  installed (it asserts the missing-dependency path). It skips, it does
  not silently pass.
- The 42-test `pmcp-conformance` suite passes against this SDK.

  Earlier drafts of this changelog, the organization README, and
  `LIMITATIONS.md` all claimed **213 tests passing**. That figure did
  not reproduce on any run, so it has been corrected to the real count
  everywhere. A number that cannot be reproduced is worse than a smaller
  one that can.
- **Fencing tokens** (Kleppmann 2016) are implemented with per-zone
  monotonic counters and staleness rejection.
- **E-Stop is a real actuation-blocking latch**, not an advisory flag —
  `pcp/estop` bypasses the gate sequence at the message-handling layer,
  and this is covered by the conformance suite rather than only
  described in the spec.

### Fixed

Defects that static review missed and that were found by **executing
the code** — listed here because they are the argument for running the
suite rather than reading it:

- An exception hierarchy bug that crashed **every** safety-block path.
- A silently duplicated middleware definition where the weaker version
  won.
- 14 async tests that were never actually executing.
- A result-discarding bug in the batch actuation path.
- `pyproject.toml` packaging fixes: six broken `console_scripts` entry
  points removed (their targets moved out of this package during the
  split, and they were coroutines regardless); the
  `packages.find.include` list cut from 19 patterns to the three that
  actually ship (`v05*`, `pcp*`, `sdk*`); and `readme` corrected from
  the non-existent `docs/README.md` to `README.md`.
- `tests/__init__.py` added to the tests directory and each
  subdirectory so pytest imports the suite as a package.

### Changed

- **The PyPI publish job is now conditional on the `release`
  environment existing.** It previously fired unconditionally on any
  `v*.*.*` tag and would have failed, because OIDC trusted publishing
  is not configured yet. Tagging `v1.0.0` is therefore safe. Once
  trusted publishing is configured on PyPI, remove the
  `needs`-style guard noted in `.github/workflows/ci.yml`.
- Split out of the monorepo into its own repository, so this SDK reads
  as a peer of `pmcp-typescript` and `pmcp-rust` rather than as the
  reference implementation they are bindings of.

### Known limitations (documented, not fixed)

- **Duplicate implementations remain.** `pcp/`, `sdk/`, and `v05/`
  each ship a `PCPServer` / `PCPClient` / `ShadowPreview`, and there
  are at least four distinct client implementations. This is the
  highest-value open contribution in this repository.
- PyPI publishing is configured but not yet enabled; `pip install pcp`
  does not resolve to this project yet.

## [Unreleased]

### Added
- Optional Gate 4 — `HNNConservationRule` checks Hamiltonian energy conservation
  via a trained neural network (`H(q, p) → ℜ`). Off by default; activate with
  `pip install pcp[hnn]` and pass `hnn_rule=` to `SafetyMiddleware`.
- New extra: `pip install pcp[hnn]` (torch + numpy).
- 12 new tests for Gate 4: off-by-default, opt-in, consistent motion passes,
  conservation violation surfaces warning, stats tracking.
- `tests/__init__.py` (and one per subdirectory) so pytest imports the
  suite as a package.

### Changed
- `SafetyMiddleware.check()` now returns all accumulated warnings (not just
  shadow or constitution) — HNN WARNING-severity messages appear in the
  violations list without blocking the call.
- `[tool.setuptools.packages.find]` reduced from 19 include patterns to
  three (`v05*`, `pcp*`, `sdk*`), matching what actually lives in this
  package.

### Removed
- Six broken `[project.scripts]` entry points (`pcp-hub`,
  `pcp-marketplace`, `pcp-depin`, `pcp-multisig`, `pcp-swarm-coo`,
  `pcp-compliance`) whose target modules moved out of this package
  during the pmcp-org split and whose target functions were coroutines
  in any case. Only `pcp-server`, `pcp-demo`, `pmcp-registry`
  remain — the three that back the shipping `v05/` code.

### Fixed
- `pyproject.toml` `readme = "docs/README.md"` pointed at a directory
  that does not exist here; corrected to `README.md`.

## [0.5.0] — 2026-06-07

The first release of the v0.5 line: MCP 2024-11-05 wire compatibility with a
mandatory three-gate safety pipeline. This release also marks a deliberate
scope cut: many experimental features from earlier versions were removed so
the protocol surface is small, testable, and honest about what it ships.

### Added
- `PCPServer` — JSON-RPC 2.0 server over stdio and HTTP, exposing robot
  actuations as MCP tools, sensors as MCP resources, and mission templates
  as MCP prompts.
- `PCPClient` — client with `connect_stdio`, `connect_http`, and
  `connect_server` (in-process) transports.
- `SafetyConstitution` with 6 default rules: speed limit, floor guard,
  workspace box, energy budget, human proximity, E-Stop, plus an optional
  force-limit rule.
- `ShadowSimulator` — three-engine ladder: PyBullet → 6-DOF forward-kinematic
  reachability → geometric bounding-box. Each preview reports which engine
  actually ran.
- `pcp_kinematics` — DH-parameter forward kinematics for 6-DOF arms, with a
  UR5e preset, joint-limit checking, and reachability tests.
- `pcp_replay` — recorded-trajectory replay harness. JSONL in, pipeline out,
  with a canonical pick-and-place reference trajectory and an unsafe
  trajectory that the pipeline correctly blocks.
- `PCPRegistry` — in-process + HTTP-backed fleet registry, MCP registry
  format compatible.
- `RobotIdentity` — W3C DID + Ed25519 (via `cryptography`).
- Three console-script entry points: `pcp-server`, `pcp-demo`,
  `pmcp-registry`.
- Three reference robot servers (arm, mobile, agricultural) as illustrative
  clients of the decorator API.
- 213-test suite with 82% coverage on the shipping `v05/` code.
- Dockerfile and docker-compose for the reference stack.

### Changed
- **Scope cut**: many non-shipping top-level modules removed from the
  packaged wheel (cloud, SDKs in Rust/TypeScript, marketplace, depin,
  multisig, swarm, compliance, edge, ledger, registry-server,
  simulator, community, reference_impl, tee-attestator,
  physics-federation, gateway, safety-loop, ros2-bridge). Earlier
  protocol versions (v0.1–v0.4) are preserved in `pmcp-labs/` for
  reference.
- **Core dependencies reduced** from `cryptography + torch + numpy` to just
  `cryptography`. The shadow sim works without numpy using a pure-Python
  math path; numpy/pybullet are now optional extras.
- **README** rewritten to be honest about scope: drops the "USB-C for Mars"
  framing and the long "Planned" list, focuses on what `pip install pcp`
  actually delivers.

### Fixed
- `pyproject.toml` build backend was `setuptools.backends.legacy:build`, a
  non-existent module that made `pip install pcp` fail silently. Now
  `setuptools.build_meta`.
- Module self-imports in `v05/` (e.g. `from v05.pcp_v5_types import …`)
  were unreachable without a working install. Now reachable.
- `CommandValidator` (test-only utility) was missing error messages for
  unknown command types and had over-verbose messages for missing fields.
- `FleetManager` (test-only utility) was not setting `status = "pending"`
  on `assign_task`, so `get_fleet_stats()["pending_tasks"]` was always 0.
- Three CLI console scripts (`pcp-server`, `pcp-demo`, `pmcp-registry`)
  referenced `main` / `main_cli` entry points that did not exist. Added.
- `v05/__init__.py` was missing `ShadowSimulator`, `PCPRegistry`, and
  `RegistryEntry` from the public surface. Added.

### Known limitations
- The shadow simulator's reachability check is a workspace-sphere test; it
  does not solve inverse kinematics. A deployment that needs joint-level
  IK validation should call into MoveIt or the vendor IK solver and feed
  the result back through the constitution.
- Reference robot servers (arm, mobile, agricultural) are illustrative.
  No real-robot driver ships in v0.5.
- The HTTP transport uses stdlib only; the HTTP *registry server* is
  optional and requires `aiohttp`.

## [0.1.0] – [0.4.0] — earlier development

See `pmcp-labs/v01/` … `pmcp-labs/v04/` for the source of the
v0.1–v0.4 series. These versions are preserved for historical reference
and are not on the supported path.
