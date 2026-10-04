# simulations/

These files are **not tests**. They are self-contained harnesses that were
previously under `tests/`, where pytest collected them and reported them as
passing. None of them import `pcp`, `v05`, or `sdk`, so they never exercised
any product code — they asserted against simulators defined in the same file
and could not fail because of a defect in `physicalcontextprotocol`.

They were moved here on 2026-10-04 so that the pytest count reflects real
coverage. `pyproject.toml` sets `testpaths = ["tests"]`, so this directory is
excluded from collection by construction.

| File | What it simulates | Collected tests it had been contributing |
|---|---|---|
| `security/test_mtls.py` | A fake CA, TLS handshakes, rate limiting, and an injection/XSS/path-traversal auditor | 6 |
| `e2e/test_fleet_workflow.py` | Fleet registration, task dispatch, telemetry streaming | 23 |
| `unit/test_command_validation.py` | Command schema validation | 15 |
| `integration/test_integration.py` | Component simulator + throughput/load runners | 7 |
| `chaos/test_chaos.py` | Failure injection | 4 |
| `utils/test_helpers.py` | Dataclasses used as fixtures by the above | 4 |
| `load/test_load.py` | Concurrent load generation | 3 |

Two real defects were fixed in `integration/test_integration.py` before the move,
because they are worth keeping regardless of where the file lives:

- Five module-level tests did `return suite.failed_tests == 0`. pytest treats any
  non-`None` return as a pass, so they reported green even when the underlying
  workflow failed. They now `assert suite.failed_tests == 0, suite.get_summary()`.
- `pytest.ini_options.python_classes` was widened from `["Test*"]` to
  `["Test*", "*Test", "*TestSuite"]`.

Note that `MTLSTestSuite`, `RateLimitTestSuite`, and `WorkflowTest` are still not
collectable: pytest refuses classes that define `__init__`, and these are also
instantiated directly. Converting them would mean moving their state setup into
`setup_method` and updating the direct call sites in `run_all_tests()`.

## If you want real coverage here

Rewrite each file against the actual SDK surface. For mTLS, that means asserting
on `physicalcontextprotocol`'s real identity and transport types — the DID and
capability-token types in `pcp`, and `TcpClientTransport` / `ConnectionPool` in
the Rust core — rather than a simulated CA defined in the test file.

## Running them directly

```bash
python -m simulations.security.test_mtls
python -m simulations.integration.test_integration
```
