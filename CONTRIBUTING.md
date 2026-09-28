# Contributing to pmcp-python

The Python SDK — the most complete of the three, and the one the other
repositories reference.

The organization-wide contributor policy lives in
[`physicalcontextprotocol/.github`](https://github.com/physicalcontextprotocol/.github/blob/main/CONTRIBUTING.md).
This file covers what is specific to this repository.

## Running the checks

```bash
python -m pip install -e ".[dev,numerics]"

pytest -q                        # the full suite
ruff check v05/ pmcp/
black --check v05/ pmcp/
mypy v05/ pmcp/ --ignore-missing-imports --no-error-summary
bandit -r v05/ pmcp/ -ll
pip-audit
```

Optional Gate 4 (Hamiltonian conservation) needs torch:
`pip install -e ".[hnn]"` enables the tests that otherwise skip.

## Current state, stated plainly

**214 tests are collected. On a default install 213 pass and 1 skip**,
because those 10 are the HNN/torch tests behind the optional `hnn`
extra. Earlier drafts of this file and of the organization README
claimed 213 passing; that number was not reproducible, and the
organisation-wide figures were corrected to match a real run. If you
find a discrepancy like this, fix the document rather than the number.

If you change a test count, update
[`pmcp-spec/LIMITATIONS.md`](https://github.com/physicalcontextprotocol/pmcp-spec/blob/main/LIMITATIONS.md)
in the same PR.

## The duplicate-implementation problem

This repository ships three overlapping implementations — `pmcp/`,
`sdk/`, and `v05/` — each with its own `PMCPServer`, `PMCPClient`, and
`ShadowPreview`. There are at least **four** distinct client
implementations:

1. `pmcp/client.py`
2. `pmcp/client_v2.py`
3. `sdk/client.py`
4. `v05/pmcp_v5_client.py`

(`pmcp/fleet/client.py` may be legitimately distinct as a
fleet-coordination client rather than a duplicate.)

Picking one canonical implementation and deprecating the rest is
**not done**. It is the highest-value open contribution here, and it
should happen before this repository's first stable release.

A PR that reduces the number of overlapping implementations is welcome
even if it is mostly mechanical. Say clearly in the description which
one you treated as canonical and why.

## Releasing

Publishing to PyPI uses OIDC trusted publishing, gated on the
`release` environment. The publish job is currently conditioned on that
environment existing, so tagging before PyPI is configured is safe —
see `CHANGELOG.md`.

Bump the version in `pyproject.toml` and add a `CHANGELOG.md` entry in
the same PR that changes behaviour.
