# Security policy — pmcp-spec

The default policy for this organization lives in
[`pmcp-spec/SECURITY.md`](SECURITY.md) and applies here in full. This
file records what is specific to the specification repository.

## Reporting

Use **private vulnerability reporting**:
**Security → Report a vulnerability** on this repository, or
[open an org-level advisory](https://github.com/physicalcontextprotocol/security/advisories/new).

Do not open a public issue.

## In scope here

- A flaw in the normative parts of the wire protocol that would let an
  implementation pass the conformance suite while violating a stated
  safety property — most importantly the gate ordering
  **E-Stop → Lease → Constitution → Shadow**.
- A gap in the formal models (`formal/`) that a careful reader would
  reasonably read as covering a case it does not cover.
- A mismatch between `schema/v0.6.0/pmcp.schema.json` and
  `docs/PROTOCOL_SPEC.md` that would let two conforming
  implementations disagree on a safety-relevant field.
- A leaked secret or credential anywhere in this repository.

## Out of scope here

- Disagreement with a design decision. That is a proposal — open an
  issue or a PR.
- Defects in an implementation. Report those against
  `pmcp-python` / `pmcp-typescript` / `pmcp-rust`.
- Incomplete coverage the repository already documents, such as the
  missing per-method JSON-RPC schema registry noted in
  `schema/v0.6.0/README.md`.

## Supported

`v0.5` protocol line and JSON Schema `v0.6.0`. Earlier drafts are not
maintained.

## Read before relying on this spec

[`LIMITATIONS.md`](LIMITATIONS.md) lists what is verified, and — more
importantly — what is not, including the open research problems that
require physical hardware to close. If Shadow validation, conformal
prediction, or UWB localization is load-bearing for your safety case,
validate those specific components on your own hardware first.
