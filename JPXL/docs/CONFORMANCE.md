# CONFORMANCE

What JPXL supports and the tests that prove it. One row per feature or clause
group. A row may only claim "supported" when a named, passing test backs it —
"the code exists" is not support, and "it decodes without error" is not
correctness.

Every lossy row must state which contract applies (bit-exact vs. Part 3
peak-error class; see the bit-exactness contract table in `PLAN.md`).

Clause numbers stay `[provisional]` (arXiv paper section numbers) until
Part 1/2/3 OCR lands — see `STANDARDS_INDEX.md`.

Last reviewed: 2026-08-02.

## Status

| Clause `[provisional]` | Feature | Contract | Test | Status |
| --- | --- | --- | --- | --- |
| — | — | — | — | Nothing supported yet. Scaffold wave in progress. |

## Official conformance corpus

Fetched by `tools/fetch-conformance.sh` into
`tests/fixtures/conformance/`, pinned by revision and hash. Not yet run.

| Stream set | Revision | Result |
| --- | --- | --- |
| — | — | not yet fetched |

## Malformed-input coverage

Decode paths are attacker-facing. Every rejection case gets a row: what is
malformed, and that JPXL rejects it without panic, unbounded allocation, or
unbounded CPU.

| Case | Expected | Status |
| --- | --- | --- |
| — | — | not started |
