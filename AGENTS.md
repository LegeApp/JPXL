# AGENTS.md — working rules for JPXL

## 1. Project identity

JPXL is a clean-room implementation of JPEG XL (ISO/IEC 18181) in Rust: a
decoder and an encoder, dual-licensed `MIT OR Apache-2.0`. The Rust workspace
lives in `JPXL/`; this file sits at the repo root and governs the whole tree.

**Decoder first.** The standard specifies the decoder — the inverse process is
the normative object. Encoding has substantial freedom, so an encoder written
before a working decoder is guesswork validated by nothing. The decoder turns
normative text into executable understanding and becomes the roundtrip oracle
for the encoder that follows.

**The standard is the source of truth.** Not libjxl, not any prior
implementation, not habit. Where the standard is unavailable (Parts 1–3 are
un-OCRed as of 2026-08-02), the fallback chain in section 2 applies and every
resulting decision is tagged `[provisional]` until confirmed against the text.

**Slow is smooth, smooth is fast.** The goal for the first phase is a correct,
readable reference pair — scalar, single-threaded, obvious. No SIMD, no rayon,
no tuning constants. But the foundations that the previous attempt lacked go in
from day one: bit-position tracing before the first field is parsed, typed
stage boundaries, and limits on attacker-controlled sizes. Those are not
optimizations; they are the things that make later optimization possible.

## 2. Clean-room rules

**The old AGPL tree is off limits.** A previous `jxl-encoder` project was
licensed `AGPL-3.0-only OR LicenseRef-Imazen-Commercial`. It is not present in
this repository. Keep it that way. Never open it, never copy source, tests,
comments, tables, or documentation from it. Its durable *lessons* are already
extracted into `JPEG_XL_CLEAN_IMPLEMENTATION_LESSONS.md` — that file is the
only permitted channel of inheritance.

**`libjxl/` is a black-box oracle.** The checkout is BSD-3-Clause, which is
permissive, but a direct translation is still derived work and — more
importantly — libjxl mixes normative decoding semantics with one encoder's
heuristics, tuning, and platform history. You cannot tell them apart by
reading. Therefore:

- Permitted: run `cjxl`, `djxl`, `jxlinfo`; diff bitstreams and pixels; dump
  headers; compare timings.
- Forbidden: reading libjxl source to learn architecture, field order, table
  values, or algorithm structure; copying code or constants.
- Codegraph and grep hits inside `libjxl/` are **oracle territory, not
  architecture guidance**. Seeing them in a search result is not permission to
  read them. Scope searches to `JPXL/` unless you are debugging an oracle
  invocation.

**Ambiguity resolution order.** When the bitstream semantics are unclear, work
down this list and stop at the first that answers:

1. `markdowns/2506.05987v2.md` — the open-access JPEG XL paper. Primary
   readable source today; has syntax figures for image and frame headers.
2. OCR markdown of the relevant part in `markdowns/standard-markdowns/`, once
   it exists.
3. Page-ranged read of the PDF in
   `original-pdfs-do-not-read-first-if-markdown-exists/` — expensive, see
   section 3.
4. A documented behavioral experiment against the oracle, written up in
   `JPXL/docs/experiments/`. This is evidence about one implementation, not
   about the standard; label conclusions accordingly.

## 3. Document access order (token economy)

PDFs in this repo are image-only scans. Reading one costs a rendered page image
per page — orders of magnitude more tokens than the equivalent markdown. So:

- **Always check `STANDARDS_INDEX.md` first** for the current conversion status
  of each part. Do not assume; the status changes as the user completes OCR.
- **Always read the markdown conversion if one exists and is not a stub.**
- Open a PDF only when the markdown is missing (Parts 1–3 today) or visibly
  garbled, and then only with an explicit page range derived from
  `STANDARDS_INDEX.md` or the PDF's own table of contents. Never read a
  standards PDF end to end.
- `markdowns/`, `original-pdfs-do-not-read-first-if-markdown-exists/`,
  `libjxl/`, and `test-set/` are gitignored. ISO text is copyrighted and must
  never enter git history.

## 4. Doc map

| Document | Purpose |
| --- | --- |
| `STANDARDS_INDEX.md` | Which spec part is where, its conversion status, what it covers, and the clause → crate crosswalk. |
| `JPXL/docs/PLAN.md` | Vertical-slice implementation plan, blocking graph, bit-exactness contract, out-of-scope list. |
| `JPXL/docs/CHANGELOG.md` | Released, user-visible changes only. Nothing else. |
| `JPXL/docs/CONFORMANCE.md` | Which clauses/features are supported and the status of the tests that prove it. |
| `JPXL/docs/PERFORMANCE.md` | Current reproducible baselines with full provenance. Not a history of attempts. |
| `JPXL/docs/HANDOFF.md` | Dated working ledger between sessions and agents. |
| `JPXL/docs/experiments/` | Immutable experiment reports, including negative results. |

**Keep living docs small.** The previous project's changelog reached 6,965
lines and became a research database nobody could read. Delete stale
hypotheses instead of accumulating them; git preserves history. Only
`docs/experiments/` is append-only, and its entries are frozen once written.

## 5. Build and test

```
cd JPXL
cargo build   --workspace
cargo test    --workspace
cargo clippy  --workspace --all-targets -- -D warnings
cargo fmt     --all --check
```

All four must pass before any handoff. Toolchain is pinned (1.97.1, edition
2024); do not bump it as a side effect of another change.

Oracle and corpus setup (both need network, both are one-time):

```
JPXL/tools/setup-oracles.sh      # build/fetch libjxl binaries used as oracle
JPXL/tools/fetch-conformance.sh  # official conformance streams, hashed
```

## 6. Engineering conventions

**Near-zero external dependencies.** The normative crates take no third-party
runtime dependencies without an explicit decision recorded in `PLAN.md`. That
includes error-handling crates: hand-roll a per-crate error enum with `From`
impls chaining the layers. No `thiserror`, no `anyhow`.

**Unit-bearing newtypes at every transform boundary.** The previous project
died on exactly this: `ForwardScale` confused with `InverseScale`, quant weight
with its reciprocal, block coordinates with pixel coordinates, raw with shifted
nonzero counts. Bare `f32`/`usize` are banned at transform, quantization, and
coordinate boundaries. Name the unit in the type.

**Validate early, serialize late.** Parse into validated typed structures at
the edge; keep raw bytes and bit offsets out of the middle of the pipeline.

**Encoder and decoder are peer trees over shared neutral crates.** Never nest
one inside the other. They may share syntax *types*; they must not share enough
*implementation* that an encoder bug is automatically accepted by the paired
decoder.

**Tracing before parsing.** Bit-position tracing exists before the first field
is read or written, feature-gated and zero-cost when off. A single misplaced
conditional field shifts every subsequent field and surfaces as a nonsense
error hundreds of bits downstream; without a trace you debug the symptom.

**Roundtrip against self, then parity against the oracle.** In that order.
And roundtrip layer by layer — entropy coding alone, transform alone, header
alone — before anything end to end. A transform that roundtrips in isolation
can still be wrong for the wire layout, so every shape needs separate tests for
values, storage order, LLF/DC extraction, and symbol order.

**The bit-exactness contract table lives in `PLAN.md`.** Know before you write
a test whether the path you are testing is bit-exact or tolerance-based.

**Decode paths are attacker-facing.** Bounds, allocation caps, checked
arithmetic, and rejection of malformed input from the first commit, not as a
hardening pass later. Lints enforce part of this: `unsafe_code` is denied,
`unwrap_used` / `indexing_slicing` / `cast_possible_truncation` warn. Do not
silence them with `#[allow]` without a comment saying why.

**Multi-group fixtures from day one.** Any path that can see more than one
group gets a ≥256×256 fixture in its tests. Single-group fixtures hide
conditional fields, region handling, and edge clipping.

**Confidence comes from proved invariants, not test count.** The old project
had hundreds of green tests while VarDCT rendered garbage. State what a test
proves; if you cannot, it proves nothing.

## 7. Handoff discipline

`JPXL/docs/HANDOFF.md` is the working ledger. Prepend dated entries (newest
first). Every entry states what changed, what is proved, and what is next.

Two sections are permanent and must be maintained:

- **"Already fixed — do not redo"**: settled decisions and repaired bugs, so
  the next session does not relitigate them.
- **"Traps — do not fix these by loosening a check"**: places where a failing
  assertion or strict limit is correct and the bug is upstream. Name the check
  and the real cause.

When a diagnosis turns out wrong, correct it **in place** and say it was
corrected. Do not leave a wrong explanation standing with a rebuttal appended
three entries later.

## 8. Multi-agent ownership

Work is dispatched as briefs that list whole files. Rules:

- A task owns the **entire** files in its brief and edits nothing outside that
  set. No opportunistic fixes in someone else's file.
- Shared types in `jpxl-core` change only through the task that owns
  `jpxl-core`. If you need a new shared type, state the requirement in your
  handoff entry rather than adding it yourself.
- If two briefs appear to overlap, stop and report the conflict; do not
  arbitrate it by editing first.

## 9. Legal

- License: `MIT OR Apache-2.0`. Full texts in `JPXL/LICENSE-MIT` and
  `JPXL/LICENSE-APACHE`.
- **Never commit ISO text.** Not in code comments, not in doc files, not in
  test data, not in commit messages. Clause-number citations ("per 18181-1
  §C.2") are fine and encouraged; quoted passages are not. Paraphrase
  requirements in original language.
- Every fixture gets a provenance sidecar: where it came from, its license, its
  hash, and how to regenerate it. Fixtures without provenance do not merge.
- The software license and the patent position are separate questions. Nothing
  here is legal advice.
