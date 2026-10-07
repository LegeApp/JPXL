# AGENTS.md — working rules for JPXL

## 1. Project identity

JPXL is a clean-room implementation of JPEG XL (ISO/IEC 18181) in Rust: a
decoder and an encoder, licensed under `MIT`. The Rust workspace
lives in `JPXL/`; this file sits at the repo root and governs the whole tree.

**Decoder first.** The standard specifies the decoder — the inverse process is
the normative object. Encoding has substantial freedom, so an encoder written
before a working decoder is guesswork validated by nothing. The decoder turns
normative text into executable understanding and becomes the roundtrip oracle
for the encoder that follows.

**The standard is the source of truth.** Not libjxl, not any prior
implementation, not habit. Where the standard does not answer, the fallback
chain in section 2 applies and the decision is tagged `[provisional]`. Existing
`[provisional]` tags predate the OCR and now require re-audit against
`part1.md` — any remaining re-audit work is tracked in AKR and surfaced in
`docs/generated/ACTIVE-WORK.md`.

**Slow is smooth, smooth is fast.** The goal for the first phase is a correct,
readable reference pair — scalar, single-threaded, obvious. No SIMD, no rayon,
no tuning constants. But the foundations that the previous attempt lacked go in
from day one: bit-position tracing before the first field is parsed, typed
stage boundaries, and limits on attacker-controlled sizes. Those are not
optimizations; they are the things that make later optimization possible.

**Phase two: measured performance, same readability bar.** The decoder is
conformant, so its hotspots may now parallelize and vectorize — but only
behind proof, and scalar-first still governs all new code: no path
parallelizes or vectorizes before its scalar form is conformant and reviewed.
`std::thread::scope` is the default threading tool (no new dependency); rayon
is allowed where work-sharing earns it, subject to §6's dependency decision —
regular row-band and map-scatter loops do not need it. SIMD is allowed
in-crate with runtime dispatch after the `jpxl-perceptual` precedent, subject
to three rules: the scalar path stays as the readable reference and the
lock-step oracle; the vectorized path is bit-identical to it (lane-mapped
work only — no reordered reductions, no fast-math, per the float policy in
`PLAN.md`); and every claim carries a before/after measurement on a pinned
fixture. Threading that only partitions independent samples is bit-identical
by construction and needs no tolerance argument; anything that reorders float
operations needs a Part 3 peak-error grade instead.

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

1. `latex/part1.tex` — the LaTeX transcription of Part 1. **Canonical for
   Part 1.** It was produced by a high-capability agent reading the scan page
   by page (not machine OCR) and carries the pseudocode bodies and syntax
   tables that `part1.md` truncates or garbles.
2. `markdowns/standard-markdowns/part1.md` … `part4.md`. `part1.md` is
   SECONDARY for Part 1: a grep/structure-search aid, not an authority.
   `part2.md`–`part4.md` are the primary source for their parts.
   One safeguard survives the ranking: if the two Part 1 sources disagree on a
   **numeric constant**, treat the disagreement itself as a signal and settle
   it via step 3 or 5 — a handful of digit slips exist in both directions
   (see the transcription decisions and policies in
   `docs/generated/CURRENT-STATE.md`), and a wrong constant parses plausibly.
3. `original-pdfs-do-not-read-first-if-markdown-exists/ISO_IEC_18181-1_2024_transcription.pdf`
   — text-only transcription of Part 1, text-searchable and cheap relative to
   the image scans. Use it when 1 and 2 disagree.
4. `markdowns/2506.05987v2.md` — the open-access JPEG XL paper. Design
   rationale and cross-checking, not normative.
5. Page-ranged read of the original image scan in
   `original-pdfs-do-not-read-first-if-markdown-exists/original/` — last resort,
   expensive, see section 3.
6. A documented behavioral experiment against the oracle, written up in
   `JPXL/docs/experiments/`. This is evidence about one implementation, not
   about the standard; label conclusions accordingly.

## 3. Document access order (token economy)

PDFs in this repo are image-only scans. Reading one costs a rendered page image
per page — orders of magnitude more tokens than the equivalent markdown. So:

- **Always check `STANDARDS_INDEX.md` first** for where each source lives and
  its status. All four parts are converted; Part 1 additionally has
  `latex/part1.tex`.
- **For Part 1, read `latex/part1.tex`.** Grep `part1.md` to find the clause
  fast, then read the clause body from the LaTeX — the OCR truncates pseudocode
  bodies and misreads digits in tables and examples.
- For Parts 2–4, the markdown is the working normative text.
- Caveat: dense syntax-table and formula pages can still scramble. Where a
  value looks implausible, cross-check against `latex/part1.tex`, then the
  text-only transcription PDF, before spending an image-scan page read.
- Open an image-scan PDF only as a last resort, and then only with an explicit
  page range against
  `original-pdfs-do-not-read-first-if-markdown-exists/original/`. Never read a
  standards PDF end to end.
- `markdowns/`, `original-pdfs-do-not-read-first-if-markdown-exists/`,
  `latex/`, `libjxl/`, and `test-set/` are gitignored. ISO text is copyrighted
  and must never enter git history.

## 4. Doc map

| Document | Purpose |
| --- | --- |
| `STANDARDS_INDEX.md` | Which spec part is where, its conversion status, what it covers, and the clause → crate crosswalk. |
| `JPXL/docs/PLAN.md` | Vertical-slice implementation plan, blocking graph, bit-exactness contract, out-of-scope list. |
| `JPXL/docs/CHANGELOG.md` | Released, user-visible changes only. Nothing else. |
| `JPXL/docs/CONFORMANCE.md` | Which clauses/features are supported and the status of the tests that prove it. |
| `JPXL/docs/PERFORMANCE.md` | Current reproducible baselines with full provenance. Not a history of attempts. |
| `JPXL/docs/HANDOFF.md` | Retired legacy pointer. Live handoff and planning state is generated from AKR. |
| `JPXL/docs/experiments/` | Immutable experiment reports, including negative results. |

**Keep living docs small.** The previous project's changelog reached 6,965
lines and became a research database nobody could read. Delete stale
hypotheses instead of accumulating them; git preserves history. Only
`docs/experiments/` is append-only, and its entries are frozen once written.

## 5. Build and test

```
cd JPXL
cargo build   --workspace
cargo test    --workspace --release
cargo clippy  --workspace --all-targets -- -D warnings
cargo fmt     --all --check
```

All four must pass before any handoff. Toolchain is pinned (1.97.1, edition
2024); do not bump it as a side effect of another change.

For the edit/build/test loop use the `fast-debug` profile
(`cargo test --workspace --profile fast-debug`): optimised like `release` but
without LTO, with 16 codegen units and incremental compilation, and with
debug assertions and overflow checks kept on, so the full workspace suite runs
in well under a minute while the codec's `debug_assert!` shape guards still
fire. `release` (thin LTO, one codegen unit) remains the profile for
benchmarks, perf profiles, promoted timings and shipped binaries. Keep target
directories on disk (`target/`, or under `.agent/scratch/` for one-off
builds), never on tmpfs.

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

## 7. AKR handoff discipline

AKR is the working ledger and the generated views under `docs/generated/` are
the reading surface. Do not prepend dated Markdown handoff entries or maintain a
parallel plan:

- Start scoped work with `knowledge.context --goal <record>` and the expected
  touched paths.
- Record settled fixes and decisions as `decision`; record traps and strict
  checks that must not be loosened as `policy`.
- Record measurements as `observation` or `evidence`, with raw logs under
  `.agent/scratch/`; close acceptance through `knowledge.complete`.
- When a diagnosis changes, use `knowledge.revise` or `knowledge.supersede` so
  the ledger exposes one current claim and preserves the history explicitly.
- Before handoff, build the generated views and run `knowledge.validate` (or
  `akr check`), reporting any remaining diagnostics.

`JPXL/docs/HANDOFF.md` is a legacy tombstone only. Historical detail remains in
Git history; durable conclusions have been migrated to AKR records.

## 8. Multi-agent ownership

Work is dispatched as briefs that list whole files. Rules:

- A task owns the **entire** files in its brief and edits nothing outside that
  set. No opportunistic fixes in someone else's file.
- Shared types in `jpxl-core` change only through the task that owns
  `jpxl-core`. If you need a new shared type, state the requirement in your
  AKR work record or checkpoint note rather than adding it yourself.
- If two briefs appear to overlap, stop and report the conflict; do not
  arbitrate it by editing first.
- **One exception:** whitespace-only reformatting by `cargo fmt`/`rustfmt` may
  land in files outside your brief. See the git notes in section 10 — do not
  revert it to keep your diff narrow.

## 9. Legal

- License: `MIT`. Full text in `LICENSE` (mirrored at `JPXL/LICENSE-MIT`).
- **Never commit ISO text.** Not in code comments, not in doc files, not in
  test data, not in commit messages. Clause-number citations ("per 18181-1
  §C.2") are fine and encouraged; quoted passages are not. Paraphrase
  requirements in original language.
- Every fixture gets a provenance sidecar: where it came from, its license, its
  hash, and how to regenerate it. Fixtures without provenance do not merge.
- The software license and the patent position are separate questions. Nothing
  here is legal advice.

## 10. Tooling — what's actually worked, from doing the work

This section is deliberately concrete and revisable: it reflects what actually
happened using these tools on this codebase, not their marketing description.
Update it when experience contradicts it. If you're porting this section to
another project's AGENTS.md, keep the "what's actually worked" framing and
replace the specifics with that project's own experience — don't just copy
these bullets verbatim into a codebase where they weren't earned.

### AKR (`.akr/`, `akr` CLI, and the `knowledge.*` MCP tools)

Durable project knowledge — plans, decisions, evidence, assessments — lives in
`.akr/` as typed records, not in Markdown. `docs/generated/` is build output.
This has worked well for exactly the case it's built for: a long chain of
scope → measure → build → verify → close-out cycles where you need each step
to cite the evidence for the one before it, and where "is this actually done"
should be a checkable fact, not a vibe.

**Before starting any task**
1. `knowledge.context --goal <milestone|work|track>`, with `--paths` for the
   files you expect to touch.
2. Read the bundle in full. Contradictions and staleness warnings are never
   noise — they're the whole point of asking first.

**While working**
- Look things up with `knowledge.get`; find them with `knowledge.search`.
  Search ranks results; it never grants authority. A record's standing comes
  from its state, its scope, and its relations.
- Scratch notes go in `.agent/scratch/` (gitignored — raw run logs, A/B
  harness output, one-off scripts live there, not in the ledger or in git).

**When something becomes durable**
- New knowledge: `knowledge.propose`. Observations need `observed_at` and, if
  they can go stale, `watches`.
- Changed knowledge: `knowledge.revise`. Never edit a `.akr` file directly,
  and never edit a record that is not `proposed`.
- Replacing a plan: `knowledge.supersede`, with a disposition for every
  unfinished child. The tool lists them; answer each one.
- Finishing work: `knowledge.complete`, with evidence for every acceptance
  check. Evidence records state what was observed; they never state what
  they verify.

**Gotchas actually hit, not hypothetical:**
- `knowledge.get`'s MCP result truncates large records around ~1500 tokens.
  When that happens, `grep`/`Read` the raw `.akr/records/**/*.akr` source
  directly for the full text — reading the source files is explicitly
  allowed (only hand-*editing* them, and reading `.akr/cache/`, are
  forbidden).
- `knowledge.revise` silently resets state to the record class's initial
  state (e.g. a `completed` work record drops back to `proposed`) unless you
  explicitly re-pass `state` in the same call. Always pass `state` when
  revising a record that's already in a terminal or non-default state.
- `assessment` records cannot target `work` records via `supported_by`
  (validation rule V-005). Cite the work record in prose inside the
  `statement`/`note` slot instead — this is an established pattern in this
  ledger, not a workaround to reinvent.
- After a batch of MCP writes, run `akr build` (regenerates `akr.lock` and
  `docs/generated/`) before `knowledge.validate` / `akr check`, or validation
  sees stale hashes. `akr build` and `akr change prepare --staged` can take
  60–120s+ on a workspace this size — that's normal, not a hang.
- **The git integration is forward-only — do not retrofit it onto history.**
  `akr change begin/prepare` + `akr git commit` works well going forward: it
  stamps commits with `AKR-Change`/`AKR-Work`/`AKR-Evidence`/`AKR-Graph`/
  `AKR-Tree` trailers so `akr git log <record>` can find what commit did it.
  But `evidence` records durably embed `observed_at: git:<sha>`. If you
  rewrite history after evidence has been recorded against those commits
  (rebase, cherry-pick to add trailers after the fact, etc.), the old SHAs
  stop being ancestors of the branch and `akr check` fails
  (`AKR-G012`/`AKR-R022`: evidence "predates" the very completion it
  verifies). This was tried and reverted in this project (2026-08-08) —
  the fix is to always use `akr change`/`akr git commit` for the *original*
  commit, never to reconstruct it after the fact once evidence exists.

**Never:** edit `docs/generated/` by hand (regenerated, CI-checked); read
`.akr/cache/` (private cache); delete a record (move it to a terminal state
instead).

**Before handing back:** `knowledge.validate` (or `akr check`). If it reports
diagnostics, fix them or say so explicitly — don't hand back silently.

**Planning is in AKR, not Markdown.** The authoritative plan and its generated
views (`ROADMAP`, `CURRENT-STATE`, `DECISION-HISTORY`, `OPEN-QUESTIONS`,
`REVIEW-REQUIRED`, `PAPERCUTS` under `docs/generated/`) come from the ledger.
`PLAN.md` and `CONFORMANCE.md` remain legacy references pending full migration;
prefer the generated views when they conflict. `JPXL/docs/HANDOFF.md` is
retired and must not accumulate new entries.

Gate before finalizing (clean tree): `scripts/ci-akr.ps1` / `scripts/ci-akr.sh`
runs `akr check`, `akr check --views-current`, and `cargo fmt --check`.
Install with `cargo install --git https://github.com/LegeApp/AKR.git akr-cli`.

### git

- Hooks are installed (`akr git install-hooks` → `commit-msg`/`pre-commit`
  wrapping `akr git-hook`). Respect them; don't `--no-verify` around them
  without a stated reason.
- **`cargo fmt`/`rustfmt` reformatting unrelated files is fine — do not revert
  it.** A crate-wide `cargo fmt -p <crate>`, and sometimes even `rustfmt` on an
  explicit file list, will also normalise pre-existing drift in sibling files
  you never intended to touch. That is an acceptable, welcome side effect: the
  tree is meant to be `cargo fmt --all` clean, so any drift it removes was a
  latent failure of the `cargo fmt --all --check` gate in
  `scripts/ci-akr.sh`. Let the reformat stand and mention it in the commit
  message; do not hand-restore the old formatting to keep a diff narrow. This
  is a deliberate exception to the file-ownership rule in section 8, and it is
  the *only* one: whitespace-only reformatting by rustfmt is exempt, every
  other edit outside your brief is not.
  Still run `git status` / `git diff --stat` after formatting, not to revert
  but to *know* what moved — and if the reformat is large or touches crates far
  from your work, commit it separately from the change you were actually
  making, so the real diff stays reviewable.
- This checkout currently has no remotes configured, so local history can be
  rewritten without affecting anyone else — but see the AKR git-integration
  gotcha above before doing that on a branch with recorded evidence.

### codegraph (`mcp__codegraph__codegraph_explore`)

Available in this workspace but not exercised in the session that wrote this
section — Bash `grep`/`Read` covered every code-exploration need that came up
(enum definitions, trait impls, call sites across crates), so there's no
first-hand verdict here yet, positive or negative. Per the user's global
default, prefer it for genuinely structural questions (call graph, cross-file
dataflow, blast radius of a change) over grep once it's actually been tried on
this codebase — and per that same default, treat what it returns as a
hypothesis to verify by reading the real code, not as ground truth. One
project-specific rule regardless of which search tool surfaces it: a hit
inside `libjxl/` is oracle territory (section 2), not architecture guidance —
seeing it in a search result is not permission to read it that way.

### fff (`mcp__fff__find_files` / `grep` / `multi_grep`)

Was disconnected for most or all of the session that wrote this section, so —
same as codegraph — no fresh first-hand verdict. Bash `grep -n <identifier>`
was a fully adequate fallback throughout, helped by this codebase's habit of
dense, greppable doc comments (`fn foo`, `struct Bar`, clause citations like
`H.5.1` all grep cleanly). If fff is connected and working, prefer it per the
global default — but don't block on it being unavailable; a plain grep with a
precise identifier gets the same answer here.
