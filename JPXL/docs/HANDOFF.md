# HANDOFF

Dated working ledger. **Prepend** new entries — newest first. Each entry: what
changed, what is now proved, what is next, what is blocked.

Two sections at the bottom are permanent and must be kept current:
"Already fixed — do not redo" and "Traps — do not fix these by loosening a
check". When a diagnosis turns out to be wrong, correct it **in place** and
mark it corrected; do not leave a wrong explanation standing.

Keep this file small. Entries whose content has landed in `PLAN.md`,
`CONFORMANCE.md`, or `docs/experiments/` get deleted from here.

---

## 2026-08-02 — standard OCR landed and audited

**State:** ISO/IEC 18181 Parts 1–4 are now complete OCR markdowns at
`markdowns/standard-markdowns/part1.md` … `part4.md` (4230 / 735 / 325 / 163
lines). A first OCR pass was **rejected**: it dropped comparison and shift
operator glyphs (`<`, `<<`, `<=`, `>>`) — fatal for bitstream pseudocode — and
lost whole pages. The accepted pass has 30–48 % more words, intact operators,
text reflowed into paragraphs and code blocks, and the lost pages recovered
(Part 1 Annex N, Part 2 A.11). Caveat: dense syntax-table and formula pages can
still scramble; spot-check them against the original scan (now in
`original-pdfs-do-not-read-first-if-markdown-exists/original/`) before treating
the markdown as sole normative source.

`part1.md` is the primary normative source from now on; the arXiv paper drops
to design rationale and cross-checking. A LaTeX conversion of Part 1 from the
original scan is incoming and will land in a new repo-root `latex/` folder — it
will supersede `part1.md` as the highest-fidelity source for formulas and
tables.

**Queued:** re-audit every `[provisional]` tag in `jpxl-bitstream`,
`jpxl-core`, and `STANDARDS_INDEX.md` against `part1.md` — the topic map and
crosswalk are marked pending verification but not yet rewritten.

**Next:** slices 2 and 3 are unblocked; slice 3 remains the critical path.

---

## 2026-08-02 — scaffold wave complete

**State:** the five-task scaffold wave has landed and the full gate is green:
`cargo build/test/clippy -D warnings/fmt --check` across the workspace, 111
tests passing. What exists and is proved:

- `jpxl-bitstream`: `BitReader` (LSB-first), `Bool`/`U32`/`U64`/`F16`/
  `ZeroPadToByte`, feature-gated bit-position tracing (`trace`), 37 tests with
  hand-derived vectors. `longU64()` definition confirmed verbatim from the
  arXiv paper. F16 is pure bit-manipulation (deterministic on all targets).
- `jpxl-core`: error style established (`JpxlError`, hand-rolled, `From`
  chains); `Limits`/`AllocGuard` (charge-before-allocate); checked geometry
  newtypes; XYB forward constants taken verbatim from the paper (p. 24),
  inverse derived by exact rational inversion (verified to 1e-5) — all
  `[provisional]`; `dct.rs` with orthonormal DCT-II/III 8/16 (1-D, 2-D square
  and rectangular), naive-reference and coefficient-layout tests. JPEG XL's
  own scaling is a wrapper prefactor at the call boundary, never baked into
  the kernels.
- `jpxl-conformance`: `sniff` (FF0A / container box), oracle discovery+runner
  (djxl, jxl-oxide; `JPXL_ORACLE_BIN` override), PPM parser + peak-error
  metrics. `jxl-oxide` CLI invocation is `[verify at first use]`.
- `jpxl-cli`: `jpxl info <file>` works on all three handmade fixtures with the
  specified exit codes (0 recognized / 2 unknown / 1 I/O error).
- `tools/setup-oracles.sh` and `tools/fetch-conformance.sh` written, NOT yet
  run (network/cmake). `fetch-conformance.sh` refuses to run until
  `PINNED_COMMIT` is set — deliberate, keep it that way.
- Slice 1 of `PLAN.md` is complete; slice 8's standalone math groundwork
  (DCT, XYB) is in place.

**Design notes for later:**
- Nonzero `ZeroPadToByte` padding maps to `BitstreamError::Overflow` (no
  dedicated variant yet); add `MalformedPadding` when header work starts if
  wanted.
- `jpxl-core::color` implements plain `B = S_gamma`; the paper's XYB′
  (`B′ = B − Y`) decorrelation step is NOT implemented — decide when the
  bitstream work reaches it.

**Blocked on the user:** ~~OCR of ISO/IEC 18181 Parts 1, 2, and 3~~ —
**resolved same day**, see the entry above. Everything derived here came from
the arXiv paper and is tagged `[provisional]`; none of it has been checked
against the real text yet.

**Next:** `PLAN.md` slice 2 (signature + `SizeHeader`/`ImageMetadata`, with
oracle header-dump cross-check) and slice 3 (entropy coding core: prefix
codes, rANS, hybrid-uint, LZ77, clustering). Slice 3 unblocks slices 4, 5, and
7, so it is the critical path.

---

## Already fixed — do not redo

- **`read_u32` wraps, it does not error** (2026-08-02). 18181-1 B.2.2:
  `(offset + v) Umod (1 << 32)`. The scaffold version returned `Overflow` on
  `offset + payload` overflow; fixed to `wrapping_add` with a clause citation
  and the test `u32_offset_plus_payload_wraps_mod_2_pow_32`. Do not "harden"
  this back into an error.
- **XYB inverse matrix is verified normative** (2026-08-02). The rationally
  derived `OPSIN_ABSORBANCE_INVERSE_MATRIX` matches 18181-1 L.2.1 Table L.1
  defaults digit-for-digit at `f32`; no longer `[provisional]`. The spec
  signals `opsin_bias0..2` as negative (decoder-side); our forward-side
  positive bias is the same convention mirrored — documented in
  `jpxl-core/src/color.rs`.

## Traps — do not fix these by loosening a check

Nothing yet. (When a strict limit or assertion fires and the real bug is
upstream, record the check and the actual cause here — the tempting fix is
almost always the wrong one.)
