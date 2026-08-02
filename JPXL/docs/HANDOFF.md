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

## 2026-08-02 — slice 7 (end-to-end lossless decode) — 6 of 9 fixtures bit-exact vs djxl

**State:** `jpxl_decode::decode()` works end to end for lossless modular:
signature → headers → frame/TOC → sections (G.1.3/G.2.3/G.4.2) → modular →
inverse transforms → pixels, plus ~90-line jxlc/jxlp container extraction and
a `jpxl decode` CLI subcommand (hand-rolled P5/P6). Bit-exact against djxl:
fixtures 03, 07, 08 (container + 16-bit), 11 (256×256 -e7), 12 (300×200), 13.
Caveat: all current fixtures are single-section (cjxl chose group_dim 512);
multi-section decode is implemented per spec but unproven.

**Experiments resolved by oracle evidence:**
- H.2 global tree: distributions are SHARED from LfGlobal; each sub-bitstream
  re-initialises only per-stream state (ANS seed, LZ77 window) after its
  ModularHeader (`SymbolDecoder::open_deferred`/`restart`;
  `GLOBAL_TREE_SHARES_DISTRIBUTIONS`). Evidence: fixture 12, 32-bit desync.
- **H.5.2 clamp guard is a defect in the standard itself**: both sources print
  `(p1 | p2) <= 0`; the operationally-correct reading (matching the prose) is
  `(p1 <= 0) or (p2 <= 0)`, differing exactly when one neighbour error is
  zero. `EXPERIMENT_CLAMP_BITWISE_OR = false`. Fixed fixtures 11 and 12.
- H.5.2 true_err is used CLAMPED; max_error tie-break is strict `>`.

**Not exercised by these fixtures** (flip-points unchanged, still open):
err_sum last column, AvgAll Idiv-vs-shift, nested-LZ77, gab_custom,
resets_canvas.

**Unresolved — do not paper over:** fixtures 05, 09, 10 diverge identically in
H.5.2 `max_error` selection (fixture 10 needs `-1896` chosen over `+2040` at
(2,1), which no magnitude rule produces; a plain minimum fixes 10 but breaks
11/12). Kept the normative `abs(x) > abs(max)`; three `#[ignore]`d tests carry
first-wrong-sample/bit forensics. Suspects: Table H.4 property numbering,
H.5 state for shift `-1` channels.

---

## 2026-08-02 — slice 5 (Modular mode, Annex H) complete

**State:** `jpxl-decode::modular` decodes modular sub-bitstreams end to end:
ModularHeader, MA trees (decode/validate/traverse, Limits-capped), all 14
predictors incl. the weighted predictor (H.5.1/H.5.2), UnpackSigned, and
inverse RCT (42 variants, round-tripped) / palette + delta-palette / squeeze
(round-tripped against an independent forward implementation over odd sizes).
113 tests incl. a 2000-case no-panic fuzz sweep. `decode_channels` is public
so slice 7 can supply its own SymbolDecoder.

**Open for slice 7 (oracle experiments, in priority order):**
1. H.2 global-tree distributions: reuse the global clustered bundle vs read a
   fresh C.1 bundle per group (spec text contradicts itself; literal second
   reading implemented). Wrong answer desynchronises a whole group.
2. Table H.3 row 13 `AvgAll` uses `Idiv 16` (differs from `>> 4` for every
   negative sample) — implemented as `Idiv`, verify.
3. H.5.2 `err_sum` last-column `+= err[i]_W` (LaTeX-only text) — one addition,
   verify.

**Resolved-by-reasoning (documented in modular/mod.rs):** H.6.2 shift restore
omission; H.6.4 `/4` as integer division; H.6.4 `(index & 1) == 0` despite
Table 1 precedence making the literal text constant-false.

---

## 2026-08-02 — slice 6 (FrameHeader/TOC/groups, Annexes F/G/J.1) complete

**State:** `jpxl-decode::frame` parses FrameHeader with its full conditional
forest, passes, blending, RestorationFilter (J.1), TOC with entropy-coded
Lehmer permutation, and group/section geometry. 102 new tests.

**Spec gotchas encoded as tests:** `HfGlobal` section exists (zero-length) in
Modular mode — `num_sections` is always `2 + num_lf_groups + num_groups ×
num_passes` regardless of encoding (F.3.1 NOTE 1); F.3.3 permutes *offsets*
computed from as-read order, not sizes; F.3.2 `GetContext` uses `min(7, …)` —
the LaTeX corrupted the 7 (second confirmed markdown-beats-LaTeX case; the
LaTeX fails specifically on numeric constants inside prose).

**Open for slice 7 (oracle experiments, one-bit differences):**
- J.1 `gab_custom` guard: implemented as `!all_default && gab` (the literal
  bare `gab` guard would cost a bit even under `all_default`, violating the
  invariant every other bundle obeys). Constant
  `GAB_CUSTOM_REQUIRES_NOT_ALL_DEFAULT` flips it in one place. Highest-value
  oracle check — differs on nearly every real frame.
- F.2 `resets_canvas`: computed once from colour blending_info and shared
  with every ec_blending_info (vs per-bundle evaluation; 2 bits per extra
  channel).

---

## 2026-08-02 — Part 2 clause map re-audited, no longer provisional

**State:** `STANDARDS_INDEX.md`'s Part 2 section replaced the arXiv-derived
provisional topic map with a real clause map verified against `part2.md`
(735 lines, full 22-page OCR, read in full). Real structure: clauses 1–9 are
the main body (1 scope, 2 normative references, 3 terms, 4 general, 5 file
organization, 6 data types, 7 graphical descriptions, 8 binary box format,
9 box types 9.1–9.11), and there are exactly two annexes, **both normative**
— A (JPEG Bitstream Reconstruction procedure, A.1–A.11) and B (JPEG XL Media
Type registration, B.1–B.2). No informative annex, unlike Part 1.

Confirmed the box set from clause 9: signature box (9.1, the 12 fixed bytes),
`ftyp` (9.2), `jxll` level box (9.3, at most one, third box if present,
default level 5), `jumb` (9.4, delegates to 19566-5), `Exif` (9.5, codestream
wins on overlap), `xml ` (9.6), `brob` Brotli-wrapper (9.7), `jxli` frame
index (9.8), `jxlc` full codestream (9.9), `jxlp` partial codestream (9.10,
index-ordered concatenation semantics), `jbrd` JPEG reconstruction data (9.11,
Tables 11–18). **`jhgm` (HDR gain map) is not in this 2nd-edition text at
all** — the old provisional entry listing it was wrong for this edition;
dropped rather than carried forward unverified.

Crosswalk gained two Part 2 rows: clause 9.1 signature box → `jpxl-conformance::sniff`
(exists) and clauses 8–9 box parsing → `jpxl-decode` (slice 9, not started).

**OCR quality note:** Table 11 (the `jbrd` `JPEGBitstream` bundle, pages
12–14) is badly garbled — subscripted field names collapse into glyph noise
(`Tyyw`, `Tpey`, `OFse`, etc.) and the marker-array loop condition reads as
nonsense. Flagged in the clause map; do not implement slice 9's `jbrd`
parsing from this table without a scan cross-check. Everything else in
`part2.md` reads cleanly, including Annex A's segment-reconstruction rules.

**Next:** unchanged — slices 2 and 3 remain ahead of slice 9 in the plan.

---

## 2026-08-02 — slice 3 (entropy, Annex C) complete; oracles live

**State:** `jpxl-entropy` covers all of Annex C with nothing stubbed: C.2.1
bundle, C.2.2 clustering + inverse MTF, C.2.3 hybrid-uint, C.2.4 prefix codes
(RFC 7932 derivation, not transcription), C.2.5/C.2.6 ANS histograms + alias
mapping, C.3.2 state machine, LZ77 with the reconstructed 120-entry
`kSpecialDistances` (validated by monotonic `dx²+dy²` ordering). 75 tests,
layer-by-layer. Oracle infra is live: djxl/cjxl v0.13.0 pinned + built,
jxl-oxide 0.12.6 (ignores output extensions — always pass `--output-format`),
conformance corpus at 4bf05352, four reproducible cjxl fixtures incl. a
300×200 multi-group case.

**Open for slice 7 (oracle experiments queued):**
- C.2.2 nested-LZ77 reading: implemented as a *constraint* (nested
  `lz77.enabled` flag is read and must be 0, stream rejected otherwise), not
  an override. One-bit difference; verify against djxl-produced streams.
- `tests/oracle_vectors.rs` harness is ready; its fixture-driven test is
  `#[ignore]`d with TODO(slice 7).

---

## 2026-08-02 — slice 2 (image headers) complete

**State:** `jpxl-decode` parses signature + the full `ImageMetadata` bundle
tree (D.2/D.3, E.2/E.3 colour encoding, L.2.1 opsin, B.3 extensions, B.2.6
enums) — 97 tests, every field traced, trace intervals proven gap/overlap-free.
Public API: `jpxl_decode::headers::decode_image_headers(&mut BitReader,
&Limits)`.

**Source-fidelity corrections (both directions now proven):**
- `latex/part1.tex` is NOT uniformly better than `part1.md`: AspectRatio
  ratio 5 reads `16 Idiv 39` in the LaTeX (wrong); the markdown's `16 Idiv 9`
  is right (16:9). Cross-check numeric constants in BOTH sources.
- `quant_bias0..2` (Table L.1): ~~sign ambiguous, taken positive~~ —
  **resolved 2026-08-02 by a clean-scan screenshot**: the printed defaults are
  the expressions `1 − 0.05465…`, `1 − 0.07005…`, `1 − 0.049935…`, i.e.
  ≈ 0.9453 / 0.9299 / 0.9501. Both OCRs had collapsed the leading `1 −`. The
  original "positive 0.05465" reading was **wrong** and is fixed in
  `headers/opsin.rs` (see Already fixed).

**Spec gotchas encoded as tests (do not relearn):** `default_m` is NOT under
`all_default` (minimal metadata is two bits, not one); `BitSet(cw_mask, b)`
takes masks 1/2/4, not bit indices; extra-channel names kept as raw bytes
(UTF-8 validity is not a conformance requirement).

---

## 2026-08-02 — Part 1 LaTeX landed; STANDARDS_INDEX re-audited

**State:** `latex/part1.tex` (6236 lines, one TeX page per source page, all 96
pages) is present, alongside a text-only transcription PDF at
`original-pdfs-do-not-read-first-if-markdown-exists/ISO_IEC_18181-1_2024_transcription.pdf`.
The LaTeX is now the highest-fidelity Part 1 source: it restores pseudocode
bodies that `part1.md` truncated (B.2.3 `U64()` continuation loop, B.2.4
`F16()`) and corrects OCR digit noise in tables and examples.

**Worked example of that noise:** B.2.2's example reads `U32(8, 16, 32, u(7))`,
bits `10` → 32, and `U32(u(2), u(4), u(6), u(8))`, bits `010111` → 7. The
markdown misreads the constants. Treat every numeric constant taken from
`part1.md` as unverified until checked against `part1.tex` — a wrong
distribution constant produces a plausible-looking parse that desynchronises
every later field.

**Done:** `STANDARDS_INDEX.md` re-audited. The `latex/` row moved from pending
to present; the transcription PDF added to the locator note; the provisional
arXiv-derived Part 1 topic map **replaced** by a real clause map (Annexes A–O
with titles, ToC page numbers, and key subclauses, each letter verified against
the text — note J is restoration filters, K image features, L colour
transforms, and simple upsampling is J.2 while non-separable upsampling is
K.2). The crosswalk now carries real clause numbers (B.2.x → `jpxl-bitstream`,
I.7/I.9 → `jpxl-core::dct`, L.2/L.3 → `jpxl-core::color`, 5.1/5.3 →
`jpxl-core::geometry`, M → `jpxl-core::limits`). `AGENTS.md` §2 and §3 updated:
the resolution chain is now part1.tex → markdowns → transcription PDF → arXiv
paper → image scans → oracle experiment; `latex/` is confirmed gitignored.

**Still provisional:** ~~the Part 2 topic map in `STANDARDS_INDEX.md` is still
arXiv-derived; `part2.md` is complete and it should be re-audited the same
way.~~ Done, see the 2026-08-02 "Part 2 clause map re-audited" entry above.

**Next:** unchanged — slices 2 and 3, slice 3 the critical path.

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
to design rationale and cross-checking. (Superseded by the entry above: the
LaTeX conversion has since landed and outranks `part1.md`.)

**Queued:** re-audit every `[provisional]` tag in `jpxl-bitstream` and
`jpxl-core`. `STANDARDS_INDEX.md` is done — see the entry above.

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

- **`quant_bias` defaults are `1 − x`, not `x`** (2026-08-02). Table L.1
  prints the defaults as literal expressions (`1 - 0.05465007330715401`, …);
  verified against a clean scan after both OCRs collapsed the `1 −` prefix.
  `DEFAULT_QUANT_BIAS` ≈ [0.9453, 0.9299, 0.9501] in `headers/opsin.rs`. Do
  not "simplify" these back to the small constants.

## Traps — do not fix these by loosening a check

- **Annex H OCR corruptions, resolved 2026-08-02 — do not re-transcribe from
  the corrupted source:** Table H.4 rows 4/5 are `abs(N)`/`abs(W)` (both
  sources garble one each); `kDeltaPalette[4]` is `{0,-12,0}` (LaTeX's
  `{0,-12,9}` is wrong); Table H.3 row 13 is `WW` not `WH` (pinned by the
  coefficients-sum-to-16 test); H.5.2 weight normalisation and `error2weight`
  exist ONLY in the LaTeX (part1.md drops the whole block); H.6.3 in part1.md
  is scrambled — use the LaTeX, where `B = B + A&A` means `B = B + A`.
- **C.2.6 alias mapping is `symbols[u] = o` (the overfull index), NOT
  `symbols[u] = 0`** (2026-08-02). The LaTeX renders it as `0` — an OCR
  corruption; only `o` is consistent with the algorithm. The invariant test
  (each symbol s appears exactly D[s] times across all slots, offsets a
  permutation of 0..D[s]) fails under `= 0`. If that test ever fires, the bug
  is in new code, not the test.
- **jxl-oxide ignores the output-file extension** and writes PNG bytes into
  any filename. The harness rejects PPM-from-jxl-oxide before spawning
  (`OracleError::UnsupportedFormat`). Do not "fix" a BadMagic PPM parse error
  by loosening the PPM parser — pass `--output-format` explicitly.
