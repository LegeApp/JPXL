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

## 2026-08-03 (VarDCT wave 0) — 8A math, 8E filters, 8F0 conformance metrics

Slice 8 (VarDCT) is underway per the approved plan (sub-slices 8A–8F + 8F0,
four waves). Wave 0 landed:

**8A** — `jpxl-core` gains `varblock.rs` (Table I.1/I.4/I.7 vocabulary,
`CoeffMatrix`/`SampleBlock` distinct types — coefficients always landscape,
I.3.2 natural order, I.8 LLF, I.9.2–I.9.8 reconstructions), block-coordinate
newtypes in `geometry.rs`, I.7.2/I.7.3 wrappers + power-of-two kernels to 256
in `dct.rs` (its `[provisional]` scaling note is RESOLVED: I.7.2 = orthonormal
× uniform `1/√s` forward / `√s` inverse per 1-D pass — do not fold the factor
into dequant matrices). 8B consumes `TransformType::{dequant_matrix_index,
coeff_rows, coeff_cols, order_id}`; 8C consumes `natural_coeff_order`.
**THIRD DEFECT IN THE PUBLISHED STANDARD:** I.8's `ScaleF` divides by zero
from DCT16x16 up, identically in all three Part 1 sources; the shipped
reading passes the varblock dimension (Dirichlet-identity derivation, exact
<1e-9), flip-point `LLF_SCALEF_ARG_IS_VARBLOCK_DIMENSION`, see
`docs/experiments/2026-08-03-i8-scalef-argument.md`. DCT8x4 half placement
settled from I.9.8's stated layout (`DCT8X4_HALF_INDEX_IS_LOW_COORDINATE`,
probe-worthy in 8F but not blocking).

**8E** — `frame/{gaborish,epf}.rs`: J.3 with sum-to-1 rescale, J.4.1–J.4.4
with all three steps; pure f32-plane functions, 8F wires them. OCR: step-0
EPF kernel coord is `{0,-2}` (part1.md right, LaTeX `{9,-2}` wrong — third
markdown-beats-LaTeX case); `epf_quant_mul=0.46` / `epf_sigma_for_modular=1.0`
LaTeX-only. FOUR OPEN FLIP-POINTS in `epf.rs` awaiting 8F's filters-on
probe: `EPF_STEPS_FROM_EXPLICIT_CONDITIONS`,
`EPF_BORDER_SAD_AT_REFERENCE_PIXEL`, `EPF_SKIP_IS_PER_VARBLOCK`,
`EPF_DISTANCE_USES_STEP_INPUT` (`docs/experiments/2026-08-03-epf-flip-points.md`).

**8F0** — `jpxl-conformance` gains Part 3 §4.2 grading: `FloatImage`,
hand-rolled NPY reader (djxl grayscale is channels=1, NOT replicated RGB —
trap), normalized f32 peak + per-channel RMSE ("root of the sum" read as
root-mean, documented). Conformance corpus references downloaded (39/39,
`bike_5` verified). Fixtures 50–57 (filters on/off × d1/d4 × gray/RGB);
zero-slack djxl-vs-djxl self-grading test proves the pipeline.

**Traps (permanent copies below):** do not "fix" `scale_f` back to the
printed I.8 call; `AFV_BASIS` is f64 on purpose (verbatim spec digits,
orthonormality to 1.5e-14 proves the two OCR repairs).

**Next:** wave 1 = 8B (I.2 parameter bundles, opus) + 8D-parse (G.2.2/G.2.4
modular sub-bitstreams, sonnet); then wave 2 = 8C + 8D-dequant; wave 3 = 8F
assembly/acceptance.

---

## 2026-08-03 (wave 2) — slices 4 and 10 complete; flip-points pinned; gab_custom dead-code bug fixed

**1. Slice 4 (ICC, E.4) done.** `jpxl-decode/src/icc/` decodes the
compressed ICC representation; fixtures 30–36 (script-built profiles, v2 and
v4, 336–6676 bytes) byte-exact vs `djxl --orig_icc_out`. Key readings, all in
`docs/experiments/2026-08-03-icc-stream-placement.md`: the E.4.1 payload is
UNALIGNED after the headers (aligned reading fails on the first symbol — no
flip-point needed); E.4.4's dictionary has 17 entries (`part1.md` truncates
to 15 — trap below); `output_size` is a constraint (growth refused), metered
by AllocGuard, capped per Table M.1 level 10 as a module-local constant
(promote to a `Limits` field if configurability is ever wanted). New API:
`extract_icc_profile()`, `DecodedImage::icc_profile`.

**2. Slice 10 (encoder breadth) done.** 16-bit gray, RGB via YCoCg-R
(`rct_type = 6` declared once in LfGlobal), multi-group via SectionStore
(each section encoded once into its own buffer → TOC from measured lengths →
bodies appended), `jxlc` container behind a CLI flag with a `jxll` level-10
box for >8-bit. Self-roundtrip + djxl + jxl-oxide sample-exact across the
full matrix (both depths, both channel counts, all four `group_size_shift`
values, naked and boxed, up to 600×520). 16-bit blocker settled by widening
the token alphabet (power-of-two sizes keep the flat prefix code free;
`token_bits` capped at 5 by C.3.3's `n < 32`); `split_exponent` bought
nothing. Externally confirmed readings: multi-section `LfGlobal` carries
ModularHeader + tree + C.1 bundle and ZERO samples; group sub-bitstreams
predict rectangle-relative (H.3 edges are the group's own).

**3. Flip-point sweep.** Real bug found and fixed: `read_restoration_filter`
returned on `all_default` before computing the gaborish fields, so
`GAB_CUSTOM_REQUIRES_NOT_ALL_DEFAULT` was dead code either way. AvgAll was
withdrawn as a flip point — both primary sources agree on `Idiv 16`, never
ambiguous. New named constants `NESTED_LZ77_REJECTS_ENABLED` and
`RESETS_CANVAS_SHARED_ACROSS_BUNDLES` (both keep the shipped reading). All
four remain UNEXERCISED by real cjxl output (~24 probes; cjxl never emits
those configurations) — documented as negative results in
`docs/experiments/2026-08-03-flip-point-fixtures.md`; each reading is pinned
by hand-built-bitstream unit tests instead. Fixture 41 (RGBA) is the first
end-to-end extra-channel decode, and it works.

**Known open bug (queued, do not lose):** a 32×32 grey source of
`x*7 + y*3` (wrapping sawtooth) encoded `cjxl -d 0 -e 3` fails to decode:
`out of bounds: 16 bit(s) requested at bit position 2112`. Reproduces
without ICC and at commit 26d8df3, so it is in the modular/frame layer, not
ICC. Needs a minimised probe fixture and a root-cause hunt.

**Next:** slice 8 (VarDCT) or slice 9 (container breadth: `jxlp`, Exif,
brob); encoder future work list is at the end of the slice-10 report themes
(ANS backend, real MA trees, palette/squeeze write side, alpha, TOC
permutation, `jpxl-encode-policy`).

---

## 2026-08-03 — H.5.2 clamp fixed, multi-section proven, slice 7.5 encoder complete

Three concurrent tasks, all landed:

**1. Fixtures 05/09/10 resolved — the H.5.2 CLAMP was the cause, not
`max_error`.** The slice-7 diagnosis below was wrong and is corrected there in
place. `max_error` is exactly the clause as written; the corrupt input was
`true_err`, because the *prediction* was being clamped by the printed
symmetric clamp. Four oracle-pinned samples (tabulated in
`docs/experiments/2026-08-03-h52-clamp-asymmetry.md`) prove no single guard
over the two printed products can be right — the two halves of the clamp are
gated differently: cap above by `max(W3,N3,NE3)` when `p1<=0 || p2<=0`; floor
below by `min(W3,N3,NE3)` only on strict disagreement (`p1<0 || p2<0`) or
when all three neighbour errors are zero. Flip-point
`EXPERIMENT_CLAMP_SYMMETRIC = false` (replaces `EXPERIMENT_CLAMP_BITWISE_OR`),
tagged `[provisional]` — this describes libjxl 0.13.0, which the printed
clause does not. All 11 lossless fixtures now bit-exact vs djxl (new debug
fixtures 20 palette-bands 24×24 and 21 gradient 260×10, generated by
`tools/make-debug-fixtures.sh`). Newly resolved flip-points:
`EXPERIMENT_MAX_ERROR_RULE = 0` (settled), `EXPERIMENT_ERR_SUM_LAST_COLUMN =
1` (now exercised — 0 and 2 break 05/09 under the corrected clamp).

**2. Multi-section decode PROVEN** (was implemented-but-unproven). Fixtures
14 (600×520 gray8, 12 sections), 15 (600×520 RGB, 12 sections), 16 (511×8,
5 sections) decode bit-exactly vs djxl; `tests/e2e_multisection.rs` asserts
`num_sections > 1` against the sidecar-recorded value so a regenerated
single-section fixture fails loudly. cjxl v0.13 has no group-size flag; its
heuristic drops to group_dim 256 when a dimension exceeds 512 or the image is
very thin — the only lever is image shape.

**3. Slice 7.5 complete — first interoperable pair.** New `jpxl-encode`
crate: gray8 lossless modular naked codestreams (no transforms, one-leaf MA
tree, gradient predictor, prefix codes with a flat 16×4-bit code via RFC 7932
§3.5's single-nonzero-length degeneracy — the alphabet lengths cost zero
bits; LZ77 off; single group/section, ≤1024×1024). All three acceptance
criteria pass sample-exact: self-roundtrip, djxl, jxl-oxide. `BitWriter`
added to `jpxl-bitstream` (chooses the first fitting U32 distribution,
rejects unrepresentable values); `jpxl encode` CLI subcommand (P5 PGM in).
`jpxl-encode` uses `jpxl-decode`/`jpxl-entropy` as dev-dependencies only.

**Still open (unexercised flip-points):** AvgAll Idiv-vs-shift, nested-LZ77,
`gab_custom`, `resets_canvas`.

**Next:** slice 10 encoder breadth (16-bit needs a wider alphabet or nonzero
`split_exponent` — `jpxl-encode::entropy` caps at 2^15−1; RGB+RCT;
multi-group via SectionStore; `jxlc` container) or slice 4 (ICC) / 8 (VarDCT).

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

**~~Unresolved~~ CORRECTED 2026-08-03 (diagnosis was wrong):** this entry
originally blamed H.5.2 `max_error` selection for the 05/09/10 divergence.
The real cause was the H.5.2 *clamp* corrupting the prediction (and hence
`true_err`) upstream; `max_error` is normative as written. See the
2026-08-03 entry above and
`docs/experiments/2026-08-03-h52-clamp-asymmetry.md`. The suspects listed
here (Table H.4 numbering, shift −1 state) were investigated and eliminated.

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

- **`EXPERIMENT_CLAMP_SYMMETRIC = false` is not a loosened check**
  (2026-08-03). Restoring the printed symmetric H.5.2 clamp "to match the
  spec" re-breaks fixtures 05/09/10/20; the contradicting oracle samples are
  tabulated in `docs/experiments/2026-08-03-h52-clamp-asymmetry.md`. The
  printed clause is wrong for libjxl 0.13.0 streams.
- **The encoder writes `RestorationFilter` explicitly OFF** (`gab = false`,
  `epf_iters = 0`, 2026-08-03). The Table J.1 *defaults* are `gab = true`,
  `epf_iters = 2` — decoder-side smoothing our decoder does not implement
  yet, so an `all_default` J.1 bundle self-roundtrips green while djxl and
  jxl-oxide return different pixels. If external decodes ever drift while
  self-roundtrip stays green, look here first.
- **`part1.md` truncates E.4.4's tag dictionary to 15 entries** (2026-08-03).
  The real list has 17 (`bTRC`, `dmda` dropped by the OCR), fixed by the
  tagcode range 4..=20 and confirmed by byte-exact fixtures. Use the LaTeX.
- **`modular_16bit_buffers = false` for >8-bit encodes is deliberate**
  (2026-08-03). It is a truthful claim about decoder working buffers (D.3);
  the paired consequence is the `jxll` level-10 box in container output
  (Annex M). Do not "restore the Table D.3 default".
- **Sawtooth 32×32 `-e 3` decode bug is real and open** (2026-08-03): grey
  `x*7 + y*3` via `cjxl -d 0 -e 3` fails with an out-of-bounds bit read at
  position 2112, independent of ICC, present at 26d8df3. Do not loosen the
  bounds check — the bug is upstream in modular/frame decoding.
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
