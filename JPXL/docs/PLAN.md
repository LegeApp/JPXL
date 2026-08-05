> **Plan of record: AKR.** Milestones, decisions, policies, constraints and the experiment
> findings now live in the AKR ledger (`.akr/`) and its generated views under
> `docs/generated/`. This file is retained as a working log / legacy reference — not the
> authoritative plan. See `AGENTS.md` and `docs/generated/ROADMAP.md`.
# PLAN.md — JPXL implementation plan

Decoder-first, vertical slices. Each slice states a goal, its spec source, and
an exit criterion that is a **test**, not an opinion. A slice is done when its
test passes and `CONFORMANCE.md` records it.

Normative sources are available: `latex/part1.tex` (highest fidelity),
`markdowns/standard-markdowns/part1-4.md` (see `STANDARDS_INDEX.md` for the
access order). "Blocked on OCR" notes below are resolved; remaining blocking is
inter-slice only.

The repo-root `RUST_JXL_ENCODER_ROADMAP.md` (original-implementation roadmap)
is the strategic companion to this file: its phase order matches these slices,
and its **first interoperable vertical slice** and **SectionStore** designs are
adopted below.

Last reviewed: 2026-08-02 (post-LaTeX).

## Slice table

| # | Slice | Scope | Spec source | Blocking | Exit criterion (test) |
| --- | --- | --- | --- | --- | --- |
| 1 | **BitReader + primitives** | Little-endian bit reader; `Bool`, `u(n)`, `U32` 4-distribution selector, `U64` extensible form, `F16` (reject NaN/Inf), `ZeroPadToByte`; bit-position tracing infrastructure. | arXiv §3.1 (syntax notation is fully stated there) | **unblocked** — IN PROGRESS this wave | Exhaustive/property roundtrip: every `U32` distribution index and `U64` continuation length encodes and decodes to the same value with the expected bit count; `F16` rejects non-finite; trace records the exact bit offset of each field. |
| 2 | **Signature + `SizeHeader` / `ImageMetadata`** | `FF 0A` sniffing, small/aspect-ratio/full size forms, `all_default` metadata path, bit depth, orientation, extra-channel list. | arXiv §3.1 `[provisional]`; confirm against Part 1 when OCRed | mostly unblocked; oracle cross-check needed | Parse a corpus of `cjxl`-produced headers; every field matches `jxlinfo` output. Bit offsets from the trace line up with total header length. Handmade fixtures cover each size form. |
| 3 | **Entropy coding** | Prefix codes, rANS decode, hybrid-uint token/extra-bits, LZ77 layer, context map + histogram clustering. | arXiv §8 for structure; **exact table layouts and histogram signaling need Part 1 OCR** | core mechanics buildable now; layouts blocked | Symbol-stream roundtrip against self across degenerate cases (single-symbol alphabet, uniform, ties in the ANS `omit_pos` selection, empty LZ77 window). Then: decode entropy-coded sections lifted from oracle files bit-exactly. |
| 4 | **ICC decode** | The compressed-ICC representation carried in the codestream. | Part 1 E.4 | **done 2026-08-03** — fixtures 30–36 byte-exact vs `djxl --orig_icc_out` | Decoded ICC bytes are byte-identical to `djxl`-extracted profiles across the fixture set. |
| 5 | **Modular mode** | MA trees over local properties, all predictors incl. Weighted/self-correcting, transforms: RCT, palette/delta palette, Squeeze. | arXiv §5; predicates and property indices need Part 1 OCR | blocked on 3 + OCR | Per-tool isolation tests first (each predictor against a hand-computed vector; each transform inverted exactly). Then a full modular group decodes bit-exactly vs. the oracle. |
| 6 | **FrameHeader / TOC / groups** | Frame header incl. conditional blocks, group geometry, TOC offsets, group permutation, passes. | arXiv §3 Fig. 8, §9 — conditionals must come from Part 1 | blocked on OCR | Frame headers from a multi-frame, multi-group, permuted-order corpus parse to the exact byte offset of the first group; group rectangles match the oracle's reported geometry. |
| 7 | **End-to-end lossless modular decode** | Wire the above into a working decode of modular-lossless files. | Parts 1 + 2 | **done 2026-08-03** — all 11 handmade lossless fixtures bit-exact vs `djxl`, incl. multi-section 600×520 | Native-depth pixel equality against `djxl` output for the whole handmade + generated fixture set, **including ≥256×256 multi-group images**. Not "decodes without error" — exact samples. |
| 7.5 | **First interoperable pair** | Deliberately tiny encoder+decoder subset: naked codestream, one frame, gray8, non-XYB, modular, single group, no transforms, one-leaf MA tree, one simple legal predictor, one context cluster, no LZ77, simplest legal entropy backend. | Part 1 (roadmap "Phase 4") | **done 2026-08-03** — `jpxl-encode` gray8 lossless; self-roundtrip + `djxl` + `jxl-oxide` all sample-exact | JPXL encodes deterministic images; JPXL decodes them exactly; **`djxl` and `jxl-oxide` decode them exactly**; corrupt variants error, never panic. This is the first proof the whole syntax stack is real — do not start VarDCT before it passes. |
| 8 | **VarDCT inverse** | XYB inverse, DCT families and varblock types, dequantization, chroma-from-luma, gaborish, EPF. | arXiv §4.2, §6, §7.2 | **core done 2026-08-03** — end-to-end kVarDCT decode, conformance-corpus grayscale cases pass vs published references; residuals: patches (K.3), extra channels, RAW matrices, untested varblock types (see HANDOFF) | Math layer: each DCT shape roundtrips, with separate assertions on coefficient storage order and LLF/DC extraction (the previous project's single largest failure class). Full path: decoded pixels within the Part 3 peak-error class vs. the oracle. |
| 9 | **Container / Part 2 boxes** | JXL signature box, `ftyp`, `jxlc`, `jxlp` concatenation, `jxll`, `Exif`, `xml `, `brob`, `jbrd` passthrough. | Part 2 | **done 2026-08-03** — full box parser, order-validating jxlp, jxlinfo cross-checked; brob decompression + jxli/jbrd parsing deferred | Box tree of every container fixture matches the oracle's box listing; split `jxlp` reassembles to a codestream byte-identical to the equivalent `jxlc`. |
| 10 | **Encoder breadth** | Grow the slice-7.5 encoder: 16-bit, gradient predictor, RGB + RCT, multi-group, minimal `jxlc` container. Encoder uses a **SectionStore** (encode each section once into bounded RAM/spill, then TOC, then stream — the TOC precedes sections on the wire, so lengths must exist before emission; never assemble a second complete codestream). | Part 1 (encoding is unconstrained; validity is what matters) | **done 2026-08-03** — 16-bit, RGB+RCT, multi-group SectionStore, `jxlc`; all three decoders sample-exact | Stage 1: JPXL encode → JPXL decode reproduces input samples exactly. Stage 2: `djxl` decodes the same file to the same samples. Both required; stage 1 alone proves nothing (paired bugs cancel). |

## Encoder phase (slices 11+) — adopted design: `docs/Encoder-plan1.md`

External-advisor architecture review (2026-08-04), adopted with the
adjustments below. Core shape: an **optimizing-compiler encoder** — a
one-way boundary between `jpxl-encode` (normative lowering + emission:
validated plan types in, exact bits out, zero heuristics) and a new
`jpxl-encode-policy` crate (all search: analysis atlas, block tiling,
adaptive quant, CfL, entropy clustering, rate control). Staged IRs
(`PreparedFrame → AnalysisAtlas → SpatialPlan → QuantizedFrameIr →
EntropyPlan → EmissionPlan`); block selection as an exact-cover search in
`BlockInfo`'s own greedy raster representation (search output IS the wire
sequence); a two-pass entropy compiler (`HfEventSink` event walk →
census → cluster → replay), never ANS calls from coefficient loops.
`jpxl-decode` stays a peer oracle, never an encoder dependency; external
decoders remain the conformance gate.

Adjustments to the advisor's plan (recorded here so briefs don't
relitigate):

1. **Slice 11.5 exists and is load-bearing: the ANS encoder.**
   `jpxl-entropy` only decodes today; the current encoder emits
   degenerate prefix codes. ANS emission (reverse-order write, C.3.2
   terminal-state contract, hybrid-uint token building) precedes
   milestone 2 and everything after.
2. **The lossless track interleaves, not follows.** Learned MA trees,
   LZ77 and predictor selection are the cheapest density wins and share
   the policy crate + entropy compiler (census/cluster/price-book applies
   to modular tokens too). Reference-level means both tracks.
3. **The §13 refactor (inverse primitives into `jpxl-core`) is its own
   gated slice** — mechanical moves only, decoder suite proves no drift.
   Share math between our crates freely; never share bitstream control
   flow (paired-bug defense).
4. SIMD/threading/spill stay out until the scalar encoder has a trusted
   R-D baseline (advisor's milestone 10 = our existing "not in scope
   yet" list; no conflict).

| # | Slice | Scope (advisor milestone) | Exit criterion (test) |
| --- | --- | --- | --- |
| 11 | **Structural split + plan IR** | Create `jpxl-encode-policy`; exact VarDCT plan types + `validate()` in `jpxl-encode` (M1) | A hand-built legal plan validates and dumps; malformed plans cannot reach the writer (typed rejection, tested per invariant). |
| 11.5 | **ANS encoder** | rANS emission in `jpxl-entropy`: reverse write, histogram serialization, hybrid-uint config, context maps | Every stream our ANS encoder emits, our decoder AND the entropy fuzz gates decode bit-exactly; terminal state exact per C.3.2. |
| 12 | **Fixed-DCT8×8 VarDCT vertical slice** | RGB8→XYB, one frame, DCT8×8 only, default matrices, constant HfMul, natural order, 1 pass/preset, filters off, ANS (M2) | JPXL + `djxl` + `jxl-oxide` all decode it; pixels within stated lossy tolerance; section trace + ANS terminal state valid. No block search before this passes. |
| 13 | **Forward transform algebra** | Allocation-free forward forms for every supported transform + `lf_from_llf` (M3) | Forward/inverse pair tests per transform; forced-transform-map encodes decode correctly end to end. |
| 14 | **Exact rate loop, fixed blocks** | global_scale, HfMul, LF quant, target bpp, exact section accounting, discrete budget fill (M4) | Hits a byte target within stated tolerance with fixed blocks; monotonicity checks bracketed, not assumed. |
| 15 | **CfL estimation** | LF correlation + per-64×64 HF factors: regression then integer refinement (M5) | Size improves at equal quality on correlated-colour corpus; zero regression on grayscale. |
| 16 | **Hierarchical block selector** | DCT8/16/32 + common rectangles, exact DP within the hierarchy (M6) | Selected maps legal (decoder placement invariant as the gate); matched-quality size beats fixed-8×8 baseline on a mixed corpus. |
| 17 | **Adaptive quant + R-D curves** | Perceptual field, `global_scale × HfMul` factorization, joint transform/quant selection (M7) | Spatial quality uniformity improves; target size stable. |
| 18 | **Entropy optimization** | Trained block context, clustering, custom orders, optional presets; ONE bounded refinement pass (M8) | Density improves with decode-verified streams; refinement loop provably bounded. |
| 19 | **Lossless density** (interleaves anywhere after 11.5) | Learned MA trees, LZ77, predictor selection for `jpxl-encode` modular | Density within a stated factor of `cjxl -e N` on a pinned corpus, all outputs decoded bit-exact by three decoders. |
| 20 | **Filter planning + high effort** | Inverse-gaborish preconditioning, EPF/sharpness policy, frontier beam search (M9–M10) | **partial 2026-08-05** — gab request + Jacobi precondition in policy (wave 20c); default still filters-off; EPF inverse / sharpness / beam search remain |

## Bit-exactness contract

Know which regime a path is in before writing its test.

| Path | Contract | Rationale |
| --- | --- | --- |
| Entropy coding (prefix, rANS, hybrid-uint, LZ77) | **Bit-exact** | Integer, fully specified. Any divergence is a bug. |
| Modular lossless roundtrip | **Bit-exact** (samples) | Integer arithmetic only, by construction of the mode. |
| Header serialization | **Bit-exact** | Given identical field values, our bits equal the reference bits. Divergence means a wrong conditional or a wrong `U32` distribution. |
| VarDCT lossy decode | **Tolerance-based** — Part 3 peak-error classes | Float pipeline; the standard defines conformance by bounded peak error, not identity. Record the class used with every result. |
| XYB and other float color math | **Tolerance-based** | Same. |

**Float policy:** all float conversions are software-defined. No reliance on
platform rounding mode, x87 excess precision, FMA contraction, or fast-math.
Conversions between float and integer are written explicitly with stated
rounding. Two hosts must produce identical output for identical input.

## Deliberately not in scope yet

Do not add these without a decision recorded here first:

- SIMD of any kind (scalar reference paths must be locked first)
- rayon / threading
- JPEG recompression and reconstruction (`jbrd` beyond passthrough)
- Animation playback semantics
- Progressive / partial decoding as a feature (the syntax must parse; the
  streaming API does not exist yet)
- GPU anything
- Perceptual metrics, encoder rate/distortion search, effort levels
- Splines and noise synthesis (patches likewise, until slice 8 is stable)

## Deferred crate splits

The workspace starts with `jpxl-bitstream`, `jpxl-core`, `jpxl-decode`,
`jpxl-cli`, `jpxl-conformance`. These splits are anticipated but not made until
the code justifies them:

The roadmap's single-crate recommendation was considered and overridden: the
workspace layout predates it, is approved, and its per-crate ownership is what
makes parallel multi-agent work safe. Do not relitigate.

| Future crate | Split out of | Trigger |
| --- | --- | --- |
| `jpxl-entropy` | `jpxl-core` | **Done** — created directly in the slice-3 wave (2026-08-02); entropy was always going to exceed a module. |
| `jpxl-encode` | new peer of `jpxl-decode` | Slice 10. Peer tree, never nested under the decoder. |
| `jpxl-encode-policy` | `jpxl-encode` | **Triggered** — slice 11 creates it (encoder phase above; design in `Encoder-plan1.md`). Policy never lives in the normative emitter. |
| `jpxl-container` | `jpxl-decode` | Slice 9, if Part 2 handling outgrows a module. |
