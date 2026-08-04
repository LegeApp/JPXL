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

## 2026-08-04 (wave 8) — cropped frames + orientation + kBlack; corpus 18/39; encoder phase planned

**`spot`/`cmyk_layers`/`sunset_logo` pass** (peaks 6e-8/1.2e-7/4.8e-7,
orders inside thresholds). All three were gated on ONE refusal: F.2
cropped frames. Built: `composite_frame` placing frames at (x0,y0) with
i64 `CropRect::intersect` (negative UnpackSigned origins, any-edge
overhang), lifted for ALL encodings (compositing is display-space,
encoding-agnostic); Table D.4 orientation — all 8 rows, derived by
inverting the first-row/first-column pair, cross-checked against the
prose, applied to integer AND float planes at the very end (D.3.2;
every in-codestream dimension is pre-orientation). **kBlack needed NO
code**: 18181-3 §4.1.2 grades every extra channel as itself in ec_info
order — CMYK conversion would FAIL shape condition 1. `bench_oriented_brg`
is NOT unlocked by orientation despite the name: its gate is do_YCbCr.
Orientation was previously silently IGNORED (not refused) — sunset_logo's
shape assertion is what caught it. Fixtures 100–109 (hand-built eXIf
orientation tags, jxlinfo-verified by the script; cjxl cannot emit
cropped non-animation frames, so crop evidence is the corpus streams +
unit tests). Mutation-tested: anti-transpose↔transpose killed by the
corner test (both are involutions — the inverse-composition test alone
cannot kill it).

**Flip point, recorded unexercised:** `CROP_LEAVES_RUNNING_IMAGE_OUTSIDE
= true` — what "the image" holds outside a cropped frame's rectangle
(F.2 names two buffers and never says). Proven undiscriminated: all
three corpus cases store to slot 1 and read source==1; both arms
implemented.

**Corpus 18/39.** Remaining gates, enumerated across all 39: animation
×5, do_YCbCr ×6 (incl. bench_oriented_brg ×2), noise ×2, `bicycles`
(xyb modular displayed frame), `patches_lossless` (patches in kModular
+ stored frame in non-XYB). Gate: 1002 tests, 0 failed, zero ignores.

**Encoder phase adopted into PLAN.md (slices 11–20)** from external
advisor doc `docs/Encoder-plan1.md`, with recorded adjustments: ANS
encoder is its own slice 11.5; lossless density (MA trees/LZ77)
interleaves as slice 19; the inverse-primitives-into-core refactor is a
gated mechanical slice; SIMD/threading stay out until a scalar R-D
baseline. DO NOT START until the user triggers it (their session-limit
budgeting).

**Next candidates (decoder):** noise (Annex K.4 — synthesis, likely
self-contained); `bicycles`/`patches_lossless` (kModular displayed-frame
gaps); YCbCr + jbrd (big, unlocks 6+); animation (scope decision);
rare-transform coverage (Hornuss/DCT4x4/≥DCT128 still no pixel proof);
the grey-Y 2.3e-4 residual family.

---

## 2026-08-04 (wave 7) — K.2 upsampling; THIRD scan-verified Part 1 defect (I.2.4 AFV transpose); corpus 15/39

**`upsampling`/`upsampling_5` pass** (peak 4.3e-5 / 1.6e-2 vs 0.004 /
0.06). Frame upsampling is **K.2**, not J.2 — J.2's triangle filter is
only for `jpeg_upsampling` (still refused). Built `frame/upsampling.rs`:
K.2 index formula, default tables (validated WITHOUT a scan read: all 84
positions' 25 weights sum to 1 within 3e-8 — a digit slip cannot
survive), 5×5 mirrored window with per-output [min,max] clamp, top-left
crop, L.4's 8×-then-f/8 split, D.3 custom weights (unit-tested; no
stream exercises `cw_mask != 0`). Pipeline order per K.1: Annex J at the
STORED frame size → K.2 → patches → Annex L. Groups/modular channels
stay on the downsampled frame grid. Flip settled by fixture:
`EC_DIMS_INCLUDE_EC_UPSAMPLING=true` (F.2 cumulative; false desyncs
fixture 94's modular stream). Fixtures 90–95 + `e2e_upsampling.rs`.

**I.2.4 AFV weight placement is a PUBLISHED DEFECT (scan-verified,
printed p.59): the text writes `weights(2*y, 2*x)` for `freqs[y*4+x]`,
the transpose of the coefficient's actual position (I.9.8 puts basis
`y*4+x` at column 2x, row 2y; the freqs table's four zero entries match
the four skipped positions).** Shipped transposed as
`AFV_FREQ_POSITION_IS_TRANSPOSED=true` with a directional unit test.
Localisation was the proof: under the literal reading the four worst
tiles on `upsampling` were AFV0–3 varblocks holding ~100% of the squared
error while 94 DCT4x8/DCT8x4 blocks sharing the same IDCT were clean;
transposing drops the corpus case 7.8e-2 → 4.3e-5 and a no-resampling
control 8.7e-3 → 3.8e-5. Scan-verified Part 1 defect tally: I.8 ScaleF,
Table I.6 index 16 (candidate), I.2.4 AFV — first AFV pixel coverage,
closing part of the rare-transform gap.

**Still open:** `bike` 2.48e-4 / `grayscale` 2.28e-4 residual family is
NOT AFV (unchanged by the fix). Unexercised: custom upsampling weights,
factors >8 beyond unit tests, `dim_shift > 0`, upsampling in kModular
(typed refusal), J.2 chroma upsampling.

**Next candidates:** cropped/oriented displayed frames + kBlack
(unlocks spot/cmyk_layers/sunset_logo); animation (scope decision);
Hornuss/DCT4x4/≥DCT128 pixel coverage; the 2.3e-4 grey-Y residual;
jxli/jbrd (scan first); Brotli decision.

---

## 2026-08-04 (wave 6) — extra channels + alpha + frame blending; corpus 6 → 13 cases green

**Seven more corpus cases pass their test.json thresholds:**
`alpha_nonpremultiplied`/`alpha_triangles` (needed only 4-channel
grading), `alpha_premultiplied` (extra channels in kVarDCT),
`patches`/`patches_5` (K.3.2 per-channel-group alpha patch blending),
`blendmodes`/`blendmodes_5` (multi-frame F.2 compositing, all five
Table F.8 modes). Corpus total: 13 of 39.

**Built:** G.1.3/G.2.3/G.4.2 extra-channel modular streams in kVarDCT
frames (selection/copy-back FACTORED into shared
`lf_group_selection`/`pass_group_selection`/`decode_group_channels` —
do not re-duplicate); `render::ExtraPlanes` appended to DecodedImage at
each channel's own ec bit depth; K.3.2 alpha in `apply_patches` (K.3.2's
`c` ranges over [0, num_extra], honours clamp); `frame/blending.rs`
`Canvas` + `blend_sample` + `composite_frame` with per-channel-group
source slots. Pre-CT reference slots (XYB, K.3) and post-CT canvases
(display space, F.2) are SEPARATE — F.2 blends after Annex L. An identity
first frame returns its own integer planes, so lossless stays bit-exact.
Animation (a presented frame with duration) is now an EXPLICIT refusal —
the old "more than one regular frame" guard no longer covers it.

**Flip points:** none settled — corpus can't discriminate
(all reference frames at origin; every patch case has exactly one extra
channel). Two NEW recorded as unexercised:
`PATCH_ALPHA_IS_THE_PATCHS_OWN`, `ALPHA_SELF_RULE_IS_THE_NAMED_CHANNEL`
(readings coincide for single-alpha images — every corpus stream).
Addendum in 2026-08-03-patches-k3.md.

**Known residual (pre-existing, not this wave):** greyscale-VarDCT Y
carries ~1.9e-4 RMSE vs oracle (fixture 84 discriminator: identical grey
source with NO alpha reproduces it to 4 s.f.) — same family as corpus
`grayscale`'s 2.3e-4 peak. A future hunt should start from grey-only
VarDCT, not alpha.

**Out of scope, enumerated:** upsampling ×2 (J.2 + K.2 ec_upsampling=4),
`spot`/`cmyk_layers`/`sunset_logo` (cropped displayed frames +
orientation + kBlack), `patches_lossless` (patches in kModular +
stored frame in non-XYB), animation ×5, `noise`, `cafe`,
`bench_oriented_brg`, `grayscale_jpeg`, `bicycles`.

**Next candidates:** J.2/K.2 upsampling (unlocks 2 cases); cropped
displayed frames + orientation (unlocks 3, incl. spot/cmyk kBlack);
animation compositing (5 cases — needs a decision on presentation
semantics, PLAN lists it out of scope); rare-transform pixel coverage;
the grey-Y 1.9e-4 residual; jxli/jbrd (scan first); Brotli decision.

---

## 2026-08-04 (wave 5) — bike + progressive corpus PASS; six corpus cases green, zero ignores

**bike divergence killed, two real bugs.** (1) Transfer functions below
zero: BT.709 evaluates its piecewise condition on the SIGNED value (the
linear toe `4.5·v`), sRGB extends with odd symmetry — measured both ways
(bike spikes fit slope 1.000/const 4.4993; an out-of-gamut sRGB fixture
matches odd at 8.1e-6 and misses literal by 0.11). Neither the standard
nor IEC/ITU define the negative domain; pinned as flip point
`NEGATIVES_TAKE_THE_LINEAR_SEGMENT = [false, true]` ([sRGB, 709]) with
directional tests + fixture 63. (2) I.5.2 adaptive LF smoothing is
FRAME-WIDE ("each LF sample of the image"), not per LF group — per-group
loops skipped every group's edge rows, leaving a 16-row band at bike's
only internal LF-group seam (y=2048). Fixed by assembling the frame-wide
LF image (dequant and LF CfL commute with assembly — LF CfL uses the
frame-wide I.2.3 factors, not per-tile). Fixture 64 (128×2176, `-d 6
-e 3` load-bearing: `-d 1` sets kSkipAdaptiveLFSmoothing) is the only
multi-LF-group fixture in the tree — the standing seam regression. bike
0.2466 → 2.5e-4. Eliminated for bike: per-tile B CfL, I.5.3 B terms.
Rare transforms (Hornuss/AFV/DCT4x4/≥DCT128) still have zero pixel
coverage.

**Progressive corpus done.** `progressive`(_5 is a symlink to it) = patch
atlas + Squeezed kModular kLFFrame (lf_level 1) + 2-pass kVarDCT with
kUseLfFrame. Built: LfFrame slots + L.2.2 kModular pre-step shared via
`xyb_from_modular` (flip `LF_FRAME_IS_XYB_PRESTEP=true`: no Quantizer in
an LF frame's LfGlobal, so raw integers have no shared scale; false gives
peak 9e10); kUseLfFrame skips ALL of G.2.2/I.5.2 incl. smoothing (the
corpus frame has 4 LF groups + smoothing bit clear and still grades
2.0e-5 — independent confirmation). Multi-pass needed NO new code, only
reachability. Flips settled: `PREV_USES_CURRENT_PASS_COEFFICIENT=true`
(five multi-pass streams: true → exact TOC exhaustion + C.3.2 terminal
states; false → entropy desync inside I.4; single-pass control byte-
identical); `G42_SIZE_TEST_IS_SHIFTED=true` (unshifted leaves channels
37/38/41/42 of a Squeeze pyramid decoded by NO rule; false runs off the
section end at exactly its TOC length). Ladder fixtures 70–74 +
`e2e_progressive.rs`.

**State:** six corpus cases pass their test.json thresholds (grayscale,
grayscale_5, bike, bike_5, progressive, progressive_5); ZERO `#[ignore]`
in jpxl-decode tests; 945 tests green. jxlinfo now built/installed by
setup-oracles.sh. Note: bike rungs cost ~78 s debug (6 s release); the
progressive corpus rung is release-always but debug-opt-in via
`JPXL_SLOW_TESTS=1`.

**Traps:** the seam bug is invisible on every ≤1-LF-group image — do not
"optimize" smoothing back into the per-group loop; fixture 64's rung is
the only thing that would catch it. `smooth_lf_image` must stay gated on
`!use_lf_frame`.

**Next candidates:** extra channels in kVarDCT + alpha blend rows
(Table K.1) — the alpha corpus cases; rare-transform pixel coverage;
`lf_level > 1` / chained LF frames (nothing exercises them); jxli/jbrd
parsing (scan first); Brotli decision; open flips still without streams:
PATCH_REFERENCE_IS_CANVAS_COORDINATES, ALPHA_GUARD_COUNTS_EXTRA_CHANNELS,
`num_hf_presets > 1`, nonzero `lf_idx`.

---

## 2026-08-03 (wave 4) — THE MODULAR BUG IS DEAD: `^` misread as `*`; slice 9; patches; RAW fixed

**Root cause of the project's oldest open bug — one character.** The original
image scan (printed p.50) shows H.5.2's clamp guard as
`((true_err_N ^ true_err_W) | (true_err_N ^ true_err_NW)) <= 0` — XOR,
sign-bit arithmetic matching its own comment. ALL THREE transcriptions
misread `^` as `*`; they descend from one scan and share one glyph
confusion, so their agreement corroborated nothing. **Both previously
claimed H.5.2 "standard defects" are WITHDRAWN** — the published standard
is correct; `weighted.rs` now has one symmetric clamp behind the literal
guard (`EXPERIMENT_CLAMP_SYMMETRIC` and the asymmetric complex deleted).
Fixed at once: sawtooth (60), LfQuant repros (61/62), fixtures 54/57 (8C
gates un-ignored, green), the silent-corruption case, and the corpus
blockage. All 11 lossless fixtures stay bit-exact; e2e_lossless 19/19,
zero ignores. Scan-verified defect tally now: I.8 ScaleF divide-by-zero
(confirmed at scan) and Table I.6 index-16 bases (candidate, scan-read).

**Slice 9 (container) done.** `BoxTree::parse` (clause-8 framing) separate
from `validate` (clause-9 shalls); order-VALIDATING jxlp reassembly
(sorting hid corruption), proven vs independent jxlc reference and both
external decoders; `jpxl boxes` subcommand; jxlinfo as box oracle (build
via `cmake --build libjxl/build --target jxlinfo`, not in setup script
yet). brob stays compressed (typed Unsupported; Brotli dependency is a
PLAN decision); jbrd unparsed pending scan cross-check. Trap: a final box
whose declared length overruns the file is REJECTED; jxlinfo lists
nonexistent bytes — do not loosen to match. Container errors ride
`JpxlError::InvalidHeader` because `FieldOutOfRange` hard-codes 18181-1.

**Patches (K.3) done.** Dictionary is the FIRST row of Table G.1 (proven
by exact LfGlobal exhaustion on bike_5, 12293/12296 bits); rendering on
XYB planes between Annex J and Annex L; kReferenceOnly kModular reference
frames in four slots via L.2.2's kModular pre-step. Traps: L.2.2 kModular
channel order is y',x',B' (luma first — XYB reading swaps patch chroma);
a missing LfGlobal bundle reports itself hundreds of bytes downstream
under an unrelated error — measure section exhaustion to localise.

**RAW dequant matrices decode (Trap removed)**: I.2.4 reads the 3-channel
modular sub-bitstream INLINE; HfGlobal is one section, H.4.1's formula is
a stream index. `read_*_with(…, Option<&RawMatrixContext>, …)`;
context-less callers refuse at the sub-bitstream's first bit. `PREV` flip
point's false arm now really consults the accumulator (do not "simplify"
`next_prev` back — the point is the `ucoeff=0, accumulated≠0` case).

**Open — one B-channel divergence on corpus bike/bike_5** (now decoding
END TO END): peak 0.2466 on B only, X/Y ≈0.017, RMSEs near-passing.
`#[ignore]`d with forensics in e2e_vardct.rs. Probe already done:
`DCT8X4_HALF_INDEX_IS_LOW_COORDINATE` is CONFIRMED and now DISCRIMINATED
(flipping fails all channels at 1.1 — bike contains DCT8x4). Suspects:
B-channel dequant of a rare transform (Hornuss/AFV/DCT4x4 have zero pixel
coverage), per-tile B CfL, B-specific I.5.3 terms.

**METHODOLOGICAL TRAP (permanent):** when every available transcription
agrees on something that reads as nonsense, that is evidence about the
transcription pipeline, not the standard. The AGENTS.md §2 chain works
only if followed TO THE IMAGE SCAN; three sessions modelled the nonsense
instead. Any "defect in the standard" claim requires a scan read first.

**Next candidates:** the bike B-channel divergence; kLFFrame + multi-pass
(progressive corpus cases); extra channels in kVarDCT + alpha patch blends
(alpha corpus cases); jxli/jbrd parsing (scan first); Brotli decision.

---

## 2026-08-03 (VarDCT wave 3) — 8F assembly: VarDCT DECODES END TO END; slice 8 core complete

**Every acceptance rung passes, 2–3 orders of magnitude inside its class:**
fixture 04 first-light peak 7e-6; filters-off 50/51/55 ≤5e-5 (class 0.004 /
1e-5); filters-on 52/53/56 ≤5.4e-5 (class 0.06 / 0.02); **conformance corpus
`grayscale`/`grayscale_5` at 2.3e-4 against PUBLISHED references** — true
standard conformance, not libjxl agreement. `bike_5` skips (needs patches,
K.3, out of slice scope). `DecodedImage` gains `float_planes` (unclipped
f32, the §4.2 surface); integer planes quantize at the edge for VarDCT.
Output colour space is the SIGNALLED encoding, except under `want_icc`
where output stays linear (clause 4 — worth 0.287 → 0.001 peak on corpus
`grayscale`).

**Flip-point probe pass: 9 tested, 1 REVERSED** —
`EPF_SKIP_IS_PER_VARBLOCK = false` (per 8×8 block; the literal per-varblock
reading cost three orders of magnitude and masked the other EPF probes).
Confirmed load-bearing by flip: LLF ScaleF argument, I.3.1 order-table
direction, LfQuant Y,X,B, three EPF readings. Still open for want of
streams: `DCT8X4_HALF_INDEX_IS_LOW_COORDINATE`,
`PREV_USES_CURRENT_PASS_COEFFICIENT` (whose false-arm is also degenerate —
8C defect, needs a real accumulator consult before it can be tested).
See `docs/experiments/2026-08-03-vardct-flip-point-probe.md`.

**Traps:**
- RAW dequant matrices are REFUSED (`Unsupported`), not missing: I.2.4
  reads the 3-channel matrix inline mid-bitstream; 8B's
  `raw_requests()`/`set_raw_matrix()` split cannot express that. Fixing it
  means restructuring `read_dequant_matrices` to take the modular decoder.
- Multi-section VarDCT is PROVEN working (264×100 / 100×264 at 2e-6);
  anything failing inside G.2.2 LfQuant is the modular bug below, upstream
  of VarDCT.

**Modular H.5 bug — sharpened, not fixed (deliberately).** A clamp
lower-gate refinement fits all 18 834 harvested clamp decisions and cuts
the sawtooth to ONE wrong sample — but that sample ((31,19): all four
true_err equal, no `max_error` reading satisfies the encoder's branch)
REFUTES the model, so it was not shipped. Minimal repros checked in:
fixture 60 (277 B sawtooth, one bad sample), 61 (644 B VarDCT LfQuant;
trigger is channel size ≥15×15 both dimensions, content-irrelevant). 8F
adds: reproduces losslessly at 300×100; can corrupt samples SILENTLY
(268×100 decodes at 500× normal error). Eliminated with evidence: err_sum
last-column, wider clamp bounds, corrected-weight symmetric clamp, bit
depth. Next surfaces: `err[i]`/`err_sum` weight computation, last-column NE
substitution. Method note: branching probes must prefer the decoder's own
context or they desync (the old (2,1) first-divergence was that artifact;
the true one is (29,17)). See
`docs/experiments/2026-08-03-h52-clamp-lower-gate-and-sawtooth.md`. 8C's
54/57 gate tests remain `#[ignore]`d.

**Slice 8 residuals (beyond the modular bug):** patches K.3 (unlocks
bike_5/bike/progressive), extra channels in kVarDCT (alpha corpus cases),
Hornuss/DCT4x4/AFV/≥DCT128 untested by any pixel comparison (cjxl never
emitted them), RAW matrices, progressive/multi-pass streams.

---

## 2026-08-03 (VarDCT wave 2) — 8C HF decode proven on real streams; 8D-dequant + CfL

**8C** — `vardct/{order,hf_coeff}.rs`: I.3.1 orders, I.3.3 histograms, I.4
full context model to quantized integers. **The ANS final-state +
section-exhaustion gate passes on fixtures 50/51/52/53/55/56 and corpus
`grayscale`/`grayscale_5`** — six Order IDs, four non-square transforms,
permutation branch exercised by the corpus (`used_orders = 20`).
Mutation-verified (channel order, `c ^ 1`, `prev` seed all caught). New
flip-point `PREV_USES_CURRENT_PASS_COEFFICIENT`. **Trap:** the passing ANS
gate is structurally blind to the order-table direction (contexts depend on
`k`, never `order[k]`); `order[k]` = destination cell per I.3.1's assignment,
and 8F's pixel comparison is the decisive evidence. Unexercised by any
stream found: LF/QF thresholds (`lf_idx ≡ 0` everywhere), `num_hf_presets
> 1`, `num_passes > 1`.

**8D-dequant** — `vardct/cfl.rs` + `lf.rs`'s dequant half: I.5.2
(dequant → LF CfL → smoothing, in that clause-stated order), I.6 (LF: one
frame-wide `(kX,kB)`; HF: per-64×64-tile via `CflFactors::for_hf`, applied
in I.5.3 by 8F). **Wave-1 flip-point REVERSED by fixture evidence:**
`LF_QUANT_CHANNEL_ORDER_IS_XYB = false` — LfQuant is Y,X,B (under the XYB
reading two channels decode exactly flat while channel 0 carries all
structure). See `docs/experiments/2026-08-03-lf-quant-channel-order-fixture-evidence.md`.

**ESCALATION — the open modular bug now blocks VarDCT:** fixtures 54 and 57
fail inside G.2.2 `LfQuant`'s own modular decode (C.3.2 terminal check),
same content-dependent family as the sawtooth trap. Siblings 55/56 pass.
Root-cause hunt dispatched alongside wave 3; 8C's two gate tests un-ignore
when it's fixed.

**Next:** wave 3 = 8F assembly + acceptance (fix HfGlobal skip, wire
everything, e2e_vardct.rs ladder) in parallel with the modular bug hunt.

---

## 2026-08-03 (VarDCT wave 1) — 8B parameter bundles, 8D-parse sub-bitstreams

**8B** — `vardct/{quantizer,block_ctx,dequant_matrix}.rs`: G.1.2, I.2.1–I.2.6
complete; reuses `DecodeError` (no error.rs wiring needed). **FOURTH DEFECT
CANDIDATE:** Table I.6's DCT128x256 Y/B bases break the per-family doubling
regularities while preserving exactly the doubled fractional parts; verified
at the image scan (not OCR). Printed values ship behind
`DCT128X256_DEFAULT_BASES_AS_PRINTED`, sentinel test
`large_dct_bases_double_per_size_step`; see
`docs/experiments/2026-08-03-i25-default-dequant-constants.md`. Eight I.2.5
OCR garbles settled at page-ranged scan reads. RAW dequant matrices expose
`raw_requests()`/`set_raw_matrix()`; `matrix()` is typed `Unsupported` until
8F wires section `3*num_lf_groups + index`. 8F integration snippet is in the
8B report (read_lf_channel_dequantization → read_lf_global_vardct →
read_hf_global_params).

**8D-parse** — `vardct/{lf,hf_meta}.rs` + `frame/stream_index.rs`: G.2.2
LfQuant to quantized planes (I.5.2 seam marked), G.2.4 four channels +
greedy varblock placement (covered-exactly-once, no LF-group crossing,
reject-not-clamp), all five H.4.1 stream-index formulas typed (decode.rs's
two inline sites migrate in 8F). Open flip-point:
`LF_QUANT_CHANNEL_ORDER_IS_XYB` (needs nonzero-LF-chroma fixture, 8F).

**Traps:** `latex/part1.tex` and the transcription PDF are BYTE-IDENTICAL
for Part 1 numeric tables — they are one source, not two; the only
independent pair is `part1.md` vs that pair. Do not "fix" Table I.6 index 16
without flipping the constant. Do not apply ×64 to I.2.5 defaults — they are
already post-scale (Hornuss 280 vs DCT8x8 3150 is the cross-check).

**Next:** wave 2 = 8C (HfPass/coefficient decode, opus) + 8D-dequant
(I.5.2/I.6, callback to the 8D agent); then wave 3 = 8F assembly.

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
- ~~Sawtooth 32×32 decode bug~~ **RESOLVED 2026-08-03** (corrected in
  place): root cause was H.5.2's clamp guard OCR'd as `*` instead of `^` in
  every transcription. Fixed in `weighted.rs`; fixtures 60/61/62 are the
  regression tests. Replacement trap: **when every transcription agrees on
  nonsense, suspect the transcription pipeline, not the standard — escalate
  to the image scan before claiming a standard defect.** Three sessions
  modelled the OCR artifact instead of reading one scan page.
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
