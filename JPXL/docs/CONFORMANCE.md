# CONFORMANCE

What JPXL supports and the tests that prove it. One row per feature or clause
group. A row may only claim "supported" when a named, passing test backs it —
"the code exists" is not support, and "it decodes without error" is not
correctness.

Every lossy row must state which contract applies (bit-exact vs. Part 3
peak-error class; see the bit-exactness contract table in `PLAN.md`).

Clause numbers cite the published Parts (1/2/3) — see `STANDARDS_INDEX.md`.

Last reviewed: 2026-08-04.

## Status

| Clause | Feature | Contract | Test | Status |
| --- | --- | --- | --- | --- |
| Annex C | Entropy decoding (prefix, rANS, hybrid-uint, LZ77, clustering) | Bit-exact | `jpxl-entropy` unit + property suites; every downstream decode | Supported |
| D, E | SizeHeader / ImageMetadata / extra-channel headers | Bit-exact | `smoke`, header unit tests, oracle cross-checks | Supported |
| E.4 | Compressed ICC | Bit-exact | fixtures 30–36 byte-exact vs `djxl --orig_icc_out` | Supported |
| F, G | FrameHeader / TOC / permutation / sections | Bit-exact | multi-section fixtures in `e2e_lossless` | Supported |
| Annex H | Modular (MA trees, all predictors incl. H.5 weighted, RCT, palette, Squeeze) | Bit-exact (samples) | `e2e_lossless` 19/19, zero ignores | Supported |
| Annex I | VarDCT (all decoded transform types, dequant, CfL, multi-pass) | Part 3 class per test | `e2e_vardct` 13/13, `e2e_progressive` 6/6, zero ignores | Supported; Hornuss/AFV/DCT4x4/≥DCT128 have no pixel-comparison coverage yet |
| I.2.4 | `RAW` dequantization matrices (inline modular sub-bitstream) | Part 3 class | corpus `bench_oriented_brg`/`_5`, `grayscale_jpeg`/`_5` (all peak <2e-6) | Supported; `RAW_MATRIX_CHANNEL_ORDER_IS_XYB`/`RAW_SUBBITSTREAM_IS_UNALIGNED` PROBED-CONFIRMED `true` (first real streams to use RAW) |
| G.1.3/G.2.3/G.4.2 | Extra channels in kVarDCT frames | Part 3 class; alpha itself lossless round-trip | `e2e_alpha` ladder 80–84 + corpus `alpha_*` | Supported (`ec_upsampling > 1` refused) |
| F.2, Table F.8 | Multi-frame compositing, all five blend modes | Part 3 class | corpus `blendmodes`/`blendmodes_5` | Supported for stored frames; animation (frames with duration) refused |
| G.2.2, F.2 | kLFFrame + `kUseLfFrame`, multi-pass accumulation | Part 3 class | `e2e_progressive` ladder 70–74 + corpus | Supported |
| Annex J | Gaborish + EPF | Part 3 with-filters class | filters-on fixtures 52/53/56 | Supported (J.2 chroma upsampling refused) |
| K.2, L.4 | Frame + extra-channel upsampling (2/4/8, default and custom weights) | Part 3 class | `e2e_upsampling` ladder 90–95 + corpus `upsampling`/`upsampling_5` | Supported in kVarDCT frames (kModular + upsampling refused) |
| F.2 | Cropped displayed frames (negative origins, any-edge overhang) | Part 3 class | corpus `spot`/`cmyk_layers`/`sunset_logo`, `CropRect` unit tests | Supported |
| D.3.2, Table D.4 | Orientation (all 8), integer + float planes | Bit-exact rearrangement | `e2e_layers` ladder 100–109, all 8 vs djxl | Supported (previews still refused) |
| D.3.6 | kBlack / kSpotColour extra channels | Graded as stored (18181-3 §4.1.2) | corpus `spot` (6ch) / `cmyk_layers` (5ch) | Supported; spot-colour rendering not performed (references don't) |
| K.3 | Patches incl. per-channel-group alpha blending, kReferenceOnly reference frames | Part 3 class | corpus `bike`/`bike_5`/`progressive`/`patches`/`patches_5`/`patches_lossless` | Supported for kVarDCT AND for a *displayed* kModular frame (`modular_displayed_pipeline`), incl. a non-`xyb_encoded` kReferenceOnly source; patches in a non-displayed kModular frame (LF/reference-only) still refused |
| K.5 | Noise synthesis (`NoiseParameters` LUT, `XorShift128Plus`/`SplitMix64`, K.5.2 rendering) | Part 3 class | corpus `noise`/`noise_5` (peak 4.9e-5 vs 0.004), fixture 110, `e2e_noise.rs` | Supported in kVarDCT frames with `upsampling == 1`; kModular frames, LF/reference-only frames, and noise combined with K.2 upsampling refused |
| L.2.2 | `kModular` pre-step for a *displayed* `xyb_encoded` frame | Tolerance | corpus `bicycles` (peak 2.5e-5 vs 9.8e-4), `e2e_modular_displayed.rs` | Supported for a 3-channel colour image with no restoration filter signalled; greyscale and gab/EPF-on combinations refused (untested) |
| L.2 | XYB inverse, transfer functions incl. negative-branch behaviour | Tolerance | color.rs tests, fixture 63, corpus | Supported |
| L.3 | `do_YCbCr` colour reconstruction | Tolerance | corpus `bench_oriented_brg`/`_5` (peak 1.9e-6), `grayscale_jpeg`/`_5` (peak 1.9e-6), `e2e_ycbcr.rs` | Supported in kVarDCT frames with `jpeg_upsampling == [0, 0, 0]`; J.2 chroma upsampling (nonzero `jpeg_upsampling`, e.g. `cafe`) and do_YCbCr in kModular frames still refused |
| Part 2 §8–9 | Box parser, `jxlc`/`jxlp`, box validation | Bit-exact framing | `container` suite, jxlinfo cross-check | Supported (`brob` decompression, `jxli`/`jbrd` parsing deferred) |
| — | Encoder, lossless (gray/RGB, 8/16-bit, RCT, multi-group, `jxlc`/`jxlp`) | Decoders must agree | self + `djxl` + `jxl-oxide` sample-exact | Supported |
| — | Encoder, lossy VarDCT (square transforms DCT8×8/16×16/32×32, CfL, per-varblock adaptive quantization, default matrices, 1 pass, filters off; exact byte-target rate loop, never-over + ≤1% undershoot fixed / ≤2% with AQ) | Decoders must agree (within output quantization) | `vardct_oracle.rs`/`vardct_roundtrip.rs`/`rate_loop.rs`: self + `djxl` + `jxl-oxide` within 1 8-bit code point on every ladder rung incl. multi-section, non-multiple-of-8, two-LF-group, HfMul>1, rate-targeted, CfL, merged-transform and varying-mul-row streams | Supported (hierarchical cover and AQ are opt-in `CoverMode::Hierarchical`/`AqMode`; rectangles, trained entropy, default flip: PLAN slice 18; no CLI wiring yet) |

Refused (typed `Unsupported`, never guessed): J.2 chroma upsampling
(`jpeg_upsampling != [0, 0, 0]`), do_YCbCr in kModular frames, upsampling
in kModular frames, animation (presented frames with a duration),
previews, splines, noise combined with K.2 upsampling, noise in
kModular/LF/reference-only frames, `lf_level > 1`, kVarDCT LF frames,
patches in a non-displayed kModular frame (LF/reference-only), a
restoration filter on a displayed kModular frame, a greyscale
`xyb_encoded` kModular displayed frame, a kVarDCT frame that is neither
`xyb_encoded` nor `do_YCbCr`, `brob` decompression.

## Official conformance corpus

Cloned by `tools/fetch-conformance.sh` into `tests/fixtures/conformance/`,
pinned by the checkout's commit. That clone brings `input.jxl`, `test.json`
and a preview `ref.png` for every test case, but the corpus's own big
reference files — `reference_image.npy`, `reference_preview.npy`,
`original.icc`, `reference.icc` — are `.gitignore`d and are fetched
separately by the corpus's own
`tests/fixtures/conformance/scripts/download_and_symlink_using_curl.sh`
(network, content-addressed by sha256, no `gsutil` dependency). Run that
script from `tests/fixtures/conformance/` after cloning; it downloads the
Google-Cloud-Storage object pool once into `.objects/` and symlinks each
test case's files in from there.

| Stream set | Revision | Result |
| --- | --- | --- |
| `tests/fixtures/conformance/` | pinned commit at clone time (see the submodule/checkout's own git log) | References downloaded via `download_and_symlink_using_curl.sh` (85 objects, ~1.2 GB, all 39 test cases linked). `bike_5`'s `reference_image.npy` (shape `(1, 2560, 2048, 3)`, sha256-verified) and `test.json` (`peak_error: 0.06`, `rms_error: 0.02`) are readable end to end by `jpxl_conformance::FloatImage::from_npy` — see `crates/jpxl-conformance/tests/corpus.rs`. **Thirty test cases pass their own `test.json` thresholds against JPXL's decoder**: `grayscale` / `grayscale_5` (peak 2.3e-4), `bike` / `bike_5` (2.5e-4), `progressive` / `progressive_5` (2.0e-5), `alpha_nonpremultiplied` / `alpha_triangles` (1.2e-7), `alpha_premultiplied` (2.6e-6), `patches` / `patches_5` (7.8e-6 / 2.4e-2), `blendmodes` / `blendmodes_5` (1.9e-6), `upsampling` / `upsampling_5` (4.3e-5 / 1.6e-2), `spot` (6.0e-8), `cmyk_layers` (1.2e-7), `sunset_logo` (4.8e-7), `noise` / `noise_5` (4.9e-5 vs a 0.004 budget), `bicycles` (2.5e-5 vs 9.8e-4), `patches_lossless` (1.2e-7 vs 9.8e-4), `delta_palette` (1.2e-7), `lz77_flower` (6.0e-8), `opsin_inverse` / `opsin_inverse_5` (6.9e-4), `bench_oriented_brg` / `_5` (1.9e-6), `grayscale_jpeg` / `_5` (1.9e-6) — tests in `crates/jpxl-decode/tests/{e2e_vardct.rs,e2e_progressive.rs,e2e_alpha.rs,e2e_upsampling.rs,e2e_layers.rs,e2e_noise.rs,e2e_modular_displayed.rs,e2e_previously_unattempted_corpus.rs,e2e_ycbcr.rs}`, zero ignores. `delta_palette`/`lz77_flower`/`opsin_inverse`/`_5` needed no new code — nobody had graded them before this wave; see `e2e_previously_unattempted_corpus.rs`'s module doc. Remaining 9: animation ×5 (`animation_icos4d`/`_5`, `animation_newtons_cradle`, `animation_spline`/`_5`), do_YCbCr-with-chroma-subsampling ×2 (`cafe`/`_5`, needs J.2), plus two probed-but-failing cases outside this wave's scope — `lossless_pfm` (peak 1.75 vs a 0.0 lossless budget) and `grayscale_public_university` (peak 0.27 vs 9.8e-4) — see `e2e_previously_unattempted_corpus.rs`'s module doc for the numbers. |

### Part 3 §4.2 grading (18181-3 §4.1.2 / §4.2 / §4.3)

`jpxl-conformance::metrics` implements the comparison Part 3 specifies for
image similarity, used both for the corpus above and for the handmade VarDCT
fixtures (50+):

* Samples are compared as `f32` on the **nominal `[0, 1]` scale, with no
  clipping** — a value outside that range (out-of-gamut, or before display
  mapping) is compared exactly as stored, never clamped first.
* Comparison is **per channel**: [`FloatImage`] holds samples C-order as
  `(frames, height, width, channels)`, matching the corpus's `.npy` layout
  and what `djxl --output_format npy` emits (verified against a real decode
  — magic `\x93NUMPY`, version `1.0`, dtype `<f4`, `fortran_order: False`).
* Three conditions, all required (§4.2):
  1. Dimensions, frame count and channel count are identical
     (`FloatImage::same_shape`).
  2. **Peak error**: the largest `|D − R|` over every sample, at most a
     stated threshold.
  3. **Per-channel RMSE**: `sqrt(mean((D − R)^2))` within each channel, at
     most a stated threshold.
* Thresholds come from each test case's `test.json` (`peak_error`,
  `rms_error` per frame) or, for handmade fixtures, from Annex A Table A.1's
  classes: VarDCT/Modular with filters is peak `0.06` / RMSE `0.02`; without
  filters, peak `0.004` / RMSE `1e-5`.

`FloatImage::from_npy` is a hand-rolled reader for the NPY subset §4.1.2
defines and djxl emits (version 1.0 header, little-endian `f32`, C-order,
rank-4 shape) — not a general NPY reader. `similarity()` returns the peak
and per-channel figures; `Similarity::conforms(peak_t, rmse_t)` applies a
class's thresholds. See `crates/jpxl-conformance/src/metrics.rs` for the
implementation and its unit tests (including a hand-computed RMSE case and
the verified djxl header layout).

## Malformed-input coverage

Decode paths are attacker-facing. Every rejection case gets a row: what is
malformed, and that JPXL rejects it without panic, unbounded allocation, or
unbounded CPU.

| Case | Expected | Status |
| --- | --- | --- |
| — | — | not started |
