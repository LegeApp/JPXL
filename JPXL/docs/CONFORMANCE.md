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
| G.2.2, F.2 | kLFFrame + `kUseLfFrame`, multi-pass accumulation | Part 3 class | `e2e_progressive` ladder 70–74 + corpus | Supported |
| Annex J | Gaborish + EPF | Part 3 with-filters class | filters-on fixtures 52/53/56 | Supported (J.2/K.2 upsampling refused) |
| K.3 | Patches, kReferenceOnly reference frames | Part 3 class | corpus `bike`/`bike_5`/`progressive` | Supported (alpha blend modes refused) |
| L.2 | XYB inverse, transfer functions incl. negative-branch behaviour | Tolerance | color.rs tests, fixture 63, corpus | Supported |
| Part 2 §8–9 | Box parser, `jxlc`/`jxlp`, box validation | Bit-exact framing | `container` suite, jxlinfo cross-check | Supported (`brob` decompression, `jxli`/`jbrd` parsing deferred) |
| — | Encoder (gray/RGB, 8/16-bit, RCT, multi-group, `jxlc`/`jxlp`) | Decoders must agree | self + `djxl` + `jxl-oxide` sample-exact | Supported |

Refused (typed `Unsupported`, never guessed): extra channels in kVarDCT,
alpha blending, YCbCr, upsampling, splines, noise, animation blending,
`lf_level > 1`, kVarDCT LF frames, `brob` decompression.

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
| `tests/fixtures/conformance/` | pinned commit at clone time (see the submodule/checkout's own git log) | References downloaded via `download_and_symlink_using_curl.sh` (85 objects, ~1.2 GB, all 39 test cases linked). `bike_5`'s `reference_image.npy` (shape `(1, 2560, 2048, 3)`, sha256-verified) and `test.json` (`peak_error: 0.06`, `rms_error: 0.02`) are readable end to end by `jpxl_conformance::FloatImage::from_npy` — see `crates/jpxl-conformance/tests/corpus.rs`. **Six test cases pass their own `test.json` thresholds against JPXL's decoder**: `grayscale` / `grayscale_5` (peak 2.3e-4), `bike` / `bike_5` (peak 2.5e-4 after the frame-wide LF smoothing fix), `progressive` / `progressive_5` (peak 2.0e-5) — tests in `crates/jpxl-decode/tests/{e2e_vardct.rs,e2e_progressive.rs}`, zero ignores. Remaining cases need extra channels/alpha, YCbCr, animation, splines, noise, or `jbrd`. |

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
