# VarDCT flip-point probe pass (slice 8F)

Date: 2026-08-03. Status: frozen.

## What this is

Slice 8 accumulated nine one-bit readings that the standard's text does not
settle. Each was implemented behind a named constant so that the alternative
reading is one edit away. Until 8F there were no pixels, so none of them could
be tested end to end; the ANS/section-exhaustion gates that settled the
*entropy* layer are structurally blind to everything downstream of it.

This is that end-to-end pass. Method, for each flip point: flip the constant
(or, for the one that is not a constant, mutate the line it governs), rebuild,
run the whole acceptance ladder in `jpxl-decode/tests/e2e_vardct.rs`, and
compare the peak/RMSE numbers against the baseline. Restore afterwards.

The ladder grades JPXL's decode against a reference decode of the *same*
stream (`djxl --output_format npy` for the handmade fixtures, the corpus's own
published `reference_image.npy` for the conformance cases), per 18181-3 §4.2.

## Baseline

With every constant at the value slice 8 shipped **except** the one this pass
reversed (`EPF_SKIP_IS_PER_VARBLOCK`, below):

| case | class | peak | RMSE (worst channel) |
| --- | --- | --- | --- |
| 04 (8x8, filters on) | — | 7.0e-6 | 1.7e-6 |
| 50 gray, no filters, d1 | 0.004 / 1e-5 | 1.4e-6 | 3.3e-7 |
| 51 gray, no filters, d4 | 0.004 / 1e-5 | 3.3e-6 | 4.6e-7 |
| 55 RGB, no filters, d4 | 0.004 / 1e-5 | 5.0e-5 | 3.9e-6 |
| 52 gray, filters, d1 | 0.06 / 0.02 | 1.4e-6 | 2.9e-7 |
| 53 gray, filters, d4 | 0.06 / 0.02 | 2.6e-6 | 3.3e-7 |
| 56 RGB, filters, d1 | 0.06 / 0.02 | 5.4e-5 | 7.1e-6 |
| corpus `grayscale` | 0.004 / 1e-4 | 2.3e-4 | 7.8e-6 |
| corpus `grayscale_5` | 0.06 / 0.02 | 2.3e-4 | 7.8e-6 |

Everything is two to three orders of magnitude inside its class, which is what
makes the probe sharp: a wrong reading does not merely eat margin, it moves the
error by a factor of a thousand.

## What the streams actually exercise

From JPXL's own `DctSelect` histograms (the 8C fixture gate prints them), the
transform types reachable by any test in this pass are

```
Dct8x8(0) Dct2x2(2) Dct16x16(4) Dct32x32(5) Dct16x8(6)
Dct32x16(10) Dct16x32(11) Dct64x64(18) Dct64x32(19)
```

Never chosen by cjxl for this content: Hornuss, Dct4x4, Dct8x16, Dct32x8,
Dct8x32, **Dct4x8, Dct8x4**, AFV0-3, and everything from Dct32x64 up. All
streams are single-pass; `num_hf_presets == 1` everywhere; `lf_idx == 0`
everywhere.

## Verdicts

| flip point | owner | verdict |
| --- | --- | --- |
| `LLF_SCALEF_ARG_IS_VARBLOCK_DIMENSION` | 8A | **PROBED-CONFIRMED** |
| `DCT8X4_HALF_INDEX_IS_LOW_COORDINATE` | 8A | **NOT-DISCRIMINATED** |
| order table direction (I.3.1) | 8C | **PROBED-CONFIRMED** |
| `PREV_USES_CURRENT_PASS_COEFFICIENT` | 8C | **NOT-DISCRIMINATED** |
| `LF_QUANT_CHANNEL_ORDER_IS_XYB` | 8D | **PROBED-CONFIRMED** (`false`) |
| `EPF_SKIP_IS_PER_VARBLOCK` | 8E | **PROBED-FLIPPED** -> `false` |
| `EPF_STEPS_FROM_EXPLICIT_CONDITIONS` | 8E | **PROBED-CONFIRMED** |
| `EPF_BORDER_SAD_AT_REFERENCE_PIXEL` | 8E | **PROBED-CONFIRMED** |
| `EPF_DISTANCE_USES_STEP_INPUT` | 8E | **PROBED-CONFIRMED** |

### `LLF_SCALEF_ARG_IS_VARBLOCK_DIMENSION` — CONFIRMED

Flipped to the literally-printed reading, nine of ten cases fail and the RMSE
of every case is `NaN`, with peak `inf` on the RGB fixtures. This is the
predicted failure mode exactly: the printed `ScaleF(c, b)` divides by
`cos(pi/2) == 0` at `c == b/2`, which is reached by every transform from
DCT16x16 up, and the fixtures are full of DCT16x16/32x32/64x64. The derivation
in `2026-08-03-i8-scalef-argument.md` is now backed by pixels.

### `DCT8X4_HALF_INDEX_IS_LOW_COORDINATE` — NOT DISCRIMINATED

Flipping it changes **no digit** of any case. The histogram above says why:
cjxl never selected DctSelect 12 or 13 for this content. The reading stays
where I.9.8's stated layout put it, and this remains open. To settle it, a
stream containing DCT4x8/DCT8x4 varblocks is needed; the content that provokes
them (fine near-1-D structure) is not what the current fixtures contain.

### I.3.1 order-table direction — CONFIRMED

Not a constant: I.4 writes `order[...][k]`, and the question is whether that
subscript is the destination cell (shipped) or a source index. Mutating the
one line to build and use the inverse permutation breaks nine of ten cases,
corpus `grayscale` going from peak 2.3e-4 to 2.6e-2 (a 110x regression that
crosses its 0.004 class). 8C flagged this as the reading its ANS gate could not
see; the pixel comparison sees it clearly. The corpus cases carry a nonzero
`used_orders`, so the signalled-permutation branch is included in the evidence,
not only the natural orders.

### `PREV_USES_CURRENT_PASS_COEFFICIENT` — NOT DISCRIMINATED

Flipping it breaks nine of ten cases, but that is **not** evidence for the
reading. Every stream reachable here is single-pass, where the two readings are
identical by construction; the flip breaks things only because the `false` arm
is implemented as `PREV_USES_CURRENT_PASS_COEFFICIENT && ucoeff != 0`, which
degenerates to a constant `false` rather than expressing the accumulator
reading. So this pass yields nothing about the intended question, and it does
surface a second, smaller finding: the constant as written cannot be used to
probe the alternative even once a multi-pass stream exists. Settling it needs
both a progressive stream and a `false` arm that consults the accumulator.

### `LF_QUANT_CHANNEL_ORDER_IS_XYB` — CONFIRMED `false` (Y, X, B)

Wave 2 reversed this on statistical evidence from greyscale fixtures alone
(two channels decoding exactly flat), and explicitly recorded that RGB was
never probed. Setting it back to `true` now fails **all ten** cases with peak
errors of 0.52 to 1.47 — including the RGB fixtures 55 and 56 and both corpus
cases. The reversal is confirmed on colour content and on a normative
reference.

### `EPF_SKIP_IS_PER_VARBLOCK` — REVERSED to `false`

The one reading this pass changed. J.4.3 says "if sigma < 0.3 for a given
varblock, the decoder skips all steps on the pixels of that block", two
sentences after defining sigma at the 8x8 rectangle containing the reference
pixel. 8E took the word "varblock" literally (`true`).

| case | `true` (as shipped) | `false` (now shipped) |
| --- | --- | --- |
| 52 gray, filters, d1 | peak 2.96e-3 | peak 1.4e-6 |
| 53 gray, filters, d4 | peak 1.11e-2 | peak 2.6e-6 |
| 56 RGB, filters, d1 | peak 1.17e-2 | peak 5.4e-5 |
| corpus `grayscale` | peak 1.25e-3, RMSE 5.1e-5 | peak 2.3e-4, RMSE 7.8e-6 |

Under `false` the filters-on fixtures reach the same 1e-6 floor as the
filters-off ones — i.e. the restoration filters stop contributing error at all.
An error that collapses by three orders of magnitude onto the float-noise floor
when a one-bit reading is flipped is that reading being wrong. The two readings
differ only where `Sharpness` varies inside a varblock larger than 8x8, which
is common in these streams.

### `EPF_STEPS_FROM_EXPLICIT_CONDITIONS` — CONFIRMED

Flipped: fixture 52 goes 1.4e-6 -> 5.8e-3, fixture 56 goes 5.4e-5 -> 1.16e-2,
and corpus `grayscale` fails its RMSE class (1.46e-4 against a 1e-4 limit).
Fixture 53 is unchanged, which is itself informative: its `epf_iters` selects
the same step set under both readings, so only the d1 streams discriminate.

### `EPF_BORDER_SAD_AT_REFERENCE_PIXEL` — CONFIRMED

Flipped, every filters-on case degrades by roughly 1000x (52: 1.4e-6 ->
2.4e-3; 53: 2.6e-6 -> 5.2e-3; 56: 5.4e-5 -> 3.3e-3; corpus `grayscale`:
2.3e-4 -> 4.1e-4). No case crosses its class threshold, so this is a
consistent-direction result rather than a pass/fail one — but the direction is
unambiguous and holds on four independent streams.

Note that under the pre-reversal `EPF_SKIP_IS_PER_VARBLOCK = true` baseline
this flip point looked nearly inert (peaks identical, RMSE moving in the fourth
digit). Fixing the skip granularity is what made it visible: with the dominant
error removed, the smaller one became measurable. That is the general shape of
a probe pass — flip points are only as discriminating as the rest of the
pipeline is correct.

### `EPF_DISTANCE_USES_STEP_INPUT` — CONFIRMED

Flipped, only fixture 53 moves, and it moves hard: 2.6e-6 -> 4.0e-3, a factor
of 1500. The other cases are byte-identical. 53 is the one stream whose
`epf_iters` runs more than one step, and the two readings can only differ
across steps — a single-step filter has one buffer either way. So the evidence
is one stream wide but exactly the stream that can carry it.

## Reproducing

```
cd JPXL
cargo test -p jpxl-decode --test e2e_vardct -- --test-threads=1 --nocapture
```

Then edit the constant named in the table and re-run; the test prints peak and
per-channel RMSE for every case whether it passes or fails, so the comparison
does not depend on a threshold being crossed.

## What this pass did not settle

* `DCT8X4_HALF_INDEX_IS_LOW_COORDINATE` — needs a stream with DCT4x8/DCT8x4.
* `PREV_USES_CURRENT_PASS_COEFFICIENT` — needs a progressive (multi-pass)
  stream *and* a non-degenerate `false` arm.
* Hornuss, DCT4x4, AFV0-3 and every transform above DCT64x64 are reconstructed
  by code that no end-to-end test has yet executed. Their unit tests
  (round-trip, DC-only, AFV orthonormality) stand, but no pixel comparison
  covers them.
* `num_hf_presets > 1` and nonzero `lf_idx` remain unexercised, as 8C reported.

---

## Addendum, 2026-08-03 — the degenerate `false` arm is repaired

*Appended after the fact; nothing above is edited.*

This report's finding that `PREV_USES_CURRENT_PASS_COEFFICIENT`'s `false` arm
read `PREV_USES_CURRENT_PASS_COEFFICIENT && ucoeff != 0` — constant `false`
rather than the accumulator reading — has been acted on. `vardct/hf_coeff.rs`
now routes the decision through a `next_prev(prev_uses_current_pass, ucoeff,
accumulated)` helper whose `false` arm tests the accumulated multi-pass
coefficient at the same order position, read after this pass's contribution has
been added. `decode_hf_group_with_prev_reading` exposes the arm as a parameter
so one bitstream can be decoded both ways, mirroring how `vardct::lf` drives its
channel-order flip point.

Three unit tests were added, none of which settles the question — they make it
*testable*:

* `the_two_prev_readings_are_genuinely_different_functions` — a truth table over
  `(ucoeff, accumulated)`. The discriminating row is `(0, 7)`: the current-pass
  arm says `false`, the accumulator arm says `true`. Under the old formulation
  both said `false`.
* `prev_shifts_the_coefficient_context_by_exactly_one` — `prev` enters I.4's
  context index additively, so a wrong `prev` selects the neighbouring histogram
  for every later symbol in the block. This is why the reading matters.
* `the_two_prev_readings_coincide_on_a_single_pass_stream` — decodes one
  hand-built stream under both arms and gets identical coefficients. The reason
  is structural: `UnpackSigned(u) == 0` exactly when `u == 0`, and F.2's left
  shift cannot turn a non-zero into a zero without an overflow that is rejected,
  so with a zero accumulator the two arms are the same function. This is what
  licenses shipping `true` with the question open, and it is why all eight
  passing ANS-gate fixtures are unaffected (verified: 50, 51, 52, 53, 55, 56,
  `corpus_grayscale`, `corpus_grayscale_5` all still green under both arms).

Still not settled: no progressive stream exists to decode. The status of this
report's conclusion is unchanged — the flip point is open, but flipping it no
longer breaks the decoder for reasons unrelated to the question.

## Addendum, 2026-08-03 — RAW dequantization matrices are no longer refused

This report and the wave-3 ledger recorded that RAW matrices were refused
because I.2.4 reads the 3-channel matrix inline and 8B's
`raw_requests()`/`set_raw_matrix()` split could not express that. That is fixed:
`read_dequant_matrices_with` / `read_hf_global_params_with` take an optional
`RawMatrixContext` and decode the sub-bitstream at the bit where the clause puts
it. Two new flip points came out of it, both untestable until a stream uses RAW:
`RAW_MATRIX_CHANNEL_ORDER_IS_XYB` and `RAW_SUBBITSTREAM_IS_UNALIGNED`. See the
addendum to `2026-08-03-i25-default-dequant-constants.md` for the wire-format
derivation.
