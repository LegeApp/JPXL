# I.2.5 default dequantization constants: which printed digits are real?

Date: 2026-08-03
Slice: 8B (VarDCT parameter bundles)
Status: frozen

## 1. Question

Table I.6 of 18181-1 (I.2.5) is a page-and-a-half of floating-point constants
that no invariant fully pins. Both text conversions of Part 1 render several of
its entries as non-numbers, as digit sequences broken by spaces, or in
contradictory orders. Which digits are the standard's, and does anything remain
inconsistent once the transcription noise is removed?

## 2. Preregistered gate

For each suspect entry, in order:

* **Pass (settled by text)** — `latex/part1.tex`, `markdowns/standard-markdowns/part1.md`
  and `original-pdfs-do-not-read-first-if-markdown-exists/ISO_IEC_18181-1_2024_transcription.pdf`
  agree on a well-formed number after allowing for known OCR classes (letter
  `O` for zero, stray intra-number spaces, dropped minus signs).
* **Escalate** — they do not agree, or the agreed value is not a number. Then a
  page-ranged read of the image scan decides, and the scan is final.
* **Anomaly** — the scan is unambiguous but the value contradicts a structural
  regularity that every other row of the same table obeys. Then the printed
  value ships, a named flip point records the alternative, and the
  contradiction is documented rather than silently "fixed".

Structural regularities available for cross-checking, chosen before looking at
the data:

* R1: each channel row is a positive base followed by `Mult()` arguments, so a
  sign or order error shows up as a non-monotone band sequence.
* R2: the three channels of a row have descending bases (X > Y > B).
* R3: the "large DCT" rows come in two families, square (indices 11, 13, 15)
  and oblong (12, 14, 16), and within a family each size step doubles the base.

## 3. Method

Sources, in the order AGENTS.md §2 prescribes:

1. `latex/part1.tex` lines 4223–4331 (I.2.5 and I.2.6).
2. `markdowns/standard-markdowns/part1.md` lines 2838–2960.
3. `pdftotext -layout` of the text-only transcription PDF, lines 3780–3880.
4. Image scan, `original/...ISO_IEC 18181, 2, 2024 jul...pdf`, pages 60, 62, 64
   and 65 only, rendered at 250 dpi with `pdftoppm` and cropped to the table
   rows in question. No other page of the scan was opened.

The transcription PDF and `latex/part1.tex` turned out to carry byte-identical
text for this table, so they count as one source, not two. That is itself worth
recording: for Part 1 numerics the only genuinely independent pair is
`part1.md` versus the LaTeX/transcription pair.

## 4. Raw results

| # | Entry | Printed in LaTeX / transcription | Printed in `part1.md` | Scan | Resolution |
| --- | --- | --- | --- | --- | --- |
| 1 | I.2.4 `Interpolate` | `A * pow(B / 4B, frac_index)` | `A * pow(B / A, frac_index)` | `A * pow(B / A, frac_index)` | `B / A` |
| 2 | I.2.4 `Mult` | `1 / (1 - wv)` | `1 / (1 - v)` | `1 / (1 - v)` | `1 / (1 - v)` |
| 3 | DCT2x2 `params[0][3]` | `64.0.0` | `640.0` (in a scrambled row) | `640.0` | `640.0` |
| 4 | DCT32x8 `params[2][0]` | `3397.776032753087 20128` | `3397.77603275308720128` | `3397.77603275308720128` | concatenated |
| 5 | DCT256x256 `params[2][0]` | `17972.0951 2039390824` | `17972.09512039390824` | `17972.09512039390824` | concatenated |
| 6 | `dct4x4_params` | `{2200, 0, 0, O}, {392, 0, 0, O}, {112, -0.25, -0.25, -0.5}` | zeros, `0.5` (minus dropped) | `{2200, 0, 0, 0}, {392, 0, 0, 0}, {112, -0.25, -0.25, -0.5}` | letter `O` is zero; the sign is negative |
| 7 | DCT16x16 / DCT32x32 rows | one order | a different order | matches LaTeX/transcription | LaTeX/transcription order |
| 8 | I.2.2 default `block_ctx_map` | 39 entries, three rows of 13 | truncated to the first line | 39 entries, three rows of 13 | 39 entries |
| 9 | DCT128x256 `params[1..2][0]` | `24209.44206460261196`, `12979.84647584004484` | same | same | **anomaly, see below** |

Item 5 is confirmed twice over: the DCT256x256 bases are exactly twice the
DCT128x128 bases in all three channels, and `2 * 8986.04756019695412` is
`17972.09512039390824` — the concatenated reading, to the last digit.

Item 9 in detail. Applying R3 to the oblong family:

```
index 12 (32x64)    15358.89804933239925   5597.360516150652990   2919.961618960011210
index 14 (64x128)   30717.796098664792    11194.72103230130598    5839.92323792002242
  2 * index 12      30717.79609866479850  11194.72103230130598    5839.92323792002242
index 16 (128x256)  61435.5921973295970   24209.44206460261196   12979.84647584004484
  2 * index 14      61435.592197329584    22389.44206460261196   11679.84647584004484
```

The X channel doubles. The Y and B channels do not — but their *fractional*
parts are exactly the doubled fractional parts, `.44206460261196` and
`.84647584004484`. Only the integer parts differ: `24209` where doubling gives
`22389`, and `12979` where doubling gives `11679`.

The same conclusion follows from the square-to-oblong ratio, which is constant
per channel across every size:

| channel | 11/12 | 13/14 | 15/16 as printed | 15/16 if doubled |
| --- | --- | --- | --- | --- |
| X | 1.56039 | 1.56039 | 1.56039 | 1.56039 |
| Y | 1.49717 | 1.49717 | 1.38462 | 1.49717 |
| B | 1.53873 | 1.53873 | 1.38462 | 1.53873 |

The scan of page 64 prints `24209.44206460261196` and `12979.84647584004484`
plainly, in a cleanly rendered row. This is not an OCR artefact.

## 5. Conclusion

Items 1–8 are transcription damage and are settled: seven by agreement between
the independent pair once the OCR classes are allowed for, and all of them
re-confirmed against the scan. Two general lessons hold up: `part1.md` beats the
LaTeX for single-glyph corruption inside formulas (items 1, 2, 3, 6), and the
LaTeX beats `part1.md` for anything whose *order* matters, because `part1.md`
interleaves two-column continuation lines (item 7) and truncates multi-line code
(item 8).

Item 9 is not transcription damage. Either the published table contains two
integer-part typos in one row, or the JPEG XL designers chose two values for
DCT128x256 that break the regularity every other row of the table follows. The
evidence for a typo is strong — a coincidence that preserves fourteen decimal
places of the doubled value while changing the integer part has no plausible
mechanism — but "strong" is not "normative", and nothing here is evidence about
what any decoder actually does.

This establishes nothing about libjxl: no oracle was run, and no
DCT128x256 varblock has been decoded end to end by anything in this tree.

## 6. Consequences

* `vardct/dequant_matrix.rs` ships the **printed** values, behind the flip point
  `DCT128X256_DEFAULT_BASES_AS_PRINTED = true`. The alternative reading is
  present as `LARGE_DCT_BASES_16_DOUBLED`; flipping the constant is the whole
  change.
* `large_dct_bases_double_per_size_step` asserts the doubling for indices
  11→13→15 and 12→14, and asserts that index 16 does *not* double in Y and B.
  It is a sentinel: an editor who "corrects" the printed values without
  flipping the constant fails that test.
* Slice 8F can settle item 9 with an oracle probe once pixels exist. It needs a
  fixture containing a 128x256 or 256x128 varblock — i.e. a large smooth image
  encoded at a distance high enough for `cjxl` to pick the largest transforms —
  graded on the Y and B channels. Until then the shipped reading is the printed
  one.
* Two further readings in the same clause are decided by argument rather than
  by evidence and are recorded in the code:
  * `DCT2_DC_WEIGHT_IS_ONE` — I.2.4's six DCT2 placement rules cover 63 of the
    64 positions and never mention `(0, 0)`, while the Hornuss paragraph one
    sentence later does set its `(0, 0)` to 1. The position is unobservable
    (it is the LLF coefficient, dequantized by I.5.2), so this cannot be
    probed; it is documented, not guessed at silently.
  * AFV's default `dct4x4_params`. Table I.6's AFV row names `dct4x8_params`
    for `dct_params` and prints the 3x9 `params`, but says nothing about the
    third input AFV's code requires. `dct4x4_params` is the only other named
    parameter set in I.2.5 and AFV's `weights4x4` is its only other consumer,
    so the two named sets exist precisely to feed AFV's two sub-matrices.
* The DCT2 rectangle bounds `((2,0),(4,2))` etc. are exclusive at the
  bottom-right. That is not a judgement call: the inclusive reading puts
  `i == 5`'s `((4,4),(8,8))` outside an 8x8 matrix, and the exclusive reading
  tiles the matrix exactly once with `(0,0)` left over. The tiling is asserted
  in `dct2_rectangles_tile_the_matrix_exactly`.

---

## Addendum, 2026-08-03 — I.2.4's RAW arm: wire format, derived

*Appended after the fact; nothing above is edited. This addendum concerns
I.2.4, not I.2.5, and is filed here because this is the report that covers the
dequantization-matrix clause.*

### Question

Where does I.2.4's `encoding_mode == RAW` read its matrix from, and in what
form?

### Derivation, from the clause alone

1. **Position.** The RAW arm is two sequential statements: `params.denominator =
   F16();` then `params = /* read a 3-channel image from a modular sub-bitstream
   of the same shape as the required quant matrix */;`. Nothing separates them,
   and the enclosing `if/else if` chain is inside the loop over the 17
   parameter sets. So the sub-bitstream begins at the bit immediately after the
   `F16()`, inside `HfGlobal`.
2. **Not a separate section.** G.3's section list (the one that also gives "one
   section for HfGlobal") has no per-dequantization-table entry: it is
   `LfGlobal`, one section per `LfGroup`, one for `HfGlobal`, and
   `num_groups * num_passes` `PassGroup` sections. H.4.1 does assign the RAW
   tables their own *stream index* — `1 + 3 * num_lf_groups + parameter_index` —
   but a stream index is property 1 of the MA tree, not a section number. The
   two are independent, and conflating them is what produced the original
   deferred-resolution design.
3. **Shape.** "the same shape as the required quant matrix" is Table I.4's
   `(rows, columns)` for that parameters index — which, as the main report's
   sibling test asserts, is exactly the transform's `(coeff_rows, coeff_cols)`.
   Three channels, each at full resolution.
4. **Value.** "For encoding mode RAW, the dequantization matrices are equal to
   the params matrices multiplied by params.denominator." RAW is therefore the
   one mode that is **not** reciprocated: for every other mode the
   dequantization matrix is the element-wise reciprocal of a weights matrix,
   here the decoded planes *are* the matrix, scaled.
5. **Validity.** The sentence "None of the resulting values are non-positive or
   infinity" closes the paragraph that covers both the RAW rule and the
   reciprocal rule, so it is read as applying to RAW too. A decoded plane
   containing a zero is a well-formed sub-bitstream carrying an ill-formed
   matrix, and is rejected.

### What the clause does not say

* **Channel order.** "a 3-channel image", nothing more. Everything else in
  I.2.4 indexes by the weights matrix's channel `c` in Table I.1's X, Y, B
  numbering, and I.5.3 consumes the matrix under the same numbering, so X, Y, B
  ships as `RAW_MATRIX_CHANNEL_ORDER_IS_XYB = true`. The alternative is not
  idle: G.2.2's `LfQuant`, which also says only "three channels", turned out to
  be Y, X, B (see `2026-08-03-lf-quant-channel-order-fixture-evidence.md`). The
  distinction relied on is that I.4 states Y, X, B for HF *coefficients*, and a
  dequantization table is a parameter table, not coefficients.
* **Alignment.** No `ZeroPadToByte()` appears on either side of the
  sub-bitstream. Annex B writes that primitive explicitly wherever it applies,
  and the two other in-frame sub-bitstreams with a field in front of them
  (G.2.2's `extra_precision`, G.2.4's `nb_blocks`) are unpadded and decode real
  streams. `RAW_SUBBITSTREAM_IS_UNALIGNED = true` ships, consistent with the
  E.4.1 ICC finding. RAW is the only construct in the frame where a modular
  sub-bitstream is *followed* by more fields in the same section, so it is the
  only thing that could decide this.

### Evidence

None from any real stream: no `cjxl` build emits `encoding_mode == 7`. The
proof is a hand-built bitstream —
`raw_matrices_decode_inline_and_keep_the_bundle_in_sync` — carrying three RAW
tables (two 8x8 with different denominators, one 8x16) interleaved with a
Hornuss set and closed by `num_hf_presets`. Its sub-bitstream writer is an
independent encoder built from the Annex H tables, the same shape `vardct::lf`
and `vardct::hf_meta` use for their own clauses and which real `cjxl` fixtures
cross-validate. Decoding it recovers the Hornuss parameters and the preset count
at the right bits, which is only possible if each sub-bitstream was consumed
exactly. A companion test truncates one symbol and asserts the result is not
silently accepted.

This establishes that the shipped reading is self-consistent and
sync-preserving. It does **not** establish the channel order or the alignment,
and it is not evidence about any other implementation.

### Consequences

* `read_dequant_matrices_with` / `read_hf_global_params_with` take an optional
  `RawMatrixContext { options, tree_source, geometry }`; the old three-argument
  entry points delegate with `None` and refuse RAW at the bit where the
  sub-bitstream starts, so a caller without frame context cannot desynchronise.
* `DequantMatrices::set_raw_matrix` is deleted. `raw_requests()` survives as an
  always-empty compatibility shim for the pre-fix `decode.rs` guard and should
  go with it.
