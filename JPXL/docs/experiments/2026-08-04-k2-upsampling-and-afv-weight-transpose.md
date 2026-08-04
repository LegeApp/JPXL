# K.2 upsampling, and the AFV dequant-weight transpose it exposed

Date: 2026-08-04. Wave 7 (J.2/K.2 upsampling; corpus `upsampling`,
`upsampling_5`).

Two findings, in the order they were reached. The second is the load-bearing
one: implementing K.2 correctly still left the target corpus case failing, and
the residual turned out to belong to a different clause entirely.

* **Part A** — the K.2 default weight tables and index formula are transcribed
  correctly, proved by an internal checksum rather than by another page read.
* **Part B** — I.2.4's AFV branch places its interpolated frequency weights at
  the transposed coefficient position. Scan-verified: the published text says
  so. Deviating from it drops the corpus error by three orders of magnitude.

---

## Part A — validating 280 transcribed constants without another scan read

### 1. Question

K.2 gives three default upsampling weight tables — `d_up2` (15 values),
`d_up4` (55) and `d_up8` (210) — as a dense block of signed 8-decimal
constants, plus an index formula that folds the filter's symmetry into a
triangular layout. Dense numeric blocks are where transcriptions fail (see
`2026-08-03-h52-clamp-xor-scan-resolution.md`, where one glyph cost three
sessions). **Are the 280 values and the index formula right, and can that be
established without spending an image-scan page read on each?**

### 2. Preregistered gate

K.2's filter reconstructs a sample plane at `k` times the resolution. Whatever
else it does, it must reproduce a constant plane exactly — an upsampler that
changes the value of a flat region is not an upsampler. That is equivalent to
the 25 weights of every output position summing to 1.

* **Pass** — for all three factors, and for every one of the
  `2*2 + 4*4 + 8*8 = 84` output positions, the 25 weights selected by the index
  formula sum to 1 within `1e-6`.
* **Fail** — any position off by more than that.
* **Inconclusive** — n/a; the test is exhaustive over the tables.

The gate covers the tables and the index formula **jointly**: a wrong index
formula selects a wrong multiset of weights, and a wrong constant appears in at
least one position's sum. It cannot distinguish which of the two is wrong, only
that neither is. Fixed before running.

### 3. Method

The three tables were transcribed from `latex/part1.tex` clause K.2 (the
canonical Part 1 source) into `crates/jpxl-decode/src/frame/upsampling.rs` as
`f64` — several values carry more precision than an `f32` holds, and truncating
a normative table to satisfy `clippy::excessive_precision` would have destroyed
the thing being checked. The index formula was transcribed verbatim:

```text
j = (ky < k/2) ? (iy + 5 * ky) : ((4 - iy) + 5 * (k - 1 - ky));
i = (kx < k/2) ? (ix + 5 * kx) : ((4 - ix) + 5 * (k - 1 - kx));
y = min(i, j); x = max(i, j);
index = 5 * k * y / 2 - y * (y - 1) / 2 + x - y;
```

The sums were computed first in a throwaway Python script (to decide whether to
commit the transcription at all) and then as the shipped test
`frame::upsampling::tests::default_weights_are_normalised`.

### 4. Raw results

```
k=2  max index reached 14   distinct sums: {1.00000001}
k=4  max index reached 54   distinct sums: {0.99999999, 1.0, 1.00000002}
k=8  max index reached 209  distinct sums: {0.99999999, 1.0, 1.00000001,
                                            1.00000002, 1.00000003}
```

Maximum deviation `3e-8`, three orders of magnitude inside the gate. The
highest index reached is exactly `len - 1` for each table, so every stored
weight is used and none is read out of range.

Two independent corroborations, not preregistered:

* K.2's worked example prints the 10x10 index matrix for `k = 2`. Its top-left
  5x5 quadrant is `[[0,1,2,3,4],[1,5,6,7,8],[2,6,9,10,11],[3,7,10,12,13],
  [4,8,11,13,14]]`, reproduced by the formula (test
  `k2_example_index_matrix`). The OCR garbles two rows of that matrix; the
  formula is what recovers them.
* `weight_count(k) = (5k/2)(5k/2+1)/2` gives 15, 55 and 210, matching the array
  lengths D.3 declares.

### 5. Conclusion

**Pass.** The tables and the index formula are jointly correct. No image-scan
read was needed or made for K.2.

This is a reusable technique, not a one-off: a normative filter table that must
preserve a constant carries its own checksum, and a single digit slip cannot
survive it. Prefer it to a page read where the invariant exists.

---

## Part B — I.2.4's AFV frequency weights are placed transposed

### 1. Question

With K.2 implemented, the `upsampling` corpus case decoded to **peak error
7.8e-2 and channel-B RMSE 5.3e-4** against its `reference_image.npy`. Its
`test.json` allows peak 0.004 and RMSE 1e-4, so it failed by a factor of 20 on
peak and 5 on RMSE. The alpha channel — which goes through the *same* K.2 code
with the same weights — was already exact at 3.6e-7. **Where does the colour
error come from?**

### 2. Preregistered gate

* **Localisation gate.** If the residual is a K.2 fault, the error must vary
  with position inside the `k x k` output block: an interpolation filter that
  is wrong is wrong at particular phases. If instead it is flat across phases,
  the error is already present in the pre-upsampling frame and K.2 is only
  magnifying it.
* **Attribution gate.** A candidate fix counts only if it reduces the error on
  **two independent streams**, one of which uses no upsampling at all, to at
  least an order of magnitude inside the corpus threshold.
* **Fail** — a fix that helps the corpus case but not the no-upsampling
  control, which would mean it was tuned to one stream.

Fixed before the second measurement.

### 3. Method

Streams (all decodes by `djxl v0.13.0 196a43d9`, libjxl revision
`196a43d996aa6ed33ebf98812a7c6d43b2b6d01b`, per
`tools/oracle-bin/PINNED_REVISIONS.txt`):

| id | stream | provenance |
| --- | --- | --- |
| A | `tests/fixtures/conformance/testcases/upsampling/input.jxl` | official corpus, 800x600, `upsampling = 4` |
| B | re-encode of that case's `ref.png` at `cjxl -d 1.0 -e 7` with **no** resampling flags | control, 800x600, `upsampling = 1` |

Steps:

1. Error binned by phase `(x mod 4, y mod 4)` within the 4x4 upsampling block,
   per channel, on stream A.
2. Error binned by 8x8 frame block (32x32 output tile) on stream A, and by 8x8
   block on stream B.
3. The varblock transform type of every varblock, dumped from
   `decode_vardct_frame` behind a temporary environment variable (since
   removed), and cross-referenced with the tile map.
4. The candidate fix applied, both streams re-measured.

### 4. Raw results

**Step 1 — phase.** Channel-B RMSE per phase, all sixteen phases:

```
(0,0) 5.04e-4   (1,0) 5.38e-4   (2,0) 5.26e-4   (3,0) 5.37e-4
(0,1) 5.05e-4   (1,1) 4.97e-4   (2,1) 6.72e-4   (3,1) 5.89e-4
(0,2) 4.95e-4   (1,2) 4.87e-4   (2,2) 5.93e-4   (3,2) 5.58e-4
(0,3) 4.89e-4   (1,3) 5.28e-4   (2,3) 4.87e-4   (3,3) 5.04e-4
```

Flat to within 40 %. **K.2 is exonerated by the localisation gate**: the error
is in the 200x150 frame, before upsampling.

**Step 2 — space.** The top twenty tiles hold **100.0 %** of the squared
channel-B error on both streams. Worst tiles on stream A:
`(16,6) 7.6e-3`, `(10,16) 6.5e-3`, `(3,2) 4.7e-3`, `(10,1) 2.6e-3`.

**Step 3 — transform types.** Stream A's 240 varblocks:

```
Dct8x4 55, Dct4x8 39, Dct8x8 37, Dct16x8 29, Dct16x16 20, Dct8x16 17,
Afv3 8, Afv2 7, Dct32x16 7, Dct16x32 5, Afv0 4, Dct2x2 4,
Dct32x32 3, Afv1 3, Hornuss 2
```

The four worst tiles are, in order, `Afv2`, `Afv1`, `Afv3`, `Afv2`. The
next-worst tiles are their immediate neighbours — error spread by the gaborish
convolution, the EPF and the upsampling window. The **94 DCT4x8/DCT8x4
varblocks, which share the same 4x8 inverse DCT that AFV's third quadrant
uses, are clean**, as are both Hornuss varblocks. Stream B reproduces the
signature independently: its four worst 8x8 blocks are AFV0, AFV3, AFV0, AFV0.

So the fault is in something AFV has and DCT4x8/DCT8x4 do not: either I.9.8's
AFV basis or I.2.4's AFV weights branch.

**Derivation.** I.2.4's AFV branch reads

```text
for (y = 0; y < 4; y++)
  for (x = 0; x < 4; x++) {
    if (x < 2 and y < 2) continue;
    val = Interpolate(freqs[y * 4 + x] - lo, hi - lo + 1e-6, bands);
    weights(2 * y, 2 * x) = val;
  }
```

Every other write in the same block of pseudocode indexes `weights` as
`(column, row)`: `weights(x, 2 * y + 1) = weights4x8(x, y)` fills the odd rows
across all eight columns, and `weights(2 * x + 1, 2 * y) = weights4x4(x, y)`
the odd columns of the even rows. Under that convention
`weights(2 * y, 2 * x)` puts the weight for basis function `y * 4 + x` at
column `2y`, row `2x`.

Which coefficient that weight belongs to is not in doubt. I.9.8 builds the AFV
quadrant as `coeff_afv[iy * 4 + ix] = coefficients(ix * 2, iy * 2)`, so basis
index `j = iy * 4 + ix` is the coefficient at column `2*ix`, row `2*iy`, and
`freqs` is indexed by that same `j`. `freqs` corroborates it from its own side:
its four zero entries are indices 0, 1, 4 and 5 — exactly the four positions
the loop skips. The correct placement is therefore `weights(2 * x, 2 * y)`.

`freqs` is strongly asymmetric (`freqs[3] = 5.378` against
`freqs[12] = 2.663`), so the two placements are not interchangeable.

**Scan check.** Because "all transcriptions agree" proves nothing when they
descend from one scan, the image scan was read: Part 1, source PDF page 63
(printed page 59). It reads `weights(2 * y, 2 * x) = val;`. **This is not an
OCR artifact — the defect is in the published text.**

**Step 4 — the fix, measured.**

| stream | literal `weights(2*y, 2*x)` | shipped `weights(2*x, 2*y)` |
| --- | --- | --- |
| A (`upsampling` corpus, threshold peak 0.004 / RMSE 1e-4) | peak 7.836888e-2, B RMSE 5.3399e-4 | peak 4.2938e-5, B RMSE 9.056e-7 |
| B (control, no resampling) | peak 8.7185e-3, B RMSE 3.0719e-5 | peak 3.8133e-5, B RMSE 5.876e-7 |

Improvement: 1800x on A's peak, 590x on A's channel-B RMSE; 230x and 52x on the
control. Both land two orders of magnitude inside the corpus threshold.

### 5. Conclusion

**Pass on both gates.** I.2.4's AFV branch, as published, places its
interpolated frequency weights at the transpose of the coefficient they
belong to. JPXL ships the transposed-back reading behind
`vardct::dequant_matrix::AFV_FREQ_POSITION_IS_TRANSPOSED = true`, with a
directional unit test (`afv_frequency_weights_sit_at_their_own_coefficient`)
that pins the two placements apart at matrix cells `(0, 6)` and `(6, 0)` —
channel B is the only channel whose AFV bands differ, so it is the only one
where the placement is observable at all.

**Defect tally.** This is the third scan-verified defect in Part 1 recorded by
this project, after I.8's `ScaleF` divide-by-zero
(`2026-08-03-i8-scalef-argument.md`) and Table I.6's index-16 bases
(`2026-08-03-i25-default-dequant-constants.md`). Unlike H.5.2's `^`/`*`
(`2026-08-03-h52-clamp-xor-scan-resolution.md`), which was withdrawn once the
scan showed the transcriptions wrong and the standard right, this one survives
the scan.

**Standing lesson.** This bug was reachable for two waves and invisible: AFV
varblocks are rare, every corpus case that exercised them had a threshold loose
enough to absorb them, and total-error metrics hid a fault confined to 2 % of
the varblocks. What found it was the *spatial* distribution — "the top twenty
tiles hold 100 % of the squared error" is a statement no aggregate RMSE can
make. When a residual will not budge, bin it before theorising about it.

---

## Provenance

```
cjxl/djxl:   v0.13.0 196a43d9 [_AVX2_,SSE4,SSE2] {GNU 15.2.0}
libjxl rev:  196a43d996aa6ed33ebf98812a7c6d43b2b6d01b (v0.12-snapshot-2-g196a43d9)
host:        Linux 6.17.0-40-generic x86_64
corpus:      tests/fixtures/conformance/testcases/upsampling/input.jxl
             reference_image.npy sha256
             9b83952c4bba9dc93fd5c5c49e27eab29301e848bf70dceccfec96b48d3ab975
scan read:   original/…18181, 2, 2024 jul… .pdf, page 63 (printed 59)
```

Streams B and the phase/tile binning were throwaway instrumentation in a
scratch directory and are not committed; the recipe above reproduces them. The
committed successors are `tools/make-upsampling-fixtures.sh` (fixtures 90-95)
and `crates/jpxl-decode/tests/e2e_upsampling.rs`.

libjxl was used strictly as a black box: `cjxl` and `djxl` were run, never
read. No conclusion here rests on libjxl's source.
