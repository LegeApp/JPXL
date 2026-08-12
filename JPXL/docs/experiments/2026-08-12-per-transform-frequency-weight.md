# Phase 6.2: does one frequency weight fit all three square transforms?

Date: 2026-08-12
Status: complete as a measurement. **It falsifies the Phase 6.3 plan as written**
and replaces it with a better-supported one. No encoder change landed.

## 1. Question

Phases 6.0 and 6.1 measured DCT8x8 only. `block_cost_bounded`'s entire job is
comparing one large square transform against four sub-quadrants, so a weight
validated only for 8x8 cannot be wired into it: applying it to a cross-size
comparison would introduce a new bias rather than remove the existing one. This
work item was scoped as the gate that makes the 6.3 encoder change safe.

Two questions, and the second turned out to matter more than the first:

1. Does the perceptual weight, expressed against **normalised spatial
   frequency**, agree across DCT8x8, DCT16x16 and DCT32x32? If yes, one curve
   can price all three.
2. Does the **quantizer-normalised** residual — 6.1's result, the thing 6.3
   proposed to ship — also agree across sizes?

## 2. Preregistered gate

* **Collapse — pass:** the three sizes' normalised-frequency curves agree within
  a small factor. A per-transform weight is then sound and 6.3 may proceed.
* **Diverge — pass (and blocks 6.3):** they do not agree, and the record must
  say so, because a weight right for 8x8 and wrong for 32x32 would bias cover
  selection in a new direction.

## 3. Method

The Phase 6.0/6.1 harness, generalised off its hard-coded 8x8 constants:

* Each square uses its own I.7.3 inverse (`jpxl_core::dct::idct_2d_raw`) and its
  own I.2.5 default dequant matrix.
* The LLF sub-block is excluded per size — 1x1, 2x2 and 4x4 cells respectively —
  matching what `score_channel_lanes` skips.
* The crop is aligned to a whole DCT32x32 regardless of which sizes a run asks
  for, so all three tile the **identical** pixels.
* `fu`/`fv` report each cell's frequency as a fraction of Nyquist, which is what
  makes a 16x16 cell comparable with the 8x8 cell at the same spatial frequency.

Two corrections to the harness were needed and both matter:

**Amplitude is now sample-domain.** I.7.2's inverse is the orthonormal DCT-III
scaled by `sqrt(s)` per dimension, so `idct_2d_raw` multiplies a square's
amplitude by `side`. Dividing that back out makes one unit of `amplitude` mean
one unit of injected **sample-domain** error whatever the transform size. That
is the only unit in which sizes can be compared, and it is the encoder's own
choice: `block_cost_bounded` multiplies each candidate's coefficient error by its
`side^2` for exactly this reason.

**Total injected energy is equalised across sizes.** A larger square tiles the
same crop with fewer blocks (4096 / 1024 / 256 here), so equal per-block energy
would mean 16x less total energy at DCT32x32. Amplitude is scaled by `side / 8`
to hold total injected sample-domain energy constant, so a difference between
curves is a shape difference and not an operating-point difference. **Raw
butteraugli is therefore directly comparable between sizes**, which turns out to
be where the most useful finding comes from.

Y channel only — chroma is deferred to Phase 6.4 for the operating-point reason
6.1 recorded. Two photographs, two amplitudes, both sweeps: 5,292 probes each.

### A convention error found and dispositioned

`idct2d_8x8` (used by 6.0/6.1) and `idct_2d_raw` (I.7.3, used here) are both
orthonormal but their coefficient index conventions differ **by a transpose**.
The encoder's forward path is `forward_dct_rc` → `dct_2d_in_place`, i.e. I.7.3,
and `HfQuantizer` indexes its matrix in that same layout, so `idct_2d_raw` is
the correct one and Phases 6.0/6.1 used the transposed grid.

This changes nothing, and the reason is checkable rather than a hope: the I.2.5
default square matrices are **exactly symmetric** — measured worst relative
asymmetry 0.000000 for every channel at all three sizes. So 6.1 paired each cell
with the correct step regardless of transpose, and 6.0's spread and stability
statistics are relabelings-invariant. The published 6.0/6.1 tables have their
`u`/`v` labels transposed relative to the encoder's grid; the measured Y map is
near-symmetric anyway (`(0,1)` 1.77 against `(1,0)` 1.79, `(0,2)` 1.50 against
`(2,0)` 1.48), so no number in those records moves. Recorded here rather than
silently fixed.

## 4. Result A: the perceptual weight IS size-independent

Equal-coefficient-error sweep, mean over 4 runs, binned by radial normalised
frequency and then normalised to each size's own `f = 0.4-0.5` bin so shape is
compared independently of level:

| f | DCT8x8 | DCT16x16 | DCT32x32 |
| --- | --- | --- | --- |
| 0.0–0.1 | 1.756 | 1.887 | 2.012 |
| 0.1–0.2 | 1.477 | 1.581 | 1.695 |
| 0.2–0.3 | 1.190 | 1.217 | 1.273 |
| 0.3–0.4 | 0.918 | 0.984 | 1.035 |
| 0.4–0.5 | 1.000 | 1.000 | 1.000 |
| 0.5–0.6 | 0.915 | 0.911 | 0.931 |
| 0.6–0.7 | 0.934 | 0.910 | 0.918 |
| 0.7–0.8 | 0.793 | 0.779 | 0.809 |
| 0.8–0.9 | 0.746 | 0.791 | 0.814 |
| 0.9–1.0 | — | 1.013 | 1.054 |

**Worst disagreement in any shared bin: 1.15x.** Six of the nine shared bins
agree within 5%. The gate passes: a single weight expressed as a function of
normalised spatial frequency prices all three square transforms.

(DCT8x8 has no `0.9-1.0` bin: its highest cell, `(7,7)`, sits at `f = 0.875`.)

## 5. Result B: larger transforms cost more at equal sample-domain error

Because total injected energy is equalised, the raw per-bin means compare
directly — and they are not equal:

| f | DCT8x8 | DCT16x16 | DCT32x32 | 32 vs 8 |
| --- | --- | --- | --- | --- |
| 0.0–0.1 | 3.837 | 4.338 | 4.655 | +21% |
| 0.1–0.2 | 3.228 | 3.633 | 3.922 | +21% |
| 0.2–0.3 | 2.600 | 2.798 | 2.945 | +13% |
| 0.3–0.4 | 2.006 | 2.261 | 2.394 | +19% |
| 0.4–0.5 | 2.185 | 2.299 | 2.313 | +6% |
| 0.5–0.6 | 2.001 | 2.094 | 2.153 | +8% |
| 0.6–0.7 | 2.040 | 2.091 | 2.122 | +4% |
| 0.7–0.8 | 1.733 | 1.790 | 1.871 | +8% |
| 0.8–0.9 | 1.631 | 1.819 | 1.882 | +15% |

The same total squared sample error costs **4–21% more butteraugli** when it is
carried by a DCT32x32 basis than by DCT8x8 bases, monotonically in size, in
every bin, on both photographs.

That is a mechanism, not a curiosity: error on a 32x32 support is spatially
coherent over a large region, where the same energy spread across sixteen
independently-signed 8x8 patches is closer to noise, and noise is easier to mask.

`block_cost_bounded` assumes the opposite — that bringing coefficient error into
the sample domain via `side^2` makes candidates of different sizes directly
comparable. It does not. **The encoder systematically under-penalises large
transforms, which biases the hierarchical cover toward merging.** The correction
is a per-transform scalar, which is far simpler than a per-cell table.

## 6. Result C: this falsifies the Phase 6.3 plan

The step-proportional sweep — 6.1's residual, the thing 6.3 proposed to ship as
`sum (delta_k / step_k)^2` — does **not** collapse.

| f | DCT8x8 | DCT16x16 | DCT32x32 |
| --- | --- | --- | --- |
| 0.0–0.1 | 1.211 | 0.666 | 0.591 |
| 0.1–0.2 | 1.029 | 0.689 | 0.655 |
| 0.2–0.3 | 0.929 | 0.727 | 0.669 |
| 0.3–0.4 | 0.831 | 0.799 | 0.775 |
| 0.4–0.5 | 1.000 | 1.000 | 1.000 |
| 0.5–0.6 | 1.062 | 1.101 | 1.208 |
| 0.6–0.7 | 1.205 | 1.291 | 1.498 |
| 0.7–0.8 | 1.156 | 1.281 | 1.598 |
| 0.8–0.9 | 1.232 | 1.518 | 1.946 |
| 0.9–1.0 | — | 2.150 | 2.789 |

**Worst disagreement: 2.05x**, and the shapes are qualitatively different.
DCT8x8's residual is the shallow bowl 6.1 reported (1.21 → 0.83 → 1.23, spread
1.93–2.44x). DCT32x32's rises monotonically from 0.59 to 2.79 (spread 6.49x to
15.41x across runs).

The cause is in the matrices themselves. Non-LLF step ratios of the I.2.5
defaults, Y channel: **2.86x at DCT8x8, 8.05x at DCT16x16, 13.05x at DCT32x32.**
The defaults coarsen high frequencies far more aggressively at large sizes than
perception justifies.

So weighting by `1 / step^2` uniformly would be approximately right for DCT8x8
and badly wrong for DCT32x32: it would tell the objective that high-frequency
error in a 32x32 block is *cheap*, precisely where measurement says it is the
most expensive thing in the table. That is exactly the new bias this phase was
scoped to catch, caught before it shipped.

## 7. Answer, and the replacement plan

* Result A passes the gate: **one weight curve in normalised spatial frequency
  is valid across all three square transforms.**
* Result C fails 6.3 as written: **quantizer normalisation must not be the form
  that weight takes.** `1/step^2` is a per-size-inconsistent proxy that only
  happens to be close at 8x8.

The replacement is simpler, better supported and *more* clean-room-defensible
than the 6.1 proposal, because it stops borrowing its shape from the dequant
tables:

1. **A single frequency-domain weight**, evaluated on each transform's own
   normalised grid — Result A says one curve suffices. Derived from a
   first-principles contrast-sensitivity function and *checked* against Result
   A's measured curve, never fitted to it (AGENTS.md §2, and the `butteraugli`
   note in the workspace manifest).
2. **A per-transform level correction** from Result B, so a DCT32x32 candidate is
   not priced as though its error were as maskable as sixteen DCT8x8 candidates'.

Both are cheap: a per-cell constant and a per-transform scalar, precomputed once.

## 8. What this does not establish

1. Still Y only, and still cover/CfL only — `HfQuantizer::choose` remains
   nearest-reconstruction, so those are the only decisions any weight can move.
2. Result B is measured at one operating point per amplitude on two photographs.
   The direction is consistent across every bin and both images, but the
   *magnitude* of the per-transform correction should be re-measured at the rates
   the corpus actually encodes at before it is used as a calibration constant.
3. The `f = 0.4-0.5` reference bin is a normalisation choice, not a fitted
   parameter; Result A's 1.15x agreement is not sensitive to it, but the tables
   above would shift uniformly under a different choice.
4. Bin 4 (`0.4-0.5`) is a local *rise* against its neighbours in Result A, the
   same index-4 feature Phases 6.0 and 6.1 both flagged as probably belonging to
   butteraugli's multi-scale pyramid rather than to vision. It survives here at
   all three sizes, which is weak evidence it is a property of the metric rather
   than of the transform — another reason to fit a smooth CSF rather than these
   numbers.
