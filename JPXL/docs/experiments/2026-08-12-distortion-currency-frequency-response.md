# Phase 6.0: the encoder's distortion currency, priced against perception

Date: 2026-08-12
Status: complete as a measurement. It establishes a magnitude, not a policy —
no encoder behaviour changed, and nothing here licenses a weight table (see §7).

## 1. Question

`jpxl-encode-policy`'s candidate objective (`block_cost_bounded`,
`crates/jpxl-encode-policy/src/lib.rs`) is

```
J = bits + lambda_c * side^2 * sum_cells (recon - target)^2
```

with `lambda_c` a single scalar per channel, calibrated once at the DCT8x8
operating point (`HfQuantizers::new_with_scales`). The distortion term is a
**flat** sum of squared dequantized-coefficient errors: cell `(0,1)` and cell
`(7,7)` are charged identically for the same error magnitude. That is a
fidelity measure.

Three gaps against `cjxl` 0.13 have been measured on the same corpus
(`jpegxl-rs.observation.lossy-vardct-headsup-2026-08-10/3`,
`jpegxl-rs.observation.lossy-perceptual-baseline-2026-08-10`):

| Metric | Gap |
| --- | --- |
| PSNR | 25–37% more bits for equal quality |
| SSIMULACRA2 | 12–20% |
| butteraugli | 39–53% |

The encoder is closest on the metric its objective *is*, and furthest on the
metric it is judged by. Phases 5J, 5K, 5L, 5M, 5N and 5O each changed *which
decision* the objective makes — the AQ field's lattice, EPF depth, EPF
spatially, chroma QM scales, the transform vocabulary — while leaving the
currency those decisions are priced in untouched. All six produced scattered,
content-dependent, mostly-regressing perceptual results.

So, before choosing another lever: **how much does a unit of the encoder's
distortion currency actually vary in perceptual cost across the DCT8x8
frequency plane?** If the answer is "not much", the flat objective is fine and
the deficit is elsewhere. If the answer is large, the objective is mispricing
its own decisions and no choice of lever can compensate.

## 2. Preregistered gate

* **Flat — pass:** the normalized perceptual response across the 63 non-LLF
  DCT8x8 cells spans less than ~1.3x. The flat objective is then approximately
  correct and Phase 6 should be abandoned in favour of the queued Phase 5P.
* **Structured — pass:** the response spans materially more than that *and* the
  shape is stable across amplitude and across content. Only both together
  count: an unstable shape is a metric artefact, not a weighting.
* **Inconclusive:** a large spread with an unstable shape, or a shape that
  tracks injected sample energy rather than frequency (see the control in §3).

No encoder change and no corpus encode was preregistered. This experiment does
not run the encoder at all.

## 3. Method

`crates/jpxl-conformance/tests/perceptual_frequency.rs`, ignored by default and
skipped without a reference image:

```
JPXL_CALIB_REF=<ref.ppm> JPXL_CALIB_SIDE=512 JPXL_CALIB_AMPS=0.25,0.5,1.0 \
  cargo test --release -p jpxl-conformance --features perceptual \
  --test perceptual_frequency -- --ignored --nocapture
```

For one centre-cropped image, per XYB channel and per DCT8x8 cell `k`:

1. Convert the crop to the encoder's own XYB planes
   (`jpxl_core::color::linear_srgb_to_xyb_planes`, via `srgb_to_linear`).
2. Build the basis function for cell `k` with the encoder's own inverse
   transform (`jpxl_core::dct::idct2d_8x8`) at amplitude `a * stddev(channel)`,
   and add it to **every** whole 8x8 block, with a deterministic per-block sign.
   The sign matters: quantization error is not coherent across blocks, and
   injecting the same signed basis everywhere would build a global texture and
   measure that instead.
3. Convert back to gamma-encoded 8-bit sRGB, as a decoder would deliver it.
4. Score against the *unperturbed round trip* — not the original file — so
   colour-conversion and 8-bit rounding loss cancel and every number is the cost
   of the injected coefficient error alone. The self-distance is asserted to be
   zero before the sweep starts.

Every cell receives the same injected coefficient energy, so the current
objective charges every cell in the table exactly the same amount. Cell 0 is
excluded from all statistics: it is the LLF, and `score_channel_lanes` skips it
(`col_start = if row < n { n } else { 0 }`) because LF is a separate path.

**Control.** The harness also reports the realized sample-domain SSE per cell. If
equal coefficient error produced unequal *sample* error, any perceptual spread
would be basis energy rather than perception. Measured spread of sample SSE
across the 64 cells: **8.2%** (from 8-bit clipping), against a perceptual spread
of 165%. The control passes.

Corpus: `small_0p8MP.ppm` and `mid_4MP.ppm` from
`.agent/scratch/realworld-bench-20260806T072202Z`, centre-cropped to 512x512
(4096 blocks each). Three amplitudes, three channels, two images: 18 runs of 64
cells. Raw output is in
`.agent/scratch/phase6-0-frequency-response-2026-08-12/`.

## 4. Result: Y

Normalized butteraugli response, mean ± stdev over the 6 Y runs (2 images x 3
amplitudes). A flat objective assumes **1.00 in every cell**. Rows are vertical
frequency `u`, columns horizontal frequency `v`.

|  | v=0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **u=0** | LLF | 1.77±0.11 | 1.50±0.10 | 1.20±0.02 | 0.82±0.03 | 1.03±0.03 | 0.90±0.05 | 0.88±0.04 |
| **u=1** | 1.79±0.10 | 1.80±0.11 | 1.39±0.09 | 1.18±0.05 | 0.90±0.02 | 1.09±0.03 | 1.04±0.02 | 1.11±0.04 |
| **u=2** | 1.48±0.09 | 1.43±0.09 | 1.22±0.11 | 1.02±0.01 | 0.88±0.04 | 1.00±0.01 | 1.03±0.02 | 1.12±0.06 |
| **u=3** | 1.19±0.02 | 1.17±0.05 | 1.02±0.01 | 0.95±0.03 | 0.84±0.02 | 0.93±0.05 | 0.94±0.01 | 1.05±0.04 |
| **u=4** | 0.81±0.03 | 0.89±0.02 | 0.89±0.05 | 0.85±0.03 | 0.75±0.03 | 0.68±0.03 | 0.71±0.03 | 0.78±0.03 |
| **u=5** | 1.02±0.02 | 1.08±0.03 | 1.00±0.02 | 0.93±0.05 | 0.68±0.03 | 0.76±0.02 | 0.71±0.02 | 0.78±0.02 |
| **u=6** | 0.90±0.05 | 1.04±0.02 | 1.02±0.03 | 0.94±0.02 | 0.70±0.02 | 0.71±0.01 | 0.75±0.04 | 0.69±0.01 |
| **u=7** | 0.88±0.04 | 1.10±0.03 | 1.12±0.06 | 1.05±0.04 | 0.78±0.03 | 0.78±0.02 | 0.69±0.01 | 0.83±0.03 |

**Range 0.68 to 1.80 — a 2.65x spread for costs the objective prices as
identical.**

Stability (Pearson `r` between normalized maps) is the part that makes this
actionable. Every one of the 15 pairwise Y comparisons — across two different
photographs, at 0.8 MP and 4 MP, over a 4x amplitude range — gives
**r ≥ 0.961**, with `r = 0.997` between the two images at amplitude 1.0. The
shape is not a property of the content or of the probe strength.

## 5. Result: X and B

| Channel | Range | Ratio | Cross-run `r` |
| --- | --- | --- | --- |
| Y | 0.68–1.80 | 2.65x | 0.961–0.997 |
| X | 0.76–1.79 | 2.35x | 0.488–0.871 |
| B | 0.88–1.37 | 1.56x | −0.057–0.951 |

X has a Y-like shape at low amplitude but saturates: its own standard deviation
is ~0.0025, so at `a = 1.0` the injected error is far outside any real operating
point and butteraugli's max-norm aggregation flattens the map. The low-amplitude
X runs (`r = 0.871` across images at `a = 0.25`) are the ones to believe.

**B is close to flat and unstable.** For the B channel the current objective is
approximately right, and this experiment gives no basis for weighting it. That
is a useful negative: it means any future weighting should be per-channel, and
should leave B alone.

## 6. Answer

The preregistered **structured** gate passes on Y, marginally on X, and fails on
B.

The encoder's distortion currency misprices Y-channel coefficient error by up to
**2.65x**, along a stable, content-independent, amplitude-independent frequency
axis. Every decision priced through `block_cost_bounded` — the hierarchical
transform cover, and CfL factor selection — is made with that mispricing in
force. This is a plausible mechanism for the PSNR/butteraugli gap split in §1,
and it is a mechanism that no choice of *lever* can compensate for, because it
is in the *ruler*.

The measured shape is a band-pass contrast-sensitivity curve: peak at the lowest
non-DC frequencies (`(1,1) = 1.80`, `(1,0) = 1.79`, `(0,1) = 1.77`), falling to
~0.70 in the mid-high band, with a mild re-rise toward the corner
(`(7,7) = 0.83`). This is the classic shape, which is a point in favour of the
measurement being real rather than an artefact of the metric.

## 7. What this does *not* establish

Recorded explicitly, because the temptation to skip straight to a weight table
is the whole risk here.

1. **It does not show that a re-weighted objective improves output.** Only that
   the current one is mispriced. `HfQuantizer::choose` picks the nearest
   reconstruction level, not an RD-optimal one, so a weight changes *cover and
   CfL selection* and nothing else until quantization itself becomes RD-aware.
   Whether that subset of decisions is worth 2.65x is unmeasured.
2. **DCT8x8 only.** The cover decision compares transforms of different sizes
   against each other; each size has its own frequency grid. A weight for
   DCT8x8 alone cannot be applied to that comparison without a per-transform
   model — and getting *that* wrong would bias cover selection in a new
   direction rather than removing the existing bias.
3. **The kink at index 4 is suspicious.** Row and column 4 are a consistent
   local minimum with a partial re-rise at 5–7. A first-principles CSF has no
   such feature; butteraugli's multi-scale pyramid plausibly does. This is a
   direct argument against fitting the measured numbers literally.
4. **Clean-room (AGENTS.md §2).** Measuring a black-box metric's response is the
   same category as running `cjxl`/`djxl` as an oracle, and is permitted; no
   libjxl source was read. But the workspace manifest's own note on the
   `butteraugli` dependency is exactly on point: *"wiring it into the encoder's
   own rate control would be a different decision — it would make our perceptual
   model a derivative of libjxl's rather than our own."* Fitting an encoder
   weight table to the table in §4 is that decision. Deriving a smooth
   first-principles contrast-sensitivity weight and then *checking* it against
   §4 is not, and is the recommended route.

## 8. Suggested continuation

In dependency order, each independently gated:

* **6.1 — measure the residual, not the raw response.** Real quantization does
  not inject equal coefficient error per cell; it injects error proportional to
  that cell's quantization step, and the standard's default dequant matrices
  (F.3/I.5) are themselves perceptually shaped. Re-run this harness injecting
  error proportional to `HfQuantizer::step(channel, cell)`. If the response
  flattens, the standard's matrices already carry the weighting and the
  objective's mistake is *only* that it measures in dequantized rather than
  quantizer-normalized units — a one-line fix with a large effect. If it does
  not flatten, the residual is the real target. **This is the cheapest and most
  decision-relevant next measurement, and it should run before any weight is
  written.**
* **6.2 — per-transform frequency grids.** Extend to DCT16x16 and DCT32x32 so
  the cover comparison can be re-priced coherently (§7.2).
* **6.3 — a first-principles CSF weight**, checked against §4 rather than fitted
  to it (§7.4), behind a feature flag, screened on the six-scene matrix under
  the existing Contract B promotion rules.
