# Phase 6.1: how much of the mispricing do the standard's own quant matrices already fix?

Date: 2026-08-12
Status: complete as a measurement. It motivates one specific, Y-only encoder
change, which is gated separately and did not land here.

## 1. Question

Phase 6.0 (`2026-08-12-distortion-currency-frequency-response.md`) injected
**equal coefficient error** into every DCT8x8 cell and found the Y-channel
perceptual cost varies by 2.65x for costs the objective prices identically.

But real quantization never injects equal coefficient error. It injects error
proportional to each cell's quantization step, and `HfQuantizer` builds that
step as `scale[channel] * matrix.at(x, y)` from the I.2.5 **default dequant
matrices**, which are themselves perceptually shaped. So 6.0's 2.65x is an upper
bound on the decision-relevant mispricing, not the mispricing itself.

The question: **under the error distribution the encoder actually produces, how
much frequency mispricing is left?**

This is not academic. The current objective is

```
D_flat = sum_k (delta_k)^2                     (delta in dequantized units)
```

and the obvious first-principles alternative, using nothing but data the encoder
already has, is to measure in **quantizer-normalised** units:

```
D_qn = sum_k (delta_k / step_k)^2
```

`D_qn` equals the true perceptual cost exactly when `w0[k] * step_k^2` is
constant — which is precisely what this experiment measures. So the residual
spread reported below *is* the mispricing that would remain after switching the
objective to `D_qn`, and 6.0's 2.65x is the mispricing that remains today.

## 2. Preregistered gate

* **Flattens — pass:** the step-proportional response is materially flatter than
  6.0's. The standard's matrices carry the weighting; `D_qn` becomes a small,
  libjxl-independent change worth screening.
* **Does not flatten — pass:** the residual is the real target, and 6.2/6.3
  proceed against it rather than against 6.0's raw table.
* Either way the answer must be per-channel: 6.0 already showed B is close to
  flat and unstable, so a single global verdict would be wrong.

No encoder change was preregistered.

## 3. Method

Identical to Phase 6.0 — same crops, same corpus, same amplitude ladder, same
per-block signs, same zero-self-distance assertion, same sample-domain SSE
control — with one change: the injected amplitude at cell `k` is scaled by that
cell's I.2.5 default DCT8x8 dequant step for that channel
(`jpxl_core::dequant::DequantMatrices::all_default().for_transform(Dct8x8, c)`).

The shape is normalised by its **root-mean-square** over the 63 non-LLF cells,
not its mean, so total injected energy matches 6.0's constant-amplitude sweep
exactly. The two experiments therefore sit at the same operating point and their
butteraugli numbers are directly comparable. `scale[channel]` is constant across
cells, so it cancels under this normalisation and the matrix *is* the step shape.

`crates/jpxl-conformance/tests/perceptual_frequency.rs`, second test:

```
JPXL_CALIB_REF=<ref.ppm> JPXL_CALIB_SIDE=512 JPXL_CALIB_AMPS=0.25,0.5,1.0 \
  cargo test --release -p jpxl-conformance --features perceptual \
  --test perceptual_frequency -- --ignored --nocapture \
  dct8x8_quantizer_normalised
```

Measured step shapes, non-LLF range after RMS normalisation:

| Channel | Step shape range | Ratio |
| --- | --- | --- |
| X | 0.4040–3.3259 | 8.23x |
| Y | 0.5756–1.6440 | 2.86x |
| B | 0.1741–3.5983 | 20.67x |

Y's step rises with frequency by 2.86x while 6.0's perceptual weight *falls* by
2.65x — the matrices are compensating in the right direction. Whether they
compensate by the right amount is the measurement.

## 4. Result

Normalised butteraugli spread across the 63 non-LLF cells, mean map over 6 runs
(2 photographs x 3 amplitudes), with the weakest pairwise stability of those runs:

| Channel | 6.0 range | 6.0 ratio | 6.0 min r | 6.1 range | 6.1 ratio | 6.1 min r |
| --- | --- | --- | --- | --- | --- | --- |
| **Y** | 0.68–1.80 | **2.65x** | 0.961 | 0.73–1.36 | **1.86x** | 0.843 |
| X | 0.76–1.79 | 2.35x | 0.488 | 0.48–3.23 | 6.76x | 0.677 |
| B | 0.88–1.37 | 1.56x | −0.057 | 0.40–4.62 | 11.60x | 0.883 |

Per-run, without averaging, every single Y run is flatter under step-proportional
injection than its 6.0 counterpart:

| Run | 6.0 ratio | 6.1 ratio |
| --- | --- | --- |
| small @0.25 | 2.49x | 2.43x |
| small @0.5 | 2.72x | 2.07x |
| small @1.0 | 2.92x | 1.83x |
| mid @0.25 | 2.63x | 2.21x |
| mid @0.5 | 2.72x | 1.90x |
| mid @1.0 | 3.00x | 1.80x |

**Y: the gate passes. The default matrices remove about 36% of the log-domain
mispricing (log 2.65 → log 1.86), leaving 1.86x.**

Residual Y weight, mean over the 6 runs. 1.00 means the default step for that
cell is exactly what butteraugli's own weighting justifies; above 1.00 means the
step is **coarser** than justified, below means finer. Mean stdev across runs
0.052.

|  | v=0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **u=0** | LLF | 1.13 | 1.03 | 0.90 | 0.75 | 0.93 | 0.98 | 1.04 |
| **u=1** | 1.21 | 1.12 | 0.92 | 0.90 | 0.73 | 1.06 | 1.13 | 1.35 |
| **u=2** | 1.02 | 0.97 | 0.88 | 0.85 | 0.79 | 1.01 | 1.17 | 1.36 |
| **u=3** | 0.90 | 0.90 | 0.85 | 0.84 | 0.76 | 0.96 | 1.10 | 1.33 |
| **u=4** | 0.74 | 0.73 | 0.79 | 0.76 | 0.80 | 0.77 | 0.85 | 1.05 |
| **u=5** | 0.93 | 1.06 | 1.02 | 0.97 | 0.77 | 0.90 | 0.96 | 1.11 |
| **u=6** | 0.98 | 1.13 | 1.16 | 1.09 | 0.85 | 0.96 | 1.06 | 1.07 |
| **u=7** | 1.03 | 1.34 | 1.36 | 1.33 | 1.04 | 1.11 | 1.07 | 1.34 |

The residual has a **different shape** from 6.0's. 6.0 fell monotonically with
frequency; this is a shallow bowl — the default matrices slightly over-coarsen
the extreme corner (1.33–1.36 along `v=7` and `u=7`) and slightly under-coarsen
the band around index 4 (0.73–0.80). The index-4 feature is the same one 6.0
flagged as suspicious and probably belongs to butteraugli's multi-scale pyramid
rather than to vision, so it is not a target.

## 5. X and B: the gate fails, but the numbers are not trustworthy

X gets worse (2.35x → 6.76x) and B much worse (1.56x → 11.60x). The direction is
almost certainly real — 6.0 already showed B's perceptual response is nearly flat
per unit coefficient error, so shaping the injected error by a 20.67x step ratio
*must* produce an uneven perceptual cost — but the magnitudes are contaminated.

At a 20.67x step shape and amplitude 1.0x the channel's standard deviation, the
highest-frequency B injections are enormous and far outside any operating point
the encoder reaches; butteraugli's max-norm aggregation saturates there. The
method's amplitude ladder was designed for a constant shape and does not control
realized distortion when the shape itself spans an order of magnitude.

**The safe reading: do not apply quantizer normalisation to chroma.** Not
because it is proven harmful, but because this experiment cannot measure chroma
at a valid operating point. A chroma verdict needs a re-run whose amplitude is
chosen per cell to hold the realized butteraugli in a fixed band.

## 6. Answer, and what it licenses

Measuring the objective's distortion in quantizer-normalised units rather than
dequantized coefficient units — `sum (delta_k / step_k)^2` instead of
`sum delta_k^2` — would cut the Y-channel frequency mispricing from 2.65x to
1.86x, using nothing but the I.2.5 default matrices the encoder already builds.
It is a per-cell precomputed constant, so it costs one multiply in the scoring
loop and nothing in the quantizer.

Three things make this the cleanest lever measured so far.

1. **It is first-principles, not fitted.** The weight comes from the standard's
   own dequant matrices. Butteraugli is used to *check* the direction and size of
   the improvement, not to supply the numbers — which is exactly the distinction
   the workspace manifest's clean-room note draws, and the opposite of fitting a
   table to 6.0's measured response.
2. **It is unit correctness, not tuning.** `HfQuantizer` documents `step` as
   existing so callers can "price distortion in the same units `choose`
   quantizes in". The scorer currently does not do that. There is an argument for
   the change that does not depend on any perceptual measurement at all; 6.1 just
   says how much it is worth.
3. **It has a sharp scope.** Y only. X and B must stay flat until §5's
   re-measurement.

What it does **not** license, unchanged from 6.0 §7: this still only moves cover
and CfL selection, because `HfQuantizer::choose` is nearest-reconstruction rather
than RD; it is still DCT8x8 only, so the cross-size cover comparison needs 6.2
before the weight can be applied coherently there; and a 1.86x residual remains,
so this is an improvement to the ruler, not a correct ruler.

## 7. Suggested continuation

* **6.2 — per-transform step shapes.** Extend the harness to DCT16x16 and
  DCT32x32. The cover decision's whole job is comparing transform *sizes*, so a
  Y weight defined only for 8x8 cannot be wired into `block_cost_bounded`
  without biasing that comparison in a new direction. **6.2 blocks the encoder
  change, and should run before it.**
* **6.3 — land quantizer-normalised Y distortion** behind a feature flag once
  6.2 supplies the other sizes, screened on the six-scene matrix under the
  existing Contract B promotion rules. Honest-negative acceptable: a better-priced
  ruler can still fail to move Butteraugli if cover and CfL are not where the
  deficit lives, and that would itself be worth knowing.
* **6.4 — chroma at a controlled operating point** (§5), amplitude chosen per
  cell to hold realized butteraugli in a fixed band, before any chroma weighting
  is considered.
