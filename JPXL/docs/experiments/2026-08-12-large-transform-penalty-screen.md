# Phase 6.2b: the per-transform distortion scale, screened alone

Date: 2026-08-12
Status: complete. **Honest negative.** Production stays neutral; the finding
redirects Phase 6.3.

## 1. Question

Phase 6.2 Result B measured that `block_cost_bounded`'s `side^2` sample-domain
normalisation does not make candidates of different sizes comparable: at equal
total injected sample-domain error a DCT32x32 basis costs 4–21% more butteraugli
than DCT8x8 bases, monotonically in size, in every one of nine radial-frequency
bins, on both corpus photographs. The objective therefore under-penalises large
transforms and the hierarchical cover is biased toward merging.

Phase 6.3 pairs two independent corrections — a first-principles CSF frequency
weight (Result A) and this per-transform level correction (Result B). Result B is
far cheaper to test: one scalar per size, no CSF derivation, no per-cell table.
It also answers the prior question 6.3 depends on: **does re-pricing the cover
objective move Butteraugli at all?**

## 2. Preregistered gate

* **Positive:** uniformly non-regressing Butteraugli without material
  SSIMULACRA2 loss. Carry into 6.3 as a promoted correction.
* **Honest negative:** it does not move Butteraugli, or moves it the wrong way.
  Recorded as such — and this is the *more* valuable outcome for 6.3's scoping,
  because it localises the deficit away from the cover decision's size axis.
* Production stays `Neutral` either way unless the direction is uniformly safe.

## 3. Method

`CoverSizePenalty` on `EncodeRequest`, default `Neutral`, plus
`--cover-size-penalty neutral|measured`. `block_cost_bounded` already computes
`to_sample_domain = side^2`; the policy multiplies that by a per-transform
constant. `Neutral` returns exactly `1.0`, so IEEE multiplication by one leaves
the shipped objective **bit-identical**, not merely close — asserted by test, and
confirmed independently below.

### Where the constants come from

Butteraugli is not linear in injected energy, so Result B's butteraugli excess
cannot be used as a distortion multiplier directly. Over the measured 4x energy
step it follows `ba ~ E^p` with `p = 0.4477`, `0.4477`, `0.4433` for the three
sizes (mean **0.4462**) — a strikingly consistent exponent, which is what makes
the conversion trustworthy. A butteraugli ratio `r` therefore corresponds to
`r^(1/p)` of distortion:

| Transform | mean BA excess | multiplier `r^(1/p)` |
| --- | --- | --- |
| DCT8x8 | — | 1.0000 |
| DCT16x16 | +3.8% | **1.0881** |
| DCT32x32 | +5.7% | **1.1331** |

Stable across both probe amplitudes: DCT16x16 identical to four decimals,
DCT32x32 within 1.4%.

### Screen

Seven `test-set` photographs (1024x768) at 1 and 2 bpp, 14 cells. Exactly
matched rate by construction: neutral is encoded at the requested bpp, its
achieved byte count becomes the cap, and measured is encoded to that exact cap.
Both decoded with the pinned `djxl` and scored with the in-repo SSIMULACRA2 and
butteraugli. Test-set inputs still lack provenance sidecars, so this table is
diagnostic rather than a promoted corpus baseline — the same status Phase 5O
recorded.

Driver and raw rows: `.agent/scratch/phase6-2b-size-penalty-2026-08-12/`.

**Baseline check.** The neutral encode of `s1` at 1 bpp reproduces 97,539 bytes,
SSIMULACRA2 62.5738 and butteraugli 5.0325 — identical to the Phase 5O
production row for the same scene and rate. The Phase 6 plumbing changed nothing.

## 4. Result

| cell | bytes N | bytes M | Δbytes | BA neutral | BA measured | ΔBA | SSIM2 N | SSIM2 M | ΔSSIM2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s1/1 | 97539 | 96681 | −0.88% | 5.0325 | 5.7491 | **+14.2%** | 62.5738 | 61.4897 | −1.0841 |
| s1/2 | 194682 | 193606 | −0.55% | 2.9610 | 2.9904 | +1.0% | 82.5346 | 82.6815 | +0.1469 |
| s2/1 | 98068 | 97647 | −0.43% | 4.4219 | 4.3460 | −1.7% | 69.1769 | 68.8973 | −0.2796 |
| s2/2 | 195817 | 194856 | −0.49% | 1.6885 | 1.6572 | −1.9% | 87.2284 | 87.0296 | −0.1988 |
| s3/1 | 98028 | 97735 | −0.30% | 3.5912 | 3.5906 | −0.0% | 74.6264 | 74.1998 | −0.4266 |
| s3/2 | 195855 | 194699 | −0.59% | 1.4990 | 1.4111 | **−5.9%** | 88.9271 | 88.9770 | +0.0499 |
| s4/1 | 97529 | 96762 | −0.79% | 3.2918 | 3.4069 | +3.5% | 76.8361 | 76.1569 | −0.6792 |
| s4/2 | 180045 | 179060 | −0.55% | 1.2492 | 1.2947 | +3.6% | 89.4898 | 89.1251 | −0.3647 |
| s5/1 | 97952 | 97651 | −0.31% | 2.2501 | 2.2476 | −0.1% | 80.7087 | 80.2562 | −0.4525 |
| s5/2 | 174550 | 173046 | −0.86% | 1.1264 | 1.2168 | +8.0% | 90.5175 | 90.3651 | −0.1524 |
| s6/1 | 98284 | 98278 | −0.01% | 4.3258 | 4.4528 | +2.9% | 58.8070 | 58.4293 | −0.3777 |
| s6/2 | 196136 | 195372 | −0.39% | 2.4621 | 2.5217 | +2.4% | 81.1712 | 81.0327 | −0.1385 |
| s7/1 | 97904 | 97182 | −0.74% | 6.8278 | 6.9211 | +1.4% | 29.9532 | 28.7875 | −1.1657 |
| s7/2 | 196384 | 194662 | −0.88% | 4.2988 | 4.3333 | +0.8% | 65.6676 | 65.0407 | −0.6269 |

**Butteraugli better in 5 of 14, worse in 9. Mean +2.03%, median +1.18%, range
−5.9% to +14.2%. SSIMULACRA2 worse in 11 of 14, mean −0.41.**

### The byte confound, quantified

Measured lands 0.55% under the cap on average — the rate loop's undershoot, not
a policy effect, but it does hand neutral a small advantage. On this corpus's
local RD slope a 0.55% byte deficit is worth roughly +0.55% butteraugli, so about
a quarter of the mean +2.03% is the confound and roughly +1.5% is real. It does
not rescue the result: it cannot explain s1/1's +14.2%, and it does not touch the
SSIMULACRA2 loss in 11 of 14 cells at all.

## 5. Why a correct measurement produced a worse encoder

Result B is not wrong. Large transforms genuinely do cost more perceptually per
unit of sample-domain error — that measurement stands, on two photographs, in
every frequency bin, at both amplitudes.

What fails is the inference that correcting it improves the encode. Charging
large transforms more makes the cover split more: on the mixed-detail unit
fixture, DCT16x16 count falls from 10 to 5 and DCT8x8 rises from 24 to 44. Those
extra splits cost bits — `DctSelect` signalling and per-varblock overhead — and
at a fixed byte cap those bits come out of quality everywhere else. The better
size decision does not pay for the bits it costs.

That is a specific, useful conclusion rather than a shrug: **the cover's
transform-size axis is not where the Butteraugli deficit lives.** The RD tradeoff
near the current operating point is flat enough that a 9–13% repricing moves the
decision without moving the outcome, and the signalling cost dominates what it
buys.

## 6. Consequences for Phase 6.3

1. **Result B is not carried into 6.3 as a promoted correction.** 6.3 screens the
   Result A CSF frequency weight *alone*. The `CoverSizePenalty` control is
   retained — production-neutral and bit-identical — so 6.3 can also test the
   combination cheaply, but the size scalar is no longer part of its proposed
   change.
2. **The prior is now weaker for 6.3, and that is worth stating before running
   it.** 6.2b re-priced one axis of the same objective, correctly, and the
   perceptual result did not follow. The frequency axis is a much larger
   mispricing (2.65x against 1.09–1.13x) and applies within every candidate
   rather than only between sizes, so it is not the same experiment — but a
   second null would strongly suggest the deficit is not in cover/CfL selection
   at all, and that the next round belongs on the quantizer, which is still
   nearest-reconstruction rather than RD.

## 7. What this does not establish

* Only cover and CfL selection can respond to any of this, because
  `HfQuantizer::choose` picks the nearest reconstruction level rather than an
  RD-optimal one. A null here is a null about those two decisions, not about
  perceptual pricing in general.
* The multipliers are calibrated from Phase 6.2's two probe amplitudes on two
  photographs. A larger or differently-calibrated penalty was not swept; this
  screens the measured value, not the family.
* Seven scenes at two rates, diagnostic status (no provenance sidecars), and one
  image size (0.79 MP). The 4 MP and 12 MP behaviour is unmeasured.
