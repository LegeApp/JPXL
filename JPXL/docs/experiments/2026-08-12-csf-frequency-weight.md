# Phase 6.3: a first-principles CSF weight in the cover objective

Date: 2026-08-12
Status: complete. The CSF **fails** its pre-registered agreement check and the
screen **fails** its promotion gate — but unlike Phase 6.2b this is a
mis-calibration with a diagnosed direction, not a null. Production unchanged.

## 1. Question

Phase 6.0 measured that the cover objective misprices Y-channel coefficient
error by 2.65x across frequency; Phase 6.2 Result A showed one curve in
normalised spatial frequency fits all three squares (agreement 1.15x). Phase
6.2b then screened the *other* half of the original plan — the per-transform
level correction — and it was an honest negative.

So: does weighting the objective by frequency move perceptual quality, and does
a first-principles contrast-sensitivity function supply the right curve?

## 2. Preregistered gates

Two, both stated before any number was computed.

**Agreement (`csf-checked-not-fitted`).** The CSF is derived from open
literature, never fitted. Its predicted response is compared to Phase 6.2's
measured curve, both binned by radial normalised frequency and levelled to the
`f = 0.4-0.5` bin:

* **pass:** worst bin ratio ≤ **1.35x** *and* Pearson `r` ≥ **0.80**
* No parameter may be adjusted to improve agreement. A failure is a result.

**Promotion (`quality-screen`).** Contract B: uniformly non-regressing
Butteraugli without material SSIMULACRA2 loss, or an honest negative.

## 3. Method

`crates/jpxl-encode-policy/src/csf.rs`. Mannos–Sakrison contrast sensitivity,

```
A(f) = 2.6 * (0.0192 + 0.114 f) * exp(-(0.114 f)^1.1)      f in cycles/degree
```

at **60 pixels per degree** — the conventional "one pixel per arcminute"
viewing assumption, fixed by that convention and not chosen to fit anything.
A size-`n` DCT-II's basis `k` carries `k/n` of Nyquist, so cell `(u,v)`'s radial
normalised frequency maps to `radial * ppd / 2` cycles/degree. Expressing it this
way is what lets one curve serve all three squares.

The objective weight is `A(f)^2` — `A` scales perceived error *amplitude* and the
objective's term is squared error — normalised to **mean 1** over the non-LLF
cells. Mean-1 is not cosmetic: `HfQuantizers::lambda` is `16 / mean(s^2)` in bits
per unit squared sample error, so a mean-1 weight leaves the rate/distortion
balance untouched and moves only the *distribution* across frequency. Asserted by
test on all three sizes.

`CoverFrequencyWeight::Flat` builds no table at all and `cell_weight` returns
exactly `1.0`, so the shipped objective is bit-identical — asserted, and
confirmed independently: the flat encode of `s1` at 1 bpp still reproduces the
Phase 5O production row (97,539 B, SSIMULACRA2 62.5738, butteraugli 5.0325).

Screen: the Phase 6.2b corpus and protocol unchanged — seven `test-set`
photographs at 1 and 2 bpp, flat encoded at the requested bpp, its achieved byte
count becoming the exact cap for the CSF arm, both decoded with the pinned
`djxl`. Diagnostic status (no provenance sidecars), as Phase 5O recorded.

## 4. Agreement check: FAIL

Predicted *measured* response is `A(f)^(2p)` with `2p = 0.8924`, because the
harness reports `ba ~ (weight * energy)^p` and `p = 0.4462` was measured
independently in Phase 6.2b.

| bin | f | measured | predicted | ratio |
| --- | --- | ---: | ---: | ---: |
| 0 | 0.0–0.1 | 1.915 | 0.830 | **2.31x** |
| 1 | 0.1–0.2 | 1.655 | 1.062 | 1.56x |
| 2 | 0.2–0.3 | 1.259 | 1.180 | 1.07x |
| 3 | 0.3–0.4 | 1.018 | 1.131 | 1.11x |
| 4 | 0.4–0.5 | 1.000 | 1.000 | 1.00x |
| 5 | 0.5–0.6 | 0.926 | 0.840 | 1.10x |
| 6 | 0.6–0.7 | 0.917 | 0.679 | 1.35x |
| 7 | 0.7–0.8 | 0.804 | 0.543 | 1.48x |
| 8 | 0.8–0.9 | 0.808 | 0.421 | 1.92x |
| 9 | 0.9–1.0 | 1.050 | 0.335 | **3.14x** |

**Worst bin ratio 3.14x (limit 1.35). Pearson r = +0.408 (limit 0.80). FAIL.**

Two distinct disagreements, both large:

1. **The CSF's low-frequency dip is absent from the measurement.** Mannos–Sakrison
   attenuates towards DC (bin 0 predicted 0.830); the measurement says the lowest
   non-LLF frequencies are the *most* sensitive (1.915). This is the known
   divergence between a **detection** CSF, which describes seeing a grating on a
   blank field, and **distortion visibility** on a masked natural image, where
   low-frequency error appears as banding and blotching and is highly visible.
2. **The CSF rolls off far too hard at high frequency.** Predicted 0.335 at the
   top bin against 1.050 measured — and the measured curve *rises* at the corner
   where the model is still falling.

## 5. Promotion screen: FAIL, but the mechanism is live

| cell | bytes flat | bytes CSF | Δbytes | BA flat | BA CSF | ΔBA | SSIM2 flat | SSIM2 CSF | ΔSSIM2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s1/1 | 97539 | 97265 | −0.28% | 5.0325 | 4.8362 | **−3.9%** | 62.5738 | 63.6154 | +1.0416 |
| s2/1 | 98068 | 97816 | −0.26% | 4.4219 | 4.1762 | **−5.6%** | 69.1769 | 69.3387 | +0.1618 |
| s3/1 | 98028 | 97535 | −0.50% | 3.5912 | 3.5097 | −2.3% | 74.6264 | 74.7679 | +0.1415 |
| s4/1 | 97529 | 97113 | −0.43% | 3.2918 | 2.7517 | **−16.4%** | 76.8361 | 77.6661 | +0.8300 |
| s5/1 | 97952 | 97518 | −0.44% | 2.2501 | 2.5145 | +11.8% | 80.7087 | 81.2555 | +0.5468 |
| s6/1 | 98284 | 97389 | −0.91% | 4.3258 | 4.1410 | −4.3% | 58.8070 | 60.2541 | +1.4471 |
| s7/1 | 97904 | 97706 | −0.20% | 6.8278 | 7.1871 | +5.3% | 29.9532 | 34.3250 | **+4.3718** |
| s1/2 | 194682 | 193078 | −0.82% | 2.9610 | 3.5931 | +21.3% | 82.5346 | 82.1496 | −0.3850 |
| s2/2 | 195817 | 195479 | −0.17% | 1.6885 | 2.2350 | **+32.4%** | 87.2284 | 86.6611 | −0.5673 |
| s3/2 | 195855 | 194266 | −0.81% | 1.4990 | 1.4886 | −0.7% | 88.9271 | 88.3912 | −0.5359 |
| s4/2 | 180045 | 172498 | −4.19% | 1.2492 | 1.2329 | −1.3% | 89.4898 | 87.8663 | −1.6235 |
| s5/2 | 174550 | 167317 | −4.14% | 1.1264 | 1.3725 | +21.8% | 90.5175 | 89.8128 | −0.7047 |
| s6/2 | 196136 | 194504 | −0.83% | 2.4621 | 2.8215 | +14.6% | 81.1712 | 80.7540 | −0.4172 |
| s7/2 | 196384 | 195971 | −0.21% | 4.2988 | 4.4512 | +3.5% | 65.6676 | 66.0349 | +0.3673 |

Overall 7 better / 7 worse on Butteraugli — which conceals the actual structure.
Split by rate it is unambiguous:

| | Butteraugli | SSIMULACRA2 | mean Δbytes |
| --- | --- | --- | --- |
| **1 bpp** | better in **5/7**, mean **−2.20%**, median −3.90% | better in **7/7**, mean **+1.22** | −0.43% |
| **2 bpp** | better in 2/7, mean **+13.10%**, median +14.60% | better in 1/7, mean −0.55 | −1.60% |

The promotion gate fails: not uniformly non-regressing.

**But this is not Phase 6.2b's null.** 6.2b moved Butteraugli by a mean of
+2.03% with everything within ±14%; the frequency weight moves it from −16.4% to
+32.4% and moves SSIMULACRA2 by up to +4.37. Frequency is a lever that actually
drives the objective. Transform size was not.

Two cells (s4/2, s5/2) undershoot the cap by ~4.2%, which inflates their apparent
2 bpp regression; the 2 bpp mean is overstated by roughly a point on that account.
It does not change the direction — s2/2 regresses 32.4% at a 0.17% byte deficit.

## 6. Why the rate split, and what it says the weight should be

The §4 disagreement predicts the §5 result exactly.

The CSF under-weights high frequency by up to 3.14x. At **1 bpp** most
high-frequency coefficients quantize to zero regardless of how they are priced,
so that error is inert, and what survives is the CSF's correct instinct that
low-frequency structure matters most — hence wins on 5 of 7 Butteraugli cells and
**all 7** SSIMULACRA2 cells. At **2 bpp** those coefficients are being coded and
are visible, and telling the objective they cost a third of what they do makes it
discard them — hence +13% mean Butteraugli.

So the measurement and the screen agree on the same correction: **the weight must
be far flatter at high frequency than a detection CSF, with no low-frequency
dip.** That is precisely the shape Phase 6.0/6.2 measured and this model failed to
reproduce.

## 7. Answer

* **Frequency weighting is the live lever** in the cover objective. It moves
  quality by an order of magnitude more than the transform-size axis did, and at
  1 bpp it improves SSIMULACRA2 on every scene tested.
* **Mannos–Sakrison at 60 ppd is not the right curve**, by a pre-registered check
  it failed on both criteria and in a way that predicts its own screen result.
* Production stays `Flat`. Both research controls are retained,
  production-neutral and bit-identical.

## 8. Continuation

The obvious next step is a weight with the measured shape rather than a detection
CSF's. That crosses the line the workspace manifest draws — a table fitted to
butteraugli's response makes this project's perceptual model a derivative of
libjxl's — so it is **a decision to record before it is taken**, not something to
slide into. Three honest options, in preference order:

1. **A distortion-visibility model from open literature** rather than a detection
   CSF: same clean-room standing as this phase, and §6 says what it must look
   like (flat-ish at high frequency, monotone from DC). This is the option that
   keeps the model our own.
2. **A two-parameter analytic shape** (low-frequency plateau plus a gentle
   rolloff) whose parameters are set from the *encoder's own* quantization
   geometry rather than from butteraugli, then checked against §4.
3. **Fit the measured curve**, explicitly recorded as a decision that our
   perceptual model becomes libjxl-derived, with the licence and provenance
   consequences stated.

Also worth noting for whoever picks this up: the 1 bpp result is already a
uniform SSIMULACRA2 win (7/7) and a 5/7 Butteraugli win. A **rate-dependent**
weight — CSF-like at low rate, flat at high rate — is not principled as stated,
but it is evidence that the correct weight interacts with which coefficients
survive quantization, which is a real effect a static table cannot capture.
