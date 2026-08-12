# Phase 6.5: the frequency weight the standard already knows

Date: 2026-08-12
Status: complete. The agreement gate **passes**; the Contract B promotion gate
**fails**, but the pre-registered prediction is confirmed and the result is the
strongest this track has produced. Production unchanged.

## 1. Question

Phase 6.3 established that frequency weighting is the live lever in
`block_cost_bounded` — it moves butteraugli from −16.4% to +32.4% where the
transform-size axis moved nothing — and that Mannos–Sakrison is the wrong curve,
failing a pre-registered check by under-pricing high frequency, which is inert at
1 bpp and destructive at 2 bpp.

Where does a better curve come from without fitting to butteraugli?

## 2. Option 1 is closed, and closing it explains Phase 6.3

The canonical DCT-domain per-coefficient models — **Ahumada & Peterson (1992)**,
**Peterson, Ahumada & Watson (1993)**, **Watson's DCTune (1993)**, **Nill
(1985)** — are all *detection-threshold* models. Every one is band-pass with a
low-frequency dip, and their high-frequency rolloff is **harder** than
Mannos–Sakrison, not softer: AP92 predicts ~0.037 of peak sensitivity at Nyquist
where measurement says ~1.05. That is ~28× off against Mannos–Sakrison's 3.1×.
**Adopting any of them would be strictly worse than what 6.3 already rejected.**

The literature also supplies the reason, which converts 6.3's failure from a
surprise into an understood result: **Georgeson & Sullivan (1975), contrast
constancy** — above threshold, perceived contrast becomes largely independent of
spatial frequency, so the band-pass detection CSF flattens. A detection CSF
describes seeing a grating on a blank field; it is the wrong regime for judging
quantization error in real content, where low-frequency error (banding, blocking)
is highly visible. That is exactly why the measured curve has no low-frequency
dip.

## 3. The candidate, and why the obvious version of it does not work

The standard's I.2.5 default dequantization matrices are a *distortion
allocation*, not a detection threshold. Scored against the measured target:

| model | median | worst | worst interior |
| --- | ---: | ---: | ---: |
| Mannos–Sakrison CSF (6.3, rejected) | 2.06x | 13.21x | 4.34x |
| **`1/step²` from the DCT8x8 matrix, resampled** | **1.41x** | 4.03x | 2.02x |
| best smooth power law *fitted to the target* (the ceiling) | 1.18x | 2.07x | 1.82x |

The donor captures most of what a butteraugli-fitted curve could, **without
fitting to butteraugli**. Fitting was therefore not pursued.

**But using each size's own matrix does not work** — that is the `1/step²`
weighting Phase 6.2 already falsified. The matrices are not a frequency model:
their step shapes disagree **3.34×** across sizes in normalised radial frequency,
against the measured perceptual curve's 1.15×. They are per-size tuned. Reading
**one** size's matrix as a curve and resampling it onto every transform's grid
repairs that by construction.

**Which size donates was settled by measurement, not argument:**

| donor | median | worst | worst interior |
| --- | ---: | ---: | ---: |
| **DCT8x8 only** | **1.43x** | 4.06x | **2.06x** |
| DCT16x16 only | 1.90x | 6.97x | 2.88x |
| DCT32x32 only | 2.62x | 11.76x | 4.80x |
| geometric mean of all three | 2.34x | 10.38x | 4.21x |

Averaging is *worse* than DCT8x8 alone — the 16/32 curves are much steeper and
drag it off target. DCT8x8 also has independent standing: it is the baseline
transform and the operating point `HfQuantizers::lambda` is calibrated at.

## 4. Two gating corrections

Both matter for reading Phase 6.3 fairly.

- **6.3's 1.35× worst-bin tolerance was unachievable by its entire model class.**
  The best *fitted* smooth two-parameter shape — power law, log-parabola,
  exponential — cannot beat **1.72× worst-bin** against this target. That gate
  was never a fair test of Mannos–Sakrison; it measured only that a smooth curve
  cannot reproduce two artifact bins.
- **Worst-bin is the wrong statistic.** It is dominated by `f < 0.1` (the steep
  low-frequency rise) and `f > 0.9` (the corner re-rise). Median is primary here,
  worst-interior secondary.

## 5. Agreement gate: PASS

Pre-registered before implementation: median ≤ 1.60×, worst-interior ≤ 2.20×.
Scored against the **shipped** table (dumped by
`tests/dump_weights.rs`, not a Python restatement):

**median 1.41×, worst-interior 2.02×, worst overall 4.03×. PASS.**

Stated plainly, because it would be easy to overclaim: **this gate cannot claim
predictive success.** The donor already scored 1.41×/2.02× on archived data
before any code was written. It exists to catch an implementation that fails to
reproduce the analysis, and it did its job — the implemented weights match the
analysis exactly.

The 4.03× worst-overall is the known limitation: the DCT8x8 donor spans only
`f ∈ [0.088, 0.875]`, so DCT16x16 and DCT32x32 cells above 0.875 clamp flat
instead of continuing to fall. Documented in the code, not silent.

## 6. Screen: promotion fails, prediction confirmed

28 exactly matched-rate cells — 7 `test-set` photographs at 0.5/1/2/4 bpp, donor
encoded to the flat arm's achieved byte count, both decoded with the pinned
`djxl`. Diagnostic status (no provenance sidecars), as Phase 5O recorded.

| requested (flat achieved) | butteraugli | SSIMULACRA2 | mean Δbytes |
| --- | --- | --- | --- |
| **0.5 bpp** (0.497) | better **5/7**, mean **−4.89%**, best −12.3% | better **7/7**, mean **+4.85** | −0.51% |
| **1 bpp** (0.996) | better 4/7, mean **−1.86%** | better **7/7**, mean **+0.70** | −0.58% |
| **2 bpp** (1.938) | better 4/7, mean +6.74%, **median −0.03%** | better 3/7, mean −0.00 | −0.66% |
| **4 bpp** (3.647) | better 4/7, mean **−0.69%** | better 1/7, mean −0.07 | −0.85% |

**Pooled: butteraugli better in 17/28, mean −0.17%. SSIMULACRA2 better in 18/28,
mean +1.37.**

**The pre-registered prediction is confirmed.** 6.3's 2 bpp arm had mean
**+13.10%**; the donor's is **+6.74%** — roughly halved, which is what the
high-frequency pricing improvement (CSF 0.41× of target, donor 0.65×) predicted.
SSIMULACRA2 at 2 bpp moved from −0.55 (1/7 better) to −0.00 (3/7), i.e. neutral.

**Contract B still fails**: not uniformly non-regressing. 11 of 28 cells regress
on butteraugli.

### Reading the 2 bpp mean honestly

Its two regressors are `s2/2` (+30.7%) and `s5/2` (+24.7%), and both sit at
already-excellent absolute quality — butteraugli 1.6885 and 1.1264 respectively.
In absolute terms those are +0.52 and +0.28. The **median** at 2 bpp is −0.03%,
so the mean is an outlier effect, not a broad regression: the other five scenes
are within ±6.7% and mostly improving.

That is a caveat in the donor's favour and should not be used to wave the failure
away — a +30.7% relative regression is still a real regression on a scene the
encoder was handling well.

### The rate structure is the interesting part

The donor is a **large win at 0.5 bpp** (SSIMULACRA2 better on every scene by
+3.45 to +5.97), a **clear win at 1 bpp** (SSIMULACRA2 7/7), **neutral at
2 bpp**, and **very nearly a no-op at 4 bpp** (most cells within ±0.1%
butteraugli).

That gradient is mechanistically consistent with Phase 6.3's hypothesis: the
weight matters most where the fewest coefficients survive quantization, because
that is where the cover decision changes what is kept. At 4 bpp almost nothing is
zeroed, so how candidates are priced barely changes what is emitted. It is
further evidence that the remaining deficit lives in the quantizer —
`HfQuantizer::choose` is still nearest-reconstruction rather than RD-optimal —
and not in cover/CfL selection, which is all any of these weights can reach.

## 7. Answer

* Option 1 is **closed** with a citation, and it explains 6.3.
* Option 3 is **not needed**: the donor reaches 1.41× against a fitted curve's
  1.18× ceiling, so buying that gap with a libjxl-derived perceptual model is not
  worth it.
* The donor is the **best-performing weight this track has produced** and the
  first to be net-positive pooled on both metrics, but it does not clear Contract
  B and production stays `Flat`.

## 8. Continuation

1. **A rate-conditioned policy is the obvious next screen and the obvious trap.**
   The donor is strongly positive below 1 bpp and inert above 2. Switching on
   rate would very likely pass a corpus gate — but "helps at low rate" is a
   description, not a mechanism, and Phase 5 is full of content-dependent
   switches that did not survive. If it is screened, the mechanism (how many
   coefficients survive quantization) should be measured directly, not proxied
   by the rate request.
2. **The quantizer is the better target.** Three phases now point the same way:
   6.2b (size axis, null), 6.3 (frequency axis, rate-split), 6.5 (frequency axis,
   fades to nothing as rate rises). All three are bounded by the fact that
   `HfQuantizer::choose` picks the nearest reconstruction level rather than an
   RD-optimal one, so cover and CfL are the only decisions any weight can move.
3. **The oblique effect is real and unexploited.** Ahumada–Peterson model an
   orientation penalty `r + (1−r)cos²θ` with `r = 0.7`. Radius-controlled against
   the Phase 6.2 data the correlation is **+0.475 over 1322 cells** — diagonals
   *are* less sensitive. But it is not constant: the diagonal/axis ratio runs
   **1.11–1.39 at f = 0.10–0.25** (diagonals *more* sensitive) and **0.49–0.68 at
   f = 0.55–0.70**. AP's constant `r` is wrong in both directions, so this needs
   a frequency-dependent form the literature does not supply. Deliberately kept
   out of this screen so the donor's contribution stayed isolated.
