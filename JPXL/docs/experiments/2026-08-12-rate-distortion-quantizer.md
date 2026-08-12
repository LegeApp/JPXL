# Phase 7.0: a rate-aware HF quantizer

Date: 2026-08-12
Status: complete. Contract B **fails** on a material SSIMULACRA2 loss — but this
is the largest quality movement any phase on this track has produced, and it
localises the remaining deficit precisely. Production unchanged.

## 1. Question

Three phases converged on the same wall. Phase 6.2b re-priced the cover's
transform-size axis (null). Phase 6.3 re-priced frequency (rate-split). Phase 6.5
re-priced frequency well (median 1.41× against the measured target) and got a
large win at 0.5 bpp fading to a **no-op at 4 bpp**. All three were bounded by
one fact: `HfQuantizer::choose` picks the nearest reconstruction level, so cover
and CfL are the only decisions any weight can move — and they matter least
exactly where the most coefficients survive.

`sources/outside-advice.md` names it directly: *"Quantization is
reconstruction-nearest, not rate-distortion optimized. It can spend bits on
visually unimportant small coefficients while still underprotecting perceptually
important structures elsewhere."*

## 2. What changed

`choose` already evaluates `[0, est-1, est, est+1]` and takes the smallest
`|recon − target|`. It now takes the smallest

```
residual_bits(q) + rd * (recon(q) - target)^2
```

where `rd` is **exactly** the coefficient `block_cost_bounded` already multiplies
squared coefficient error by — `lambda[channel] * side² * size_penalty *
frequency_weight[cell]`. Minimising that per coefficient minimises the same total
the cover search sums, so the quantizer and the cover finally agree on one
currency.

The `rd` table is installed on the quantizer at construction rather than threaded
through `choose`'s signature, because the decision rule is a property of the
operating point. `None` means nearest, and the branch on it keeps the shipped
path bit-identical. Installation happens after `lambda` is calibrated, since
`lambda` is derived from the finished DCT8x8 baseline quantizer.

Two mechanical consequences, both handled rather than discovered later:

- The **zero shortcut** (`|target| ≤ 0.5·bias·step ⇒ 0`) is only sound under the
  nearest rule. Under RD it would still return the right answer *inside* the
  threshold, but taking it would skip the wider zeroing the mode exists for, so
  the RD path runs the full comparison.
- The **SIMD `choose_lane4`** vectorises the nearest rule. Under RD it delegates
  to four scalar `choose` calls — what the non-SIMD build already does — so the
  two stay bit-identical by construction rather than by maintaining a second
  hand-vectorised comparison.

`cell_lower_bound`'s provable prune assumes the nearest rule, so the
`s8-cover-prune` feature is not validated against this mode and stays off.

### Stated limitation, up front

`residual_bits(q)` charges magnitude bit length plus sign. `outside-advice.md` is
explicit that it knows nothing about entropy context, zero-run behaviour,
coefficient order, neighbouring nonzeros, token probabilities or table overhead.
**The classic rate-distortion quantization win is zeroing a coefficient to extend
a zero run, and this proxy cannot see runs at all.** So this captures only the
first-order "is this coefficient worth any bits" decision — a widened dead zone.

## 3. Proofs

- `Nearest` is the default; an explicitly-nearest request is byte-identical to
  it, and the RD arm provably reaches the wire.
- `rate_distortion_choice_minimises_its_own_objective_and_widens_the_dead_zone`
  checks, over every square and 24 target magnitudes per cell, that **no
  candidate beats the chosen one** on `residual_bits + rd·error²` (recomputed
  independently from the public surface), and that RD **never** picks a larger
  magnitude than nearest — a rate term can only make coefficients cheaper, so
  the dead zone can only widen. It also asserts the mode is not inert.
- Full workspace suite green, `cargo fmt --all --check` clean.

## 4. Result

28 exactly matched-rate cells, same corpus and protocol as Phases 6.2b/6.3/6.5.

| requested | butteraugli | SSIMULACRA2 | mean Δbytes |
| --- | --- | --- | --- |
| **0.5 bpp** | better **5/7**, mean **−2.08%**, best −14.3% | 4/7, mean −0.72 | −0.34% |
| **1 bpp** | better **5/7**, mean **−8.79%**, best **−29.9%** | **0/7**, mean **−3.00** | −0.43% |
| **2 bpp** | better 4/7, mean **−4.29%** | **0/7**, mean −2.36 | **−3.01%** |
| **4 bpp** | better **6/7**, mean **−5.66%** | **0/7**, mean −1.10 | **−5.72%** |

**Pooled: butteraugli better in 20/28, mean −5.21%. SSIMULACRA2 better in 4/28,
mean −1.80.**

For scale, on the identical protocol: Phase 6.2b moved butteraugli +2.03% with
everything inside ±14%; Phase 6.5 moved it −0.17%. **This is by far the largest
lever measured on this track** — a single cell moves −29.9%.

### The metric split is real, not a byte artifact

At 0.5 and 1 bpp the byte deltas are −0.34% and −0.43%, i.e. properly matched.
At 1 bpp butteraugli improves by a mean of **−8.79%** while SSIMULACRA2 regresses
on **every single scene** by a mean of **−3.00**. That disagreement is the
finding, and it is not explainable by rate.

The mechanism is over-smoothing. Widening the dead zone removes small nonzero
coefficients; butteraugli's max-norm aggregation rewards the reduced peak error,
while SSIMULACRA2 — which is structure- and detail-sensitive — penalises the lost
texture. A rate proxy that cannot see zero runs zeroes **uniformly** rather than
selectively, so it takes texture everywhere instead of where runs would actually
pay.

### A second, separate finding: RD makes the high-rate undershoot worse

At 2 and 4 bpp the RD arm lands **3.0%** and **5.7%** under the cap on average,
up to **−7.8%** on a single cell. Those cells (`s3/2`, `s4/2`, `s5/2`, and most
of the 4 bpp row) are the same ones whose *nearest* encode already undershot —
the known Phase 4M high-rate undershoot. RD produces a smaller stream at every
quantizer setting, which pushes the rate loop further from a cap it already
struggled to reach.

**Those two rate rows are therefore confounded** and their metric deltas should
not be read as matched-rate comparisons. Notably the butteraugli improvement at
4 bpp (−5.66%) is achieved while spending 5.7% *fewer* bytes, which is a genuine
rate-distortion improvement on that axis; the SSIMULACRA2 loss at those rates is
partly the byte deficit (roughly a third of it, on this corpus's local RD slope)
and partly real.

## 5. Answer

**Contract B fails**: material SSIMULACRA2 loss, 0/7 scenes better at 1, 2 and
4 bpp. Production stays `QuantizerChoiceMode::Nearest`.

But the pre-registered informative outcome is the one that landed. The work
record said a null would mean the deficit is not in per-coefficient decisions and
the track should turn to the entropy model. It was not a null — the quantizer
moved quality an order of magnitude more than any cover-side lever — **and it
still points at the entropy model**, for a sharper reason: the rate term works,
and its *proxy* is what is wrong. A rate estimate that knew about zero runs and
context would zero selectively, keeping the texture SSIMULACRA2 is measuring and
the coefficients that extend no run.

Three phases said "the quantizer is where the leverage is." This one confirms it
and hands the next phase a specific target.

## 6. Continuation

1. **A run-aware rate estimate is the next thing to build**, and it is now the
   critical path rather than one option among several. The entropy stage already
   tokenises coefficients (`entropy.rs`) with the exact C.2.3 hybrid-uint
   arithmetic the writer uses, so a context- and run-aware `residual_bits`
   replacement has a foundation. Re-running this exact screen against it is the
   cleanest possible A/B, since everything else is held fixed.
2. **The rate loop needs to be re-checked under any quantizer change.** The
   undershoot regression here was not anticipated by the protocol and would have
   silently confounded a weaker result. Any future mode that changes stream size
   at fixed quantizer settings should report achieved-versus-requested bpp as a
   first-class number, not a footnote.
3. **The metric split deserves its own attention.** Butteraugli and SSIMULACRA2
   disagreeing this sharply and this consistently (20/28 against 4/28) is
   information about the metrics as much as the encoder. Phase 5 promoted on
   butteraugli with SSIMULACRA2 as a guard; if a lever can move them in opposite
   directions this reliably, that promotion rule needs restating before the next
   promotion, not after.
