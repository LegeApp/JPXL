# Phase 7.1: block-level trailing-nonzero truncation

Date: 2026-08-12
Status: complete. The design hypothesis is **confirmed** — selectivity more than
halves the SSIMULACRA2 damage — but the gate still **fails**, which fires the
pre-registered redirect to lambda. Production unchanged.

## 1. What was built, and why this shape

Phase 7.1a established the lever: 18181-1 I.4 emits a coefficient token for every
order position up to the last nonzero and then **stops**. So zeroing a block's
last nonzero frees its own token *plus every interior zero back to the previous
nonzero* — 1.56 extra tokens on average at 1 bpp — while zeroing mid-run frees
nothing. `residual_bits` credits a flat ~2 bits for both, which is why Phase
7.0's per-coefficient rate term zeroed by magnitude alone and stripped texture
uniformly.

Because the last-nonzero position is a property of the **whole block**, no
per-coefficient rule can express this. So `HfQuantizer::truncate_trailing` is a
backward pass: quantize greedily as before, then walk back from the last nonzero
considering each trailing nonzero for removal, pricing the drop at

```
own token bits + (interior zeros it exposes) * ZERO_TOKEN_BITS
```

against `weight[cell] * (target² − (recon − target)²)`, and accepting while
rate-distortion favourable. It only ever removes from the end, so it *cannot*
strip texture uniformly the way a magnitude rule does.

`ZERO_TOKEN_BITS = 1.0` is a stated estimate, not a measurement: token counts are
exact but token bits are not, since each symbol's real cost depends on an ANS
cluster the census builds after the walk. One bit is deliberately conservative —
it under-credits truncation, so the pass errs toward keeping coefficients.

Ordering detail that matters: Y is truncated *before* `d_y_hf` is read, so the
chroma CfL targets decorrelate against the Y the decoder will actually
reconstruct rather than against coefficients the pass then drops.

## 2. Proof that it does what it claims

`tests/truncation_saving.rs` checks the claim against the **real** walk
(`jpxl_encode::vardct::walk_frame`), not a restatement of the same assumption.
On a 256×256 mixed-content fixture at 1 bpp:

| | pass off | pass on | change |
| --- | ---: | ---: | ---: |
| `non_zeros` symbols | 678 | 678 | unchanged |
| nonzero tokens | 5,513 | 4,863 | **−11.8%** |
| zero tokens | 13,907 | 7,988 | **−42.6%** |
| total symbols | 20,098 | 13,529 | −32.7% |

That asymmetry **is** the signature being tested. Removing 11.8% of nonzeros
collapses 42.6% of zero tokens, because each removal truncates a walk. A pass
that zeroed by magnitude would move the two counts together; one that zeroed
mid-run would *raise* the zero-token count. The `non_zeros` symbol count is
unchanged, confirming the pass touches coefficients and never the cover.

## 3. Screen

28 exactly matched-rate cells, same corpus, rates and protocol as Phase 7.0 so
the two are directly comparable.

| rate | SSIMULACRA2 | butteraugli | mean Δbytes |
| --- | --- | --- | --- |
| 0.5 bpp | better 2/7, mean **−0.95** | 4/7, +1.55% | −1.38% |
| 1 bpp | better 1/7, mean **−0.71** | **6/7, −8.00%** | −0.36% |
| 2 bpp | better 0/7, mean **−0.60** | 5/7, −6.90% | −2.00% |
| 4 bpp | better 0/7, mean **−0.46** | 4/7, −0.48% | −3.98% |

**Pooled: SSIMULACRA2 better in 3/28, mean −0.68. Butteraugli better in 19/28,
mean −3.46%.**

Against the two prior arms on the identical protocol:

| arm | SSIMULACRA2 | butteraugli |
| --- | --- | --- |
| 7.0 per-coefficient rate term | 4/28, mean **−1.80** | 20/28, −5.21% |
| **7.1 selective truncation** | 3/28, mean **−0.68** | 19/28, −3.46% |
| 6.5 donor weight (promoted) | 18/28, mean +1.37 | 17/28, −0.17% |

## 4. Answer

**The design hypothesis is confirmed.** Selectivity cut the SSIMULACRA2 damage
from −1.80 to −0.68 — **62% less harm** — while keeping most of the butteraugli
gain (−3.46% against −5.21%). Restricting removals to those that actually
truncate a walk did exactly what 7.1a predicted it would.

**The gate still fails.** Under
`jpegxl-rs.decision.ssimulacra2-is-the-primary-promotion-metric`, SSIMULACRA2
decides, and it regresses in 25 of 28 cells. Production stays
`QuantizerChoiceMode::Nearest`.

The pre-registered fallback therefore fires, and it is worth quoting as written
because the result is exactly the case it anticipated: *"If SSIMULACRA2 still
regresses, the over-smoothing is not the proxy's fault and the target becomes
lambda calibration."*

### Why lambda, specifically

The finding is now sharper than "rate-aware quantization hurts". A pass that
removes **only** coefficients whose removal collapses a run — the most
rate-efficient removals available anywhere in the block — still costs
SSIMULACRA2 on 25 of 28 cells. So the problem is not *which* coefficients are
dropped. It is that dropping **any** HF coefficient at this operating point costs
more structure than the bits are worth, on SSIMULACRA2's accounting.

That is a statement about the exchange rate, i.e. `lambda`. It is derived in
`HfQuantizers::new_with_scales` as `16 / mean(s²)` over the non-LLF DCT8x8 cells,
from a uniform-quantizer argument — halving a step costs one bit per coefficient
and moves `s²/12` to `s²/48`. **It has never been calibrated against a perceptual
metric.** If it over-values bits, every rate-aware decision will over-zero
regardless of how selective it is, which is precisely the pattern across 7.0 and
7.1.

Note also that `ZERO_TOKEN_BITS` and `lambda` are the same knob in effect — both
scale the rate side against distortion — so sweeping the former is not an
independent experiment. Calibrating lambda is.

## 5. Caveats

- **The high-rate byte undershoot recurs**, milder than 7.0's: −2.00% at 2 bpp
  and −3.98% at 4 bpp, worst cell −7.00%. Those rows are partly confounded, and
  a smaller stream at fixed quantizer settings again pushes the rate loop away
  from a cap it already struggles to reach. The 1 bpp row (−0.36%) is clean and
  carries the clearest signal: butteraugli better on 6 of 7 by a mean of −8.00%,
  SSIMULACRA2 worse on 6 of 7.
- **`s7/0.5` is an outlier and should not be read as a normal cell.** Its
  SSIMULACRA2 is already *negative* (−10.08) at the flat baseline — the scene is
  destroyed at that rate — and it lands 7.00% under cap. Its −5.17 SSIMULACRA2
  delta dominates the 0.5 bpp mean; excluding it the 0.5 bpp mean is −0.25.
- **Token counts are exact; token bits are not**, and the cross-block context
  coupling (a block's `non_zeros` feeds later blocks' contexts) is ignored by a
  within-block pass. Both were recorded as known approximations before the screen
  ran, and neither is the explanation for a 25-of-28 result.
- The pass allocates a natural-order vector per varblock. Acceptable for an
  opt-in research mode; it would need hoisting before any promotion.

## 6. Continuation

**Phase 7.2: calibrate lambda against SSIMULACRA2.** The three phases now form a
clean argument for it. 6.2b showed the cover's size axis is inert. 6.5 showed the
frequency axis helps and is now promoted. 7.0 and 7.1 showed the rate axis moves
quality hard in butteraugli's favour and against SSIMULACRA2's, and 7.1 showed
that persists even when only the most rate-efficient removals are taken. The one
term never calibrated against any perceptual metric is the exchange rate between
the two.

The experiment is cheap and well-posed: sweep lambda's multiplier over the same
28-cell protocol with the truncation pass on, and find whether an operating point
exists where SSIMULACRA2 is non-regressing and butteraugli keeps its gain. If
none exists, rate-aware quantization is genuinely not available to this encoder
at any exchange rate, which would itself be worth knowing and would end the line
of enquiry cleanly.
