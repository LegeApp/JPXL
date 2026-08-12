# Phase 7.2: calibrate lambda against SSIMULACRA2

Date: 2026-08-12
Status: complete. An operating point **exists** and is **promoted** on the
target-rate path: trailing truncation at `lambda_scale = 4.0`. Production
fixed-quantizer defaults stay nearest at unit lambda.

## 1. Why this experiment

Phases 7.0 and 7.1 established that a rate-aware quantizer moves quality hard
(butteraugli better in ~20 of 28 cells) but against SSIMULACRA2. Phase 7.1's
selective truncation cut the SSIMULACRA2 damage by 62% while keeping most of
the butteraugli gain — and still regressed SSIMULACRA2 in 25 of 28 cells.

That is a statement about the exchange rate, not about *which* coefficients
drop. `lambda` is derived in `HfQuantizers::new_with_scales` as
`16 / mean(s²)` over non-LLF DCT8x8 cells from a uniform-quantizer argument. It
had never been checked against any perceptual metric. If it under-weights
distortion (equivalently, over-values bits in the `R + λD` form this encoder
uses), every rate-aware decision over-zeroes regardless of selectivity.

`ZERO_TOKEN_BITS` and `lambda` are the same knob in effect — both scale rate
against distortion — so only `lambda` needs a control.

## 2. What was built

`EncodeRequest::lambda_scale` (default `1.0`) multiplies the derived `lambda`
before it is used by the cover objective and by the Phase 7.0/7.1 RD weights.
CLI: `--lambda-scale <f>`. Non-positive or non-finite values fall back to `1.0`.

The unit scale is bit-identical to the pre-Phase-7.2 path. Under
`QuantizerChoiceMode::Nearest` a non-unit scale still moves cover selection
(because `block_cost_bounded` always multiplies by `lambda`); that is
intentional for research.

## 3. Protocol

Same 28-cell matched-rate protocol as Phases 7.0 and 7.1:

- 7 scenes (`test-set/2024050{1_110934,2_151356,2_151800,2_184356,2_192515,3_105655,3_105759}`)
- rates 0.5 / 1 / 2 / 4 bpp
- flat (production nearest) at requested bpp sets the byte cap
- each arm re-encodes to that cap with
  `--quantizer-choice trailing-truncation --lambda-scale <s>`
- metrics via `jpxl compare` (SSIMULACRA2 primary, butteraugli secondary)

Scales swept: **0.5, 1.0, 2.0, 4.0, 8.0**.

Pre-registered reading rules carried from 7.1:

- the **1 bpp row is primary** (high-rate rows partly confounded by undershoot)
- **s7/0.5 is reported separately** (flat baseline SSIMULACRA2 already negative)

Raw rows and driver:
`.agent/scratch/phase7-2-lambda-2026-08-12/`.

## 4. Results

Pooled (28 cells) against flat, mean SSIMULACRA2 delta and mean butteraugli
percent (negative = better):

| scale | SSIMULACRA2 | butteraugli | both better | mean Δbytes |
| --- | --- | --- | --- | --- |
| 0.5 | 1/28, mean **−2.98** | 14/28, **+1.54%** | 1/28 | −2.24% |
| 1.0 (Phase 7.1) | 3/28, mean **−0.68** | 19/28, **−3.46%** | 2/28 | −1.93% |
| 2.0 | 15/28, mean **+1.10** | 23/28, **−4.94%** | 12/28 | −1.21% |
| **4.0** | **24/28, mean +1.94** | **23/28, mean −6.81%** | **22/28** | −0.95% |
| 8.0 | 21/28, mean **+2.21** | 17/28, **−3.06%** | 16/28 | −2.16% |

Primary 1 bpp row (7 cells):

| scale | SSIMULACRA2 | butteraugli | both |
| --- | --- | --- | --- |
| 0.5 | 0/7, −2.20 | 6/7, −4.47% | 0/7 |
| 1.0 | 1/7, −0.71 | 6/7, −8.00% | 1/7 |
| 2.0 | 5/7, +0.53 | 5/7, −3.66% | 4/7 |
| **4.0** | **7/7, +1.42** | **7/7, −9.06%** | **7/7** |
| 8.0 | 7/7, +1.86 | 4/7, −4.04% | 4/7 |

Byte deltas at scale 4.0 by rate: −0.73 / **−0.55** / −0.60 / −1.93%. The
1 bpp row is clean. High-rate undershoot is milder than Phase 7.1's.

s7/0.5 at scale 4.0: SSIMULACRA2 **+6.79**, butteraugli **−6.51%** — no longer
an outlier in the adverse direction.

## 5. Decision

**An operating point exists.** `lambda_scale = 4.0` under trailing truncation
is the first rate-aware quantizer setting that is non-regressing on
SSIMULACRA2 *and* keeps (in fact enlarges) the butteraugli gain:

- pooled: SSIMULACRA2 better in **24 of 28** (mean **+1.94**), butteraugli
  better in **23 of 28** (mean **−6.81%**), both better in **22 of 28**
- primary 1 bpp: **7 of 7** on both metrics

Scale 2.0 also clears the mean gate but is weaker (SSIM 15/28, both 12/28).
Scale 8.0 keeps the SSIM win but loses butteraugli on the 4 bpp row
(+6.46% mean) and deepens undershoot (−6.99% at 4 bpp). **4.0 is the chosen
point.**

### Direction of the correction

In this encoder's `R + λD` form, higher `λ` values distortion more and
truncates less. The unit-scale arms over-zeroed for SSIMULACRA2 because the
uniform-quantizer derivation under-weighted sample-domain distortion relative
to what SSIMULACRA2 accounts as structure. Multiplying by four is an empirical
correction, not a new derivation — the uniform-quantizer argument remains the
starting point.

### Promotion

Under `jpegxl-rs.decision.ssimulacra2-is-the-primary-promotion-metric`:

- `EncodeRequest::for_target` now sets
  `quantizer_choice = TrailingTruncation` and `lambda_scale = 4.0`
- `EncodeRequest::defaults` stays nearest at unit lambda (Contract A
  fixed-quantizer path unchanged)

## 6. What this does *not* claim

- It does not claim the uniform-quantizer derivation is wrong in absolute
  units — only that, for this encoder's rate proxy and truncation decision, a
  4× scale is the operating point that matches the primary metric.
- It does not re-open Phase 7.0's per-coefficient rate term. That mode still
  cannot see runs; the promoted path is selective truncation only.
- Cross-block context coupling and the `ZERO_TOKEN_BITS = 1.0` estimate remain
  known approximations; calibrating lambda absorbed their first-order effect
  into one scale rather than measuring them separately.
- The high-rate undershoot is reduced but not eliminated. Rate-loop fill is a
  separate lever.

## 7. Continuation

With a working rate-aware quantizer on the target-rate path, the track's next
unexhausted levers are the entropy model (still crude as a rate proxy for
anything mid-run) and modular planning (Phase 4). Phase 5P (exact-distortion
special transforms) remains independent research.
