# Phase 5P: exact sample-domain special transforms

Date: 2026-08-12
Status: complete. **Honest negative.** Exact sample-domain scoring of
same-8×8 specials does **not** remove Phase 5O's Butteraugli regressions; it
makes SSIMULACRA2 and butteraugli worse on almost every discriminating cell.
Temporary research mode removed; production fingerprint restored. Writer
support for the same-8×8 special vocabulary remains (wire evidence from 5O).

## 1. Hypothesis

Phase 5O selected Hornuss / DCT2x2 / DCT4x4 / DCT8x4 / DCT4x8 / AFV0–3 at 8×8
leaves using the cover's square-DCT distortion currency
(`Σ λ · side² · (recon − target)²` in coefficient space). Specials are not
Parseval, so that map mis-prices them. Phase 5P re-scored each candidate with
exact inverse-transform sample SSE after the same quantizer choice, plus the
same eight-bit DctSelect transition proxy 5O measured, under a research flag.

## 2. What was built (then removed)

- `SpecialTransformMode::ExactSampleDistortion` on `EncodeRequest` and
  `--special-transforms exact-sample` on the CLI.
- `block_cost_exact_sample`: quantize HF, inverse via
  `TransformType::samples_from_coefficients`, charge
  `Σ λ_c · (recon − sample)²`.
- Leaf selection over the ten same-footprint transforms.
- Writer gate widened permanently to accept those transforms (Phase 5O wire
  evidence); large/rectangular families stay refused.

## 3. Distortion proof (kept)

`exact_sample_distortion_is_not_parseval_for_hornuss` shows that for Hornuss
the Parseval map `coeff_SSE · 64` is not equal to inverse-recon sample SSE.
Any future special-transform scorer that reuses the square-DCT coefficient map
is known-wrong on at least one live transform family.

## 4. Quality screen (discriminating set)

Eight cells: Phase 5O's strongest winners and its regressors, matched-rate
protocol (production `--bpp` sets the cap; exact arm re-encodes to that cap).
Production is the post–Phase-7.2 target-rate policy (trailing truncation at
`lambda × 4`).

| cell | Δ SSIMULACRA2 | Δ butteraugli |
| --- | ---: | ---: |
| small1/2 | −5.81 | **−7.2%** |
| small2/1 | −10.43 | +3.5% |
| small3/1 | −15.83 | +28.3% |
| small4/1 | −26.89 | +35.9% |
| small1/1 | −12.34 | +25.9% |
| small2/2 | −3.52 | +12.5% |
| small3/2 | −7.28 | +8.2% |
| small4/2 | −16.62 | +20.0% |

**Butteraugli better in 1 of 8. SSIMULACRA2 better in 0 of 8.**

The single BA win (small1/2) still loses 5.8 SSIMULACRA2. Under
`ssimulacra2-is-the-primary-promotion-metric` and Contract B (no BA regression
on the matrix), the gate fails. Full six-scene matrix not run.

Raw rows: `.agent/scratch/phase5p-exact-2026-08-12/phase5p-results.txt`.

## 5. Decision

**No operating point.** Exact sample-domain distortion is the right *currency*
for specials (Parseval is false) but is not a sufficient *selector* at this
encoder's rate proxy and DctSelect charge. The research mode is removed;
production stays square hierarchical cover only.

Wire vocabulary for same-8×8 specials stays open so a later scorer (map-cost,
perceptual residual, or entropy-aware rate) can emit them without re-proving
the writer.

## 6. What this does not claim

- It does not claim specials are useless — only that exact sample SSE + an
  8-bit global DctSelect proxy, under the current rate loop and quantizer, is
  not a promotion path.
- It does not re-open Phase 5O's coefficient-SSE special scorer.
- Cross-block DctSelect coupling and ANS-level rate remain unmodelled.

## 7. Continuation

Special-transform selection needs a better rate model or a perceptual residual
before another quality screen. Until then, the unexhausted quality levers are
elsewhere (entropy-aware rate in the quantizer path; modular 4C/4D density).
