# H.5.2's clamp guard is XOR: one scan read closes three sessions of chasing

Date: 2026-08-03
Status: **resolved.** Supersedes the clamp conclusions of
`2026-08-03-h52-clamp-asymmetry.md`,
`2026-08-03-h52-clamp-lower-gate-and-sawtooth.md` and
`2026-08-03-h52-subpredictor-localisation.md` (each carries a dated addendum;
their frozen bodies are unchanged).

## 1. Question

Three investigations had concluded, from oracle evidence, that H.5.2's clamp
was *defective as published* — first that its guard could not be read
literally, then that its two halves must be gated differently, then that the
sub-predictors themselves must be wrong. Each fit the evidence available and
each was refuted by the next fixture. What does the clause actually say?

## 2. Method

The document-access order of `AGENTS.md` §2 was followed to its last resort:
a page-ranged read of the **original image scan** of Part 1, printed page 50
(scan page 54), by the orchestrator.

## 3. Raw result

The published clause reads:

```text
// if true_err_N, true_err_W and true_err_NW don't have the same sign
if (((true_err_N ^ true_err_W) | (true_err_N ^ true_err_NW)) <= 0) {
    prediction = clamp(prediction, min(W3, N3, NE3), max(W3, N3, NE3));
}
```

The operator is **`^` (XOR), not `*`**. All three text renditions available in
this repository — `latex/part1.tex`, `markdowns/standard-markdowns/part1.md`
and the text-only transcription PDF — render it as `*`. They are not
independent: they all derive from the same scan, and a caret between two
subscripted identifiers is exactly the glyph that degrades into an asterisk.

## 4. Why XOR makes the clause exactly its own comment

For two's-complement integers:

* `a ^ b` has its sign bit set **iff** the sign bits of `a` and `b` differ;
* `x | y` has its sign bit set iff either operand does;
* `x | y == 0` iff both are zero, i.e. `N == W` and `N == NW` exactly.

So the guard fires iff the signs of `N` and `W` differ, **or** the signs of
`N` and `NW` differ, **or** all three errors are bit-identical. That is
precisely "don't have the same sign", plus the degenerate all-equal case.
Under the `*` misreading the same expression becomes a product whose sign
carries the same information only when no operand is zero — and the
zero cases are exactly where every previous model broke.

Verification against every sample the three prior reports had pinned:

| fixture | `true_err (W, N, NW)` | XOR guard | required | ✓ |
| --- | --- | --- | --- | --- |
| 08 (2,0) | (−8456, 0, 0) | `0^−8456 < 0` → fires | clamp to `hi` | ✓ |
| 20 (3,0) | (+864, 0, 0) | `0^864 > 0`, `0^0 = 0`, OR > 0 → no | leave alone | ✓ |
| 10 (2,0) | (+2040, 0, 0) | OR > 0 → no | leave alone | ✓ |
| 21 (251,5) | (0, 0, 0) | both XORs 0 → fires | clamp to `lo` | ✓ |
| 60 (31,19) | (−56, −56, −56) | both XORs 0 → fires | clamp 142 → `hi` 120 | ✓ |

The last row is the sample that refuted the asymmetric model. It is also why
the 18 834-sample harvest in the second report found the decision to be "a
pure function of the sign triple with zero contradictions": it *is* sign
arithmetic — the harvest had measured the right invariant and drawn the wrong
conclusion from it, because a product cannot express "all three identical".

The `subpredictor-localisation` impossibility proof dissolves for the same
reason: its `true_err` state had evolved under a wrong clamp, so the
`subpred` values it proved things about were computed from poisoned inputs.

## 5. Implementation

`modular/weighted.rs` now contains one symmetric clamp behind the literal
guard, and nothing else:

```rust
let prediction = if ((te_n ^ te_w) | (te_n ^ te_nw)) <= 0 {
    prediction.clamp(w3.min(n3).min(ne3), w3.max(n3).max(ne3))
} else {
    prediction
};
```

**Arithmetic width.** H.5.1 narrows `true_err` to 32 bits; this decoder holds
those values sign-extended in `i64`. Sign extension preserves both the sign
bit and zero-ness, so the `i64` XOR has the same sign and the same zero-ness
as the `i32` XOR the clause describes. The wider evaluation is exact, not
merely close, and the choice is documented at the call site.

The `EXPERIMENT_CLAMP_SYMMETRIC` flip-point is **deleted** — the scan is
decisive, so there is nothing left to flip. It is replaced by a doc note,
`_H52_CLAMP_GUARD_IS_XOR`, recording the misreading so the next reader of the
LaTeX does not "fix" the code back.

A grep across `modular/**` for any other guard transcribed as a product
between `true_err` terms found exactly one more, in the `replay_predict` test
helper, which is corrected to XOR as well. No other clause in the module
compares `true_err` values multiplicatively.

## 6. Result

| fixture / suite | before | after |
| --- | --- | --- |
| 11 lossless (03/05/07–13/20/21) | bit-exact | **bit-exact** |
| multisection 14–16 | pass | **pass** |
| 60 sawtooth (277 B) | 1 wrong sample | **bit-exact** |
| 61 VarDCT LfQuant (644 B) | C.3.2 failure | **decodes** |
| 62 mixed 24×24 (305 B) | wrong from (11,5) | **bit-exact** |
| VarDCT 54 / 57 — 8C ANS gate | `#[ignore]`d | **pass** |
| `e2e_lossless` | 16 pass, 3 ignored | **19 pass, 0 ignored** |

All three forensic tests became regression tests, and both of 8C's gate tests
were un-ignored.

## 7. Consequences, and the lesson

* Two claimed "defects in the published standard" — the nonsensical bitwise-OR
  guard and the asymmetric clamp — were **OCR artifacts**. The standard is
  correct. Any ledger or doc text asserting otherwise is wrong and is
  corrected in place.
* `AGENTS.md` §2's resolution order works, but only if it is actually followed
  to the end. Three sessions treated "all available text sources agree" as
  strong evidence. They are not independent: they share one scan and one class
  of glyph confusion. **When multiple transcriptions agree on something that
  reads as nonsense, that is evidence about the transcription pipeline, not
  about the standard** — escalate to the image scan rather than modelling the
  nonsense.
* The oracle evidence gathered along the way was never wrong; only the
  hypotheses fitted to it were. The 18 834-sample harvest and the fixture
  ladder are what made the scan reading immediately checkable, and they are
  why this took one commit to verify rather than another session.
