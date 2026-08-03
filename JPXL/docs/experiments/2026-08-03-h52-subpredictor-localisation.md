# H.5.2: the defect is in the sub-predictors, not the weights, clamp or max_error

Date: 2026-08-03
Status: **localised, not fixed.** No code shipped.

Supersedes the "next surfaces" guidance of
`2026-08-03-h52-clamp-lower-gate-and-sawtooth.md` (which named `err[i]`/
`err_sum` and the last-column `NE` substitution). Both of those are now
**eliminated**, along with three more layers. That report's other findings
stand; this one replaces its §4 next-steps.

## 1. Question

After two passes fitting H.5.2's clamp gating, which layer of the weighted
predictor is actually wrong?

## 2. Preregistered gate

* **Pass**: a layer identified such that no setting of the *other* layers can
  produce the encoder's behaviour — i.e. an impossibility proof, not a fit.
* **Fail**: another rule that fits the evidence but is refuted by a later
  sample (the outcome of the previous two passes).

## 3. Method

### 3.1 A better reproducer

The known repros were fixture 60 (sawtooth, `SelfCorrecting` leaves) and 61
(VarDCT `LfQuant`). Bisecting fixtures 54/57's source **losslessly** produced
a third and much better one, now checked in as fixture 62: 24x24, 305 bytes,
`cjxl -d 0 -e 7`.

Minimality is established in both axes: 20/21/22/23 decode bit-exactly and 24
fails; and at 24x24, checkerboard-only, gradient-only, flat-right-half,
black/white and greyscale variants all decode bit-exactly. Both halves and the
1-pixel period are required.

Unlike fixtures 60 and 61 this one fails on a channel whose MA leaves are
`Gradient`, which is what makes the impossibility argument below available:
the residual and the neighbourhood are then both known exactly.

### 3.2 Establishing that the entropy stream is still synchronised

Fixture 62's coded image is three 1-row palettes (36/24/12 entries) plus three
24x24 index channels, with 11 leaves over 5 entropy clusters. All three
palettes and all 576 samples of the first index channel decode bit-exactly.

For every sample up to the divergence, all 11 contexts were probed (clone the
`SymbolDecoder` and `BitReader`, decode once per leaf) and the set of clusters
that reproduce the known-correct sample recorded:

* **233 samples had exactly one viable cluster** — a forced checkpoint — and
  our decoder chose it every time;
* 474 were ambiguous (several clusters give the same value);
* our chosen cluster was never outside the viable set.

Passing 233 consecutive forced checkpoints makes an earlier desynchronisation
untenable. This matters because the previous two passes were repeatedly misled
by desyncs masquerading as decode errors.

### 3.3 The impossibility proof

The first sample no context can produce is index channel 2 at **(11, 5)** —
the last column of the gradient half, whose `NE` neighbour is the first
checkerboard sample.

```
neighbours   W 5  N 4  NW 4  NE 23      →  W3 40  N3 32  NW3 32  NE3 184
true_err     (W 0, N -32, NW -15, NE -120)
err_sum      [7, 35, 29, 34]
subpred      [192, 108, 55, 69]
our prediction 151   (weights [13,2,3,2] after normalisation)
```

The correct sample is 5. Every `Gradient` leaf predicts
`clamp(W+N-NW, min, max) = 5`, and no context decodes a residual of 0. Every
`SelfCorrecting` leaf decodes a residual of 0, so each requires
`(prediction + 3) >> 3 == 5`, i.e. **prediction ∈ [37, 44]**.

That value is unreachable:

* any weighted average of `subpred` with non-negative weights lies in
  **[55, 192]** — the minimum sub-prediction is 55;
* the clamp can only additionally yield `min(W3,N3,NE3) = 32` or
  `max(W3,N3,NE3) = 184`.

`[37,44] ∩ ([55,192] ∪ {32,184}) = ∅`.

So **no** choice of weights, `err_sum` terms, clamp gating, clamp bounds drawn
from `{W3, N3, NE3}`, or `max_error` rule can produce the encoder's
prediction. The error is upstream of all of them: in the four sub-predictor
formulas of H.5.2, or in the `true_err` values they consume.

### 3.4 Fresh re-derivation of the transcription

The whole H.5 state-update block was re-read from `latex/part1.tex` from
scratch and diffed line by line against `modular/weighted.rs` and
`modular/predictor.rs`:

* Table H.2's seven edge rules (`W`, `N`, `NW`, `NE`, `NN`, `NEE`, `WW`) match
  `Neighbours::gather` exactly, including `W`'s two-level fallback at the
  origin and `NEE`'s fallback to `NE`.
* H.5.2's *error* edge rules ("W, N or WW → 0; NW or NE → the value of N",
  with the explicit rightmost-border example) match the four `true_err_*` and
  five `err_*` accessors exactly.
* `err_sum`'s five terms, the last-column `+= err[i]_W`, `error2weight`, the
  `log_weight` normalisation, `s = (sum_weights >> 1) - 1` and the final
  `Idiv`/`>> 24` all match.
* `part1.md` is too garbled here to serve as an independent check
  (`err[i] Hm` for `err[i]_NW`, `err(i] il)` for the last-column term), so the
  LaTeX stands alone — but nothing in it is ambiguous at this level.

The transcription is therefore not the defect, which is what makes §3.3's
conclusion actionable: one of the four `subpred` expressions is being
evaluated with the wrong operands, not typed in wrongly.

## 4. Conclusion

Eliminated, each with evidence rather than by argument:

| layer | how eliminated |
| --- | --- |
| `max_error` selection | §3.3 — no rule changes the prediction |
| clamp gating (both halves) | §3.3 — no gate reaches [37,44] |
| clamp bounds from `{W3,N3,NE3}` | §3.3 — only 32 and 184 are reachable |
| `err[i]` / `err_sum` / weights | §3.3 — every weighting lies in [55,192] |
| last-column `NE` substitution | §3.4 — matches the clause exactly |
| Table H.2 sample edge rules | §3.4 — match the clause exactly |
| transcription error | §3.4 — fresh independent re-read |

What remains: the four `subpred` expressions of H.5.2 and the `true_err`
values feeding them. Note `subpred[0] = W3 + NE3 - N3 = 192` is the outlier
here and carries 13 of the 20 normalised weight, precisely because `NE` is the
first checkerboard sample; the required prediction (~40) is close to
`W3 = 40`, which is what `subpred[0]` would be if `NE3` were replaced by `N3`.
That is a *hypothesis for the next session*, not a finding: substituting it
gives 52 (estimate 6), not 5, so it is not sufficient on its own.

What this does not establish: nothing about which of the four sub-predictors
is wrong, and nothing about fixtures 60/61 beyond family membership — their
divergences have not been re-derived under this framing.

## 5. Consequences

* No source change. `crates/jpxl-decode/src/modular/**` and
  `crates/jpxl-entropy/**` are byte-identical to their pre-investigation state.
* New fixture 62 (305 B, lossless) + provenance sidecar, via
  `tools/make-debug-fixtures.sh`.
* `tests/e2e_lossless.rs` carries it as a third `#[ignore]`d forensic test.
* 8C's two ANS-gate tests stay `#[ignore]`d.

---

**ADDENDUM 2026-08-03 (later the same day).** The clamp conclusions of this
report are SUPERSEDED by
`2026-08-03-h52-clamp-xor-scan-resolution.md`. H.5.2's guard is
`((true_err_N ^ true_err_W) | (true_err_N ^ true_err_NW)) <= 0` — **XOR, not
multiplication**. Every text rendition in this repository misread `^` as `*`,
which is what made the clause look defective. The published standard is
correct and there is no asymmetric clamp; read as XOR, one symmetric clamp
explains every sample cited here. The *evidence* recorded below stands and was
what made the resolution checkable — only the conclusions drawn from it are
withdrawn.

Its §3.3 impossibility proof is also withdrawn: the `true_err` state it used
had evolved under the wrong clamp, so `subpred` was computed from poisoned
inputs. The sub-predictor formulas are correct.
