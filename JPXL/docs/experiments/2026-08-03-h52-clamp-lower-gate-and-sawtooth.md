# H.5.2's clamp, second pass: the lower gate is wrong, and one sample refutes the model

Date: 2026-08-03
Status: **incomplete — the bug is not fixed.** No code shipped. This report
records new evidence, a refuted model, and two minimal reproducers.

Relationship to the earlier report:
`2026-08-03-h52-clamp-asymmetry.md` is frozen and its central finding stands
(the clamp is asymmetric; the printed symmetric clamp is contradicted by four
oracle samples). This report **narrows** that entry's §4.3 rule: the lower
gate it proposes is demonstrably too narrow. It does not supersede the earlier
entry, because the replacement rule derived here is itself refuted by §4
below. When a correct rule is found, that report should supersede both.

## 1. Question

Two open failures were believed to be one family:

* the "sawtooth trap" — a 32x32 grey `(x*7 + y*3) mod 256` source, `cjxl -d 0
  -e 3`, failing with `out of bounds: 16 bit(s) requested at bit position
  2112`;
* VarDCT fixtures 54 and 57, whose G.2.2 `LfQuant` modular sub-bitstream fails
  its own C.3.2 terminal-state check, blocking VarDCT acceptance.

Are they the same bug, and what is it?

## 2. Preregistered gate

* **Pass**: one uniform change after which the sawtooth and 54/57 decode, all
  eleven previously bit-exact lossless fixtures stay bit-exact, and no fixture
  is special-cased.
* **Fail**: any candidate contradicted by a stream that currently decodes.
* **Inconclusive**: a rule consistent with all evidence gathered but unable to
  decode a reproducer end to end. **This is the outcome.**

## 3. Method and raw results

### 3.1 They are the same family

Both failures are in Annex H channels whose MA tree selects the
`SelfCorrecting` predictor and branches on property 15 (`max_error`):

* fixture 60 (sawtooth): one 32x32 channel, tree = one decision node
  `property[15] > 0` with two `SelfCorrecting` leaves.
* fixture 61 / 54 / 57 (`LfQuant`): three channels, tree =
  `property[1] > 2 ? West : (property[15] > 0 ? … : …)`, both non-`West`
  leaves `SelfCorrecting`. Property 1 is the stream index, which is 1 for
  `LfQuant` of LF group 0, so the `West` branch is never taken.

So both exercise the H.5 weighted predictor with nothing else able to absorb
an error, and both are content-dependent through the size of `true_err`.

### 3.2 Minimisation (fixture 61)

Bisecting fixtures 54/57's 128x128 half-gradient/half-checkerboard source:

| variant | result |
| --- | --- |
| 128x128 checkerboard only | decodes |
| 128x128 gradient only, 644 bytes | **fails** |
| gradient at 16/32/48/64/72/80/88/96/104/112 | decodes |
| gradient at 120 / 128 / 136 | **fails** |
| 128x64, 64x128, 128x16, 16x128 | decodes |

The trigger is smooth content with **both** dimensions past a threshold
between 112 and 120 — i.e. `LfQuant` channels of 14x14 (works) versus 15x15
(fails). Structure either side of the transition is otherwise identical: same
tree, same three clusters, same LZ77-off ANS configuration.

### 3.3 Forensic technique

The sawtooth's true samples are known in closed form, which makes the probe
from the first clamp hunt exact: at each sample, clone the `SymbolDecoder` and
`BitReader`, decode once per MA leaf, and record which contexts reproduce the
known target.

**A correction to that technique, recorded because it cost real time.**
Greedily substituting `hits.first()` desynchronises the stream: two contexts
can decode to the same *value* while consuming different bits. The probe must
prefer the decoder's own context whenever it is viable and only substitute
when it is not. With the greedy version the sawtooth appeared to diverge at
(2,1); with the sticky version the true first divergence is (29,17).

### 3.4 The lower gate is too narrow

Harvesting every clamp decision that actually changes the prediction, over all
currently-passing fixtures (18 834 lower-binding samples), the decision is a
**pure function of the sign triple** `(sign(true_err_W), sign(true_err_N),
sign(true_err_NW))` — 18 distinct triples, zero contradictions. All 18 agree
with the shipped rule (`p1 < 0 || p2 < 0 || quiescent`).

The sawtooth adds one triple the shipped rule gets wrong. At (29,17):

```
true_err = (W -1, N -20, NW 0, NE 1790)     p1 = +20, p2 = 0
lo = min(W3,N3,NE3) = 16      unclamped = -16      required prediction ∈ [13,20]
```

Both contexts decode the same symbol there, so this is the prediction, not the
context. Clamping up to `lo = 16` gives estimate `(16+3)>>3 = 2` and the
correct sample 254. The shipped rule does not clamp.

The only harvested triple of the same shape, `(+5, +8, 0)`, must **not**
clamp. The two differ only in the common sign of W and N, giving the
refinement

```
lower fires iff  disagree && (p1 < 0 || p2 < 0 || quiescent || (te_W < 0 && te_N < 0))
```

which fits all 19 triples. Physically: `true_err = prediction − value`, so two
negative neighbour errors mean the predictor has been under-shooting, and a
further downward extrapolation is not trusted.

Measured effect: all eleven lossless fixtures stay bit-exact, and the sawtooth
improves from *206 wrong samples plus an entropy blow-up at bit 2112* to
**one wrong sample**. It does not fix fixtures 54/57/61.

### 3.5 The sample that refutes the model

The sawtooth's last wrong sample is (31, 19), the last column of row 19:

```
neighbours  W 11  N 15  NW 8  NE→N 15  (last column)      values are correct
true_err    (-56, -56, -56, -56)        p1 = p2 = +3136 > 0  → no clamp at all
unclamped = weighted prediction = 142   lo = 88   hi = 120
```

The encoder's context there is 0, which requires `property[15] > 0`; our
`max_error` is −56. Two escapes exist and **both are refuted**:

* **(A) `max_error` is wrong.** Impossible: all four candidates are −56, so no
  selection rule, walk order or tie-break yields a positive value. Solving for
  the tree threshold over the 639 samples decoded before the divergence gives
  `T < −56` and `T ≥ 0` simultaneously — infeasible. Alternative definitions
  (previous sample's `max_error`, dropping NE, ties-to-last) are each
  infeasible on the same data.
* **(B) the prediction is wrong** and should be ≈120 = `hi`, making context 1
  correct. Achievable only by firing the *upper* cap where the printed guard
  says not to. Three variants were tried — `both_under` on the upper half,
  all-three-negative, all-four-negative — and each breaks between 5 and 11
  currently-bit-exact lossless fixtures. Decisively refuted.

The three neighbour predictions are each pinned to an 8-wide window by their
own correct decodes, and every value in every window is negative, so
`max_error ∈ [−59, −52]` under any correct decoder. Something outside the
model of "H.5.2 clamp gating plus `max_error` selection" is wrong.

### 3.6 Eliminated, with the evidence

* **`err_sum` last-column term** (`EXPERIMENT_ERR_SUM_LAST_COLUMN`). Swept
  0/1/2 against the refined lower gate: the sawtooth fails identically at
  (31,19) in all three; variant 2 additionally breaks 8 fixtures. The failing
  sample *is* at a last column, so this was the strongest structural lead and
  it is now dead.
* **A wider clamp bound set.** Adding `NW3` to `min(W3,N3,NE3)` was tested
  against the 18 834 harvested decisions: 5 545 samples that currently must
  *not* clamp would clamp, and 31 that do clamp would clamp to a different
  value. Refuted.
* **A symmetric clamp with corrected weights.** Refuted independently: in
  thousands of harvested samples the unclamped prediction is verified correct
  while sitting outside `[lo, hi]`, so no weight correction can rescue a
  symmetric clamp.
* **Bit depth** (already eliminated by 8C, re-confirmed): probing
  `bits_per_sample` 1..32 leaves fixture 54 failing identically.
* **Content** for fixture 61: high-frequency content decodes; smooth content
  fails. The reverse of the usual expectation.

## 4. Conclusion

The sawtooth and the `LfQuant` failures are the **same family** — the H.5
weighted predictor on `SelfCorrecting` channels branching on property 15 —
but this report does **not** establish they are the same defect, because the
sawtooth's remaining sample is refuted by evidence that the `LfQuant` failure
has not yet been traced to.

What is now established:

1. The shipped lower gate is **too narrow**; §3.4 gives a strictly better rule
   backed by 18 834 harvested decisions plus one new triple, with no
   regressions. It was **not shipped**, because it fixes no fixture end to end
   and adding a second unexplained epicycle to a normative clause without a
   passing reproducer is exactly the trap the first clamp entry fell into.
2. The model "asymmetric clamp gating + `max_error` selection" is
   **incomplete**: sample (31,19) of fixture 60 cannot be explained inside it.
3. Two minimal reproducers now exist as checked-in fixtures, 60 (277 bytes)
   and 61 (644 bytes), replacing hand-rebuilt repros.

What this does **not** establish: it says nothing about which clause is
misread. The next session should start at fixture 60 (31,19) and at fixture
61's 112→120 transition, and should treat the `err[i]`/`err_sum` weight
computation and the `NE`-substitution at the last column as the two surfaces
not yet excluded — the clamp gating itself has now been mapped exhaustively
against 18 834 samples and is not sufficient on its own.

## 5. Consequences

* No source change. `crates/jpxl-decode/src/modular/**` is byte-identical to
  the state before this investigation.
* New: `tools/make-debug-fixtures.sh` grew fixtures 60 and 61 (and an
  `encode_lossy` helper, plus a fix so a grey source is round-trip-checked as
  PGM rather than PPM).
* New: fixtures 60/61 with provenance sidecars.
* `tests/e2e_lossless.rs` carries both as `#[ignore]`d forensic tests.
* 8C's two ANS-gate tests in `vardct/hf_coeff.rs` stay `#[ignore]`d.
