# H.5.2's clamp is not one symmetric clamp behind one guard

Date: 2026-08-03
Status: complete. Supersedes the slice-7 diagnosis recorded in
`HANDOFF.md` ("fixtures 05, 09, 10 diverge in H.5.2 `max_error` selection"),
which is **corrected here: `max_error` was never wrong.**

## 1. Question

`jpxl_decode::decode` was bit-exact against `djxl` on six of the nine handmade
lossless-modular fixtures. Fixtures 05, 09 and 10 diverged. Slice 7 attributed
the divergence to H.5.2's `max_error` walk, having shown that fixture 10
*requires* `max_error = -1896` at a sample where the clause's
`abs(x) > abs(max_error)` walk yields `+2040`.

Question: what single reading of Annex H decodes all nine fixtures bit-exactly?

## 2. Preregistered gate

* **Pass**: one change, uniform across all fixtures and all samples, after
  which `cargo test --workspace` is green with the three `#[ignore]`s removed,
  `matches_djxl_output` compares every fixture sample-for-sample against
  `djxl`, and no fixture is special-cased anywhere in the decoder.
* **Fail**: any candidate that needs a per-fixture branch, a tolerance, or a
  rule that a passing fixture contradicts.
* **Inconclusive**: a rule that fits the nine fixtures but is contradicted by
  a newly generated `cjxl` stream.

Two hypotheses were preregistered as most likely, both from the slice-7
handoff: (a) Table H.4 property numbering, (b) H.5 weighted-predictor state
for channels with `hshift`/`vshift` of `-1`. **Both were eliminated** (§5).

## 3. Method

Host: Linux 6.17.0-40-generic, x86-64.
Oracle: `JPXL/tools/oracle-bin/cjxl` and `djxl`, JPEG XL v0.13.0 196a43d9
(libjxl rev `196a43d996aa6ed33ebf98812a7c6d43b2b6d01b`), as pinned in
`tools/oracle-bin/PINNED_REVISIONS.txt`. libjxl source was not read; both
binaries were used only to encode and to decode.

Four techniques, in order:

1. **Instrumented trace.** A temporary `JPXL_DBG` env-gated `eprintln!` in
   `modular::decode_channels` dumped, per sample: the four neighbouring
   `true_err`, the full property vector, the MA leaf, the raw symbol, the
   clamped and unclamped weighted prediction, and the decoded value. The
   instrumentation was removed before commit; `modular/mod.rs` is unchanged.
2. **Context recovery by ANS branching.** For fixture 10's 4x3 palette
   meta-channel, a temporary probe cloned the `SymbolDecoder` and `BitReader`
   at every sample, decoded once per MA-tree leaf, and kept the leaves whose
   result matched a target value; a depth-first search over the resulting
   choice tree enumerated *every* context sequence consistent with a candidate
   palette. Run over all six orderings of the image's four colours, exactly
   one ordering and exactly one context sequence survived, which makes the
   encoder's context sequence a measured fact rather than a guess.
3. **Constraint harvesting.** For the six already-passing fixtures the decode
   is known-correct, so every MA-tree decision on property 15 is a constraint
   `(true_err tuple, threshold) -> required outcome`. 228 031 such tests over
   152 distinct `true_err` tuples were extracted from the traces and used to
   test candidate rules offline.
4. **Minimisation with new `cjxl` fixtures.** Synthetic sources were generated
   and encoded to find smaller streams with the same failure, and to find
   streams that isolate the weighted predictor (a palette meta-channel whose
   MA leaf is `SelfCorrecting` reads the prediction out directly). The two
   decisive ones are now `tests/fixtures/handmade/20_…` and `21_…`, produced
   by `tools/make-debug-fixtures.sh`.

## 4. Raw results

### 4.1 The `max_error` walk cannot be repaired

Fixture 10's meta-channel is 4 wide, 3 tall, palette
`(255,255,255) (0,114,255) (237,28,36) (0,0,0)`; its MA tree is one node,
`property[15] > -255`. Recovered context sequence (technique 2, unique):

```
[0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, *]
```

so property 15 must be `<= -255` at samples 1, 3, 4, 5, 6, 7, 10 and `> -255`
at 0, 2, 8, 9. Under the slice-7 code the `true_err` tuples `(W, N, NW, NE)`
were:

| sample | (x, y) | true_err (W, N, NW, NE) | required |
| --- | --- | --- | --- |
| 5 | (1,1) | (-132, +2040, -2040, -1896) | <= -255 |
| 6 | (2,1) | (+625, -1896, +2040, +1896) | <= -255 |
| 7 | (3,1) | (+578, +1896, -1896, +1896) | <= -255 |
| 8 | (0,2) | (0, -132, -132, +625) | > -255 |

Sample 6 is the killer: `+2040` is the unique largest magnitude, so **no**
magnitude rule, walk order or tie-break can return a value `<= -255`. A brute
force over all 24 orders x 9 comparison predicates x all 15 non-empty subsets
of the four candidates, scored against the 152 harvested tuples plus these
eleven, returned **zero** consistent rules; the best near-misses failed on
exactly this sample. A plain signed minimum fits all of fixture 10 and fails
9 samples of fixtures 11 and 12 (three tuples: `(0,4,-1,-4)`,
`(0,-8,1108,-8)`, `(0,1108,-3,-8)`, all with threshold `-4`).

Conclusion: the walk is not the free variable. One of the `true_err` inputs is
wrong, and `true_err` is `prediction - (value << 3)`, so the *prediction* is
wrong.

### 4.2 Four samples that pin the clamp

Each row below is a sample where the H.5.2 clamp changes the prediction and
where the required prediction is known independently — either because the MA
leaf is `SelfCorrecting` (so the decoded value reads the prediction out) or
because the property-15 comparison downstream is forced.

| fixture | sample | true_err (W, N, NW) | (lo, hi) = min/max(W3,N3,NE3) | unclamped | required | so |
| --- | --- | --- | --- | --- | --- | --- |
| 08 | ch0 (2,0) | (-8456, 0, 0) | (8456, 8456) | 10092 | 8456 | upper bound **applies** |
| 20 | ch0 (3,0) | (+864, 0, 0) | (576, 576) | 400 | 400 | lower bound **does not** |
| 10 | ch0 (2,0) | (+2040, 0, 0) | (0, 0) | -346 | -346 | lower bound **does not** |
| 21 | ch2 (251,5) | (0, 0, 0) | (-32, 2008) | -40 | -32 | lower bound **applies** |

Rows 1 and 2 have identical printed guard products (`true_err_N` is 0, so both
products are 0) and differ only in the sign of `true_err_W`, yet require
opposite outcomes. Since the printed guard is a function of those two products
alone, **no reading of "one guard, one symmetric clamp" fits**. Rows 2 and 4
show the same for the lower bound alone: all-zero errors clamp, a nonzero
`true_err_W` with zero `true_err_N` does not.

Supporting derivations (independent of the trace):

* fixture 20 (3,0): `W = 72`, so `W3 = N3 = NE3 = 576`; `subpred` is
  `[576, 144, 306, 576]`, `err_sum` is `[180, 162, 168, 180]`, normalised
  weights `[4, 4, 4, 4]`, giving `prediction = 400` and sample estimate
  `(400 + 3) >> 3 = 50`. The decoded residual is `+94` and the true palette
  entry is `144 = 94 + 50`. The clamped 576 gives `72`, hence `166`, which is
  not one of the six colours in the source.
* fixture 08 (2,0): `subpred` is `[8456, 12684, 11099, 8456]` with equal
  `err_sum` 1057, so the unclamped prediction is 10092 and the estimate 1261;
  the palette is the ramp `k * 1057`, and only the clamped 8456 (estimate
  1057, value 2114) lands on it.

### 4.3 The rule that fits everything

```text
p1 = true_err_N * true_err_W
p2 = true_err_N * true_err_NW
lo = min(W3, N3, NE3);  hi = max(W3, N3, NE3)

if (p1 <= 0 || p2 <= 0)                                   prediction = min(prediction, hi)
if (p1 <  0 || p2 <  0 || (N == 0 && W == 0 && NW == 0))   prediction = max(prediction, lo)
```

i.e. the printed guard gates the upper half; the lower half needs a *strict*
sign disagreement, or the degenerate case where the three neighbouring errors
carry no information at all.

Variants tried and rejected, each against all nine fixtures plus seven
`cjxl`-generated probes:

| variant | fixtures failed |
| --- | --- |
| printed clause, guard `(p1 \| p2) <= 0` | 05, 08, 09, 10, 13, 20 |
| printed clause, guard `p1 <= 0 \|\| p2 <= 0` (slice 7) | 05, 09, 10, 20 |
| printed clause, guard `p1 < 0 \|\| p2 < 0` | 05, 08, 09, 10, 11, 12, 13 |
| upper bound only, printed guard | 05, 11 |
| bounds widened to include 0 | 05, 09, 10 |
| upper always, lower on printed guard | 05, 08, 09, 10, 11, 12, 13 |
| upper on printed guard, lower on strict only | 05 |
| upper on printed guard, lower off on row 0 | 05, 09 |
| symmetric, guard `strict \|\| (N == 0 && W <= 0 && NW <= 0)` | 05, 09, 10, 20 |
| **§4.3 above** | **none** |

The `true_err`-from-clamped-vs-unclamped flip-point and the three `err_sum`
last-column readings were re-swept against §4.3 as a cross-product; clamped
`true_err` and the LaTeX's `+= err[i]_W` remain the only settings that pass.

## 5. Conclusion

**`max_error` (H.5.2) is the clause as written** —
`abs(x) > abs(max_error)`, walked W, N, NW, NE, strict `>` so ties keep the
earlier candidate. The slice-7 report that fixture 10 needs a plain minimum is
**withdrawn**: it was measuring a corrupted `true_err`, not a wrong walk.

**H.5.2's clamp, as printed, is contradicted by the oracle.** The upper and
lower halves of `clamp(prediction, min(W3,N3,NE3), max(W3,N3,NE3))` are gated
by different conditions (§4.3). Two `cjxl` streams that agree on every input
the printed guard can see require opposite outcomes, so this is not a reading
of the clause — it is a divergence between the clause and libjxl 0.13.0.

What this does **not** establish:

* It does not establish what the normative text intends. This is an oracle
  experiment: it describes libjxl 0.13.0. The decision is tagged
  `[provisional]` at `EXPERIMENT_CLAMP_SYMMETRIC`.
* It does not establish that §4.3 is libjxl's actual formulation, only that it
  agrees with libjxl on every sample of eleven streams (≈ 470 000 samples).
  The `quiescent` disjunct in particular is pinned by one fixture family; a
  cleaner equivalent may exist and would decode identically.
* It says nothing about clamping behaviour for `bits_per_sample > 16`, for
  channels with positive `hshift`/`vshift` (squeeze), or for VarDCT streams —
  none are exercised here.

Hypotheses eliminated, with the evidence:

* **Table H.4 property numbering.** For fixture 10's meta-channel every
  property was evaluated on the *correct* decode (technique 2) and tested
  against every threshold: no property other than 15 can produce the recovered
  context sequence, and property 15 with any threshold cannot either under the
  slice-7 `true_err`. The numbering is right; the input was wrong.
* **Weighted-predictor state for `hshift = vshift = -1` channels.** Fixture 21
  fails identically at a plain `hshift = vshift = 0` colour channel, and
  fixture 20's meta-channel decodes bit-exactly once the clamp is fixed, with
  no shift-dependent branch anywhere.
* **Per-image versus per-channel weighted state.** Fixture 10's divergence is
  in the first channel of the first stream, where the two readings coincide.
* **`err_sum` last-column term.** All three readings were swept before and
  after the clamp change; the LaTeX's `+= err[i]_W` is the only one that
  passes, and only after the clamp change (it was neutral before).
* **`true_err` from the unclamped prediction.** Swept as a cross-product with
  every guard and bounds variant: 5 of 13 tests pass at best.

## 6. Consequences

* `crates/jpxl-decode/src/modular/weighted.rs`: the clamp is implemented as
  §4.3 behind the new flip-point `EXPERIMENT_CLAMP_SYMMETRIC` (set `true` to
  restore the literal clause). `EXPERIMENT_CLAMP_BITWISE_OR` is gone — it
  asked which of two symmetric readings applies, and the answer is neither.
  `EXPERIMENT_MAX_ERROR_RULE` stays 0 and is now **resolved** rather than
  "partially resolved with a known contradiction".
* `crates/jpxl-decode/tests/e2e_lossless.rs`: the three `#[ignore]`s are
  removed; fixtures 05, 09 and 10 join `BIT_EXACT_FIXTURES`, so
  `matches_djxl_output` now compares all eleven against `djxl`.
* `tools/make-debug-fixtures.sh` (new) and fixtures
  `20_modular_palette_bands_24x24_lossless.jxl`,
  `21_gradient_260x10_lossless.jxl` with provenance sidecars.
