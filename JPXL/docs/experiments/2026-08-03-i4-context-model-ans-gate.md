# I.4's context model, proved by ANS exhaustion — and what that proof misses

Date: 2026-08-03
Sub-slice: 8C (coefficient order + HF coefficient decode)
Code: `jpxl-decode/src/vardct/{order.rs, hf_coeff.rs}`

## 1. Question

Two questions, one experimental and one textual.

**(a)** Does JPXL's implementation of 18181-1 I.4 — the `BlockContext` /
`NonZerosContext` / `CoefficientContext` model, the Y-X-B channel loop, the
`prev` rule and the `non_zeros` bookkeeping — reproduce a real encoder's
context sequence exactly, with no reference coefficients available?

**(b)** Which parts of I.3.1 and I.4 does that proof *not* reach, so that a
later sub-slice knows what is still resting on a reading of the text alone?

## 2. Preregistered gate

An entropy-coded stream in ANS form must end in the single terminal state
C.3.2 names. The decoder's context choice determines which alphabet each
symbol is drawn from, so a wrong context desynchronizes the arithmetic
decoder and the final state is wrong with overwhelming probability. The gate,
fixed before running:

* **Pass** — for every available `kVarDCT` stream, `SymbolDecoder::finish()`
  succeeds after the last HF coefficient of every `PassGroup` section, and
  fewer than 8 bits (byte padding only) remain unread in the section.
* **Fail** — any final-state mismatch, or whole bytes left unread.
* **Inconclusive** — the stream never reaches I.4 (blocked in an earlier
  clause), which is recorded separately and is not counted either way.

A second, adversarial gate was preregistered for question (b): deliberately
mutate one decision in the implementation, re-run, and record whether the
first gate catches it. A decision the gate cannot catch is not proved by it.

## 3. Method

Test harness: `fixture_gate` inside
`crates/jpxl-decode/src/vardct/hf_coeff.rs`. It parses signature, image
headers, an optional ICC profile, `FrameHeader`, TOC, `LfGlobal` (G.1.2 +
the three `kVarDCT` rows + G.1.3's leading `Bool()`), each `LfGroup` (G.2.2
`LfQuant`, G.2.4 `HfMetadata` and its greedy placement), `HfGlobal` (I.2.4,
I.2.6, then I.3's `hf_pass[]`), and finally every `PassGroup` section through
`decode_hf_group`. It stops at quantized integers: no dequantization, no
IDCT, no pixels, no reference data of any kind.

Inputs:

* handmade fixtures 50–57 (`cjxl` v0.13.0 196a43d9, 128x128, gray and RGB,
  filters on/off, `-d 1` and `-d 4`; provenance in each `.jxl.txt` sidecar);
* the official conformance corpus at the pinned commit, all 39 test cases
  (gitignored — the corpus tests skip when it is not fetched).

Run: `cargo test -p jpxl-decode --lib vardct::hf_coeff::tests::fixture_gate
-- --nocapture --test-threads=1`.

## 4. Raw results

### 4.1 First gate

| stream | varblocks | DctSelect histogram (value, count) | `used_orders` | unread bits |
| --- | --- | --- | --- | --- |
| 50 gray nofilters d1 | 10 | (5,8) (18,2) | 0 | 6 |
| 51 gray nofilters d4 | 14 | (5,4) (10,8) (18,2) | 0 | 6 |
| 52 gray filters d1 | 25 | (0,2) (2,2) (4,1) (5,11) (6,4) (10,3) (11,1) (19,1) | 0 | 1 |
| 53 gray filters d4 | 22 | (4,2) (5,5) (10,12) (11,1) (19,2) | 0 | 6 |
| 55 rgb nofilters d4 | 11 | (5,7) (10,2) (18,2) | 0 | 3 |
| 56 rgb filters d1 | 21 | (5,5) (10,14) (19,2) | 0 | 5 |
| corpus `grayscale` | 82 | — | 20 | 0 |
| corpus `grayscale_5` | 82 | — | 20 | 0 |

All eight **pass**. Every one of these frames is a single TOC section
(F.3.1), so `LfGlobal`, the `LfGroup`, `HfGlobal` and the `PassGroup` are one
consecutive bit stream: the "fewer than 8 unread bits" column is a
whole-frame bit-position check, not just an HF one.

DctSelect coverage across the set: 0 (DCT8x8), 2 (DCT2x2), 4 (DCT16x16),
5 (DCT32x32), 6 (DCT16x8), 10 (DCT32x16), 11 (DCT16x32), 18 (DCT64x64),
19 (DCT64x32) — six distinct Order IDs including four *non-square*
transforms, whose coefficient arrays are landscape while their footprints are
not.

Inconclusive (never reach I.4):

| stream | blocked at |
| --- | --- |
| 54 rgb nofilters d1, 57 rgb filters d4 | G.2.2 `LfQuant`: the modular sub-bitstream fails its **own** C.3.2 check (`Modular(Entropy(Malformed(...)))`) |
| 16 corpus cases | extra channels — `GlobalModular` then carries a real channel list, which this harness does not build |
| 9 corpus cases | not `kVarDCT` |
| 2 corpus cases | `jpeg_upsampling != 0` (subsampling, out of slice-8 scope) |
| 8 corpus cases | other Annex H / G.1 failures upstream of I.4 |

Fixtures 54 and 57 are worth calling out: 55 and 56 are the same encoder,
same geometry, same source content, the other two (distance, filters)
combinations, and they pass. So the 54/57 failure is content-dependent
Annex H behaviour, of the same family as the open modular sawtooth bug, and
not a VarDCT issue. `bits_per_sample` was ruled out by probing 1, 8, 16, 24
and 32 — all fail identically.

### 4.2 Second gate (mutation testing)

One decision changed at a time, gate re-run over all eight streams:

| mutation | caught? |
| --- | --- |
| channel loop X, Y, B instead of Y, X, B | **yes** — 8/8 fail |
| drop `BlockContext`'s `c ^ 1` swap | **yes** — 8/8 fail |
| invert the `k == num_blocks` seed of `prev` | **yes** — 8/8 fail |
| walk the LF thresholds 1, 2, 1 instead of 0, 2, 1 | **no** — 8/8 still pass |
| invert the I.3.1 permutation composition | **no** — 2/2 corpus cases still pass |

## 5. Conclusion

**(a) Answered, positively.** The context model reproduces a real encoder's
context sequence bit-exactly on every stream that reaches it, including
streams that use six Order IDs, four non-square transforms, and (in the two
corpus cases) I.3.1's shared permutation stream. Because the frames are
single-section, the result also pins the bit position of every field from
`LfGlobal` to the end of the frame.

**(b) Two decisions are outside the proof's reach, for structural reasons,
and one is outside it for want of a stream.**

1. **The order table's direction is invisible to this gate.** `order[k]` says
   *where* a coefficient is stored; the entropy contexts depend on `k`, not on
   `order[k]`. Inverting the permutation therefore leaves the ANS stream
   perfectly synchronized and only moves coefficients within the block. The
   shipped direction — `order[k]` is the destination cell of order position
   `k`, with the permutation composed inside the natural order
   (`order[i] = natural[nat_ord_perm[i]]`) — rests on I.3.1's assignment
   statement, which defines `order[...][i]` as an *element of*
   `natural_coeff_order[b]`, i.e. a cell position. It is pinned by unit tests
   using an asymmetric permutation, and it will be *decided by evidence* only
   at 8F's pixel comparison. This is the single most important thing a reader
   of the passing gate should not over-read.
2. **The LF-threshold walk order (0, 2, 1) is unexercised.** Every stream
   found uses the default `block_ctx_map`, which has no LF and no QF
   thresholds, so `lf_idx` is identically zero and the walk order cannot
   matter. The same is true of the `qf_thresholds` comparison and of
   G.2.2's `kUseLfFrame` "`lf_idx` is always zero" rule. These are implemented
   from the clause and covered only by unit tests.
3. **`num_hf_presets > 1` is unexercised.** Every stream found has
   `num_hf_presets == 1`, so `hfp` is a zero-bit field and the histogram
   offset is always 0.

This is not an oracle experiment: it makes no claim about libjxl. It is a
self-consistency proof against streams libjxl happened to produce, and the
property it checks (C.3.2's terminal state) is normative.

## 6. Consequences

* `vardct/hf_coeff.rs` ships the model as described, with
  `PREV_USES_CURRENT_PASS_COEFFICIENT = true` as the one named flip point
  (see below).
* `vardct/order.rs`'s module documentation states the direction verdict and
  records that the ANS gate cannot decide it.
* Fixtures 54 and 57 are `#[ignore]`d with the upstream blocker named in the
  attribute, per the "skip, don't loosen" rule.
* The two corpus cases are permanent tests, and are the *only* coverage of
  I.3.1's permutation branch by real encoder output; they skip when the
  gitignored corpus is absent.

## 7. The one flip point this sub-slice adds

`PREV_USES_CURRENT_PASS_COEFFICIENT` (`hf_coeff.rs`). I.4 computes `prev`
from whether "the decoded coefficient at position `k - 1` is 0". Three lines
later the clause says a non-first pass *adds* its coefficients to the
previously decoded ones. For a single-pass frame the two readings coincide,
so no available stream distinguishes them; for a progressive frame they
differ. The shipped reading (`true`) uses the symbol decoded by the current
pass, on the grounds that the accumulator is never mentioned inside the
symbol loop and a decoder consulting it would have to keep prior passes live
there. The order-space question in the same sentence is *not* a flip point:
`k` is the loop variable, so "position `k - 1`" can only be the previous
order position.
