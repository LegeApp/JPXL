# G.2.2 LfQuant channel order: fixture evidence supersedes the earlier reading

Date: 2026-08-03. Slice 8D-dequant (I.5.2 LF dequantization, I.6
chroma-from-luma). Supersedes the "provisionally settled" conclusion of
`2026-08-03-lf-quant-channel-order.md` (that entry is frozen per AGENTS.md's
append-only rule for `docs/experiments/`; this entry corrects it in a new
file rather than editing it in place).

## 1. Question

Same question as the earlier entry: what order does G.2.2 read `LfQuant`'s
three channels in? That entry settled `X, Y, B` from I.5.1/I.5.2's repeated
`qX, qY, qB` phrasing and Table I.1's channel numbering, while flagging the
argument as "provisional... a real fixture resolves immediately."

## 2. Preregistered gate

* **Settled** — real fixture bytes, decoded under both channel-order
  hypotheses, produce a plausibility signal clean enough to read off which
  hypothesis is right (not just "consistent with").
* **Unsettled** — the signal is ambiguous or contradicts itself across
  fixtures; ship the flip point as before and wait for 8F's full pixel path.

Fixed before running the probe below.

## 3. Method

8D-dequant's brief allowed (but did not require) attempting this once real
dequantization math exists, using fixtures 50-57 without needing the full
IDCT/coefficient path — I.5.2's LF plane is already a 1:8-downsampled view of
the actual image (the G.1 NOTE: "The LF coefficients (G.2.2) always
correspond to a 1:8 downsampled image"), so a real fixture's dequantized
`(dX, dY, dB)` can be inspected directly.

`vardct::lf::read_lf_quant_ordered` (private, added for this purpose) takes
the channel order as an explicit `bool` parameter instead of reading the
`LF_QUANT_CHANNEL_ORDER_IS_XYB` constant, so both hypotheses can be decoded
from the same bytes without recompiling.

A new test-only harness (`vardct::lf::tests::probe_lf_quant`) hand-decodes a
real fixture up through `LfQuant`: `decode_image_headers` → (ICC if
signalled) → `read_frame_header` → `FrameGeometry::from_header` → `read_toc`
→ the single-section byte slice → G.1.2 `LfChannelDequantization` → G.1's
`kVarDCT` rows (`Quantizer`, `HfBlockContext`, `LfChannelCorrelation`, all
8B's) → G.1.3's leading tree `Bool()` (and the tree itself, since fixture 50
signals one) → `read_lf_quant_ordered`. It deliberately does not call
`decode.rs`, which this task does not own and which does not implement
VarDCT yet; every function it calls is public API already exercised
elsewhere.

Fixture 50 (`128x128` greyscale, VarDCT, `-d 1`, filters off) and fixture 51
(same source, `-d 4`) were decoded under both hypotheses. For each, the
per-channel **population variance** of the dequantized LF plane (multiplier
fixed at `1.0` so only the shape of the data matters, not its scale) was
computed. The reasoning: JPEG XL's VarDCT path always converts to XYB
(opsin) regardless of the source's own colour space, and XYB is a
luma/chroma-like decomposition (`Y` sum-like, `X`/`B` difference-like); for a
*genuinely achromatic* source, the two chroma-like channels should carry
essentially nothing while the luma-like channel carries the actual image
structure (both fixtures' sources are described in their provenance
sidecars as a smooth ramp against a checkerboard — real, non-trivial
structure).

RGB fixture 54 was also attempted, as a corroborating "should look
different" case, but the harness fails to decode its `LfQuant` sub-bitstream
(an ANS final-state mismatch) past a point where every earlier field —
`read_frame_header`, `read_lf_global_vardct`, the global MA tree — succeeded
and reported plausible values. That failure is identical under both
hypotheses (so it is not itself evidence about channel order) and was not
chased further; it is a gap in this hand-rolled harness for a genuinely
multi-channel source, tracked as 8F's problem once `decode.rs` grows real
VarDCT wiring, not a new open question for this flip point.

## 4. Raw results

Population variance per channel, `multiplier = 1.0` (i.e. variance of the
raw dequantized-but-unscaled samples), read off the **first-decoded**,
**second-decoded**, **third-decoded** channel in stream order (labels
avoided on purpose — see below):

| fixture | pos0 | pos1 | pos2 |
| --- | --- | --- | --- |
| 50 (`-d 1`) | 7958.09 | 0.0 | 0.0 |
| 51 (`-d 4`) | 832.27 | 0.0 | 0.0 |

Both fixtures: the first-decoded channel alone carries substantial,
non-trivial variance; the other two are **exactly** `0.0` — not "small", the
literal IEEE-754 zero, meaning every dequantized sample in those two
channels is the identical value. This is consistent across two independent
distances of the same source.

## 5. Conclusion

**Settled, with the caveat that it rests on greyscale content only.** The
first-decoded `LfQuant` channel is Y (luma), not X: a genuinely achromatic
source's chroma-like channels have nothing to encode and quantize to a flat
constant, and that is exactly what positions 1 and 2 show, while position 0
carries the source's real structure. That makes the shipped order **Y, X, B**
— the same order I.4 states explicitly for HF coefficients — contradicting
the earlier entry's textual (`qX, qY, qB` → X-first) reading.

This does not by itself prove X sits at position 1 rather than B at position
1 (both chroma channels are flat, so their relative order is not
distinguishable from this signal alone) — but I.4's Y, X, B order is the only
attested 3-channel order in Annex I other than X, Y, B, and the evidence
here rules out X, Y, B specifically (X-first would put the flat channel
first and the structured one second, the opposite of what was observed), so
Y, X, B is taken by elimination against the two orders Annex I actually
names anywhere.

**What would still overturn this:** a fixture with genuine chroma content
(fixture 54, RGB, once the harness gap above is fixed) showing a different
pattern, or an oracle byte-level comparison against `djxl`'s own LF-plane
dump if one becomes available.

## 6. Consequences

* `jpxl_decode::vardct::lf::LF_QUANT_CHANNEL_ORDER_IS_XYB` flipped from `true`
  to **`false`** (shipped = Y, X, B). The constant's doc comment and the
  module doc's "Channel list and order" section were updated in place (code
  comments are not covered by the `docs/experiments/` append-only rule).
* `read_lf_quant_ordered` is the load-bearing implementation now — the
  constant genuinely branches on it, rather than being descriptive text next
  to code that always assumed one order (which is what it was before this
  entry: a real gap, now closed).
* `lf_quant_channel_order_probe_against_fixture_50` and `..._51` in
  `vardct::lf`'s own tests turned this investigation into a standing
  regression check (`assert_eq!(var_x, 0.0)`, `assert_eq!(var_b, 0.0)`,
  `assert!(var_y > 1.0)`) rather than leaving it as a one-off finding.
* The two hand-built-bitstream tests from 8D-parse
  (`a_hand_built_lf_quant_stream_decodes_to_the_expected_planes`,
  `extra_precision_is_read_before_the_sub_bitstream`) wrote their token
  streams in the old X-first order; both were updated to Y-first to match
  the shipped reading (their *assertions* — which value ends up in
  `planes.x`/`.y`/`.b` — are unchanged, since those fields are named by
  semantics and `read_lf_quant_ordered` does the reassignment).
