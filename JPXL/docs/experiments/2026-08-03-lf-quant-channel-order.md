# G.2.2: what order do LfQuant's three channels come in?

Date: 2026-08-03. Slice 8D-parse (LF and HF-metadata modular sub-bitstreams).
Not an oracle experiment: no VarDCT pixel path exists yet to probe against
`djxl`; settled (provisionally) from cross-clause consistency alone.

## 1. Question

G.2.2 says only that the LF-coefficients sub-bitstream "consists of three
channels", without ever naming which of X, Y, B comes first, second, third.
Annex H's modular decode reads channels strictly in list order, so the order
has to be fixed by something outside G.2.2 itself.

## 2. Preregistered gate

* **Settled** — some other clause names the trio in a specific order more than
  once, consistently, and nothing contradicts it.
* **Unsettled** — ship a flip point and defer to an oracle probe once 8F's
  pixel path exists.

Fixed before reading past I.5.1.

## 3. Method

Read I.4 (HF coefficient decode order), I.5.1, I.5.2 (LF dequantization) and
I.6 (chroma from luma) in `latex/part1.tex`, and Table I.1's channel
numbering, looking for every place the three channels are named together.

## 4. Raw results

Two different orders are attested in Annex I, for two different sub-bitstreams:

* I.4's HF coefficient decode is explicit: "it reads channels Y, X, then B".
  That is the order for the *pass-group* HF coefficient stream, not for
  `LfQuant`.
* I.5.1 introduces "quantized LF and HF coefficients qX, qY and qB", and
  I.5.2's dequantization code repeats the same order three times in a row:
  `dX = mxDC * qx / ...`, `dY = myDC * qy / ...`, `dB = mBDC * qB / ...`
  (the OCR renders `qx`/`qy` as `gx`/`ay` in one transcription, but the X,
  Y, B ordering of the three lines is unaffected). Table I.1's own channel
  numbering is `X = 0, Y = 1, B = 2`, matching this order and not I.4's.

No clause ever writes `qY, qX, qB`. The X, Y, B order is attested three times
in I.5.1/I.5.2 for the LF-specific multipliers and matches the channel-index
table; the Y, X, B order is attested once, explicitly for the *different*
HF sub-bitstream, which even has its own explicit statement precisely
because it deviates from the natural channel order (mirrored in I.4's
`BlockContext()`'s `c < 2 ? c ^ 1 : 2` swap, which exists to convert *back*
from X,Y,B-numbered channel indices to a Y,X,B context ordering).

## 5. Conclusion

**Provisionally settled**, weaker than the DCT8x4 half-placement precedent:
`LfQuant`'s three channels are read in the order X, Y, B, matching Table I.1's
channel numbering and never contradicted anywhere. This is an argument from
repetition and consistency, not a direct statement — G.2.2 could in principle
still intend the HF order for its own reasons the way I.4 needed. It is the
kind of one-bit ambiguity a real fixture resolves immediately, since the two
readings are byte-for-byte different bitstreams for any group with
`jpeg_upsampling != [0,0,0]` (identical for isotropic subsampling, since the
three channels would have the same shape and only their *values* would
differ) and always differ in value for any content with real chroma-from-Y
signal.

## 6. Consequences

* `jpxl_decode::vardct::lf::LF_QUANT_CHANNEL_ORDER_IS_XYB` (default `true`)
  carries both readings; flipping it swaps the second and third channel specs
  and the corresponding slice of `into_channels()`.
* This slice's own tests fix data through the constant's shipped reading (they
  decode a stream and check `planes.x`/`planes.y`/`planes.b` against values
  chosen to make cross-channel confusion visible), so they pin the reading
  without asserting it is *correct* — see `vardct::lf`'s module doc for the
  same caveat stated in code.
* 8D-dequant and 8F inherit the probe: the first real VarDCT fixture with
  `x_factor_lf != 0` (nonzero LF chroma-from-luma) is decisive, because a
  wrong channel order corrupts the reconstructed X/B planes but leaves Y
  untouched — an easy signature to look for.
