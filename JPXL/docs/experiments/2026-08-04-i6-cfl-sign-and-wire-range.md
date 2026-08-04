# I.6 chroma-from-luma: sign convention and the HF factor wire range

*Dated 2026-08-04. Encoder slice 15 (CfL estimation). Immutable — see AGENTS.md
section 4.*

## Question

Slice 15 puts the first non-neutral I.6 correlation factors this project has
ever written on the wire: a frame-wide LF pair (`x_factor_lf`/`b_factor_lf`,
I.2.3) and per-64x64-tile HF factors (`XFromY`/`BFromY`, G.2.4). Two things had
to be established behaviourally before trusting them:

1. the **sign convention** of the encoder's residual against the decoder's
   reconstruction, and
2. the **representable range** of the HF factors, which G.2.4 stores as generic
   Modular samples (any `i32`) but which a conformant decoder may constrain.

## Method

A 256-block-wide multi-group RGB fixture was encoded and decoded by three
independent decoders: `jpxl-decode` (this repo), `jxl-oxide` (independent Rust),
and `djxl` (libjxl, the reference). The oracle gate compares decoder against
decoder on ONE stream, so any disagreement is a bitstream/decoder fault, not
rate-distortion. A single-tile probe forced exactly one 64x64 HF tile to a
chosen factor value and zeroed the rest, so a divergence localises to that tile
and that value.

## Result 1 — the sign convention is correct as written

The encoder subtracts `k * dY_recon` from the chroma coefficient before
quantizing (`residual = C - k*dY`), where `dY` is the **reconstructed** luma the
decoder will add back, and `k = base_correlation + factor/colour_factor`
computed in `f32` exactly as the decoder computes it. The decoder reconstructs
`X = dX + kX*dY`, `B = dB + kB*dY`. With HF factors held at zero and only LF
enabled — and vice versa — every decoder agreed with every other, i.e. the sign
is not merely self-consistent (encoder vs our decoder) but agrees with the
reference. A shared encoder/decoder sign flip would have shown up as our decoder
and `jxl-oxide` agreeing with each other but not with `djxl`; that pattern did
NOT appear for either arm once the range constraint below was applied.

## Result 2 — HF factors must lie in signed-8-bit range [-128, 127]

With the sign correct, `djxl` still diverged from `jpxl-decode` and `jxl-oxide`
(which agreed with each other) whenever an HF factor exceeded a magnitude
threshold. The single-tile probe pinned the threshold exactly:

| stored factor | djxl vs jpxl-decode |
| --- | --- |
| +127 | agree |
| +128 | diverge (peak 15) |
| -128 | agree |
| -129 | diverge (peak 17) |

i.e. `djxl` reads `XFromY`/`BFromY` as **signed bytes**; a value outside
[-128, 127] is interpreted differently there than by the two lenient decoders.
This is an interoperability constraint on the encoder, not a coding cost: the
reference simply does not represent factors outside that range. Our own decoder
is *more* permissive, which is exactly why a self-roundtrip and even a second
independent decoder could not have caught this — only the reference oracle did.

**Conclusion:** the policy clamps the HF factor search to [-128, 127]
(`HF_FACTOR_MIN`/`HF_FACTOR_MAX` in `jpxl-encode-policy`), and the writer's
`check_supported` rejects any plan whose HF factor escapes that range, so the
constraint is enforced at the typed boundary rather than trusted. The LF factors
are `u(8)` biased by 128 (I.2.3) and are already confined to the same effective
range by the field width.

This report is evidence about three decoder implementations, not a normative
statement about ISO/IEC 18181-1; the range is treated as the common
interoperable subset the reference honours.
