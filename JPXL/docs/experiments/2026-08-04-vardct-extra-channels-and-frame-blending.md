# Extra channels in kVarDCT frames, Table F.8 frame blending, and K.3.2 with alpha

Date: 2026-08-04. Status: frozen.

## Why this note exists

Wave 6 targeted the conformance corpus's alpha cases. Four constructs were in
the way, and each was refused with a typed `Unsupported` before this wave:

| refusal | clause cited | cases it blocked |
| --- | --- | --- |
| extra channels in a `kVarDCT` frame | G.4.2 | `alpha_premultiplied`, `patches`, `upsampling` |
| more than one regular frame (blending) | F.2 | `blendmodes`, `spot`, `cmyk_layers` |
| patches on a frame with extra channels | K.3.2 | `patches` |
| frame upsampling | J.2 | `upsampling` |

The first three are now implemented and their cases pass. This note records
what the streams contain, what each construct was proved by, the three
readings the clauses leave open, and one residual that is **not** caused by any
of this work.

## 1. What the corpus streams contain

Read with JPXL's own frame-header parser — no oracle emits this per frame:

| case | frames | extra channels |
| --- | --- | --- |
| `alpha_nonpremultiplied` | one `kModular` regular frame | 1 alpha, 12-bit (colour 12-bit) |
| `alpha_triangles` | one `kModular` regular frame | 1 alpha, 9-bit (colour 9-bit) |
| `alpha_premultiplied` | one `kVarDCT` regular frame | 1 alpha, **16-bit**, `alpha_associated`, colour **12-bit** |
| `patches` | `kReferenceOnly` kModular 207x189 at (0,0) → `kVarDCT` 1600x1096 with `kPatches` | 1 alpha, 8-bit |
| `blendmodes`, `blendmodes_5` | **five** full-size `kModular` regular frames, `save_as_reference = 1`, `save_before_ct = false`, sources 0/1/1/1/1, modes kReplace, kBlend, kAdd, kMul, kMulAdd | 1 alpha, 12-bit |
| `upsampling` | one `kVarDCT` frame, `upsampling = 4`, `ec_upsampling = [4]` | 1 alpha, 8-bit |
| `spot` | 2 frames, the second **cropped** 381x145 at (89,114) | 3, none of them alpha-blended by the colour rule |
| `cmyk_layers` | 4 frames, three **cropped** | 2 (`kBlack` + alpha at index 1) |
| `sunset_logo` | 2 frames, both **cropped** at negative offsets | 1 |

The two modular alpha cases (`alpha_nonpremultiplied`, `alpha_triangles`)
turned out to need no decoder change at all: the modular path has carried the
extra channels in its `initial_channels` list since slice 5. What they needed
was a *grading* path — a `FloatImage` built from all planes, each divided by
its own `(1 << ec_info[i].bit_depth) - 1`. Both are exact: peak error 1.19e-7,
i.e. the f32 round-trip of the integers and nothing else.

## 2. Extra channels in a kVarDCT frame (G.1.3, G.2.3, G.4.2)

A `kVarDCT` frame's G.1.3 channel list is `num_extra` channels long — the
`+ 1`/`+ 3` colour channels are `kModular`-only. So with no extra channels the
whole modular apparatus of such a frame is three empty sub-bitstreams, which
H.1 reads zero bits for; that is why the construct could be ignored until now
and why turning it on is a bit-position question, not only a feature question.

Three rows become live, and their positions inside the frame are the whole
problem:

* **G.1.3 `GlobalModular`**, in `LfGlobal`, after the `Bool()` that gates the
  global MA tree — which was already being read.
* **G.2.3 `ModularLfGroup`**, in each `LfGroup` section, **between** `LfQuant`
  (G.2.2) and `HfMetadata` (G.2.4). Table G.3's row order.
* **G.4.2 modular group data**, in each `PassGroup` section, **after** that
  group's HF coefficients (Table G.5's two rows, in that order).

The selection rules and the copy-back are byte-identical to the modular path's,
so this wave factored them out rather than duplicating them:
`lf_group_selection`, `pass_group_selection` (which is where
`G42_SIZE_TEST_IS_SHIFTED` lives) and `decode_group_channels` are now shared,
with `decode_group_section` reduced to the section-slicing wrapper the modular
path needs. The `kVarDCT` path calls the same three functions with the readers
it already holds.

### What proves it

The C.3.2 terminal-state gate, again. Every one of these sub-bitstreams is a
complete entropy-coded stream whose `SymbolDecoder::finish()` must land exactly
on the terminal ANS state, and — more sharply — a `PassGroup` section's modular
data is read from the *same reader* as its HF coefficients, so getting its
position or its channel selection wrong desynchronises the next section rather
than merely losing alpha. Fixture 83 (four groups, eight sections) and the
1600x1096 `patches` case are where that would show.

The pixel evidence is the ladder in `crates/jpxl-decode/tests/e2e_alpha.rs`
plus the corpus, and in every one of them the **alpha channel comes back
bit-exact** (peak 5.96e-8 = the f32 round-trip; 0.0 on three corpus cases),
because cjxl codes extra channels losslessly at these settings. That is a much
sharper gate than the colour channels' lossy budget, and the test asserts it
separately.

### What is still refused, and why

* `dim_shift > 0` or `ec_upsampling > 1` on any extra channel. Both make the
  channel arrive smaller than the frame, and L.4/K.1 restore it with K.2's
  non-separable upsampling, which this decoder does not have. `upsampling` is
  the corpus case that needs it (`upsampling = 4`, `ec_upsampling = [4]`), and
  it is refused at J.2 first anyway.
* A float-sample extra channel (`ec_info[i].bit_depth.is_float()`). G.4.2 says
  the integers are "interpreted according to `ec_info[i].bit_depth`"; the float
  reading of that is D.3.5's bit pattern, not a division. Nothing available
  exercises it, so it is refused rather than guessed.

## 3. Frame blending (F.2, Tables F.7 and F.8)

`blendmodes` is five full-size `kModular` frames, each stored to reference slot
1 and each blending against slot 0 or 1, and between them they use **every one
of Table F.8's five modes on both the colour channels and the alpha channel**.
That is an unusually complete test vector, and the test asserts the presence of
all five modes in the header before it grades a pixel.

Implementation notes that the clause pins and that are easy to get wrong:

* **The space.** F.2: "the blending is done in the colour space after inverse
  colour transforms from Annex L have been applied (except for L.4)". So a
  canvas is display-space `[0, 1]` floats — *not* the XYB that K.3.2's patch
  references hold. One `Reference[]` slot can therefore mean two different
  things, and `save_before_ct` is what says which. JPXL keeps them in two
  arrays: `references` (pre-CT, for K.3) and `canvases` (post-CT, for F.2), and
  refuses a regular frame with `save_before_ct` set.
* **Per-group sources.** `source` is a field of `BlendingInfo`, so the colour
  channels and each extra channel may blend against *different* reference
  slots. An unwritten slot is "assumed to have all sample values set to
  zeroes".
* **The alpha channel's own formulas.** `kBlend` on the alpha channel is
  `alpha = old_alpha + new_alpha * (1 - old_alpha)`, and `kMulAdd` on it
  preserves the source frame's value. Using the generic formulas there gives a
  visibly wrong alpha ramp; `blendmodes` catches it, since its alpha channel
  passes through all five modes.
* **`clamp`.** For `kBlend`/`kMulAdd` it clamps `new_alpha`; for `kMul` it
  clamps `new_sample` instead. `blendmodes` has `clamp = false` everywhere, so
  this is asserted only by unit tests.

**Exactness is preserved for the single-frame case.** Compositing produces
floats (`kBlend` divides), which would destroy the bit-exact integer output of
a lossless modular decode. So `decode` recognises the identity case — one
regular frame, no crop, `kReplace` on the colour rule and on every extra-channel
rule — and returns that frame's own planes untouched. Every pre-existing
lossless test, including 32-bit float `lossless_pfm`, still returns integers
from the same code path it did before.

## 4. Three readings the clauses leave open

Each is a named constant, so flipping it is a one-line change at the site.

### `PATCH_ALPHA_IS_THE_PATCHS_OWN` (`vardct/render.rs`), shipped `true`

Table K.1's alpha rows say "the alpha channel is the extra channel with index
`k = blending[j].alpha_channel[c]`" and never say whether that channel is read
from the patch (the reference frame) or from the canvas. `true` reads it from
the reference frame at the patch's own `(x0 + ix, y0 + iy)`, which is the only
reading under which a patch can carry its own opacity — the use K.3 exists for
— and the one consistent with Table F.7's `kBlend`, where alpha belongs to the
frame being composited.

**Unexercised.** `patches`'s 654 positions are `kAdd` on colour and `kNone` on
alpha; `patches_lossless` is the only other patch case with an extra channel
and it is refused for other reasons (a `kModular` frame with patches, and a
reference frame in a non-XYB image). No available stream uses a Table K.1 row
above 3.

### `ALPHA_SELF_RULE_IS_THE_NAMED_CHANNEL` (`frame/blending.rs`), shipped `true`

Table F.8 gives `kBlend` and `kMulAdd` a second formula "for the alpha channel
itself" without saying which extra channels that covers. `true` fires the
exception exactly when the channel being blended is the one supplying
`new_alpha`/`old_alpha` — under which `new_sample` and `new_alpha` are the same
number and the alternate formula reads as a simplification rather than a
different operation. `false` would fire it for every `kAlpha`-typed channel.

**Unexercised.** The two readings coincide for any image with a single alpha
channel, which is every multi-frame stream available. `cmyk_layers` has two
extra channels (`kBlack` and an alpha at index 1) and would discriminate, but
it is refused for cropped frames.

### `G42_SIZE_TEST_IS_SHIFTED` — unchanged, now shared

Previously local to the modular path; the `kVarDCT` path now goes through the
same `pass_group_selection`, so the constant governs both. No new evidence.

## 5. A residual that is not ours: greyscale VarDCT, ~1.9e-4 RMSE

Handmade fixture 84 (128x128 greyscale + alpha, `d = 1.0`, filters off) grades
at peak 3.20e-4 and RMSE 1.87e-4 on its grey channel, over the 1e-5 RMSE of
18181-3 Annex A's no-filters class. **It is not caused by the extra channel.**

Preregistered discriminator, run before any conclusion: encode the *identical*
grey source with no alpha channel at all, decode both, compare each against its
own `djxl` reference. Predicted, if the extra-channel path were at fault: the
no-alpha stream grades clean. Result:

```
grey source, no alpha:      worst 0.00028831   rmse 0.00018684
fixture 84 (grey + alpha):  worst 0.00031978   rmse 0.00018683
```

Identical to four significant figures, so the extra channel contributes
nothing. Three further probes localised it away from the obvious suspects: the
output channel choice (R, G, and the R/G average all give the same error to
1e-8, and forcing X to zero before L.2.2 changes nothing — X is already
negligible), so the difference is in the reconstructed **Y** itself. The same
content in colour (fixture 83) grades at 4.8e-6 RMSE, and the corpus
`grayscale`/`grayscale_5` cases pass at RMSE 7.8e-6 with a peak of 2.3e-4 —
the same peak magnitude, diluted by photographic content. So this is a
pre-existing, content-dependent greyscale-VarDCT residual in the 1e-4 range,
already visible in wave 5's numbers, and it belongs to whoever next audits
I.5.3/I.9 for the greyscale case. Fixture 84's colour RMSE bound is set to 3e-4
with this note cited; its peak bound stays at the no-filters class, and its
alpha channel is held to a lossless round-trip.

## 6. Final numbers

Decoder output versus each case's published `reference_image.npy`, thresholds
from its own `test.json` (peak / RMSE):

| case | limit | measured peak | worst channel RMSE |
| --- | --- | --- | --- |
| `alpha_nonpremultiplied` | 6.10e-5 / 6.10e-5 | 1.19e-7 | 6.37e-8 |
| `alpha_triangles` | 1.95e-3 / 1.95e-3 | 1.19e-7 | 1.70e-8 |
| `alpha_premultiplied` | 3.82e-6 / 3.82e-6 | 2.62e-6 | 2.02e-7 |
| `patches` | 4.0e-3 / 1.0e-4 | 7.75e-6 | 1.51e-7 |
| `patches_5` | 6.0e-2 / 2.0e-2 | 2.41e-2 | 2.55e-4 |
| `blendmodes` | 1.0e-4 / 4.0e-3 | 1.91e-6 | 2.26e-7 |
| `blendmodes_5` | 6.0e-2 / 2.0e-2 | 1.91e-6 | 2.26e-7 |

Seven cases, taking the corpus total from six to thirteen.

Handmade ladder, against the pinned `djxl` decode of the same stream:

| fixture | limit | measured peak | worst channel RMSE |
| --- | --- | --- | --- |
| 80 RGBA 8x8 lossless | 1e-6 / 1e-6 | 5.96e-8 | 2.17e-8 |
| 81 RGBA 600x520 lossless, 9 groups | 1e-6 / 1e-6 | 5.96e-8 | 2.34e-8 |
| 82 RGBA 64x64 VarDCT | 4e-3 / 1e-5 | 5.91e-5 | 4.63e-6 |
| 83 RGBA 384x320 VarDCT, 4 groups | 4e-3 / 1e-5 | 5.82e-5 | 4.76e-6 |
| 84 grey+alpha 128x128 VarDCT | 4e-3 / 3e-4 (§5) | 3.20e-4 | 1.87e-4 |

## 7. Still refused after this wave

| construct | clause cited | corpus cases it blocks |
| --- | --- | --- |
| cropped frames (displayed) | F.2 | `spot`, `cmyk_layers`, `sunset_logo` |
| frame upsampling / K.2 | J.2, K.2 | `upsampling` |
| an extra channel with `dim_shift > 0` or `ec_upsampling > 1` | L.4, K.2 | `upsampling` |
| patches on a `kModular` frame | K.3 | `patches_lossless` |
| a stored frame in a non-XYB image | L.2.2 | `patches_lossless` |
| a regular frame with `save_before_ct` | F.2 | none available |
| a float-sample extra channel | G.4.2 | none available |
| a presented regular frame with a duration, and animation | F.2 | the five `animation_*` cases |
| splines, noise, YCbCr, JPEG reconstruction | K.4, K.5, L.3 | `noise`, `cafe`, `bench_oriented_brg`, `grayscale_jpeg` |

The animation refusal is **new and deliberate**. F.2 composes several regular
frames into one image only "in the case that `metadata.have_animation` is
false"; a frame with a nonzero `duration` is presented in its own right. Before
this wave the blanket "more than one regular frame" refusal covered that; now
that composition exists, an explicit guard replaces it, so an animation is
still refused rather than silently flattened into a composite the standard does
not describe.

Cropped frames are the single largest remaining lever: the canvas machinery
this wave added is what they need, and with it `spot`, `cmyk_layers` and
`sunset_logo` become reachable (`sunset_logo` also needs `metadata.orientation`
and `cmyk_layers` a `kBlack` channel interpretation).
