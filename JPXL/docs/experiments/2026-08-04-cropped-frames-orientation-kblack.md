# Cropped displayed frames, image orientation, and `kBlack` extra channels

*Date: 2026-08-04. Frozen once written; corrections go in a dated addendum.*

Scope: 18181-1 F.2 (cropped frames and their compositing), D.3.2 / Table D.4
(image orientation), D.3.6 (`kBlack`), graded under 18181-3 §4.2.
Target corpus cases: `spot`, `cmyk_layers`, `sunset_logo`.

---

## 1. Refusal enumeration before the work

Every corpus case was run through `jpxl decode` and its **first** refusal
recorded. The three target cases all stopped at the same gate:

| case | first refusal |
| --- | --- |
| `spot` | `18181-1 F.2: cropped frames` |
| `cmyk_layers` | `18181-1 F.2: cropped frames` |
| `sunset_logo` | `18181-1 F.2: cropped frames` |

Nothing else refused them. In particular neither `kBlack` nor `kSpotColour`
was ever a gate: `check_supported_extra_channels` rejects only float-sample
extra channels and out-of-range subsampling factors, and none of the three
cases has either. Orientation was never a gate either — it was silently
ignored, which is worse than a refusal and is why `sunset_logo` needed the
shape check in the new test to catch it.

The enumeration was re-run afterwards. Exactly three cases moved, and the
residual refusals are unchanged:

| refusal | cases |
| --- | --- |
| `F.2: a presented regular frame with a duration` (animation) | `animation_icos4d(_5)`, `animation_newtons_cradle`, `animation_spline(_5)` |
| `L.3: do_YCbCr colour reconstruction` | `bench_oriented_brg(_5)`, `cafe(_5)`, `grayscale_jpeg(_5)` |
| `L.2: xyb_encoded colour reconstruction` (modular displayed frame) | `bicycles` |
| `K.5: noise synthesis` | `noise(_5)` |
| `L.2.2: a stored frame in a non-XYB image` | `patches_lossless` |

**`bench_oriented_brg` is not unlocked by orientation.** Despite the name, its
gate is `do_YCbCr` — it is a JPEG-reconstruction stream. Orientation was never
what stopped it.

## 2. What the reference arrays actually contain

Read first, before writing any conversion code — `test.json` plus the NPY
header of each case:

| case | `extra_channel_type` | NPY shape (F, H, W, C) |
| --- | --- | --- |
| `spot` | `[Alpha, SpotColor, SpotColor]` | `(1, 400, 600, 6)` |
| `cmyk_layers` | `[Black, Alpha]` | `(1, 512, 512, 5)` |
| `sunset_logo` | `[Alpha]` | `(1, 1386, 924, 4)` |

So 18181-3 §4.1.2 stores **every extra channel as itself**, in `ec_info`
order, after the colour channels. `kBlack` is graded as the fourth plane it
is; there is no CMYK→display conversion anywhere in the expected output, and
spot colours are likewise not rendered into the colour channels. A decoder
that did either would fail §4.2's condition 1 (shape) before a single sample
was compared. **No `kBlack`-specific code was written, and none is correct to
write at the codestream layer.**

`sunset_logo`'s shape is the orientation evidence: the codestream's
`SizeHeader` is 1386×924 and the reference is 924 wide by 1386 tall.

## 3. Frame structure of the three cases

Parsed by JPXL's own frame-header reader, not by an oracle:

```
spot          600x400, orientation 1, want_icc, ec [Alpha, Spot, Spot] @16 bit
  frame 0  kModular  no crop     600x400   kReplace  -> Reference[1]
  frame 1  kModular  crop (89,114) 381x145 kBlend src=1;  ec: Blend, Add, Add

cmyk_layers   512x512, orientation 1, want_icc (CMYK), ec [Black, Alpha] @8 bit
  frame 0  kModular  no crop     512x512   kReplace  -> Reference[1]
  frame 1  kModular  crop (143,166) 200x107 kBlend src=1 alpha_channel=1 -> Ref[1]
  frame 2  kModular  crop  (98,311) 300x88  kBlend src=1 alpha_channel=1 -> Ref[1]
  frame 3  kModular  crop (134, 13) 110x68  kBlend src=1 alpha_channel=1

sunset_logo   1386x924, orientation 7 (anti-transpose), no ICC, ec [Alpha] @10 bit
  frame 0  kModular  crop (-662,-100) 2048x1024  kReplace  -> Reference[1]
  frame 1  kModular  crop (-662,-100) 2048x1024  kBlend src=1
```

Three things worth naming.

* `cmyk_layers` puts alpha at extra-channel index **1**; `kBlack` is index 0.
  Every blending rule names `alpha_channel = 1` explicitly. A decoder that
  assumed index 0 would alpha-blend against the black separation.
* `sunset_logo`'s offsets are negative, and the rectangle's far edges land
  **exactly** on the image's (−662 + 2048 = 1386, −100 + 1024 = 924). So
  `have_crop` is true and F.2's `full_frame` is *also* true: the frame covers
  the image, it just does not start at the origin. `have_crop` and
  `full_frame` are different questions and this case separates them.
* In all three, the running composite is stored into `Reference[1]` and the
  next frame reads `source = 1`. That coincidence is what makes §5's flip
  point unexercised.

## 4. Table D.4, derived and then checked against the oracle

The clause gives each row as a "first row"/"first column" pair plus a prose
transform. Inverting the pair — stored row `r` lands on the displayed edge
named by "first row", stored column `c` on the edge named by "first column" —
gives, for a stored grid `W x H` and a displayed position `(x, y)`:

| orientation | source `(sx, sy)` | displayed size |
| --- | --- | --- |
| 1 Identity | `(x, y)` | `W x H` |
| 2 FlipHorizontal | `(W-1-x, y)` | `W x H` |
| 3 Rotate180 | `(W-1-x, H-1-y)` | `W x H` |
| 4 FlipVertical | `(x, H-1-y)` | `W x H` |
| 5 Transpose | `(y, x)` | `H x W` |
| 6 Rotate90Cw | `(y, H-1-x)` | `H x W` |
| 7 AntiTranspose | `(W-1-y, H-1-x)` | `H x W` |
| 8 Rotate90Ccw | `(W-1-y, x)` | `H x W` |

Cross-checked against the prose in the same row: row 7's "flip horizontally
then rotate 90 clockwise" composes to `S[H-1-x][W-1-y]`, which is the entry
above; row 5's "rotate 90 clockwise then flip horizontally" composes to
`S[x][y]`, the transpose. The two descriptions agree, so **no scan read was
needed** — this is not a suspected defect, just a table that has to be
inverted carefully.

**Oracle A/B, all eight rows.** One 24×16 RGBA payload was encoded eight
times, differing only in a PNG `eXIf` orientation tag (`cjxl` has no
orientation flag; Table D.4's values are Exif's, which the clause notes). Each
was decoded by `djxl --output_format npy` and by JPXL. All eight matched at
peak `5.96e-8` — pure `f32` rounding — with the transposing four reporting the
swapped shape. The eight reference decodes are pairwise distinct, which the
test asserts, so the agreement is not eight decoders agreeing to do nothing.

**Where the turn goes.** F.2 says a frame's `width`/`height` "are interpreted
according to the sample grid before taking `metadata.orientation` into
account". So `SizeHeader`, every frame rectangle, every group coordinate and
every crop offset are pre-orientation, and the turn is a single permutation of
the finished image — applied to the integer planes and the float planes alike.
Turning earlier would still produce a correctly *shaped* image on
`sunset_logo`, which is the trap the corpus rung exists to catch.

## 5. Flip point: what a cropped frame leaves outside its rectangle

`CROP_LEAVES_RUNNING_IMAGE_OUTSIDE` (jpxl-decode `decode.rs`).

F.2 says a cropped frame "updates the rectangle of the image", and separately
that a blend's previous sample comes from `Reference[source]`. Those are two
different buffers, and the clause never says what the image holds *outside*
the updated rectangle.

* `true` **(shipped)** — the running composite is left untouched outside the
  rectangle. This is what "updates the rectangle of the image" says: a crop is
  a partial update.
* `false` — `Reference[source]` shows through outside the rectangle, i.e. the
  frame is composited over the whole canvas with the crop only masking the new
  frame's contribution.

**Unexercised, and demonstrated to be so.** The two readings coincide whenever
the running composite equals `Reference[source]`, which is exactly how all
three corpus cases are built (§3). Flipping the constant to `false` and
re-running `e2e_layers.rs` leaves all seven tests green. Discriminating would
need a stream whose cropped frame blends against a slot other than the one
holding the running image; `cjxl` does not emit one, and hand-writing a
multi-frame modular codestream is a larger job than the ambiguity is worth
today. Both arms are implemented and compiled, so the flip is one constant.

## 6. Why there is no handmade cropped-frame fixture

`cjxl` emits a cropped **displayed** frame only from animation input (GIF /
APNG), whose frames carry a non-zero `duration` and are therefore a separate
presented image — refused, and out of scope for this wave. Every other input
path produces `have_crop == false`. There is no encoder flag for a crop.

So the crop evidence is:

* the three corpus streams, whose rectangles the test asserts from the
  bitstream before grading (including `sunset_logo`'s negative origins), and
* six unit tests on `CropRect::intersect` covering the inner rectangle, the
  negative origin, overhang past the far edges, a rectangle entirely off the
  canvas on each of the four sides, and `i32::MIN`/`i32::MAX` offsets against
  a `u32::MAX`-wide frame.

The handmade ladder (fixtures 100–109) covers orientation, which `cjxl` *can*
be made to emit. `tools/make-orientation-fixtures.sh` verifies with `jxlinfo`
that each stream really carries the orientation asked for, and fails the build
of the fixture set otherwise — without that check the whole ladder could
silently be orientation 1.

## 7. Mutation testing of the new gates

| mutation | killed by |
| --- | --- |
| `AntiTranspose` → plain transpose | 3 of 7 e2e rungs; unit test `the_displayed_origin_comes_from_the_corner_table_d4_names` |
| clamp the rectangle but not the source coordinate (`origin.max(0)`) | `corpus_sunset_logo_…`; unit test `negative_origins_clip_and_still_offset` |
| ignore `have_crop`, composite at the origin | all three corpus rungs |
| flip `CROP_LEAVES_RUNNING_IMAGE_OUTSIDE` | **nothing** — see §5 |

Note that the inverse-composition unit test does *not* kill the first
mutation: anti-transpose and transpose are both involutions, so composing
either with itself is the identity. The corner test is what pins the table.
Two tests that look redundant are not.

## 8. Results

| case | our peak | `test.json` peak | our worst channel RMSE | `test.json` RMS |
| --- | --- | --- | --- | --- |
| `spot` | 5.96e-8 | 3.815e-6 | 8.0e-10 | 3.815e-6 |
| `cmyk_layers` | 1.19e-7 | 9.77e-4 | 6.2e-9 | 9.77e-4 |
| `sunset_logo` | 4.77e-7 | 2.44e-4 | 2.1e-8 | 2.44e-4 |

All three are `f32` round-trip error on a lossless modular decode: the error
is the quantization of `value / ((1 << bits) - 1)`, not a decoding
difference. Handmade rungs 100–108 grade at zero to 1e-6; rung 109 (`kVarDCT`)
inside 4e-3 peak.

## 9. What this does not prove

* The flip point in §5, on any stream.
* Cropped frames in a `kVarDCT` frame. The refusal was lifted for every
  encoding because compositing happens in display space and is
  encoding-agnostic, but nothing available exercises the combination.
* Crop interacting with `upsampling > 1`. `FrameGeometry` divides the frame
  rectangle by `upsampling` and the crop offsets are on the image grid; no
  stream has both.
* Custom orientation on a preview frame. D.3.2 says the turn applies to the
  preview too; previews are refused outright.
* Spot-colour *rendering*. The channels are carried through as samples, which
  is what the conformance references hold; the K-annex compositing of a spot
  colour into the colour channels is a display concern and is not implemented.
