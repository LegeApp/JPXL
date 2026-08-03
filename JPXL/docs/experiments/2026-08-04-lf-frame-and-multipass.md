# Progressive decoding: kLFFrame, kUseLfFrame and multi-pass HF (wave 5)

Date: 2026-08-04. Status: frozen.

## What this is

The conformance cases `progressive` and `progressive_5` were the last two
VarDCT cases refused outright. This report records what those streams actually
contain, the three one-bit readings the work had to settle, and the evidence
for each. Two of the three were open flip points inherited from slice 8; one is
new and was found by the stream rather than by reading.

`progressive_5/input.jxl` is a **symlink** to `progressive/input.jxl`. The two
cases are one codestream graded at two error classes (peak 0.02 / RMSE 1e-4 and
peak 0.06 / RMSE 0.02).

## What the stream actually contains

Parsed by JPXL's own frame-header reader, not by any oracle:

```
image 4064x2704, xyb_encoded, want_icc, 0 extra channels, 896-byte ICC

frame 0  kReferenceOnly  kModular  flags 0x00  crop 29x28 @ (0,0)
                                   save_as_reference 0, save_before_ct
                                   1 section, 2207 bytes
frame 1  kLFFrame        kModular  flags 0x00  lf_level 1
                                   frame dimensions 508x338 (F.1 divides by 8)
                                   7 sections: 76773, 0, 0, 16012, 17303, 6490, 6910
frame 2  kRegularFrame   kVarDCT   flags 0x22 = kPatches | kUseLfFrame
                                   num_passes 2, shift [1],
                                   downsample [2], last_pass [0]
                                   176 groups, 4 LF groups, 358 sections
```

So five things at once: patches (K.3, already implemented in wave 4), a
`kLFFrame`, `kUseLfFrame`, two HF passes with a nonzero shift, and `want_icc`
output (clause 4's linear output, already implemented).

The `kLFFrame`'s `GlobalModular` image is **Squeezed**: 43 channels, of which
`GlobalModular` decodes the first 36 and the rest are group data. That detail
is what produced the third flip point below.

## Handmade first-light ladder

`cjxl` can emit each construct separately, which makes a
one-feature-at-a-time ladder possible before the corpus file. Fixtures 70-74,
recipe in `tools/make-progressive-fixtures.sh`:

| fixture | cjxl flag | structure |
| --- | --- | --- |
| 70 | `--progressive_ac` | 3 passes, `shift [0, 0]`, no LF frame |
| 71 | `--qprogressive_ac` | 2 passes, `shift [1]`, no LF frame |
| 72 | `--progressive_dc=1` | `kLFFrame` at `lf_level` 1, 1 pass |
| 73 | `-p --progressive_dc=1 --qprogressive_ac` | both, 1 group |
| 74 | as 73, 384x320 RGB | both, 4 groups, colour |

All five decode with every section consumed to its exact TOC length and every
ANS stream landing on C.3.2's terminal state, and grade at the no-filters class
(peak 0.004 / RMSE 1e-5):

| case | peak | worst-channel RMSE |
| --- | --- | --- |
| 70 | 1.4e-6 | 3.3e-7 |
| 71 | 1.4e-6 | 3.3e-7 |
| 72 | 1.1e-6 | 2.9e-7 |
| 73 | 1.1e-6 | 2.9e-7 |
| 74 | 5.5e-5 | 4.9e-6 |
| corpus `progressive` (class 0.02 / 1e-4) | 2.0e-5 | 4.4e-7 |
| corpus `progressive_5` (class 0.06 / 0.02) | 2.0e-5 | 4.4e-7 |

Everything is two to three orders of magnitude inside its class, which is what
makes the probes below sharp: a wrong reading does not eat margin, it moves the
error by orders of magnitude or stops the decode outright.

## Verdict 1 — `PREV_USES_CURRENT_PASS_COEFFICIENT` = `true`, PROBED-CONFIRMED

I.4's coefficient loop takes a `prev` that is 0 when the decoded coefficient at
order position `k - 1` is 0. Three lines later the clause says that for a
non-first pass the decoder adds the decoded coefficients to the
previously-decoded ones. The open question since slice 8C: does `prev` consult
the symbol **this pass** decoded, or the **accumulated** multi-pass value?

Slice 8F could not discriminate — every stream then reachable was single-pass,
where the two readings coincide by construction (`UnpackSigned(u) == 0` exactly
when `u == 0`, and F.2's left shift cannot turn a non-zero into a zero without
an overflow that is rejected). It also found the `false` arm degenerate, and
that was repaired the same day: `next_prev(prev_uses_current_pass, ucoeff,
accumulated)` now has a `false` arm that really reads the accumulator.

With five multi-pass streams in hand the probe is decisive. Flipping the
constant to `false` and rebuilding:

| stream | passes | `true` (shipped) | `false` (accumulator) |
| --- | --- | --- | --- |
| fixture 70 | 3, shift 0 | peak 1.4e-6 | **I.4 error**: 125 undelivered non-zero coefficients |
| fixture 71 | 2, shift 1 | peak 1.4e-6 | **I.4 error**: 96 undelivered |
| fixture 73 | 2, shift 1 | peak 1.1e-6 | **I.4 error**: 96 undelivered |
| fixture 74 | 2, shift 1, 4 groups | peak 5.5e-5 | **I.4 error**: 125 undelivered |
| corpus `progressive` | 2, shift 1, 176 groups | peak 2.0e-5 | **I.4 error**: 24 undelivered |
| fixture 72 | **1** | peak 1.1e-6 | peak 1.1e-6 — *unchanged* |

The failure mode is the informative part. "Undelivered non-zero coefficients"
means the `non_zeros` count the block promised was never reached: a wrong
`prev` selects the neighbouring histogram, so every later symbol in the block
decodes to a different value and the promised non-zeros never arrive. This is
not a tolerance drift, it is the entropy layer losing synchronisation.

Fixture 72 is the control: single-pass, so the two readings are the same
function, and it is byte-identical under both. That is exactly the pattern a
correct discrimination should show — the flip must move only what it can
possibly move.

The textual reading agrees. I.4 says the decoder *sets* the position to
`UnpackSigned(ucoeff)` and only afterwards, as a separate sentence, adds the
pass to the accumulator; "the decoded coefficient" names the value just set,
which for the current pass's own array is `ucoeff != 0`.

**Verdict: `PREV_USES_CURRENT_PASS_COEFFICIENT = true`. Settled.**

## Verdict 2 — `LF_FRAME_IS_XYB_PRESTEP` = `true` (new flip point)

F.2 records "the samples of the frame before any colour transform is applied"
as `LFFrame[lf_level - 1]`. G.2.2 says those samples are used *instead of* what
G.2.2 and I.5.2 would have produced — i.e. instead of **dequantized** LF, which
is float XYB. For a `kModular` LF frame the decoded samples are integers, so
something has to bridge the two, and the clause does not say what.

The reading shipped is that L.2.2's `kModular` pre-step is that bridge:
`X = x' * m_x_lf_unscaled`, `Y = y' * m_y_lf_unscaled`,
`B = (B' + Y') * m_b_lf_unscaled`, with L.2.2's own channel naming
(**y', x', B'** — channel 0 is luma). It is the same step
`reference_from_modular` already applies to `save_before_ct` reference frames,
under the same clause.

A derivation forces it independently of any measurement: I.5.2's multiplier is
`mXDC = (1 << 16) * m_x_lf_unscaled / (global_scale * quant_lf)`, and
`global_scale`/`quant_lf` come from the `Quantizer` bundle, which Table G.1
puts in `LfGlobal` **only for `encoding == kVarDCT`**. A `kModular` LF frame
carries `LfChannelDequantization` but no `Quantizer`, so `mXDC` is not
computable for it and the raw integers are in no shared scale.

Probed anyway, because a derivation that is never executed is a hypothesis.
With the constant `false` (raw integers stored, no pre-step):

| stream | `true` | `false` |
| --- | --- | --- |
| fixture 72 | peak 1.1e-6 | peak **2.9e4** |
| corpus `progressive` | peak 2.0e-5 | peak **9.3e10** |

**Verdict: `LF_FRAME_IS_XYB_PRESTEP = true`.**

Two consequences of the same skip, both taken and neither separately probed
(no stream distinguishes them, because both follow from I.5.2 being skipped):

* **LF chroma-from-luma does not run.** I.6 states it directly: LF
  coefficients are not modified when `flags.kUseLfFrame` is true.
* **Adaptive LF smoothing does not run**, because I.5.2 is where it lives and
  I.5.2 is skipped in its entirety. Note that the corpus stream's
  `kSkipAdaptiveLFSmoothing` bit is *clear* — under a reading where the
  smoothing survived the skip, it would run. It does not, and the stream grades
  at 2.0e-5.

Also taken from G.2.2: with `kUseLfFrame` every `LfQuant` sample counts as
`-inf`, so no `lf_threshold` is ever exceeded and I.4's `lf_idx` is zero
regardless of the thresholds. That is `HfGroupParams::lf_idx_is_zero`, which
slice 8C had already provided and nothing until now could set.

## Verdict 3 — `G42_SIZE_TEST_IS_SHIFTED` = `true` (new flip point)

This one the stream found, not the text.

G.1.3 stops the `GlobalModular` decode at the first channel that is not "at
most `group_dim`" in both dimensions. G.4.2 then admits a channel to group
decoding only if "the channel dimensions exceed `group_dim` x `group_dim`".
Read with unshifted dimensions the two tests leave a hole, because a Squeeze
pyramid's channel list is **not** monotone in size. The `progressive` LF frame
is exactly that shape:

```
index 36   254x338   hshift 1  vshift 0     > group_dim, group-decoded
index 37   127x169   hshift 2  vshift 1     <= group_dim -- decoded by nothing
index 38   127x169   hshift 2  vshift 1     <= group_dim -- decoded by nothing
index 39   254x338   hshift 1  vshift 0
index 40   254x338   hshift 1  vshift 0
index 41   254x169   hshift 1  vshift 1     <= group_dim -- decoded by nothing
index 42   254x169   hshift 1  vshift 1     <= group_dim -- decoded by nothing
```

`GlobalModular` stopped at index 36; channels 37, 38, 41 and 42 sit after it,
so they are not decoded there, and under the literal unshifted comparison they
are not decoded in groups either. They would stay zero — and, worse, the
pass-group sub-bitstream would be missing four channels, so it desynchronises
immediately.

The reading shipped compares against the channel's **own** grid,
`group_dim >> hshift` by `group_dim >> vshift` — the same right shift G.2.3 and
G.4.2 apply to the group rectangle two sentences later, and the shift that
makes the four group rectangles tile the channel exactly (verified: channel 37
is tiled 64+63 by 128+41 = 127x169). Under it every channel after G.1.3's stop
exceeds its shifted group, the hole closes, and "remaining channel" means what
the NOTE at G.1.3 says it means.

Evidence, and it is the section-exhaustion gate at its sharpest: under `false`
the LF frame's first pass-group section fails with

```
out of bounds: 16 bit(s) requested at bit position 128096
```

`128096 / 8 = 16012`, which is *exactly* the TOC length of that section. The
decode runs off the end of the section it is reading, by the amount four
missing channels would cost. Under `true` the section is consumed to the bit
and the whole frame decodes.

The two readings agree on every channel whose shifts are zero, which is every
non-Squeeze stream — all of slices 5-9's modular fixtures are unaffected, and
the full suite is green under `true`.

**Verdict: `G42_SIZE_TEST_IS_SHIFTED = true`.** Caveat: this is one stream's
worth of evidence for a reading that departs from the literal words of G.4.2.
It is the reading that makes the clause self-consistent, but a second Squeezed
multi-group stream with a different pyramid shape would strengthen it.

## What this pass did not settle

* `PATCH_REFERENCE_IS_CANVAS_COORDINATES` stays unexercised: `progressive`'s
  reference frame is at crop origin `(0, 0)`, where both readings coincide,
  exactly as `bike`/`bike_5` are.
* Whether L.2.2's pre-step or something else applies to a **`kVarDCT`** LF
  frame. No encoder emits one; `decode_lf_frame` refuses it rather than
  guessing.
* `lf_level > 1`, and an LF frame that itself consumes a lower-level LF frame.
  `cjxl --progressive_dc=2` was not pursued; the slot array is sized for it
  (`NUM_LF_FRAME_SLOTS = 4`) and the read index is `frame_header.lf_level`,
  but nothing exercises the chained case.
* `num_hf_presets > 1` and a nonzero `lf_idx` remain unexercised, as 8C and 8F
  reported. `progressive` pins `lf_idx` to zero by construction, so it moves
  that one *further* out of reach rather than closer.

## Reproducing

```
cd JPXL
tools/make-progressive-fixtures.sh          # once, needs the pinned oracles
cargo test --release -p jpxl-decode --test e2e_progressive -- --nocapture
```

The corpus rung is 4064x2704: about 20 s in a release build and about
4.5 minutes unoptimized, so it runs unconditionally under `--release` and
otherwise only when `JPXL_SLOW_TESTS` is set. Every rung prints its peak and
per-channel RMSE whether it passes or fails, so a probe comparison does not
depend on a threshold being crossed. Then edit the constant named in a verdict
above and re-run.

---

## Addendum, 2026-08-04 — the LF-group seam is fixed; smoothing is frame-wide

*Appended after the fact; nothing above is edited. Follow-up task in the same
file set.*

### The diagnosis this confirms is not mine

`2026-08-04-negative-transfer-function-branch.md` §6 localised `bike`'s last
failure to a 16-row band at `y = 2039..2056` — the frame's only internal
LF-group boundary — and named the cause: `vardct::lf::adaptive_smoothing` was
driven once per LF group, so it skipped the first and last row and column of
*every group*, not only the frame's own edges. That agent also committed the
corpus-free reproducer, fixture 64
(`64_vardct_lfgroupseam_rgb_128x2176_nofilters_d6.jxl`, `-d 6 -e 3` because at
`-d 1` cjxl sets `kSkipAdaptiveLFSmoothing` and the pass never runs). The
evidence and the localisation are theirs; this addendum records the clause
check, the fix, and the numbers.

### What the clause says

Re-read from `latex/part1.tex`, I.5.2, source page 68. Two sentences scope the
pass, and both name the frame:

* the pass runs "for each LF sample **of the image** that is not in the first
  or last row or column";
* it closes by having I.8 read "the corresponding samples from the dequantized
  **LF image**".

Neither says LF *group*. The preceding sentence does — "for each LF group, the
dequantization process is influenced by the number `extra_precision`" — and
that is exactly right, because `extra_precision` is a per-LF-group field
(G.2.2). So the clause distinguishes the two scopes in adjacent sentences: the
dequantization multiply is per group, the smoothing pass is per image.

### Why assembling after I.6 is sound

The order I.5.2 fixes is dequantize, then I.6 chroma-from-luma, then smooth.
The first two are per-sample:

* dequantization uses the group's own `extra_precision` and the frame's
  `mXDC`/`mYDC`/`mBDC`;
* I.6's **LF** arm uses `x_factor_lf - 128` and `b_factor_lf - 128`, which are
  frame-wide constants from I.2.3 — unlike the HF arm, which reads `XFromY`
  and `BFromY` at the 64x64 tile containing the sample.

Neither reads a neighbouring sample, so running them per group and assembling
afterwards produces the same image as assembling first. Smoothing is the only
stage with a spatial footprint, and it is the only one moved. That is what
`decode::smooth_lf_image` does: gather the per-group planes into one
frame-wide LF image, run `adaptive_smoothing` once, scatter back.

### Interaction with `kUseLfFrame`

None, and that is normative rather than incidental: I.5.2 opens by saying the
subclause is skipped in its entirety for a frame with `kUseLfFrame` set, and
the smoothing paragraph is inside I.5.2. So `smooth_lf_image` is gated on
`!use_lf_frame` as well as on `!kSkipAdaptiveLFSmoothing`.

Worth stating because the corpus stream would otherwise look like a
counter-example: `progressive`'s regular frame has **four** LF groups and
therefore internal seams, and its `kSkipAdaptiveLFSmoothing` bit is *clear* —
under a reading where smoothing survived the `kUseLfFrame` skip it would run,
and run across those seams. It does not run at all, and the case grades at
2.0e-5. The main report above already recorded that as the untested half of
the skip; this is the same conclusion from the other side.

### Numbers

| case | before | after | limit |
| --- | --- | --- | --- |
| fixture 64 (128x2176 RGB, one internal seam) | peak 1.06e-2, all of it in rows 2039..2056 | **peak 2.7e-6** | 0.004 / 1e-5 |
| corpus `bike` | peak 3.17e-2 (B) / 1.79e-2 (X) / 1.34e-2 (Y) | **peak 2.5e-4**, RMSE 6.6e-7 | 0.007 / 1e-4 |
| corpus `bike_5` | identical to `bike` | **peak 2.5e-4**, RMSE 6.6e-7 | 0.06 / 0.02 |

The band does not shrink, it disappears: fixture 64's post-fix peak (2.7e-6)
is the same float-noise floor as its off-seam error was before the fix
(3.3e-6), and is in line with every single-LF-group fixture in
`e2e_vardct.rs`. `bike`'s residual 2.5e-4 is on X only and is the same order
as corpus `grayscale`'s long-standing 2.3e-4.

No case regressed. The progressive ladder (fixtures 70-74) and corpus
`progressive`/`progressive_5` are unchanged to the digit, as they must be:
every one of them either has a single LF group or takes the `kUseLfFrame`
path, so none of them can reach the smoothing pass.

### Consequences

* `decode::smooth_lf_image` added; `dequantize_lf` is now called with
  `smoothing_enabled = false` on the production path, and its doc comment and
  `vardct/lf.rs`'s module **Scope** note say why.
* `e2e_vardct.rs`: `fixture_64_rgb_lf_group_seam_nofilters_d6`, `corpus_bike`
  and `corpus_bike_5` un-ignored. That file now has **13 passing tests and
  none ignored**. Fixture 64 is the standing regression test — it is the only
  fixture in the tree with more than one LF group, so nothing else would
  notice a relapse.
* One cost worth knowing: un-ignoring the two `bike` cases (2048x2560 each)
  takes `cargo test -p jpxl-decode --test e2e_vardct` from under a second to
  about 78 s in a debug build (6 s in release). That is the price of grading
  two 5-Mpixel corpus cases by default.
