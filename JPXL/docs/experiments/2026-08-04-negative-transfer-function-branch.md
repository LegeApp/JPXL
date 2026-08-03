# Table E.6 transfer functions below zero: sRGB is odd, BT.709 is not

Date: 2026-08-04
Status: **RESOLVED by evidence**, in both directions, against a published
conformance reference (BT.709) and the pinned oracle (sRGB). Flip point
`jpxl_core::color::NEGATIVES_TAKE_THE_LINEAR_SEGMENT = [false, true]`.

## 1. Question

18181-1 L.2.2 hands the decoder **linear** light and says nothing about
range: the inverse XYB transform is explicitly allowed to produce values
outside the coded gamut, and 18181-3 §4.2 forbids clipping before the
conformance comparison. The signalled transfer function (Table E.6) is then
applied to those samples, so the OETF has to be evaluated at **negative**
arguments.

Neither 18181-1 nor the standards Table E.6 names — IEC 61966-2-1 for
`kSRGB`, ITU-R BT.709-6 for `k709` — defines the curve below zero. Both print
a two-branch function whose stated domain begins at zero:

```text
sRGB     E' = 12.92 E                for E <= 0.0031308
         E' = 1.055 E^(1/2.4) - 0.055 otherwise
BT.709   E' = 4.5 E                  for E < 0.018
         E' = 1.099 E^0.45 - 0.099    otherwise
```

Two extensions are defensible:

* **literal** — evaluate the printed condition on the *signed* value. Every
  negative argument is below the toe threshold, so it takes the linear branch:
  `12.92 * v`, `4.5 * v`.
* **odd** — extend with odd symmetry, `f(-v) == -f(v)`, so a negative argument
  takes the *power* branch on its magnitude.

They disagree by a lot, and the disagreement grows without bound: at
`v = -0.13` BT.709 gives `-0.585` literally and `-0.341` odd; at `v = -0.5`
sRGB gives `-6.46` literally and `-0.74` odd.

JPXL originally chose **odd** for both curves, with a docstring arguing it was
"the only extension that is continuous, monotonic and sign-preserving". That
argument is wrong on its own terms — the literal reading is also continuous,
monotonic and sign-preserving, and it is continuous *at the same knot the
standard already places*. Neither argument decides anything; only measurement
does.

## 2. Preregistered gate

For each curve separately:

* **Resolved:** one reading reproduces an independent reference decode to
  within the "no filters" Part 3 error class (peak 0.004) over an image with
  a large population of strongly negative samples, and the other misses by
  more than an order of magnitude.
* **Unresolved:** both readings land in the same class, or neither does (which
  would mean the divergence is somewhere else).

The two curves are gated independently and *may answer differently* — nothing
requires libjxl, or the standard, to treat them the same way.

## 3. Method

### BT.709 — the `bike` conformance case

`tests/fixtures/conformance/testcases/bike` signals `Transfer function: 709`
(confirmed with `jxlinfo -v`). Its `reference_image.npy` is **published with
the corpus**, not produced here, which makes it the strongest evidence
available: it is the one artefact in this repository that is not an oracle
run.

JPXL's decode of `bike` was dumped to `.npy` and differenced against the
reference sample by sample. The residual split cleanly into two populations,
which is what made the measurement possible at all:

* **Isolated 1-6 sample spikes**, 47 of them above 0.02, peak **0.2466**, all
  in the output **B** channel, all at pixels whose B is strongly negative. R
  and G at those same pixels agreed to `~1e-6` — the residual lay along the
  "linear blue only" direction.
* A **band at `y = 2039..2056`**, all three channels, peak 0.0317. That is a
  different bug (see §6) and is excluded from this measurement.

That R and G agree to `1e-6` while B is off by 0.2466 is by itself decisive
about *where* the divergence is. The inverse XYB transform mixes all three
XYB components into all three outputs; no perturbation of X, Y or B upstream
of L.2.2 can move one output channel by 0.25 and leave the other two at
`1e-6`. Whatever differs, differs **after** L.2.2 — and the only per-channel
stage after L.2.2 is the transfer function.

Taking three spikes, inverting JPXL's (odd) encoding to linear and fitting a
power law between JPXL's linear value and the reference's *encoded* value:

| JPXL linear | reference encoded |
| --- | --- |
| -0.130490 | -0.5871054 |
| -0.097893 | -0.4404343 |
| -0.059726 | -0.2688831 |

log-log slopes between consecutive pairs: **1.00007** and **0.99927**. A slope
of 1 means the reference's negative branch is *linear*, and the constant is
`0.5871054 / 0.130490 = 4.4993` — i.e. **4.5**, BT.709's toe slope.

### sRGB — a synthetic out-of-gamut fixture

No corpus case with `kSRGB` in this repository has a large negative
population, so one was constructed: 64x64 fully saturated primaries in 8x8
tiles separated by 2-pixel black rules, encoded at `-d 4.0 --gaborish=0
--epf=0 -e 7`. In the pinned `djxl` reference decode, 3288 of 12288 samples
are below -0.0031308 and 1196 are below -0.05, with a minimum of -0.325. This
is fixture `63_vardct_outofgamut_rgb_64x64_nofilters_d4`.

JPXL's decode (using the literal branch) was inverted to linear, and the
reference was inverted under each candidate rule; the two linear images were
then differenced.

## 4. Result

| curve | reading | linear-domain peak residual vs reference |
| --- | --- | --- |
| BT.709 (`bike`) | odd | **0.2466** (encoded domain) |
| BT.709 (`bike`) | literal | **0.0317** (encoded domain; entirely the §6 band, zero elsewhere) |
| sRGB (fixture 63) | literal | **0.1105** |
| sRGB (fixture 63) | odd | **8.11e-6** |

Both gates pass, and **they pass in opposite directions**:

* `k709` takes the **literal** reading: negative samples encode as `4.5 * v`.
  Applying it to JPXL's `bike` decode removed *every* one of the 2433
  over-threshold B samples outside the §6 band, leaving nothing above 0.007
  anywhere except that band. This is not a fit to three points; it is a
  whole-image identity.
* `kSRGB` takes the **odd** reading, to `8e-6` over an image built to make the
  question as loud as possible. Under the literal reading the same fixture is
  off by more than 1.0 at its worst sample.

## 5. Decision

```rust
pub const NEGATIVES_TAKE_THE_LINEAR_SEGMENT: [bool; 2] = [false, true];
//                                            sRGB ---^      ^--- BT.709
```

`linear_to_gamma` (Table E.7's pure power law) is not covered by the flip
point: it has no linear branch to fall back on, so odd symmetry is the only
continuous extension available and it stays odd.

Pinned by `color::tests::negatives_take_the_branch_each_curve_was_measured_to_take`
(the constant and both curves' values at `v = -0.13`) and end to end, for the
sRGB half, by fixture 63's rung in `tests/e2e_vardct.rs`. The BT.709 half is
pinned by the `bike` / `bike_5` corpus rungs, which need the corpus fetched.

**Caveat on scope.** This is a measurement of what the reference decodes do,
not a derivation from normative text — the normative text does not answer.
The asymmetry between the two curves is itself the reason to distrust any
argument-from-elegance here: whatever produced these references treats the two
curves differently, so a future third curve (`kPQ`, `kHLG`, `kDCI`) must be
measured, not inferred from these two.

## 6. Not settled by this experiment

`bike`'s residual band at `y = 2039..2056` survives this fix at peak 0.0317
(B), 0.0179 (X), 0.0134 (Y), against a threshold of 0.007. `y = 2048` is
`bike`'s only internal **LF-group** boundary (LF groups are 2048x2048; the
frame is 2048x2560). The band is 16 rows wide — the two 8x8 block rows either
side of the seam — plus the filters' spread.

Reproduced minimally, outside the corpus: a 256x2176 synthetic encoded at
`-d 6 -e 3 --gaborish=0 --epf=0` has **its entire error budget** at that seam
(overall peak 0.0134, band peak 0.0134, everything else exact); a 2176x128
one shows the same at the vertical seam, columns 2040..2055. At `-d 1` the
seam vanishes, which is consistent with `kSkipAdaptiveLFSmoothing`.

Diagnosis: `vardct::lf::adaptive_smoothing` is invoked per LF group and skips
the first and last row and column *of the LF group's own plane*. I.5.2 says
the pass runs "for each LF sample **of the image** that is not in the first or
last row or column" — the frame-wide LF image, so only the frame's own edges
are excluded and an internal LF-group seam must be smoothed across. Fixing it
means assembling the frame-wide LF image before I.6/I.5.2 rather than after,
which is `vardct/lf.rs` plus `decode.rs` — outside the file set of the brief
that produced this note, and therefore reported rather than fixed.
