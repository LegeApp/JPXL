# J.4 edge-preserving filter: four flip points and the J.3/J.4 OCR reconciliation

Date: 2026-08-03
Status: complete as a source-reconciliation record; the four flip points are
**UNRESOLVED by evidence** and carry the reading argued below until a probe
pass runs (see §6).

## 1. Question

Sub-slice 8E implements the restoration filters (18181-1 J.3 and J.4.1–J.4.4)
as pure functions over f32 planes. Two questions had to be settled before a
line of pixel code could be written:

1. Where do `latex/part1.tex` and `markdowns/standard-markdowns/part1.md`
   disagree on J.3/J.4, and which source wins in each place?
2. Which of the underdetermined readings of J.4 does the clause best support,
   and can any of them be decided today?

## 2. Preregistered gate

* **Source reconciliation — pass:** every numeric constant, coordinate list
  and default used by `gaborish.rs`/`epf.rs` is traced to one of the two
  sources, and every disagreement is resolved by an argument that does not
  depend on which source is being read (self-consistency of the clause,
  symmetry, or a count the prose states independently). **Fail:** a value is
  taken from one source because it was read first.
* **Flip points — resolved:** an argument from the clause text alone forces
  one reading, or a decoding experiment separates them. **Unresolved:** the
  reading is chosen on the balance of the clause's own wording and pinned by a
  named constant plus a unit test that fails if the constant is flipped
  without changing the test. Wave 3 (8F) has an end-to-end VarDCT decode and a
  reference `.npy`; only then can a flip point be decided by evidence.
* No oracle run was preregistered: nothing in 8E is reachable end to end at
  the time of writing, so any oracle comparison would have been of a pipeline
  that does not exist.

## 3. Method

Sources read first-hand: `latex/part1.tex` lines 4930–5150 (Table J.1 and
clauses J.1–J.4.4 in full, plus 5.2 `Mirror1D` at 675–690) and
`markdowns/standard-markdowns/part1.md` lines 3350–3495 (the same clauses).
No PDF page read was needed; no libjxl source was read; no binary was run.

Per `AGENTS.md` §2 the LaTeX is canonical for Part 1 and the markdown is a
secondary grep aid, with the standing exception that a **numeric or tabular**
disagreement is itself a signal to be settled on the merits.

## 4. Raw results

### 4.1 Where the two sources disagree

| item | `latex/part1.tex` | `part1.md` | taken | why |
| --- | --- | --- | --- | --- |
| J.4.4 step-0 kernel, 13th coordinate | `{9, -2}` | `{0, -2}` | **markdown** | `{9,-2}` has L1 distance 11, but J.4.1 says the step covers "the twelve neighbouring pixels that have an L1 distance of at most 2". `{0,-2}` is the unique entry that makes the set the complete L1<=2 ball (1 + 4 + 8 = 13 taps) and makes it closed under negation and axis swap. The prose count is an independent check, so this does not rest on trusting the markdown. |
| Table J.1 `epf_quant_mul` row | present, `F16()`, default `0.46` | **row absent** (only the stray fragment `kVarDCT` survives) | **LaTeX** | The markdown lost the row wholesale; a missing row is not a disagreement about a value. |
| Table J.1 `epf_sigma_for_modular` row | present, `F16()`, default `1.0` | **row absent** (only `encoding == kModular` survives) | **LaTeX** | As above. |
| Table J.1 `epf_pass0_sigma_scale` default | `0.9` | `09` | **LaTeX** | The markdown dropped the decimal point. |
| Table J.1 `epf_weight_custom` trailing `u(32)` "(ignored)" row | present | present | both agree | Already consumed by `read_restoration_filter`, and covered by the existing test `weight_custom_consumes_the_ignored_u32`. No action for 8E. |
| J.4.2 `DistanceStep2` body | readable | collapsed into a mangled markdown table | **LaTeX** | Structural garble, not a value disagreement. |

Both `0.46` and `1.0` were already the defaults in
`frame/restoration.rs::EpfParams::default`, so the reconciliation confirms the
parser rather than changing it. `epf_channel_scale = {40.0, 5.0, 3.5}`,
`epf_border_sad_mul = 2/3`, `epf_pass2_sigma_scale = 6.5`, the sharpness LUT
`{0, 1/7, …, 1}` and the gaborish defaults `0.115169525`/`0.061248592` agree
in both sources.

### 4.2 J.3, and one thing that is *not* ambiguous

The rescale is stated plainly: the unnormalized kernel is centre 1, four edge
neighbours `gab_C_weight1`, four corners `gab_C_weight2`, and the nine weights
are "rescaled uniformly … such that [they] sum to 1". Uniformly means one
factor `1 / (1 + 4*w1 + 4*w2)` applied to all nine, the centre included —
there is no reading in which the centre stays 1. Both sources agree verbatim.
`GaborKernel::new` is the single place this is computed; the exit test asserts
`sum() == 1` for the defaults and for three adversarial weight pairs, and
`1 + 4*w1 + 4*w2 == 0` is rejected rather than divided by.

### 4.3 The four flip points

**F1 — `epf_iters` to step selection.** J.4.1 gives three explicit conditions:
step 0 iff `epf_iters == 3`; step 1 "is always done (if `epf_iters > 0`)";
step 2 iff `epf_iters >= 2`. That yields `1 -> {1}`, `2 -> {1,2}`,
`3 -> {0,1,2}`. The competing pull is the field *name*, which suggests "run
the first N steps": `1 -> {0}`, `2 -> {0,1}`.

Observation that settles it on the text alone as far as text can: the explicit
conditions already run **exactly `epf_iters` steps** for every legal value, so
they satisfy everything the name suggests, while the prefix reading has to
contradict the sentence "the second step is always done". Constant
`EPF_STEPS_FROM_EXPLICIT_CONDITIONS = true`.

**F2 — the `epf_border_sad_mul` predicate.** J.4.3: `position_multiplier` is
`rf.epf_border_sad_mul` when "either coordinate of the reference sample is 0
or 7 UMod 8". Four combinations were considered (frame-absolute vs
block-relative coordinates × reference pixel vs per-tap).

*The coordinate axis collapses.* The 8x8 block grid is aligned to the frame
origin, so a frame coordinate's residue mod 8 **is** its offset inside its
block. The two "readings" are the same predicate and no constant is needed;
this is recorded because the ambiguity was raised in scoping and is now
retired.

*The remaining axis is reference-pixel vs per-tap.* `Weight(distance, sigma)`
takes no tap argument — it cannot evaluate a per-tap predicate — and the
clause says "the reference sample", not "the neighbouring sample". Constant
`EPF_BORDER_SAD_AT_REFERENCE_PIXEL = true`; `false` evaluates the predicate at
each tap's unmirrored frame coordinate.

**F3 — `sigma < 0.3` skip granularity.** J.4.3 defines sigma at "the 8 x 8
rectangle containing the reference pixel" and then says "if sigma < 0.3 for a
given **varblock**, the decoder skips all steps on the pixels of that block".
The two differ only where `Sharpness` varies within a varblock larger than
8x8, since sigma's other factor (`mul`, I.5.3) is constant per varblock.

The clause's own word is "varblock", so `EPF_SKIP_IS_PER_VARBLOCK = true`: the
skip test consults an optional per-varblock sigma the caller supplies
(`SigmaField::with_varblock_sigma`), while the *weights* keep using the 8x8
block's own sigma, which is what the sentence two lines earlier defines. With
no varblock sigma supplied (Modular, where sigma is the uniform
`epf_sigma_for_modular`) the two readings coincide, so the flip is inert
outside VarDCT.

**F4 — guide buffer vs input buffer.** J.4.2 measures distances on
`sample(x, y, c)`; J.4.4 accumulates `input(x + ix, y + iy, c)`. Neither name
is defined in terms of the other. J.4.1 mentions "guide or input pixels",
which hints at two buffers — but no clause anywhere constructs a guide buffer
or says what it would contain, while J.4.1 does state plainly that each step's
output is the next step's input.

Taken literally there is one buffer per step and the two names denote it.
`EPF_DISTANCE_USES_STEP_INPUT = true`; `false` measures every step's distances
against the filter's original (pre-EPF) input, which is the only concrete
two-buffer reading the text admits.

### 4.4 What the unit tests pin

* `rescaled_kernel_sums_to_one`, `constant_plane_is_unchanged` — the J.3
  normalization; a missing or partial rescale breaks both.
* `hand_computed_impulse_response`, `hand_computed_mirroring_on_a_one_pixel_wide_plane` —
  the kernel written out longhand, and the collapse of the horizontal taps on
  a width-1 plane.
* `mirror1d_terminates_on_a_one_sample_axis`, `mirror1d_matches_the_clause_on_a_wide_plane` —
  5.2 including the degenerate 1- and 2-wide axes, where a single reflection
  lands outside the opposite edge.
* `step0_kernel_is_the_thirteen_pixels_within_l1_two` — the `{0,-2}` decision,
  by count, L1 bound and symmetry rather than by citing a source.
* `centre_weight_is_exactly_one`, `output_is_a_convex_combination_of_the_input` —
  `sum_weights >= 1`, hence no division blow-up.
* `sigma_below_threshold_is_the_identity` — the skip rule.
* `steps_follow_the_explicit_conditions` — F1, including the "exactly
  `epf_iters` steps" property that motivates the choice.
* `per_varblock_skip_overrides_the_block_sigma` — F3, and it asserts the
  *other* branch when the constant is flipped, so flipping it does not
  silently pass.
* `hand_computed_step2_on_a_three_pixel_row` — the whole distance → weight →
  weighted-average chain on numbers computed on paper (including the border
  multiplier and the mirroring of the vertical taps).

## 5. Conclusion

The J.3/J.4 constants are reconciled across both transcriptions, and in every
disagreement the chosen value is forced by something the clause states
independently (a count, a symmetry, or a missing row), not by source priority.
The step-0 kernel's 13th coordinate is `{0, -2}`: a markdown-beats-LaTeX case,
the second one on record after Table I.7's Order-ID column.

None of the four flip points is decided by evidence. Each is a named constant
with the argued reading, each has a test that would notice a silent flip, and
F2's coordinate axis is retired as a non-ambiguity. This establishes what the
clause says; it does not establish what any encoder emits, and no implementation
was consulted.

## 6. Consequences

* New `jpxl-decode/src/frame/gaborish.rs` (J.3 plus the shared 5.2 `Mirror1D`)
  and `frame/epf.rs` (J.4.1–J.4.4), both pure functions over f32 planes.
* Four named constants, all in `frame/epf.rs`:
  `EPF_STEPS_FROM_EXPLICIT_CONDITIONS`, `EPF_BORDER_SAD_AT_REFERENCE_PIXEL`,
  `EPF_SKIP_IS_PER_VARBLOCK`, `EPF_DISTANCE_USES_STEP_INPUT`.
* **For 8F (wave 3):** once a filters-on VarDCT fixture decodes end to end,
  each constant is a one-line flip and a re-run of the peak/RMSE comparison
  against the reference `.npy`. F1 and F4 should move the whole image and be
  easy to separate; F2 should show up as a per-block-edge pattern; F3 needs a
  fixture whose varblocks exceed 8x8 *and* whose `Sharpness` varies inside
  one, so it may stay unexercised — in which case that is a negative result to
  record, not a failure.
* `frame/restoration.rs` was not modified: its `epf_quant_mul = 0.46`,
  `epf_sigma_for_modular = 1.0` and the trailing `u(32)` read-and-discard are
  all confirmed correct by this reconciliation.
