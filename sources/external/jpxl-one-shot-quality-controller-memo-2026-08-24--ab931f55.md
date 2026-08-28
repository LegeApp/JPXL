# Design memo: JPXL score-targeted encoding

## Executive verdict

JPXL should implement **common-case one shot**, not claim true one shot.

A source- or transform-derived statistical predictor cannot guarantee the canonical decoded SSIMULACRA2 score of an arbitrary image. A conservative reserve can reduce the miss rate, but it cannot turn an empirical predictor into a proof. The only generally defensible true-one-shot guarantees would be:

1. encode losslessly, or
2. derive a proven worst-case relationship between quantization and SSIMULACRA2.

Neither is presently available as a byte-competitive lossy controller.

The existing hard-floor contract can nevertheless remain intact:

> A successful result is emitted only after its actual reconstructed pixels have been canonically scored at or above the requested target.

The normal path should perform one predicted pixel plan, one reconstruction/metric evaluation, and one entropy attachment/emission. A miss should receive one slope-based corrective plan. Broad uncertainty, out-of-distribution content, likely saturation, or a second miss should route into the existing bounded controller while reusing all work already performed.

This directly preserves the handoff’s requirement that successful output not silently fall below the target, while keeping saturation and work exhaustion explicit. 

---

## Review scope and limitations

I reviewed:

* the attached advisor handoff;
* `quality.rs`, `quality_features.rs`, `quality_predictor.rs`, `rate.rs`;
* the public `jpxl` facade and CLI quality path;
* `CandidateSearchContext`, `CandidateForwardCache`, the cover/CfL planning order, and entropy attachment;
* `calibrate_initial_rung.py`;
* `quality-corpus.json`;
* the included AKR decisions and evidence summaries.

I did not inspect or derive anything from the included libjxl source. That remains consistent with the clean-room constraint. 

The raw `.agent/scratch` evidence named in the handoff was excluded from the agent pack, so I could not independently recalculate those rows from their JSONL files. I treated the handoff and AKR records as authoritative for those measurements. The environment also lacks the Rust toolchain, so this is a static code and evidence review rather than a fresh test run.

---

# 1. Immediate code findings that should be fixed first

## 1.1 The public facade currently violates the hard-floor contract

This is the most important finding.

In `crates/jpxl-encode-policy/src/quality.rs:1109-1128`, when no probe meets the requested score, the controller deliberately selects the finest verified under-target probe and exact-prices it:

```rust
// Nothing met the target: emit the finest verified probe and say so
under_target = true;
...
ordered.push(index);
```

At `quality.rs:1684-1716`, both `SaturatedTop` and `UnderTargetWorkCap` still produce a normal `QualityOutcome` containing `codestream: winner.bytes`.

The public facade then wraps those bytes and returns `Ok` at:

* `crates/jpxl/src/lib.rs:692-736`
* especially `lib.rs:708` and `lib.rs:723-736`.

The public status documentation is itself explicit:

```rust
/// The bounded controller ran out of probes before any candidate met the
/// score; the emitted stream's `achieved_score` is below the request.
UnderTargetWorkCap,
```

Finally, `crates/jpxl-cli/src/main.rs:2455-2495` treats every `Ok(Perceptual(...))` as a successful encode and returns the bytes for writing.

That is incompatible with the handoff’s stated minimum-score semantics. 

### Required correction

The default hard-floor API must not return an under-target codestream as a successful result.

A minimally disruptive API would be:

```rust
pub enum PerceptualFailureKind {
    LossyLadderSaturated,
    WorkBudgetExhausted,
}

pub struct PerceptualFailure {
    pub kind: PerceptualFailureKind,
    pub requested_score: f64,
    pub best_verified_score: f64,
    pub best_rung: u32,
    pub probes: u32,
    pub structural_builds: u32,
    pub metric_version: MetricVersion,
    pub trace_json: Option<String>,
}
```

Then either:

```rust
Error::PerceptualTargetNotMet(PerceptualFailure)
```

or a dedicated result enum should be returned. The CLI should exit nonzero and should not create or replace the requested output file.

Two explicitly different fallback modes may be offered:

* `--quality-fallback=lossless`: emit a lossless stream and report that the lossy ladder saturated.
* `--quality-fallback=best-effort`: knowingly emit the best verified under-target stream, with a distinct contract and status.

Neither should happen silently.

`SaturatedFloor` remains a successful result. `SaturatedTop` and `UnderTargetWorkCap` do not.

---

## 1.2 The nominal probe budget is not actually hard

`quality.rs:907-921` defines:

```rust
/// One probe beyond the budget
fn rescue_probe(...)
```

The status documentation at `quality.rs:265` also refers to “the probe budget (plus its one rescue probe).”

This conflicts with comments and tests that describe `pixel_probes` as a hard cap. It also makes the requested work-count table misleading.

The correction should be one of:

* reserve the final slot for rescue inside `pixel_probes`, or
* expose separate counters and limits such as `navigation_probes` and `rescue_probes`.

I recommend the first. Balanced should have a **total maximum of five canonical pixel probes**, not five plus a conditionally hidden sixth. The new predictive path should continue the same total budget rather than receiving one or two prediction attempts and then restarting a five-probe controller.

---

## 1.3 The current predictor is trained on the wrong endpoint

The current generated predictor is not merely too small. Its labels do not correspond to the production decision that the proposed model must make.

`quality_predictor.rs:8-11` states that it predicts fixed-quantizer `global_scale` with `HfMul = 1`.

`calibrate_initial_rung.py:34-46` confirms that the calibration ladder ends at `global_scale = 73728`, while the production effective-scale ladder continues far beyond that by increasing `HfMul`.

The production decision also includes quantizer-dependent structure:

```text
quantizer
  -> AQ setup
  -> HF quantizers
  -> cover selection
  -> selected transforms
  -> CfL
  -> quantization
  -> reconstruction
```

A fixed-global-scale calibration point is therefore not necessarily the same operating point as the final production Balanced stream, especially when a fresh-structure rescue changes cover or CfL.

The replacement labels should be:

> The coarsest effective rung whose **fresh production Balanced pixel plan** meets the target, over the complete effective-scale ladder, with saturation represented as censoring rather than as a clamped crossing.

The current final corrected result can be recorded as a secondary label, but the oracle training label should not inherit the bounded navigator’s own approximation errors.

---

## 1.4 The runtime predictor ignores most of its available features

Although `SourceFeatures` contains:

* dimensions;
* grayscale;
* luma q10, q50, and q90;
* chroma q50;
* flat fraction;
* edge proxy;

`predicted_effective_scale()` in `quality.rs:613-663` uses only:

* `luma_variance_q50`;
* `flat_fraction`;
* target score.

The generated table’s `support` field is also unused. A cell supported by one image receives the same runtime trust as a well-supported cell.

This explains why the current predictor is useful as a navigation seed but not as a final controller. Its recorded median absolute log-scale error is 0.30, while p90 is 2.68—a roughly 14.6× scale-ratio error at the tail. High targets are also heavily censored by the fixed-scale ceiling. 

The replacement should not be an expanded bucket table. It needs a calibrated curve prediction, uncertainty, and explicit saturation/OOD handling.

---

## 1.5 Current “cheap” source features are not entirely free

`quality_features.rs:91-139` creates two full vectors of per-atom values and sorts both to obtain quantiles.

For a 50 MP image there are approximately 781,250 8×8 atoms. Two full `f32` arrays are not a major part of the encoder’s total memory, but two `O(n log n)` sorts are unnecessary for a supposedly cheap controller feature stage.

Use deterministic fixed-bin log histograms or a deterministic selection algorithm. Histograms are preferable because they also provide useful tail and distribution features without additional storage.

---

## 1.6 Existing controller tests are too idealized for this redesign

Most `quality.rs` controller tests use `CurveEvaluator`, where score is a monotone quantizer-only power law. That validates navigation mechanics but cannot expose:

* cover/CfL discontinuities;
* fresh-versus-reused structure changes;
* local score nonmonotonicity;
* saturation;
* content-class OOD;
* entropy-size inversions;
* public under-target success behavior.

The public quality test uses one synthetic image at targets 50, 70, and 85. It does not test the public API’s terminal miss semantics.

Add real-trace replay tests and explicit public tests asserting:

```rust
assert!(matches!(
    result,
    Err(Error::PerceptualTargetNotMet(...))
));
```

for both work exhaustion and lossy-ladder saturation.

---

## 1.7 The literal byte-minimization wording overstates the current controller

The current controller exact-prices at most the two coarsest feasible finalists:

```rust
let max_finalists = budget.exact_prices.clamp(1, 2);
```

It selects the smallest exact stream among those retained candidates, not the globally smallest JPXL codestream satisfying the score.

This is reasonable for a bounded-effort encoder, but the contract should say so:

> Produce a canonically verified stream meeting the requested score. Within the selected effort’s declared policy family and bounded evaluated candidate set, select the smallest exact-priced feasible finalist.

If “minimize exact codestream bytes” is interpreted literally over all possible quantizers, structures, restoration settings, and entropy outcomes, neither the current controller nor a one-shot predictor satisfies it.

This matters because exact entropy size can itself be locally nonmonotone. A one-shot controller can aim for the coarsest feasible rung, but it cannot prove that an unpriced neighboring rung would not entropy-code smaller.

---

# 2. Contract comparison

| Approach                                                      | Existing hard floor preserved?                                    | Main limitation                                                                         | Recommendation                                  |
| ------------------------------------------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ----------------------------------------------- |
| Conservative source-only prediction plus reserve              | **No**, not by itself                                             | Reserve gives empirical coverage, not a per-image guarantee; large reserve wastes bytes | Do not use as the correctness mechanism         |
| One predicted encode, verify once, explicit failure on miss   | **Yes for successful outputs**                                    | No repair; potentially poor completion rate                                             | Useful diagnostic mode, not primary public mode |
| Common-case one shot with one corrective re-plan              | **Yes**                                                           | Some inputs still require two plans or exact fallback                                   | **Recommended**                                 |
| Reduced-resolution or cheap pilot, then one final full encode | Only if the final result is also verified and failure is explicit | Pilot/full-resolution relationship is content-dependent; still adds work                | Test only if transform-derived prediction fails |
| Existing bounded exact controller for uncertainty/OOD         | **Yes after the facade fix**                                      | Expensive but bounded                                                                   | Recommended fallback                            |
| Percentile or expected-score promise without verification     | **No**                                                            | Changes `--quality` semantics                                                           | Only as a separately named API contract         |
| Always route uncertain cases to lossless                      | Yes                                                               | Potentially catastrophic byte expansion                                                 | Explicit opt-in fallback only                   |

A “score reserve” is still useful for choosing the first candidate or correction aim. It simply must not be described as the reason the floor is guaranteed. The canonical verification gate is the guarantee.

---

# 3. Proposed runtime architecture

## 3.1 High-level flow

```text
source pixels
  -> PreparedFrame + production AnalysisAtlas
  -> deterministic source summary
  -> request-scoped CandidateSearchContext
  -> fill/reuse quantizer-independent transform candidate cache
  -> deterministic transform summary
  -> QualityPredictionV2(target, effort, source, transform)
       outputs:
         median crossing
         risk-adjusted candidate crossing
         calibrated interval
         local loss slope
         saturation risk
         OOD flags
         structural-instability risk
  -> route:
       uncertain / OOD / likely saturated
         -> existing bounded exact navigator, seeded by prediction
       otherwise
         -> one fresh predicted pixel plan
         -> full reconstruction + canonical SSIMULACRA2
             meets tightly
               -> one entropy attachment
               -> one emission
             misses
               -> one slope-based corrective plan on warm transform cache
               -> reconstruct + score
               -> emit if feasible
               -> otherwise continue existing navigator with existing observations
             meets but substantially overshoots
               -> optional one coarsening attempt when predicted byte saving is material
               -> retain first feasible plan as fallback
  -> if terminal result is below target:
       explicit failure, lossless fallback, or explicit best-effort mode
```

The current architecture already provides most of the needed mechanical separation:

* `CandidateSearchContext::pixel_plan_for()` builds scored pixels without entropy.
* `attach_entropy_for()` trains entropy without changing pixels.
* `CandidateForwardCache` is request-scoped and quantizer-independent.
* cover/CfL rebuilds can reuse cached DCT coefficients.

That means a failed prediction does not need entropy training and does not need to recompute already cached transforms.

---

## 3.2 Suggested pseudocode

```rust
fn encode_score_targeted(
    ctx: &mut CandidateSearchContext<'_>,
    source: SourceFeatureSummary,
    target: f64,
    budget: QualityBudget,
) -> Result<MetStream, PerceptualFailure> {
    // Uses the same cache that final cover/CfL and quantization will consume.
    let transform = ctx.prepare_transform_summary()?;

    let prediction = QUALITY_MODEL.predict(
        target,
        ctx.request().rate_preset,
        &source,
        &transform,
    );

    if prediction.ood.any()
        || prediction.interval_log_width > FALLBACK_LOG_WIDTH
        || prediction.saturation_risk.is_high()
    {
        return continue_exact_controller(
            ctx,
            target,
            prediction.median_rung,
            Vec::new(),
            budget,
        );
    }

    let first_rung = prediction.candidate_rung.ceil_finer();
    let first = plan_and_score_fresh(ctx, first_rung, target)?;

    if first.score >= target {
        if first.score - target <= MET_OVERSHOOT_BAND
            || prediction.predicted_tightening_saving < 0.03
        {
            return entropy_attach_and_emit(ctx, first);
        }

        // Optional rare byte-tightening attempt. Keep `first` until the
        // coarser candidate has been verified.
        let tighter_rung =
            correction_rung(first_rung, first.score, target, prediction.local_beta, false);

        if tighter_rung < first_rung {
            let tighter = plan_and_score(ctx, tighter_rung, ReusePolicy::ByDistance)?;
            if tighter.score >= target {
                return entropy_attach_and_emit(ctx, tighter);
            }
        }

        return entropy_attach_and_emit(ctx, first);
    }

    let corrected_rung =
        correction_rung(first_rung, first.score, target, prediction.local_beta, true);

    let corrected = plan_and_score(ctx, corrected_rung, ReusePolicy::ByDistance)?;

    if corrected.score >= target {
        return entropy_attach_and_emit(ctx, corrected);
    }

    // Do not restart. Seed the existing navigator with both measured probes,
    // the warm transform cache, and the remaining total budget.
    continue_exact_controller(
        ctx,
        target,
        prediction.median_rung,
        vec![first.into_probe(), corrected.into_probe()],
        budget.remaining_after(2),
    )
}
```

The correction formula should use the same perceptual-loss domain already used by the navigator:

```text
L(q) = max(100 - q, 1e-3)
β    = -d ln(L) / d ln(scale), β > 0

ln(scale₂) =
    ln(scale₁) + [ln L(observed_score) - ln L(aim_score)] / β
```

Then:

* round to the finer legal rung when correcting a miss;
* clamp the jump to the existing bounded ratio;
* fall back to the current extrapolation logic when `β` is invalid;
* rebuild cover/CfL when the scale displacement exceeds the existing structural threshold or the model marks the region as structurally unstable.

Unlike the current first-rung predictor, this model supplies the local slope needed to use the first measured score efficiently rather than geometrically exploring again.

---

# 4. Exact work counts

“One forward transform” here means one request-scoped population of the candidate transform banks: each legal `(LF group, origin, transform type)` is computed at most once and reused. It does not mean that the image contains only one DCT operation.

| Runtime route                                          |   Transform-bank fills |  Pixel plans | Full-frame reconstructions | Canonical metric evaluations | Entropy trainings | Codestream emissions/exact prices | Fresh cover/CfL builds |
| ------------------------------------------------------ | ---------------------: | -----------: | -------------------------: | ---------------------------: | ----------------: | --------------------------------: | ---------------------: |
| Predicted normal success                               |                      1 |            1 |                          1 |                            1 |                 1 |                                 1 |                      1 |
| Predicted miss, one successful correction              |               1 shared |            2 |                          2 |                            2 |                 1 |                                 1 |                    1–2 |
| Predicted overshoot, one byte-tightening attempt       |               1 shared |            2 |                          2 |                            2 |                 1 |                                 1 |                    1–2 |
| Preflight uncertainty/OOD to Balanced exact controller |               1 shared |           ≤5 |                         ≤5 |                           ≤5 |                ≤2 |                                ≤2 |                     ≤2 |
| Prediction followed by exact continuation              |               1 shared | **≤5 total** |               **≤5 total** |                 **≤5 total** |          ≤2 total |                          ≤2 total |               ≤2 total |
| Saturation or work-cap failure under hard-floor API    |               1 shared |           ≤5 |                         ≤5 |                           ≤5 |                 0 |                                 0 |                     ≤2 |
| Explicit lossless fallback                             | Separate lossless path |            — |                          — |        optional verification |  lossless backend |                                 1 |                      — |

Important implementation details:

* Prediction attempts consume the normal total probe budget.
* There is no hidden sixth rescue probe.
* Failed plans receive no entropy training.
* The normal path never holds two frame-sized pixel plans.
* The byte-tightening path may temporarily retain two plans, but only under a rare explicit trigger.
* An under-target terminal result is not exact-priced unless the caller explicitly requested best effort.

The handoff shows that current Balanced usually performs four probes and two exact prices, while the wall ratio is 3.07× to 8.00× the matched-rate path. Removing approximately three reconstructions/metric evaluations from the median request is therefore the correct performance target.  

---

# 5. What the model should predict

## Primary output: the image-specific score/scale crossing curve

Do not predict bitrate as the primary output.

Use:

```text
x = ln(effective_scale)
z = ln(max(100 - SSIMULACRA2, 1e-3))
```

For the seven target knots 30, 50, 70, 80, 85, 90, and 95, predict:

1. median fresh-structure crossing `x50`;
2. risk-adjusted candidate crossing, initially approximately `x90`;
3. lower and upper calibrated crossing quantiles;
4. local positive loss exponent `β`;
5. saturation/reachability risk;
6. out-of-distribution flags;
7. optional estimated exact bytes at the crossing;
8. optional structural-instability risk.

For arbitrary target values, interpolate between target knots in log perceptual loss, not directly in score.

A suitable runtime result is:

```rust
pub struct QualityPredictionV2 {
    pub median_rung: Rung,
    pub candidate_rung: Rung,
    pub interval_low: Rung,
    pub interval_high: Rung,
    pub local_loss_exponent: f32,
    pub saturation_risk: SaturationRisk,
    pub structural_risk: StructuralRisk,
    pub ood: OodFlags,
    pub predicted_bytes: Option<u64>,
    pub model_version: QualityModelVersion,
}
```

## Why not predict bitrate?

The existing rate controller still performs multiple exact writer prices and bounded corrections. Passing a predicted bitrate to it would reorganize the search rather than remove it.

It also leaves two mappings to solve:

```text
source + score target -> bitrate
bitrate -> quantizer/policy
```

Direct effective-rung prediction solves the actual expensive decision.

The rate module remains useful for:

* legal rung/effective-scale mappings;
* quantizer construction;
* auxiliary byte labels;
* comparative evaluation.

It should not be the normal score-target backend.

## Why not predict a local or per-frequency quantization field yet?

That introduces many more degrees of freedom and changes the codec’s psychovisual policy at the same time as the controller is being replaced. It would make failures difficult to attribute.

First establish that a scalar effective-rung predictor can remove repeated score probes without losing the byte advantage. Local/per-frequency optimization can then become a separate quality-efficiency project.

## Why not predict policy choices initially?

Keep the production Balanced policy fixed:

* restoration fixed by the request/default;
* hierarchical cover;
* normal CfL behavior;
* Balanced entropy effort;
* no Quality policy bank.

Entropy policy does not affect reconstructed pixels and therefore does not need score prediction.

A later model may select between a very small, independently validated set of source-only policies, but each policy would need its own calibrated crossing curve. The failed Quality promotion screen is a reason not to make policy prediction part of the first controller.

---

# 6. Model form and deterministic implementation

## Recommended model

Use a small, code-generated monotone generalized additive model:

```text
crossing_k =
    target_intercept_k
    + Σ piecewise_linear_feature_term_jk(feature_j)
    + a small number of predeclared interactions
```

Fit separate models for:

* the median crossing;
* the candidate quantile;
* the upper uncertainty bound;
* local slope;
* saturation risk.

The first implementation should contain no more than a small set of measured interactions, such as:

* noise × flatness;
* high-frequency energy × bit depth;
* flatness × orientation coherence;
* chroma/luma energy ratio × CfL correlation.

Do not add interactions simply because the trainer can fit them.

## Monotonicity

For each image, predict the seven target-knot crossings, then project them to a nondecreasing effective-scale sequence:

```text
x30 <= x50 <= x70 <= x80 <= x85 <= x90 <= x95
```

Round the candidate scale toward the finer legal rung.

The same projection must be applied to median and upper-quantile curves.

This guarantees that the **chosen predicted rung** does not become coarser as target increases. It does not provide a mathematical proof that canonical SSIMULACRA2 itself is monotone across every independently encoded request, because the real score curve can have local reversals. Promotion should therefore require zero achieved-score inversions on the full target grid, while the documentation should not describe this as a theorem for arbitrary images.

## Determinism

Model inference should:

* run in scalar Rust;
* use generated constants;
* reduce per-group feature accumulators in fixed LF-group order;
* avoid worker-order floating-point reductions;
* use explicit rounding before converting to a rung;
* include exact model and feature-schema versions in traces.

Fixed-point model constants are worth considering if f64 boundary behavior differs across supported architectures. A scalar f64 implementation with conservative rung rounding may already be sufficient, but this must be tested across SIMD modes and worker counts.

---

# 7. Feature design

The current production atlas is a useful foundation. The diagnostic `AnalysisAtlasV2` also already contains several promising feature candidates, but it currently performs a separate pass and stores a large diagnostic structure. It should be used for offline ablation first, not installed wholesale in the production path.

## Proposed features and costs

| Feature group                                                                           | Purpose                                     | Source-only analysis | Final forward transforms | Candidate reconstruction | Entropy/emission |
| --------------------------------------------------------------------------------------- | ------------------------------------------- | -------------------: | -----------------------: | -----------------------: | ---------------: |
| Width, height, log area, aspect ratio, edge-partial-atom fraction, bit depth, grayscale | Size and metric-scale effects               |                  Yes |                       No |                       No |               No |
| Luma variance histogram and q10/q50/q90/q99                                             | Flat/texture distribution                   |                  Yes |                       No |                       No |               No |
| Chroma variance and chroma/luma ratios                                                  | Colour complexity                           |                  Yes |                       No |                       No |               No |
| Flat fraction plus low-variance tail                                                    | Smooth fields and gradients                 |                  Yes |                       No |                       No |               No |
| Channel clipping/saturation fractions and dynamic range                                 | High-target and saturated-content risk      |                  Yes |                       No |                       No |               No |
| Horizontal/vertical gradient energy and cross term                                      | Edge orientation and anisotropy             |                  Yes |                       No |                       No |               No |
| Laplacian energy and affine-plane residual                                              | Detail versus smooth ramp separation        |                  Yes |                       No |                       No |               No |
| Noise MAD and noise-to-edge ratio                                                       | Low-light/noise quantization sensitivity    |                  Yes |                       No |                       No |               No |
| Orientation coherence and flat-side asymmetry                                           | Text, line art, and edge leakage risk       |                  Yes |                       No |                       No |               No |
| Channel covariance                                                                      | CfL benefit and chroma reconstruction risk  |                  Yes |                       No |                       No |               No |
| DCT energy by channel and radial frequency band                                         | Direct quantization sensitivity             |                   No |                      Yes |                       No |               No |
| Frequency-energy q50/q90/q99 and tail ratios                                            | Sparse detail versus broadband texture      |                   No |                      Yes |                       No |               No |
| Directional AC energy                                                                   | Lines, hatching, directional textures       |                   No |                      Yes |                       No |               No |
| DCT8/DCT16/DCT32 candidate energy or cost gaps                                          | Cover preference and structural instability |                   No |                      Yes |                       No |               No |
| Coefficient zero/nonzero counts at a few reference scales                               | Approximate scale sensitivity               |                   No |                      Yes |                       No |               No |
| Transform-domain luma/chroma correlation                                                | CfL sensitivity                             |                   No |                      Yes |                       No |               No |
| Cheap token/run histogram proxy                                                         | Auxiliary byte prediction                   |                   No |                      Yes |                       No |               No |

No proposed final-decision feature requires a candidate reconstruction, canonical score, entropy training, or exact emission.

## Production feature extraction rules

1. Replace full vector sorting with deterministic histograms.
2. Accumulate source features during the existing atlas pass where practical.
3. Use `AnalysisAtlasV2` only to identify useful signals initially.
4. Fuse winning diagnostic signals into a compact streaming production summary.
5. Build transform summaries directly from `CandidateForwardCache`; do not copy all coefficient banks into a second representation.
6. Reduce transform summaries per LF group and combine them in fixed raster order.
7. Reject any feature set whose additional normal-path wall exceeds 5% of the matched-rate path.

---

# 8. One final transform can support prediction and emission

The precise dependency cycle in the current planner is:

```text
quantizer
  -> AqSetup::build
  -> LfQuantizer and HfQuantizers
  -> quantizer-dependent cover objective
  -> selected transforms
  -> CfL over selected coefficients
  -> quantization
```

Therefore, the final cover and CfL cannot be selected before the quantizer is known.

However, the forward transform coefficients themselves are quantizer-independent. The existing `CandidateForwardCache` is explicitly designed so that a given `(origin, transform)` is computed at most once per request. `ensure_cover_candidates_cached()` can populate aligned DCT8, DCT16, and DCT32 candidates.

The cycle can therefore be broken as follows:

```text
source
  -> quantizer-independent candidate transform bank
  -> aggregate transform features
  -> predict quantizer
  -> quantizer-dependent AQ and cover selection
  -> selected transforms read from same bank
  -> CfL
  -> quantization
```

This is genuinely one transform bank supporting both prediction and final emission.

### Required refactor

Add a method approximately like:

```rust
impl CandidateSearchContext<'_> {
    pub(crate) fn prepare_quality_transform_summary(
        &mut self,
    ) -> Result<TransformFeatureSummary>;
}
```

It should:

* prepare geometry;
* fill the transform candidates required by the Balanced cover;
* calculate compact feature accumulators;
* leave all cache entries available to the later pixel planner.

There is one caveat: eagerly filling every candidate may compute more than the current serial lazy cover path on some images. The existing parallel hierarchical path already has a complete-cache mechanism, but the exact incremental cost and 50 MP memory behavior must be measured. If full prefill is too expensive, begin with DCT8 summaries or another subset that the final cover path already necessarily computes.

Restoration policy must remain fixed in the first version because Gaborish changes the transform frame itself. Predicting restoration would move the policy decision before this shared transform stage and reopen the dependency problem.

---

# 9. Training and calibration procedure

## 9.1 Training labels

Create a new oracle dataset from the actual production Balanced pixel policy.

For each independent source family:

1. build source and transform features once;
2. evaluate the full effective-scale ladder, including `HfMul` segments;
3. adaptively densify around target crossings;
4. rebuild cover/CfL fresh for oracle crossing points;
5. record canonical SSIMULACRA2;
6. record the coarsest fresh rung meeting each target;
7. record local score/loss slope;
8. exact-price the crossing and nearby feasible rungs;
9. record top-rung score and whether the target is reachable;
10. optionally record anchor-reuse versus fresh-structure differences.

The seven target rows from one image are correlated observations, not seven independent samples. Weight each source family equally, dividing its weight across targets and derived resolutions.

## 9.2 Saturation is censored data

Do not assign the top rung as if it were the actual crossing when the top score misses.

Record:

```text
crossing > top_rung
```

and train a separate transparent reachability/saturation model.

At runtime:

* model saturation risk is only a routing hint;
* actual `SaturatedTop` requires canonically scoring the real top candidate below target;
* uncertainty and saturation remain separate concepts.

This is especially important because the current calibration reports saturation fractions of 0.81 and 0.94 at targets 90 and 95. 

## 9.3 Loss functions

Fit at least three crossing models:

### Median crossing

Use Huber or median quantile loss on:

```text
ln(required_effective_scale)
```

### Candidate crossing

Initially use quantile loss at approximately `τ = 0.90`:

```text
ρτ(error)
```

At `τ = 0.90`, predicting too coarse is penalized approximately nine times as much as predicting too fine.

This quantile is not a safety guarantee. It is a wall-versus-byte operating point. The verification gate supplies safety.

### Upper uncertainty bound

Fit a higher quantile such as `τ = 0.95` and calibrate its empirical coverage by whole family on development data.

### Byte-regret term

Add a modest penalty for unnecessarily fine predictions:

```text
λbytes * max(0, ln(bytes(predicted_rung) / bytes(crossing_rung)))
```

This prevents the asymmetric floor penalty from collapsing into a permanently overfine controller.

### Slope model

Use robust regression for:

```text
β = -d ln(perceptual_loss) / d ln(effective_scale)
```

with a positivity constraint and bounded runtime range.

## 9.4 Uncertainty and OOD detection

Use both model residuals and feature support.

Fallback should fire when any of the following occurs:

* calibrated log-scale interval width exceeds `ln(1.5) ≈ 0.405`;
* a hard feature lies outside the calibrated production envelope;
* distance to the nearest training prototype exceeds the development-set 99th percentile;
* predicted upper crossing reaches the top 2% of the effective ladder;
* saturation risk exceeds the development-calibrated threshold;
* transform-size preference is unusually ambiguous;
* target/bit-depth/content combination has insufficient family support.

The runtime OOD implementation can remain simple:

* robust min/max ranges;
* median and MAD standardization;
* a small code-generated set of feature-space medoids;
* deterministic L1 or diagonal-distance calculation.

Do not describe these intervals as universal confidence guarantees. They are empirical in-domain calibration used to decide whether the cheap path is worthwhile.

## 9.5 Generated-model provenance

Every generated model should embed or accompany:

* model schema version;
* feature schema version;
* SSIMULACRA2 metric version;
* encoder git revision;
* trainer git revision;
* corpus-manifest hash;
* split-manifest hash;
* label-generation command;
* label dataset hash;
* training configuration;
* generated Rust checksum;
* full development and holdout report.

A single reproducible command should regenerate both the Rust constants and the report. Generated diffs should be human-reviewable.

---

# 10. Corpus assessment and expansion

The current corpus contains 47 images split 19/15/13 across calibration, development, and holdout. It includes the requested broad classes, but it is not sufficient to calibrate production uncertainty.

It is sufficient for:

* rejecting obviously inadequate features;
* comparing source-only versus transform-derived predictors;
* determining whether the median and tail errors improve materially;
* building the first shadow model.

It is not sufficient for:

* a trustworthy 90th or 95th percentile crossing model;
* OOD thresholds;
* high-target saturation calibration;
* public promotion across every content class.

## Required manifest changes

Add:

```json
{
  "family_id": "...",
  "variant_id": "...",
  "generator_family": "...",
  "source_capture_id": "..."
}
```

All crops, resolutions, colour variants, recompressions, and derivatives of one source must remain in the same split.

The existing manifest includes multiple resolutions of individual photographs. They appear to remain within one split, but without `family_id` that cannot be mechanically audited or weighted correctly.

## Expansion targets

For a first feature-gated implementation:

* at least **150 independent source families**;
* at least 25 independent families in each critical non-photo class;
* at least 30 blind holdout families.

For public promotion:

* at least **300 independent source families**;
* at least 60 never-tuned promotion-holdout families;
* 8-, 10-, 12-, and 16-bit inputs;
* dimensions from metric minimums through 50 MP;
* multiple camera and rendering sources;
* real and synthetic text/UI;
* line art and diagrams;
* smooth ramps and sky gradients;
* saturated wide-colour stress;
* low-light and structured noise;
* grayscale;
* tiny images;
* high-detail natural photographs.

The current 13-image locked holdout should remain untouched as a legacy regression holdout. The larger expansion should create a new blind promotion holdout rather than recycling those 13 images for threshold selection.

---

# 11. Quantitative experiment and promotion gates

The handoff requires Balanced wall no more than 2.0× the matched-rate path on the two anchors. 

The following thresholds should be frozen before running the experiment.

## 11.1 Prediction quality, non-saturated family holdouts

| Metric                                                     |                                 Required |   |       |
| ---------------------------------------------------------- | ---------------------------------------: | - | ----- |
| Median `                                                   | ln(predicted crossing / oracle crossing) | ` | ≤0.10 |
| p90 absolute log error                                     |                                    ≤0.30 |   |       |
| p99 absolute log error                                     |                                    ≤0.70 |   |       |
| First-plan success, reachable in-domain targets 30–90      |                                     ≥90% |   |       |
| First or one-correction success                            |                                     ≥98% |   |       |
| Exact-controller continuation on reachable in-domain cells |                                      ≤5% |   |       |
| Successful outputs below requested score                   |                                    **0** |   |       |

Target 95 should be reported separately until its current saturation rate is reduced or properly routed.

A quick falsification gate may be looser:

* p90 log error ≤0.45;
* at least 80% first-plan success;
* no more than 1.5% simulated byte regression.

Failure there means the transparent one-shot model is not yet viable and production controller work should stop.

## 11.2 Byte efficiency

The current Balanced score controller has a 0.99281 geometric-mean ratio to the interpolated rate curve. That leaves only approximately 0.724% before the ratio reaches 1.0. 

Therefore the public promotion limits should be:

| Metric                                               | Required |
| ---------------------------------------------------- | -------: |
| Geometric-mean bytes versus current score controller |   ≤1.007 |
| Geometric-mean bytes versus same-effort rate curve   |   ≤1.000 |
| Mean per-image BD-rate versus same-effort rate curve |      ≤0% |
| p95 per-cell byte ratio versus current controller    |    ≤1.03 |
| Worst reachable cell byte ratio                      |    ≤1.08 |
| Median score overshoot on first-plan successes       |    ≤0.35 |
| p90 score overshoot                                  |     ≤1.0 |

A shadow prototype may initially allow a 1.5% geometric-mean regression, but that is not enough for promotion because it would discard the controller’s existing aggregate rate advantage.

## 11.3 Wall

| Measurement                               | Acceptance | Design target |
| ----------------------------------------- | ---------: | ------------: |
| 4.3 MP anchor / matched-rate              |      ≤2.0× |        ≤1.50× |
| 12 MP anchor / matched-rate               |      ≤2.0× |        ≤1.75× |
| Incremental source/transform feature wall |        ≤5% |           ≤3% |
| Normal-path full reconstructions          |          1 |             1 |
| Normal-path entropy trainings/emissions   |          1 |             1 |

A canonical metric evaluation remains unavoidable. Therefore the practical lower bound is:

```text
one Balanced pixel plan
+ one full reconstruction
+ one canonical metric
+ one entropy attachment/emission
```

If that isolated path still exceeds 2.0×, the remaining problem is metric/render cost rather than controller navigation. That should be measured directly rather than hidden behind further prediction work.

## 11.4 Memory

| Measurement                                   |                         Required |
| --------------------------------------------- | -------------------------------: |
| Normal-path peak RSS versus matched-rate path |                           ≤1.10× |
| Fallback peak RSS                             | no worse than current controller |
| Normal retained frame-sized pixel plans       |                                1 |
| Normal duplicate transform payloads           |                                0 |

## 11.5 Determinism and monotonicity

Promotion requires:

* byte-identical output at supported worker counts;
* identical decision path and status across worker counts;
* identical result across supported SIMD modes;
* nondecreasing predicted rung across the full target grid;
* zero achieved-score inversions on the promotion corpus;
* zero byte inversions on the target grid unless explicitly justified by a smaller exact stream with a higher achieved score;
* independent decoding and re-scoring equal to the reported score within the established metric tolerance.

---

# 12. Minimal implementation sequence

## PR 0: repair the existing contract

This is separate from the predictive experiment.

1. Do not return under-target bytes as `Ok`.
2. Make saturation and work exhaustion structured failures.
3. Make CLI output atomic and avoid creating an output file on failure.
4. Add explicit lossless and best-effort fallback policies if desired.
5. remove the hidden extra rescue probe or put it inside the documented cap.
6. Add public API tests for saturation and work exhaustion.

This changes behavior only where the current implementation already contradicts the public hard-floor contract.

## PR 1: trace and corpus schema, no encoding change

Create `jpxl.quality-trace/2` with:

```text
model_version
feature_schema
median_rung
candidate_rung
interval_low/high
local_loss_exponent
saturation_risk
ood_flags
fallback_reason
first_observed_score
correction_rung
decision_path
total pixel plans
total reconstructions
total metric evaluations
total entropy trainings
total emissions
```

Add `family_id` and related fields to the corpus manifest.

The current controller remains authoritative.

## PR 2: production-endpoint label generator

Replace or supplement `calibrate_initial_rung.py` with a trainer that:

* operates over the full effective-scale ladder;
* labels production Balanced fresh-structure crossings;
* represents saturation as censoring;
* records local slopes and neighboring exact bytes;
* splits and weights by family;
* emits reproducible dataset and model provenance.

Existing controller traces can be used as provisional labels for the first falsification pass, but not as final oracle labels.

## PR 3: source-only shadow model

Implement `QualityPredictionV2`, but only log its counterfactual decision.

Compare:

* current two-feature table;
* all existing `SourceFeatures`;
* histogram-based source features;
* selected `AnalysisAtlasV2` aggregates.

Do not let the model affect a bitstream.

## PR 4: transform-summary shadow model

Add the request-scoped transform-summary API to `CandidateSearchContext`.

Again, only log:

* candidate rung;
* interval;
* predicted correction;
* whether exact fallback would have fired;
* counterfactual one-shot work counts;
* predicted byte regret.

The current exact controller still chooses output.

## Gate before production integration

Proceed only when family-held-out results satisfy:

* median log-scale error ≤0.10;
* p90 ≤0.30;
* at least 80% first-plan success in the quick screen;
* no more than 1.5% simulated byte regression;
* incremental feature cost ≤5%;
* no opaque model required.

If transform features cannot reach that gate, do not move candidate reconstructions into the feature extractor and call the result one shot.

## PR 5: feature-gated common-case path

Only after the shadow gate:

* make the predicted plan the first real plan;
* use canonical verification before entropy;
* add one slope correction;
* continue the existing navigator from measured observations;
* maintain the same total work caps;
* retain current exact controller as the default fallback.

Public promotion then uses the stricter wall and byte gates above.

---

# 13. Expected tradeoffs

## Wall

The proposed normal route eliminates:

* geometric expansion probes;
* repeated bracket tightening;
* repeated candidate reconstruction;
* one of the usual exact finalist prices.

Given the current median of four probes and two exact prices, it should materially approach the single Balanced path, although one full canonical metric remains unavoidable.

## Memory

Normal-path memory should improve because only one pixel plan needs to survive until entropy attachment. The transform cache is already request-scoped and shared.

The main memory risk is eagerly completing all transform candidates before prediction on 50 MP inputs. That must be measured before promotion.

## Score floor

The score floor becomes stronger than the current public implementation because:

* under-target streams are no longer returned as ordinary successes;
* every emitted lossy stream has passed canonical verification;
* uncertainty changes the work path, not the correctness promise.

## Bytes

Byte efficiency is the hard part.

A highly conservative predictor can easily turn a target of 85 into an achieved score above 90 and waste substantial bytes. The model therefore needs both:

* an asymmetric candidate quantile to keep first-plan misses uncommon;
* an explicit byte-regret term and optional rare coarsening correction.

The existing controller’s modest rate advantage leaves little room for aggregate regression. A model that is fast but routinely overfine should not be promoted.

## Complexity

Runtime complexity remains modest:

* one compact feature summary;
* one generated additive model;
* one prediction structure;
* one continuation path into the existing navigator.

No new heavyweight dependency or neural runtime is justified at this stage.

---

# 14. Principal risks

1. **Family leakage.** Resolutions or synthetic variants of one source can make prediction appear substantially better than it is.

2. **High-target censoring.** Treating top-rung saturation as an ordinary crossing corrupts both the scale model and uncertainty.

3. **Structural discontinuities.** Cover and CfL can change around the crossing, making a smooth source-only score curve inaccurate.

4. **Metric nonmonotonicity.** Running-max training curves hide local reversals that the runtime may still encounter.

5. **Entropy nonmonotonicity.** The coarsest feasible rung is not always the smallest exact stream.

6. **Overconservative calibration.** A large reserve can meet the floor while silently discarding the present byte advantage.

7. **Understated feature cost.** A second source pass, full coefficient copy, or eager transform work can erase the saved controller wall.

8. **Transform-cache memory at 50 MP.** Full candidate prefill may alter peak RSS even when it avoids recomputation.

9. **Model staleness.** Changes to quantization, AQ, cover cost, CfL, restoration, metric version, or entropy policy invalidate labels.

10. **Misrepresented uncertainty.** Empirical quantiles do not provide arbitrary-image guarantees. The canonical score gate must remain authoritative.

11. **Hidden fallback work.** Restarting the exact controller after one or two prediction attempts would make the apparent one-shot path more expensive than the current controller.

12. **Benchmark-specific routing.** Tuning fallback thresholds to the 13 locked images would create a demonstration rather than a general controller.

13. **Literal contract mismatch.** Neither current bounded search nor one-shot prediction globally minimizes all possible exact codestreams. The bounded effort domain must be explicit.

---

# Final recommendation

Implement this in three conceptual layers:

1. **Correctness layer:** under-target output is never a normal success.
2. **Prediction layer:** a transparent monotone model predicts the production Balanced fresh-structure crossing, local slope, interval, saturation risk, and OOD state from source plus shared transform features.
3. **Execution layer:** one predicted plan is canonically verified; one correction is allowed; the existing navigator continues from those observations under the same total cap.

Do not route through the bitrate controller, do not predict a per-frequency field yet, and do not make policy-bank selection part of the first model.

The key design principle is:

> The predictor determines where JPXL should look first. Canonical verification determines whether JPXL is allowed to emit. The existing bounded controller remains the recovery mechanism, not the normal path.

That is the narrowest design that can plausibly move Balanced from four full-frame probes toward one while honestly preserving the hard per-image score floor and the encoder’s existing byte-efficiency advantage.
