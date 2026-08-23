# JPXL SSIMULACRA2-Driven Perceptual Quality Controller Plan

## Executive decision

JPXL should make **perceptual quality**, rather than bitrate, the normal contract for lossy encoding.

The production objective should be:

```text
minimize exact codestream bytes

subject to:

SSIMULACRA2(reference, decoded_candidate) >= requested_score
```

`--bpp` and `--bytes` should remain available as expert, benchmarking, and constrained-delivery modes. They should stop defining the normal lossy policy.

A request for quality 85 should produce the smallest JPXL stream that scores 85 or better, regardless of whether a particular image requires 0.6 bpp or 2.3 bpp.

Achieving that requires three separate mechanisms:

1. **Quality-target navigation:** find the global quantization region that meets the requested score.
2. **Perceptual policy selection:** choose chroma allocation, LF/HF balance, cover, CfL, restoration, and related settings by comparing their byte cost at the same score.
3. **Finalist perceptual optimization:** remove the least valuable coded information while preserving the score floor.

The first mechanism gives JPXL a quality-target interface. The second and third are what create a real perceptual encoder and close the equal-quality bitrate gap.

---

## 1. Why the current rate controller cannot solve the problem

The current public lossy path is defined by `RateTarget::Bytes` or `RateTarget::BitsPerPixel` in:

```text
crates/jpxl-encode-policy/src/request.rs
```

The high-level facade exposes the same model through:

```text
Encoder::with_target_bpp
Encoder::with_target_bytes
```

in:

```text
crates/jpxl/src/lib.rs
```

Bitrate is also leaking into quality policy rather than being used only as a search coordinate. In particular:

```text
effective_b_qm_scale
effective_x_qm_scale
at_most_one_bpp
```

select chroma behavior according to the requested bitrate.

That is an empirically calibrated bitrate policy. It assumes that “below 1 bpp” has approximately the same perceptual meaning for foliage, faces, screenshots, low-light noise, gradients, text, and ordinary photographs. It does not.

The correct formulation is:

```text
Given:
    source image x
    requested minimum score Q
    legal encoder configuration θ

Find:
    θ* = argmin Bytes(Encode(x, θ))

Subject to:
    SSIM2(x, Decode(Encode(x, θ))) >= Q
```

Bitrate does not disappear. It moves to its proper role: **the cost being minimized**, rather than the user-visible definition of quality.

The existing experiments already support this direction:

- Naive coefficient-level rate-distortion selection improved Butteraugli but frequently damaged SSIMULACRA2 by removing texture too uniformly.
- Terminal coefficient truncation performed better because it removed coefficients where zero-run savings actually existed.
- Variance-based adaptive quantization paid substantial multiplier-plane overhead and failed to allocate quality consistently.

Those outcomes show that the controller needs both:

1. a perceptual judgment of what can be removed; and
2. an accurate estimate of whether removing it saves real bytes.

A bitrate target supplies neither.

---

## 2. Public quality contract

### 2.1 Target types

Replace the lossy-only `Mode::Lossy(RateTarget)` design with an explicit target family:

```rust
pub enum LossyTarget {
    Perceptual(PerceptualTarget),
    Rate(RateTarget),
    FixedQuantizer(FixedQuantizerTarget),
}

pub struct PerceptualTarget {
    pub metric: PerceptualMetric,
    pub minimum_score: f64,
}

pub enum PerceptualMetric {
    Ssimulacra2,
}
```

The normal high-level API should be:

```rust
let encoded = Encoder::new()
    .with_ssimulacra2_score(85.0)?
    .with_effort(Effort::Balanced)
    .encode_rgb8(width, height, rgb)?;
```

CLI:

```text
jpxl encode --quality 85 input.ppm output.jxl
jpxl encode --quality 90 --effort quality input.ppm output.jxl
```

Keep explicit technical modes:

```text
--ssimulacra2 85
--bpp 1.0
--bytes 500000
--global-scale ...
```

`--quality`, `--bpp`, `--bytes`, and fixed-quantizer controls must be mutually exclusive.

### 2.2 Score semantics

Use the SSIMULACRA2 score directly. Do not imitate libjxl’s distance scale and do not present the value as an arbitrary “quality percentage.”

The official metric uses a fixed scale ending at 100 and describes approximately 70 as high quality, 80 as very high quality, 85 as excellent, 90 as visually lossless, and 100 as mathematically lossless. SSIMULACRA2 also explicitly penalizes both added-edge artifacts such as ringing and lost-edge artifacts such as smoothing. 

Define edge behavior:

- Targets below zero are rejected.
- Targets from zero to below 100 use VarDCT perceptual encoding.
- A target of exactly 100 routes to lossless encoding.
- Images below SSIMULACRA2’s minimum supported dimensions route to lossless or return an explicit unsupported-target error.
- No silent fallback to PSNR, SSIM, or another metric.
- Initial production scope is opaque SDR sRGB and grayscale.
- HDR, alpha-sensitive scoring, non-sRGB primaries, animation, and extra-channel semantics receive explicit later contracts.

### 2.3 Quality and effort are independent

Quality answers:

> How good must the decoded result be?

Effort answers:

> How much work may the encoder spend finding the smallest qualifying result?

Use a generic effort enum:

```rust
pub enum Effort {
    Fast,
    Balanced,
    Quality,
}
```

The same score target must mean the same minimum quality at every effort. Higher effort may reduce bytes or reduce quality overshoot. It may not lower the achieved score.

### 2.4 Outcome and reporting

Return the actual perceptual result:

```rust
pub struct PerceptualOutcome {
    pub requested_score: f64,
    pub achieved_score: f64,
    pub exact_bytes: u64,
    pub quantizer: QuantizerChoice,
    pub policy: PerceptualPolicyChoice,
    pub metric_version: MetricVersion,
    pub trace: Vec<QualityProbe>,
    pub saturated: bool,
}
```

CLI output:

```text
quality_target=85.0000 achieved=85.1372 bytes=412883
metric=ssimulacra2-2.1 effort=balanced probes=4 policy_trials=3
```

---

## 3. Clean-room and dependency boundary

The production controller must not depend on:

- Butteraugli;
- libjxl source;
- libjxl’s controller constants;
- libjxl’s AQ, quantizer, cover, or search implementation.

Record the derivation boundary before implementation:

- The SSIMULACRA2 algorithm and score semantics may be taken from its independent specification, reference implementation, or a permissively licensed Rust implementation.
- JPEG XL reconstruction and legal encoder choices remain derived from ISO/IEC 18181-1 and JPXL’s existing clean-room implementation.
- Search algorithms are based on general constrained optimization, coordinate descent, finite perturbation, and JPXL’s own experiments.
- libjxl remains a black-box comparator. Its files may be encoded, decoded, measured, and scored, but its internal controller is not a design source.
- The metric implementation, source revision, patches, license, and test vectors are pinned.
- Changing the metric version is treated as an encoder-behavior change because it may change selected codestreams.

Do not add `jpxl-decode` as a normal dependency of `jpxl-encode-policy`. The quality search should consume a scoring interface, while a separate optional perceptual layer owns rendering and SSIMULACRA2.

---

## 4. Required architecture

### 4.1 Refactor the common candidate context

`PreparedSearch` in `rate.rs` already retains:

- source and transformed frames;
- analysis;
- the forward-transform cache;
- quantization workspace;
- executor;
- diagnostics.

Move that shared state into a neutral module:

```text
crates/jpxl-encode-policy/src/candidate.rs
```

Suggested shape:

```rust
pub struct CandidateSearchContext<'a> {
    frame: &'a PreparedFrame,
    transform_frame: &'a PreparedFrame,
    atlas: &'a AnalysisAtlas,
    request: &'a EncodeRequest,
    executor: &'a EncodeExecutor,
    forward_cache: CandidateForwardCache,
    quant_workspace: QuantizationWorkspace,
}
```

Both `rate.rs` and the new `quality.rs` should use this context. Do not duplicate source preparation, analysis, transforms, or scratch allocation.

### 4.2 Split pixel decisions from entropy decisions

A quality probe needs reconstructed pixels. It does not need:

- histogram training;
- ANS tables;
- coefficient-order alternatives;
- entropy clustering alternatives;
- section layout;
- a serialized codestream.

The current `EmissionPlan` combines:

```text
spatial
quantized
entropy
sections
```

Introduce a validated pre-entropy plan:

```rust
pub struct PixelPlan {
    pub spatial: Arc<SpatialPlan>,
    pub quantized: Arc<QuantizedFrameIr>,
}

pub struct ValidatedPixelPlan(PixelPlan);
```

Entropy is attached later:

```rust
fn attach_entropy(
    pixels: &ValidatedPixelPlan,
    search: EntropySearch,
    context: &mut CandidateSearchContext,
) -> Result<ValidatedEmissionPlan>;
```

`ValidatedPixelPlan` should enforce all non-entropy invariants:

- dimensions and geometry;
- cover validity;
- quantizer ranges;
- LF-group order;
- coefficient dimensions;
- grid dimensions;
- spatial metadata consistency.

`ValidatedEmissionPlan` then adds entropy and section validation.

This split is critical. A global quality search may require three to six reconstructions, but only two or three finalists should pay for full entropy training and exact writer counting.

### 4.3 Preserve a canonical source reference

`PreparedFrame` currently retains XYB after source conversion. Do not create the metric reference by converting those XYB planes back to RGB.

That would introduce an RGB→XYB→RGB round trip into the reference and bias the metric toward the codec’s own transform.

In perceptual mode, preserve a separate source representation:

```rust
pub struct PerceptualReferenceFrame {
    pub width: u32,
    pub height: u32,
    pub linear_rgb: Arc<[[f32; 3]]>,
    pub color_encoding: CanonicalColorEncoding,
}
```

Build it from the normalized original input before XYB conversion. Keep it optional so rate-target and fixed-quantizer modes do not pay the memory cost.

The initial canonical scoring pipeline should be:

1. Accept SDR sRGB/BT.709.
2. Preserve the original normalized source.
3. Reconstruct the candidate through inverse transforms, CfL, Gaborish, EPF, XYB-to-RGB, transfer, and final clamping.
4. Put source and candidate into the same canonical representation.
5. Score exactly that representation.
6. Verify that in-memory scoring agrees with scoring an emitted and independently decoded codestream.

### 4.4 Add an encoder-side plan renderer

Production quality search cannot serialize and decode a complete codestream for every probe.

Add:

```text
crates/jpxl-plan-render
```

It consumes `ValidatedPixelPlan` and reconstructs canonical RGB pixels.

It must not call `jpxl-decode` in production. It may reuse genuinely neutral mathematical primitives moved into `jpxl-core`, but encoder-side reconstruction orchestration remains independent.

Required parity tests:

- plan-rendered RGB against emitted output decoded by `jpxl-decode`;
- plan-rendered score against emitted-and-decoded score;
- comparison against `djxl` and `jxl-oxide` output within measured decoder tolerances;
- fixtures covering all supported transform sizes;
- chroma-from-luma;
- QM scales;
- Gaborish;
- EPF;
- LF smoothing;
- coefficient and quantization extremes.

A temporary research implementation may emit and call the in-tree decoder to prove the search logic. It must not be promoted as the production controller.

### 4.5 Add a dedicated perceptual crate

Create:

```text
crates/jpxl-perceptual
```

Responsibilities:

- wrap and pin the SSIMULACRA2 implementation;
- prepare the source reference;
- retain reusable metric context and scratch;
- render candidates using `jpxl-plan-render`;
- return scalar scores;
- later return metric breakdowns and attribution data;
- provide a deterministic canonical scoring mode.

`jpxl-encode-policy` should see only an evaluator:

```rust
pub trait PerceptualEvaluator {
    fn evaluate(
        &mut self,
        candidate: &ValidatedPixelPlan,
    ) -> Result<PerceptualObservation>;
}
```

This keeps third-party metric code and decoder-like rendering out of the policy layer.

### 4.6 Precompute the reference

Every candidate is compared against the same source. Source-only metric work must therefore be computed once.

The current project uses `ssimulacra2 0.5.1` as a measurement dependency. Benchmark that implementation against an independent SIMD implementation with:

- precomputed references;
- reusable comparison contexts;
- allocation reuse;
- bounded-memory scoring for large images.

`fast-ssim2`, for example, exposes a precomputed-reference batch path and reusable comparison context specifically suited to repeated encoder-search comparisons. 

Do not switch implementations merely because one is faster. Require:

- official-vector parity;
- corpus score parity;
- high-quality-region parity near thresholds;
- deterministic final-gate behavior.

A useful design is:

```rust
enum MetricExecution {
    FastNavigation,
    CanonicalFinal,
}
```

Fast navigation may use SIMD and cached approximations. Candidate selection is finalized using the deterministic canonical evaluator.

---

## 5. Stage A: exact quality-target navigation

This stage creates a correct quality-target encoder under one fixed policy. It does not yet close the bitrate gap.

### 5.1 Search coordinate

Use the existing effective quantizer ladder:

- `global_scale`;
- extended through `hf_mul` where required.

Hold structural policy fixed during this stage.

For interpolation, use metric loss rather than raw score:

```text
loss = max(epsilon, 100 - score)

x = log(effective_quantizer_scale)
y = log(loss)
```

The score compresses near 100. Interpolating raw scores in that region will be unstable. Log loss should give a smoother local crossing model without changing public semantics.

### 5.2 Initial rung prediction

Do not translate quality into a fixed bpp and do not begin every image from the same quantizer.

Build a small offline predictor using JPXL’s own encode traces:

```text
(target score, source features) -> initial quantizer rung
```

Use existing `AnalysisAtlas` information:

- luma variance quantiles;
- chroma variance;
- edge-energy quantiles;
- flat-region fraction;
- texture fraction;
- transform-energy summaries;
- grayscale flag;
- dimensions.

Start with a deterministic table, monotone spline, or low-order regression. This predictor only reduces search work. Exact scoring always verifies the result.

### 5.3 Bounded bracket and interpolation

For a fixed policy:

1. Evaluate the predicted rung.
2. Move coarser or finer according to the score.
3. Expand until one feasible and one infeasible point are known, or the legal range saturates.
4. Fit a crossing in log-scale/log-loss space.
5. Evaluate the predicted crossing.
6. Permit a bounded correction.
7. Retain every evaluated point in a frontier.
8. Exact-price only the closest feasible candidates and immediate competitors.
9. Select the smallest exact stream whose canonical score remains above the target.
10. Store the winner once.

The target is a hard minimum, not an undershoot band.

A small internal score guard may be necessary to prevent CPU- or SIMD-dependent threshold crossings. Determine it from measured variation rather than choosing an arbitrary permanent constant.

### 5.4 Do not assume monotonicity

Finer quantization will normally improve score and increase bytes, but structural and entropy decisions may create local reversals.

Maintain a Pareto frontier:

```text
(score, provisional_rate, exact_bytes, quantizer, policy)
```

Monotonicity can guide navigation. It cannot decide final selection.

Exact-price:

- the nearest feasible rung;
- the next finer rung;
- any evaluated feasible point with competitive estimated rate;
- optionally the immediate coarser point to confirm infeasibility.

### 5.5 Effort budgets

| Effort | Metric probes | Full entropy prices | Initial behavior |
|---|---:|---:|---|
| Fast | normally 2, maximum 3 | 1–2 | prediction plus one correction |
| Balanced | normally 3, maximum 5 | 2–3 | bounded bracket and correction |
| Quality | bounded 6–10 | bounded finalists | deeper bracket and later policy work |

These are hard budgets enforced in code and surfaced in diagnostics.

No hidden exhaustive fallback should remain in Fast or Balanced.

When the target cannot be bracketed within the legal quantizer range, return the best verified feasible result with `saturated=true`. Never silently emit a below-target result.

### 5.6 Controller skeleton

```rust
fn solve_fixed_policy(
    context: &mut CandidateSearchContext,
    evaluator: &mut impl PerceptualEvaluator,
    target: f64,
    effort: Effort,
    policy: PerceptualPolicy,
) -> Result<QualityCandidate> {
    let predicted = predict_rung(context.atlas(), target, policy);
    let mut frontier = QualityFrontier::new(target, effort);

    frontier.insert(evaluate_pixel_candidate(
        context,
        evaluator,
        predicted,
        policy,
    )?);

    while !frontier.has_score_bracket() && frontier.metric_budget_left() {
        let next = frontier.next_expansion_rung()?;
        frontier.insert(evaluate_pixel_candidate(
            context,
            evaluator,
            next,
            policy,
        )?);
    }

    while frontier.metric_budget_left() && !frontier.crossing_is_tight() {
        let next = frontier.predict_log_loss_crossing()?;
        frontier.insert(evaluate_pixel_candidate(
            context,
            evaluator,
            next,
            policy,
        )?);
    }

    for candidate in frontier.near_target_feasible_candidates() {
        exact_price_with_full_entropy(context, candidate)?;
    }

    frontier.smallest_exact_feasible()
}
```

---

## 6. Stage B: perceptual policy selection

A fixed-policy navigator can find:

> The current encoder configuration at score 85.

It cannot find:

> The smallest legal JPXL encoding at score 85.

That requires comparing policies at the same perceptual score.

### 6.1 Add `EncodeRequest::for_quality`

Do not reuse `for_target` unchanged.

Quality mode must not call:

```text
at_most_one_bpp
```

and must contain no other requested-bitrate quality gates.

The initial baseline can inherit current Balanced settings, but those settings must be labeled as a starting policy, not as perceptually optimal defaults.

### 6.2 Define a bounded policy bank

Do not search the Cartesian product of every encoder setting.

Define coherent alternatives over axes already known to affect quality or rate:

- neutral versus refined X/B QM scales;
- a small `quant_lf` set;
- current and conservative restoration settings;
- valid cover alternatives;
- CfL policy alternatives;
- terminal-truncation policy;
- entropy effort, after pixel qualification.

Each policy is solved independently to the same target score. Only then are exact bytes compared.

This replaces “B=5 below 1 bpp” with the actual question:

> Does this chroma allocation produce fewer bytes for this image at the requested perceptual quality?

### 6.3 Coordinate descent

Balanced:

1. Solve the baseline policy.
2. Select the two most relevant alternatives from source analysis.
3. Start each alternative near the baseline score crossing.
4. Solve each to the same score.
5. Exact-price feasible finalists.
6. Keep the smallest.

Quality:

1. Perform the Balanced pass.
2. Vary one remaining policy axis around the winner.
3. Accept the strongest byte reduction.
4. Permit one bounded second pass.
5. Stop when no trial exceeds a minimum saving threshold.

The threshold prevents spending several full metric and entropy evaluations to save negligible space.

### 6.4 Policy prediction

After exact policy traces accumulate, train a small source-feature ranker to order policy trials.

The predictor may skip obviously irrelevant trials under Fast or Balanced. It may not bypass final score verification.

---

## 7. Stage C: SSIMULACRA2-guided finalist optimization

This is the phase most likely to close the remaining equal-quality bitrate gap.

The selected candidate will normally overshoot the target because global quantizer rungs and policy settings are discrete. That overshoot is a **quality reserve** that can be exchanged for bytes.

Closed-loop perceptual optimization is a proven general architecture: Guetzli minimized JPEG size while using a perceptual metric as feedback, though at substantial computational cost. The relevant lesson is the feedback loop, not its JPEG decisions or Butteraugli implementation.

### 7.1 Begin with terminal HF truncation

The first edit family should be terminal coefficient removal.

The existing experiments showed why:

- Removing a terminal nonzero can eliminate its token.
- It can eliminate preceding interior-zero tokens.
- It may reduce the `non_zeros` symbol.
- Zeroing a coefficient in the middle of a run may save little or nothing.

Replace fixed global `lambda_scale` behavior with a finalist-only constrained reducer:

1. Begin from a candidate above the score target.
2. Train or retain its entropy model.
3. Enumerate legal terminal-truncation edits by varblock and channel.
4. Estimate byte savings using the actual coefficient walk and trained entropy tables.
5. Estimate SSIMULACRA2 loss.
6. Rank edits by:

```text
bytes_saved / estimated_perceptual_loss
```

7. Apply a bounded batch of non-overlapping edits.
8. Reconstruct and calculate the full score.
9. Accept the batch when the target still holds.
10. Otherwise roll it back and halve the batch.
11. Periodically retrain entropy after accepted batches.
12. Retrain, exact-price, and full-score the final candidate before emission.

Do not retrain entropy for every proposed coefficient edit. Use the current model for ranking and exact repricing at checkpoints.

### 7.2 Expand edit families one at a time

After terminal truncation proves useful, test:

- region-specific `HfMul` coarsening where signaling cost is amortized;
- reduced chroma precision in low-sensitivity regions;
- revision of low-margin cover merge/split decisions;
- bounded CfL changes;
- coherent restoration changes.

Each edit family must independently pass equal-score corpus gates.

Do not revive naive variance-only adaptive quantization. Existing evidence shows that multiplier-plane signaling may consume 1–3% of the file before providing any benefit, and that simple variance allocation can damage the metric in important regions.

### 7.3 SSIMULACRA2 is not additive by block

SSIMULACRA2 is multiscale and nonlocal. A frame score cannot be divided into independent block-owned quality points.

The correct loop is:

```text
local estimate
    -> propose edits
    -> reconstruct candidate
    -> full SSIMULACRA2 score
    -> accept or roll back
```

Local attribution decides what to test. Only the complete canonical score decides what survives.

---

## 8. Extend the SSIMULACRA2 wrapper for encoder use

The current conformance wrapper returns one scalar. That is sufficient for Stage A but not enough for efficient finalist optimization.

### 8.1 Expose the metric breakdown

Add an internal result:

```rust
pub struct Ssimulacra2Result {
    pub score: f64,
    pub raw_error: f64,
    pub aggregates: [f64; 108],
    pub contribution_atlas: Option<ContributionAtlas>,
}
```

SSIMULACRA2 evaluates three error-map types over six scales and three color components, then computes both mean and fourth-norm aggregates, yielding 108 aggregate terms before final weighting and remapping. 

Those terms provide a better search signal than the final score near 100.

The public contract remains the final score. Raw error and aggregates remain internal implementation details.

### 8.2 Contribution atlas

Retain sufficient intermediate information to identify regions dominated by:

- structural mismatch;
- ringing or added-edge error;
- smoothing or lost-edge error;
- X, Y, or B channel error;
- fine-scale or coarse-scale error.

Project this information onto LF groups or 32×32/64×64 regions.

The atlas is an attribution estimate, not an exact decomposition. Validate it through finite perturbations:

1. deliberately alter one region;
2. recompute the complete score;
3. compare actual score movement with predicted sign and ranking.

### 8.3 Metric execution requirements

The production backend must:

- precompute source-only work once;
- reuse candidate buffers;
- avoid allocations after warm-up where practical;
- use deterministic reduction order for final decisions;
- support bounded-memory operation for large images;
- expose source preparation, conversion, pyramid, and aggregation timings separately.

Do not optimize metric kernels until end-to-end profiling shows the metric is the limiting stage.

### 8.4 Incremental scoring comes later

Do not make incremental SSIMULACRA2 a prerequisite.

A later implementation may:

1. retain candidate pyramids;
2. recompute edited regions with conservative scale-dependent halos;
3. update aggregate statistics;
4. periodically verify against a full score;
5. disable incremental mode on any parity failure.

Multiscale filtering makes local updates substantially more complicated than local DCT reconstruction. Full scoring remains the correctness oracle.

---

## 9. Train a JPXL-specific perceptual surrogate

Exact SSIMULACRA2 belongs in the outer loop. It is too expensive for every cover branch or coefficient decision.

The eventual inner encoder should use a cheap surrogate trained from JPXL’s own distortions.

### 9.1 Generate training data from JPXL

For a broad source corpus:

- encode across quantizer rungs and score levels;
- perturb terminal coefficients;
- perturb chroma allocation;
- alter cover decisions;
- alter LF/HF allocation;
- alter CfL;
- alter restoration;
- record exact byte change;
- reconstruct and record full SSIMULACRA2 change;
- retain source, transform, frequency, channel, coefficient-context, and metric-attribution features.

Do not train on libjxl’s internal choices. libjxl remains a final black-box curve comparator.

### 9.2 Start with a transparent model

Begin with a bounded model such as:

```text
predicted perceptual loss =
    channel weight
  × frequency weight
  × local structure weight
  × ringing/smoothing asymmetry
  × reconstruction error
```

Use:

- lookup tables;
- monotone splines;
- linear models;
- generalized additive models.

Do not begin with a neural network. The first goal is reliable edit ranking, inspectability, and deterministic execution.

### 9.3 Surrogate purpose

The surrogate predicts:

> Which candidate edit is most promising?

It does not predict:

> What is the final frame score?

Every accepted batch still passes exact full-frame SSIMULACRA2.

### 9.4 Validation

Split data by complete source image and source family, not by blocks.

Report:

- sign accuracy of predicted score changes;
- rank correlation among competing edits;
- false-safe rate;
- bytes saved after rollback;
- results by content class;
- results by score band.

The false-safe rate is particularly important. A model that repeatedly predicts unsafe edits wastes full metric evaluations and destabilizes the controller.

---

## 10. Calibration corpus and comparisons

The controller must be calibrated across quality scores, not bitrate rows.

Use targets such as:

```text
30, 50, 70, 80, 85, 90, 95
```

Expand the corpus to include:

- daylight photographs;
- skin and faces;
- foliage, hair, and fabric;
- low-light and high-ISO noise;
- strong edges beside flat regions;
- saturated colors;
- gradients;
- screenshots and text;
- line art and synthetic graphics;
- grayscale;
- small images;
- very large images.

Use disjoint sets:

1. **Calibration:** initial-rung and policy predictors.
2. **Development:** implementation iteration.
3. **Locked holdout:** promotion decisions.

No near-duplicate or source family should cross those boundaries.

For every image and target score, collect:

- current JPXL bitrate curve interpolated to the score;
- fixed-policy quality controller;
- policy-optimized controller;
- finalist-refined controller;
- libjxl black-box curve interpolated to the score;
- exact bytes;
- achieved score;
- complete encode time;
- guard metrics.

The primary aggregate becomes:

- BD-rate over SSIMULACRA2; or
- geometric-mean byte ratio at matched SSIMULACRA2.

“Same requested bpp” stops being a promotion criterion.

---

## 11. Implementation sequence

### PR 1 — Target semantics and provenance

Implement:

- `LossyTarget`;
- `PerceptualTarget`;
- generic `Effort`;
- `with_ssimulacra2_score`;
- CLI `--quality`;
- target mutual exclusion;
- target-100 lossless routing;
- clean-room provenance;
- metric-version policy;
- perceptual outcome types.

No quality encode behavior needs to change yet.

**Exit gate:** API semantics and documentation are complete.

### PR 2 — Metric engine and canonical reference

Implement:

- optional `jpxl-perceptual`;
- source RGB preservation in perceptual mode;
- pinned SSIMULACRA2 wrapper;
- official and corpus golden tests;
- precomputed-reference benchmarks;
- canonical deterministic final mode.

**Exit gate:** in-memory source/candidate scoring matches the current conformance harness for 8-bit and 16-bit fixtures.

### PR 3 — Pixel plan and plan renderer

Implement:

- `PixelPlan`;
- `ValidatedPixelPlan`;
- separate entropy attachment;
- `jpxl-plan-render`;
- neutral shared reconstruction primitives;
- render/emit/decode parity tests.

**Exit gate:** plan-rendered pixels and score match emitted-and-decoded results throughout the supported VarDCT matrix.

### PR 4 — Fixed-policy quality navigator

Implement:

- `CandidateSearchContext`;
- `quality.rs`;
- initial-rung predictor;
- bounded bracket;
- log-loss interpolation;
- bounded correction;
- Pareto frontier;
- finalist-only entropy;
- hard effort budgets;
- achieved-score reporting.

**Exit gate:** every holdout encode meets the target, higher targets are quality-monotone, and no production effort has an unbounded fallback.

### PR 5 — Perceptual policy bank

Implement:

- `EncodeRequest::for_quality`;
- removal of bpp-dependent quality branches;
- bounded chroma, LF, restoration, cover, and CfL alternatives;
- coordinate descent;
- exact equal-score comparison.

**Exit gate:** no worse than the fixed-policy controller at equal score and a material mean byte reduction.

### PR 6 — Metric breakdown and attribution

Implement:

- raw error;
- 108 aggregate terms;
- contribution maps;
- finite-perturbation attribution audits;
- detailed metric timings.

**Exit gate:** attribution predicts useful regional rankings without being treated as additive truth.

### PR 7 — Terminal coefficient reducer

Implement:

- terminal edit enumeration;
- context-faithful rate estimates;
- metric-based edit ranking;
- batch acceptance;
- rollback and halving;
- entropy retraining checkpoints;
- final exact price and score.

**Exit gate:** lower bytes at the same score on the locked corpus, no target violations, and bounded work.

### PR 8 — Broader edits and surrogate

Implement one edit family at a time, train the JPXL-specific surrogate, and add incremental metric work only when profiling justifies it.

**Exit gate:** the remaining equal-SSIMULACRA2 gap to libjxl reaches the declared threshold without violating speed, determinism, or decoder compatibility.

---

## 12. Acceptance gates

### 12.1 Quality contract

- Final canonical score is never below the requested target.
- Numerical guard is based on measured platform variation.
- Higher requested scores never produce lower achieved scores.
- Score and byte curves are Pareto-monotone after removing dominated outputs.
- Quantizer saturation is explicit.
- Budget exhaustion is explicit.
- No fallback silently changes the meaning of the target.

### 12.2 Compression efficiency

Promotion is measured at equal SSIMULACRA2:

- PR 4 should be byte-neutral or better than the current controller at equal score. Its main purpose is correct targeting.
- PR 5 must reduce mean equal-score bytes without a material content-class regression.
- PR 7 must produce a further stable reduction and close a declared portion of the libjxl gap.
- A reasonable final target is within roughly 1–2% equal-score BD-rate of libjxl on the locked broad corpus, or better. Replace that provisional number with the measured current gap before implementation starts.

A controller that hits the score accurately but uses the same number of bytes is not a compression improvement.

### 12.3 Speed and boundedness

- Every effort has hard metric, render, policy, and entropy budgets.
- Fast and Balanced have no hidden exhaustive fallback.
- Metric and reconstruction are included in end-to-end time.
- A provisional Balanced target is no more than 25% wall-time overhead over current Balanced at the same achieved SSIMULACRA2 once direct plan rendering exists.
- Quality may spend more but remains bounded and reports all work.

### 12.4 Determinism and compatibility

- Same codestream across supported thread counts.
- Same codestream across scalar and SIMD encoder paths.
- Canonical score selection remains stable across supported CPUs.
- Final threshold selection uses deterministic metric reduction or measured hysteresis.
- Every stream is accepted by `jpxl-decode`, `djxl`, and `jxl-oxide`.
- Plan-render parity remains continuously tested.
- Metric version and test-vector fingerprints are recorded.

### 12.5 Anti-gaming safeguards

SSIMULACRA2 is the production objective, but promotion still reports:

- optional external Butteraugli measurements;
- PSNR and RMSE;
- text and line-art screens;
- gradient and banding screens;
- ringing and smoothing screens;
- color-shift screens;
- worst-delta visual review.

Butteraugli does not enter the production controller. It remains an optional external diagnostic.

A policy that gains SSIMULACRA2 through an obvious systematic defect is rejected and the defect is added to the guard corpus.

---

## 13. Telemetry

Every perceptual encode should be able to report:

- metric implementation and version;
- source-reference preparation time and memory;
- source feature bucket;
- predicted initial rung;
- each candidate’s quantizer;
- each candidate’s policy;
- score and render time;
- bracket endpoints;
- crossing prediction;
- pixel-plan count;
- entropy-training count;
- exact-price count;
- policy trials;
- policy winner margin;
- initial score reserve;
- local edits proposed;
- edits accepted;
- edits rejected;
- rollback count;
- bytes saved by edit family;
- final score;
- final guard margin;
- final exact bytes;
- saturation reason;
- budget-exhaustion reason;
- wall time by analysis, planning, reconstruction, metric, entropy, and emission.

Use a machine-readable trace format. These traces become the training data for initial-rung, policy, and local-edit predictors.

---

## 14. Work that should not be done first

Do not begin with:

- a fixed mapping from `--quality 85` to a preselected bpp;
- an unbounded binary search that fully encodes and decodes every rung;
- a per-block division of the scalar SSIMULACRA2 score;
- another variance-only AQ plane;
- a Cartesian sweep over all policy settings;
- copying libjxl’s controller and substituting SSIMULACRA2 for Butteraugli;
- a neural predictor before exact traces exist;
- metric micro-optimization before profiling the full loop.

Those approaches either preserve the original defect, create unacceptable runtime, or obscure whether perceptual feedback improved byte efficiency.

---

## 15. First practical implementation

The first experimental quality encoder should perform:

```text
--quality Q
    -> preserve canonical source
    -> predict quantizer rung
    -> build pixel plan
    -> directly reconstruct candidate
    -> calculate SSIMULACRA2
    -> establish bounded score bracket
    -> interpolate in log SSIM2 loss
    -> score one finalist or correction
    -> attach full entropy to 2–3 feasible finalists
    -> select smallest exact stream meeting Q
    -> emit once
    -> report achieved score
```

That proves the public contract and generates the traces needed for deeper work.

The minimum production architecture likely to improve equal-score bitrate is:

```text
fixed score target
    + bounded perceptual policy selection
    + exact-rate terminal truncation using score reserve
```

The policy bank and terminal reducer should therefore follow immediately after the fixed-policy controller. They should not be deferred indefinitely as optional future quality work.

---

## Definition of done

JPXL has a perceptual quality controller when:

1. Normal lossy encoding accepts a minimum SSIMULACRA2 score.
2. The emitted image is verified against that score using a pinned independent metric implementation.
3. Metric probes reconstruct from validated pixel plans without unnecessary entropy training or serialization.
4. Quality-mode settings contain no requested-bpp policy branches.
5. Multiple legal policies are compared at the same score.
6. The smallest exact qualifying codestream wins.
7. Finalist edits exchange measured score reserve for exact byte savings.
8. Full-frame scoring controls acceptance and rollback.
9. Higher effort improves compression rather than changing target semantics.
10. Equal-score curves show the gap to libjxl closing.
11. The implementation remains clean-room, deterministic, bounded, and independently decodable.

Until policy selection and finalist refinement exist, JPXL has a quality-target interface.

Once they exist, JPXL has a perceptual encoder.