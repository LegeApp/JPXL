# Review verdict

I found the intended file at `JPXL/docs/optimize.md`. I reviewed it alongside the rate controller, quantization, cover selection, CfL, VarDCT event walking, entropy emission, diagnostics, and AKR evidence.

The optimization work is real. The two-anchor controller has removed the old 23–25-probe catastrophe, and the measured Fast and Balanced improvements are substantial. The project has now moved into a different phase:

* **The main speed problem is no longer the outer rate loop.** It is redundant work inside a fixed or mostly fixed plan: quantization replay, event generation, census construction, table building, ANS preparation, and final writing.
* **The main quality problem is not entropy coding.** It is the relatively coarse spatial/perceptual policy: limited atlas features, disabled AQ, approximate cover rate costs, approximate CfL scoring, limited transform vocabulary, and mostly global restoration choices.
* **Fast is near cjxl e7 speed only in an unmatched-quality comparison.** That is useful progress, but not parity.
* **Balanced is promising but not yet validated.** It has only two audit images and, more seriously, its implementation currently contradicts the document’s rejection of globally frozen cover/CfL.

Nothing below requires reading or adopting libjxl source architecture. These recommendations follow from JPXL’s own code, its standard-defined choices, its measurements, and generic dataflow and rate-distortion methods.

---

# Important code-review findings before the nine questions

## 1. `optimize.md` contains current results followed by substantially obsolete advice

The “Original advisor verdict” beginning around `docs/optimize.md:160` still describes Fast as repeatedly running nearly complete planning. That was accurate before the two-anchor work but no longer describes the current implementation.

The document should be split into:

* `docs/optimize.md`: current state, current profiles, current questions.
* `docs/history/optimization-advisor-2026-08-13.md`: superseded diagnosis and historical plan.

At minimum, mark the old sections explicitly as superseded. Otherwise an agent can reasonably follow the old advice and rebuild mechanisms that already exist.

The questions also mix profiles:

* The **18% cover** and **12.6% quantization** figures are from the older hierarchical high-quality profile.
* Current fixed-DCT8 Fast instead reports quantization, writing, ANS, tables, census, and CfL as its leading costs.
* Cover-sharing questions primarily concern Balanced and Quality, not current Fast.

Each profile result and open question should be tagged `[Fast]`, `[Balanced]`, or `[Quality]`.

## 2. Balanced currently performs the global structural freeze that the document says was rejected

`docs/optimize.md:121-125` says reusing initial cover/CfL for the finalist caused up to a 3.95-point SSIMULACRA2 regression and that global freezing is rejected.

The current two-anchor implementation nevertheless does this:

* The finalist is created with `Some(&anchor)` at `jpxl-encode-policy/src/rate.rs:1129-1139`.
* The correction also receives an anchor at `rate.rs:1193-1201`.
* When an anchor is supplied, `jpxl-encode-policy/src/lib.rs:436-445` clones both `anchor.groups` and `anchor.cfl` without rebuilding either.
* The comment at `lib.rs:1064-1069` says a correction reuses a “freshly planned full-CfL finalist,” but the finalist was not freshly planned spatially.

This means “one Full plan” in the Phase 17 diagnostics actually means:

> One plan with **Full entropy alternatives** operating on globally reused cover and CfL.

It does not mean a full spatial/CfL finalist.

That does not automatically make Balanced bad. The two audit images happened to score well. It does mean the earlier 3.95-point failure and the current positive two-image result must be reconciled before Balanced is promoted.

I would replace the ambiguous `Option<&StructuralAnchor>` with an explicit policy:

```rust
enum AnchorReuse {
    None,
    CoverOnly,
    CoverAndCfl,
}
```

Then add assertions such as:

```rust
debug_assert!(
    !matches!(reuse, AnchorReuse::CoverAndCfl)
        || request.rate_preset != RateSearchPreset::Quality
);
```

Also rename `EntropySearch::Full` to something like `EntropyEffort::Exhaustive`, so “Full” no longer sounds like full encoder planning.

## 3. Fallback telemetry loses most of the attempted work

In `rate.rs:877-905`, when the two-anchor path rejects and falls back, only a small subset of the attempted statistics is copied into the exhaustive result. It drops attempted:

* `fast_prices`
* `full_prices`
* DCT cache hits and misses
* candidate allocations and payload bytes
* nested Fast and Full diagnostics
* writer diagnostics

That will make fallback cases look materially cheaper than they were. It also prevents building a reliable confidence model because the pathological samples are exactly the ones whose work is misreported.

Separate the telemetry into:

```rust
struct RateWorkStats {
    // Additive work performed, including abandoned attempts.
}

struct RateDecisionStats {
    // Final selected path, bytes, correction reason, fallback reason.
}
```

`RateWorkStats` can implement an additive merge. Avoid trying to make fields such as “selected finalist bytes” additive.

The current names are also misleading:

* `full_prices` can count finalist Store emissions that did not use Full entropy.
* `exact_candidates` explicitly excludes retained Store emissions.
* `structural_builds` does not distinguish cover builds from CfL builds.

Add counters for:

* outer Count emissions
* internal candidate Count emissions
* Store emissions
* full-entropy training passes
* coefficient walks
* raw event walks
* token replays
* cover builds and partial refreshes
* CfL builds and partial refreshes
* dirty and frozen cover nodes

## 4. Rate controller, spatial effort, quantization effort, CfL effort, and entropy effort are conflated

At `jpxl-encode-policy/src/lib.rs:397-427`, this condition:

```rust
request.rate_preset == RateSearchPreset::Fast
    && entropy_search.uses_fast_entropy()
```

simultaneously chooses:

* nearest quantization
* fixed DCT8 cover
* Fast/default entropy behavior

These are independent decisions. Their coupling makes it difficult to identify whether quality loss comes from the rate controller, spatial policy, quantization policy, CfL, or entropy search.

Use orthogonal effort fields:

```rust
struct EncoderEffort {
    rate: RateController,
    spatial: SpatialEffort,
    quantization: QuantizationEffort,
    cfl: CflEffort,
    entropy: EntropyEffort,
}

enum RateController {
    Exhaustive,
    TwoAnchor,
}

enum SpatialEffort {
    FixedDct8,
    SelectiveLargeTransforms,
    Hierarchical,
}

enum QuantizationEffort {
    Nearest,
    Trailing,
    FullRd,
}

enum CflEffort {
    Neutral,
    Captured,
    Refined,
}

enum EntropyEffort {
    Default,
    Ranked,
    Exhaustive,
}
```

Then named presets become compositions rather than hidden bundles. This is important for both research and production.

## 5. Several output-preserving optimizations should precede more speculative work

These are visible directly in the current code:

| Change                                                                      | Current issue                                                                                       | Relevant code                                          |
| --------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| Construct only DCT8 quantizers for fixed-DCT8 Fast                          | `HfQuantizers::new_with_scales` builds DCT8, DCT16, and DCT32 before cover mode is considered       | `lib.rs:397-427`, `1395-1414`                          |
| Build default dequant matrices once per request                             | Every `HfQuantizer::new` calls `DequantMatrices::all_default()` and rebuilds derived arrays         | `quantize.rs:187-264`                                  |
| Remove transform scratch from completed-cache cover jobs                    | `CoverForwardBank::Complete` ignores `ForwardScratch`, but each small region job allocates it       | `lib.rs:1560-1577`, `3455-3487`, `3893-3947`           |
| Split structural anchors into immutable geometry plus a small HfMul overlay | Every anchor reuse deep-clones `Vec<PlannedGroup>` and `Vec<VarblockDecision>`                      | `lib.rs:436-445`, `1057-1094`                          |
| Replace recursive cover-result `Vec`s with a fixed tree or bitmask          | Current 4×4-atom square hierarchy has only five internal decisions, yet recursion allocates vectors | `tile_region_with` around `lib.rs:3695-3808`           |
| Replace tiny CfL vectors with fixed buffers                                 | LF/CfL collection repeatedly allocates small `Vec<f32>` buffers                                     | `lib.rs:2476-2501`                                     |
| Cache pass-group membership descriptors                                     | Census, table construction, and writing reconstruct pass-group walks repeatedly                     | `jpxl-encode/src/vardct/write.rs:1157-1311`            |
| Remove the second ANS symbol vector                                         | `SymbolEncoder` stores Events and then constructs another `Vec<AnsSymbol>`                          | `jpxl-entropy/src/encode/stream.rs:713-730`, `910-919` |

These changes are lower risk than learned prediction or new visual heuristics. They should be measured first.

---

# Answers to the nine architectural questions

## 1. How should the finalist refresh only structurally unstable cover decisions?

**Yes: use winner/runner-up evidence and a top-down dirty frontier, but margins alone are insufficient unless their exactness is recorded.**

The current square hierarchy over one aligned 4×4-atom region has only five internal decisions:

* one DCT32-versus-split root
* four DCT16-versus-DCT8 children

That can be represented without recursive heap allocation:

```rust
struct CoverRegionSummary {
    valid_nodes: u8,
    nodes: [CoverNodeEvidence; 5],
}

struct CoverNodeEvidence {
    winner: CoverWinner,
    delta_anchor_0: f32, // merge_cost - split_cost
    delta_anchor_1: f32,
    evidence: CostEvidence,
}

enum CostEvidence {
    Exact,
    SplitCertifiedByLowerBound,
}
```

A node is provisionally stable when:

1. It has the same winner at both anchors.
2. The signed cost difference does not approach or cross zero between anchors.
3. Its conservative margin exceeds a calibrated guard.
4. Any child uncertainty cannot overturn the parent’s split cost.
5. The finalist rung lies inside, or close enough to, the anchor interval.

The lower-bound distinction matters. `tile_region_with` can stop scoring a merge once it is guaranteed to lose. In that case you have useful one-sided evidence—“split definitely wins by at least X”—but not an exact runner-up cost. Preserve that as a bound rather than inventing an exact margin.

At the finalist:

* Begin at each region root.
* Reuse a stable merged root immediately.
* For a stable split root, descend only into children marked uncertain.
* Recompute exact candidate costs only for the dirty frontier.
* Materialize the final `Vec<VarblockDecision>` after the fixed decision tree is resolved.

This should be developed against the existing `regret.rs` harness. That harness already computes exact split/merge alternatives and records regret. Extend it to measure:

* winner stability across anchor rungs
* false-stable decisions
* bytes and metric regret caused by freezing
* dirty-node percentage by image and rate
* exact versus lower-bound-certified margins

CfL needs a separate stability mechanism. A stable cover does not prove a stable CfL factor. Return the best and runner-up factor costs for each tile/channel. Recompute CfL when:

* its cover changed
* the best factor differs between anchors
* the factor margin is small
* chroma residual statistics fall outside the calibrated range

The first production version should be conservative. Recomputing 20–30% of nodes is still useful if it replaces a full second cover pass.

## 2. Can the two anchors and finalist be quantized in one coefficient traversal without tripling storage?

**Not literally under the current adaptive controller. There is a data dependency that prevents it.**

The second anchor rung is selected only after the first anchor’s exact size is known:

```text
first quantization and exact price
    ↓
choose second rung
    ↓
second quantization and exact price
    ↓
predict finalist rung
    ↓
finalist quantization
```

Therefore neither the second rung nor the finalist rung is known during the first traversal. A true one-pass implementation would require either:

* speculative quantization of several possible rungs, or
* a retained parametric representation of how every coefficient quantizes over arbitrary rungs.

The latter becomes complicated because:

* trailing truncation is block-global rather than purely coefficient-local
* X/B quantization depends on rung-specific reconstructed Y through CfL
* last-nonzero positions and contexts change by rung
* a structurally refreshed finalist may use different transforms

The practical target is **one transform computation plus two or three cheap quantization replays**, not one quantization replay.

Recommended structure:

1. Keep the current request-scoped forward coefficient banks.
2. For navigation anchors, do not allocate a complete `QuantizedFrame`.
3. Quantize a group, stream its results into a compact event tape or census builder, then reuse the workspace.
4. Retain full frame-sized quantized storage only for the selected finalist.
5. Once selective structural refresh exists, run the finalist quantizer only through dirty regions plus one streamlined pass over stable regions.

There is one optional controller experiment: select two fixed bracketing anchors before either is priced, quantize both in one loop or SIMD lane group, and fall back when they fail to bracket. That makes batching possible but changes the current adaptive behavior. It should be evaluated separately rather than smuggled into a low-level optimization.

For current Fast, the best attainable near-term reduction is:

* first anchor replay
* second anchor replay with no full arena
* finalist replay into the retained arena

That changes three expensive, allocation-heavy plans into three sequential scans over shared coefficients, only one of which produces a full output object.

## 3. What compact token representation can serve census, Count, and Store?

Start with a **raw per-pass-group event tape**, not an elaborate universal IR.

A suitable initial representation is a structure-of-arrays:

```rust
struct RawHfTape {
    group_offsets: Vec<u32>,
    contexts: Vec<u16>, // only after proving the maximum bound
    values: Vec<u32>,
}
```

Using separate arrays avoids padding. If the maximum pre-context count cannot be proven to fit `u16`, use `u32` initially.

This tape can serve:

1. raw-value census
2. token census after choosing hybrid-uint configurations
3. exact encoded-size calculation
4. final Store replay

It should be built per pass group or in fixed slabs so parallel workers can append independently and deterministic reduction remains straightforward.

The current generic `SymbolEncoder::Event` contains a `usize` plus three `u32`s and is likely expensive per event on a 64-bit target. Measure `size_of::<Event>()` explicitly rather than assuming. The ANS path then constructs a second `Vec<AnsSymbol>`, creating another full stream-sized allocation.

After the raw tape is proven, add a packed token tape:

```rust
struct TokenTape {
    cluster: Box<[ClusterIndex]>,
    token: Box<[TokenIndex]>,
    extra_bits: Box<[u8]>,
    extra: Box<[u32]>,
}
```

Then teach the ANS encoder to consume those arrays directly, eliminating `Vec<AnsSymbol>`.

There is one limitation: a raw natural-order tape cannot be reused for an arbitrary custom coefficient order or a materially different block-context model, because both event order and context IDs may change.

For Full entropy alternatives, the later representation should therefore be a **block descriptor**, not merely an event tape:

```rust
struct HfBlockDescriptor {
    transform: TransformType,
    hf_mul: u32,
    qdc: [i32; 3],
    channel_payloads: [CoefficientPayload; 3],
}
```

Each channel payload can choose sparse or dense storage based on nonzero density and retain cell indices. A custom order can then replay the same values in a different order without returning to the quantized frame.

Do this in two stages:

* Raw event tape for Fast/default entropy first.
* Richer block descriptor only after measurements show it is worthwhile for Full entropy ranking.

Also fix census memory before introducing another large buffer. The default block model has thousands of possible contexts, while every worker currently allocates dense `RawHistogram` arrays. Use lazy contexts or a touched-context bitset with deterministic sorted merging. Add a values-only `TokenCensus` constructor where direct pre-tokenized LZ77 symbols cannot occur.

## 4. Can winner/runner-up summaries share cover scoring across anchors and the finalist?

**Yes, but the summary should represent a small cost response curve, not only the selected block map.**

The forward candidate bank already stores the expensive transform results. What is currently lost after cover selection is the decision evidence.

For every internal cover node, retain:

* merge transform identity
* split child identities
* merge and split cost at anchor 0
* merge and split cost at anchor 1
* distortion and rate components separately
* exact or lower-bound status
* selected HfMul or candidate HfMul range

Separating distortion and rate allows a conservative finalist estimate when the rate multiplier or quantizer changes:

```rust
struct CandidateCostSample {
    distortion: f32,
    coefficient_rate: f32,
    metadata_rate: f32,
}
```

Do not assume linear interpolation is exact. Use it only to classify clearly stable versus uncertain nodes. Dirty nodes still receive exact scoring.

An efficient evaluation strategy is:

1. Traverse each cached candidate’s coefficients for the current anchor.
2. Accumulate costs for all relevant node candidates.
3. Resolve the five-node region tree without allocations.
4. Add the second sample when its rung becomes known.
5. At the finalist, freeze only nodes whose cost intervals cannot overlap.
6. Re-score the rest exactly.

This shares the transform bank, node topology, metadata, and most scheduling state even though quantization itself remains sequential.

A more advanced version can retain a few quantization breakpoints for each candidate. I would not begin there. The current two-point samples plus exact dirty refresh are much easier to validate.

## 5. Can a cheap confidence signal identify the class where Fast loses approximately 0.8 SSIMULACRA2?

There is not yet evidence that this is a “class.”

The AKR record states that seven canonical 1 bpp scenes had a mean delta of −0.1362 and one worst result of −0.7806. It does not establish that the worst image represents a repeatable semantic or photographic category. Training a content classifier from this would overfit immediately.

Use codec-internal risk signals first:

* Whether the target is bracketed or extrapolated.
* Anchor slope and anchor span.
* Predicted-versus-actual finalist residual from prior corpus samples.
* Percentage of low-margin cover nodes.
* Cover winner disagreement between anchors.
* CfL factor disagreement or weak factor margins.
* A deterministic sparse sample of exact DCT16/DCT32 regret.
* Flat-gradient, edge-coherence, high-frequency-tail, and noise statistics.
* Chroma-luma covariance and residual outliers.

The most valuable signal will probably be a **sampled structural-regret test**:

* Select a fixed deterministic subset of regions, such as one out of every 32 or 64.
* On those regions, compare fixed DCT8 with the hierarchical choice exactly.
* Measure missed rate-distortion gain.
* Route high-regret images to `SelectiveLargeTransforms` or Balanced.
* Route to exhaustive Quality only when structural risk and rate-model risk are both high.

This preserves Fast for easy images without requiring a semantic scene model.

Once the corpus is large enough, a small monotone decision tree or calibrated linear classifier can combine these signals. Train it to predict actual Fast-versus-Quality regret, not scene labels.

## 6. Can Fast selectively restore larger transforms cheaply?

Yes. **Variance alone is not an adequate gate.**

High variance may indicate noise, grass, hair, or other texture where a larger transform is unhelpful. A useful large-transform prefilter should look for energy that is both structured and compressible:

* low residual after fitting a mean or plane
* coherent gradient direction
* low high-frequency tail energy
* low estimated noise
* neighboring blocks with consistent orientation or smoothness
* low chroma disagreement
* low DCT8 split cost but a plausible merge lower bound

Start with DCT16 only. Introduce DCT32 only when all four DCT16-scale quadrants and the parent region satisfy a stronger gate.

A reasonable profile is:

```text
Fast:
    fixed DCT8

Fast-Hybrid:
    DCT8 everywhere
    exact DCT16 scoring only on atlas-approved regions
    DCT32 only on strongly coherent parent regions

Balanced/Quality:
    full hierarchical scorer
```

The prefilter must run before completing every DCT16/DCT32 candidate bank. Otherwise the transform work has already been paid and only the scorer was avoided.

The existing `AnalysisAtlas` was clearly designed for this evolution. Its comments list gradients, Laplacian energy, anisotropy, noise, covariance, masking, and saliency as future features, but it currently stores only mean and variance at `analysis.rs:1-60`.

Use `regret.rs` to tune the gate:

* false-negative transform regret
* fraction of regions admitted
* DCT time spent
* recovered SSIMULACRA2/Butteraugli
* leave-one-scene-out validation

Do not optimize the threshold against the current seven-image screen.

## 7. Can the undershoot margin be learned from bounded rate-slope confidence?

Yes, but **two points cannot provide their own confidence interval**. They define a slope, not uncertainty.

Confidence must come from either:

* an offline error distribution, or
* a cheap third anchor when the first two are unusual.

Record, for a broad corpus:

```text
anchor rungs
anchor sizes
log-rate slope
anchor span
target interpolation or extrapolation
spatial/quant/CfL/entropy effort
predicted finalist bytes
actual finalist bytes
correction rung and bytes
```

Fit a small quantile model for:

```text
log(actual_finalist_bytes / predicted_finalist_bytes)
```

Useful predictors include:

* slope magnitude
* distance from target to nearest anchor
* whether the target is bracketed
* anchor size ratio
* zero-coefficient fraction
* event entropy
* image dimensions
* transform distribution
* navigation-versus-final entropy mode

At runtime, produce a conservative finalist-byte interval:

* Narrow interval: emit one finalist.
* Wide but monotone interval: add one cheap midpoint or directional anchor.
* Non-monotone or implausible interval: fall back.

The undershoot reserve becomes the calibrated upper error quantile instead of a fixed one-eighth of the tolerance.

The correction calculation should also use the nearest compatible points. Current correction logic derives from two far Fast anchors plus a finalist that may use Full entropy. Those are not exactly the same rate curve. For Balanced, explicitly model the Fast-to-Full entropy offset or use a local secant involving the actual finalist.

One practical correction to the question: a single correction will usually be much cheaper than an exhaustive Quality fallback. The confidence system should generally choose among:

1. one finalist
2. one additional cheap anchor
3. one correction
4. exhaustive fallback

It should not fall back merely to avoid a correction.

## 8. Is Balanced’s SSIMULACRA2/Butteraugli trade stable, and should it have a guard?

The two-photo result is insufficient to establish stability, particularly while Balanced is globally reusing the first anchor’s cover and CfL.

First fix the structural-reuse semantics. Then evaluate across:

* several hundred photographs and synthetic images
* low-light, noise, foliage, skin, text, graphics, sky, gradients, saturated colors
* at least 0.5, 1, 2, and 4 bpp
* mean, median, p95, and worst-case metric deltas
* per-category tails, not only aggregate means

A runtime guard should use quantities already available inside the encoder rather than running a complete Butteraugli comparison:

* total weighted coefficient reconstruction error
* p95/p99 and maximum block error
* coherent low-frequency error across neighboring blocks
* gradient-band or ringing-risk estimates
* chroma/CfL residual outliers
* percentage of low-margin frozen cover nodes
* deterministic sampled transform regret
* percentage of local HfMul decisions at extreme values

When the guard trips, do not immediately rerun the entire exhaustive rate loop. Use a staged response:

1. Rebuild full cover and CfL at the predicted finalist rung.
2. Reuse the already known target neighborhood.
3. Re-price that fresh plan.
4. Apply one corrected rung if needed.
5. Fall back to exhaustive Quality only if it still fails the target or risk gate.

This “fresh spatial finalist” is likely to capture most of Quality’s result without paying for its entire quantizer ladder.

## 9. Can a trained cost ranker eliminate losing entropy alternatives?

Yes, and this is probably necessary for Balanced. But a ranker alone will not remove the full 0.8 seconds because one baseline census, model build, ANS encode, and Store still remain.

Do not start with an opaque machine-learning ranker. Start with an analytical estimator derived from JPXL’s own event statistics:

```text
estimated candidate cost =
    raw extra bits
  + cross-entropy under candidate distributions
  + exact context-map signalling
  + exact hybrid-uint configuration signalling
  + exact order signalling
  + exact block-context signalling
  + exact preset and assignment signalling
```

For each class of entropy alternative:

* **Coefficient order:** estimated event/token reduction minus order signalling.
* **Block context:** cross-entropy reduction from retagging descriptors minus model and map overhead.
* **Presets:** per-group histogram divergence and expected gain minus duplicated model and assignment overhead.

Balanced should:

1. Always build the default baseline.
2. Rank alternatives cheaply.
3. Exact-price only the strongest challenger.
4. Exact-price additional candidates only when the estimate’s uncertainty overlaps zero gain.
5. Use the baseline if no challenger clears a conservative gain threshold.

Quality remains exhaustive and becomes the oracle used to calibrate the estimator. Record false-negative regret:

```text
bytes lost because an omitted candidate would actually have won
```

A calibrated linear correction or small tree can later learn the residual between analytical estimates and exact prices. Keep the analytical signalling costs exact.

Also audit parallelism. Natural-order and custom-order candidates can compete at the candidate level while each census also parallelizes across groups. Use one parallel dimension at a time based on candidate count and group count; otherwise scheduler and cache overhead may erase the benefit.

---

# The most important next architecture: Anchored Quality

The current presets leave a missing middle:

* Fast: cheap controller and deliberately cheap spatial tools.
* Balanced: cheap controller but globally frozen initial structure, then Full entropy.
* Quality: exhaustive controller and full structural policy.

Add an **Anchored Quality** path whose purpose is to isolate rate-controller savings from structural quality compromises:

1. Build the request-scoped atlas and forward caches.
2. Run two cheap navigation anchors.
3. Predict the target rung.
4. At that rung, build a **fresh hierarchical cover and fresh CfL** using the Quality spatial and quantization policies.
5. Run default entropy to obtain an exact local rate point.
6. Run exhaustive or ranked entropy only for that finalist.
7. If necessary, perform one correction using selective structural refresh.
8. Fall back to exhaustive Quality when the exact target or confidence gates fail.

This is the cleanest experiment in the project because it answers:

> How much of exhaustive Quality’s runtime is necessary for quality, and how much is merely its rate-search controller?

I expect this to be more valuable than immediately trying to perfect global anchor reuse. If Anchored Quality stays close to exhaustive Quality across a broad corpus, the exhaustive ladder can remain an oracle and rare fallback rather than the production high-quality path.

---

# Further speed work, in priority order

## Phase 0: repair semantics and measurement

Before another optimization:

1. Reconcile Balanced with the rejected global-freeze experiment.
2. Introduce explicit cover/CfL reuse modes.
3. Fix fallback telemetry.
4. Rename entropy “Full” terminology.
5. Tag every profile and question by preset.
6. Add counters for coefficient walks, event generation, token replays, candidate training, dirty nodes, and retained bytes.

Without this, later improvements may be credited to the wrong subsystem.

## Phase 1: output-preserving cleanup

Implement and benchmark these independently:

1. DCT8-only quantizer construction for fixed-DCT8 Fast.
2. Request-scoped default dequant matrix cache.
3. Shared immutable RD tables where they are identical across HfMul values.
4. Completed-cache cover API that does not request unused `ForwardScratch`.
5. Worker-local batches of cover regions rather than one scratch allocation per region.
6. Fixed cover trees and final materialization instead of recursive temporary vectors.
7. Fixed or reusable CfL sample buffers.
8. Immutable `Arc` structural geometry plus compact per-probe HfMul overlays.
9. Cached pass-group walk descriptors.
10. Sparse/lazy census contexts.
11. Direct ANS encoding from packed event storage.

Require byte-identical outputs and one-thread/four-thread determinism for each change.

## Phase 2: raw event tape for Fast

Implement the raw tape only for:

* natural coefficient order
* default block context
* one preset
* Fast/default entropy

This confines complexity and directly attacks the current Fast profile’s census, table, ANS, and writing costs.

Measure:

* events per pixel
* bytes per event
* peak RSS
* number of coefficient walks removed
* table-build time
* ANS time
* final writing time

Do not proceed to a universal descriptor until these numbers show the memory trade is favorable.

## Phase 3: Anchored Quality

Implement the fresh full spatial/CfL finalist described above.

This is the next major quality-preserving speed milestone.

## Phase 4: selective cover and CfL refresh

Add fixed node summaries, anchor margins, and the dirty frontier. Initially compare against Anchored Quality, not against globally frozen Balanced.

The acceptance condition should be:

* meaningful cover-time reduction
* no statistically meaningful metric regression
* very low false-stable regret
* bounded worst-case fallback rate

## Phase 5: analytical entropy ranker

Use the exhaustive path to build a corpus of candidate estimates versus exact prices. Promote ranking only after its false-negative byte regret is understood.

---

# Further quality work required for genuine matched-quality parity

Micro-optimization can bring Fast and Balanced closer in speed. It cannot by itself eliminate the quality gap. The current quality ceiling is visible in the policy code.

## 1. Complete the AnalysisAtlas

`AnalysisAtlas` currently contains only per-atom mean and variance. Add request-scoped, multi-scale integral statistics for:

* horizontal and vertical gradient energy
* gradient orientation coherence
* Laplacian or high-frequency energy
* robust noise estimates
* X/Y/B covariance
* local plane-fit residual
* flat-gradient risk
* masking and edge density

Compute them once and reuse them for:

* transform gating
* AQ
* CfL confidence
* restoration/filter policy
* distortion guards

This is not a scene-recognition system. It is a compact local signal model for encoder decisions.

## 2. Replace variance AQ with actual local rate-distortion choices

Production target-rate currently disables AQ at `request.rs:501-523` because the earlier activity heuristic lost against AQ Off.

Do not revive the same variance/x265-style activity formula with different constants. Build small discrete RD curves from JPXL’s own coefficients:

```text
for each region and candidate HfMul:
    estimate exact or near-exact rate
    measure weighted reconstruction distortion
    retain non-dominated choices
```

Then select under a Lagrange multiplier:

```text
J = D + λR
```

Add a small spatial regularizer so neighboring HfMul choices do not fluctuate arbitrarily.

Because the curves are discrete and small, λ can be adjusted against aggregate predicted rate without re-transforming or exhaustively re-quantizing the image. This can become both an AQ system and a more accurate target-rate model.

## 3. Replace hardcoded cover-rate proxies with model-derived costs

Current cover scoring uses values such as:

* `PER_VARBLOCK_BITS = 2`
* `NON_DCT8X8_SIGNAL_BITS = 32`
* a fixed non-baseline HfMul signalling cost
* `residual_bits(q)` as a magnitude-class proxy

The comment at `lib.rs:3278-3284` itself says the non-DCT8 cost should be re-derived after entropy-model training.

Use a bounded two-stage process:

1. Initial cover with the cheap proxy.
2. Train a provisional default entropy model.
3. Compute expected token cost from actual local raw-value distributions.
4. Re-score only low-margin cover nodes.
5. Retrain once if the map changed materially.

Do not create an open-ended cover/entropy iteration. One provisional model plus one local refresh is enough for the first version.

This can improve compression and transform selection simultaneously because the cover objective will finally reflect the encoder’s actual entropy behavior.

## 4. Improve CfL scoring for larger transforms

The current CfL collection folds larger-transform coefficients onto an 8×8 frequency grid and scores factors with a DCT8 baseline quantizer at `lib.rs:2503-2517` and `2752-2767`.

That is a reasonable initial approximation but can mis-rank factors for DCT16/DCT32 blocks.

Keep the current regression as a cheap seed, then:

1. Generate a small candidate set around the seed plus neutral.
2. Re-score the strongest candidates with the actual transform and HfMul quantizer.
3. Return best and runner-up costs for the stability guard.
4. Recompute only tiles containing changed cover nodes.

A bounded cover/CfL alternation is also reasonable:

```text
neutral-CfL cover
→ CfL estimate
→ re-score low-margin cover nodes with CfL-aware chroma error
→ final CfL
```

One alternation, not an uncontrolled loop.

## 5. Expand transform use based on anisotropy

After DCT16/DCT32 gating works, add independently designed selection for the standard’s rectangular or special VarDCT transforms.

Do not enable the entire vocabulary globally. Use:

* anisotropy
* edge direction
* plane-fit residual
* transform-specific lower bounds
* exact scoring only for admitted candidates

This is use of standard-defined coding tools, not adoption of another encoder’s architecture.

## 6. Make LF and restoration decisions content-aware

Production currently uses fixed choices such as:

* `quant_lf = 8`
* one EPF iteration
* uniform sharpness 7
* one cover frequency weighting policy

Those were sensible corpus promotions, but global constants will eventually limit worst-case quality.

Use the atlas and final quantization error to select at least frame- or group-level:

* LF quantization
* EPF strength
* sharpness
* Gaborish/restoration policy
* risk-based treatment of flat gradients and ringing-prone edges

Optimize metric tails, not only means. A small improvement to worst-case gradients, skin, text, and saturated edges may matter more than a larger average PSNR gain.

---

# How to benchmark against libjxl without importing its architecture

The public libjxl effort documentation makes clear that effort levels do not merely run the same encoder longer: different efforts use different VarDCT tools and increasingly exhaustive heuristics. Higher effort generally improves quality and quality consistency at a given size. Therefore fixed-DCT8 Fast versus cjxl e7 is a legitimate absolute speed observation, but not a matched-tool or matched-quality claim. ([GitHub][1])

Use three benchmark lanes.

## Lane A: comparable fast-tool lane

Compare:

* JPXL fixed-DCT8/default entropy Fast
* cjxl e1/e3

The public effort description identifies e1/e2 as 8×8 VarDCT and e3 as adding better ANS, making these more informative fast-effort references than e7. This is comparator selection, not encoder-design guidance. ([GitHub][1])

## Lane B: production matched-quality lane

Compare:

* Anchored Quality
* exhaustive JPXL Quality
* cjxl e6/e7

For each image and target rate, tune the cjxl quality/distance parameter outside the timed region until compressed size matches JPXL within a fixed tolerance. Then time the calibrated encode.

This answers:

> At the same bytes and comparable decoded quality, which encoder is faster?

## Lane C: target-rate service lane

JPXL is doing an exact target-rate search, while a single calibrated cjxl encode is not necessarily providing the same service.

Build a common black-box target-size controller around both encoders:

* same starting information
* same byte tolerance
* same maximum attempts
* same never-over or closest-size rule

Time the entire controller plus encoding. Report this separately from fixed-parameter core encode speed.

JPXL may be able to beat libjxl first in this lane through its bounded controller even before its single-plan core is faster.

## Required protocol

Use:

* a pinned cjxl commit or binary hash
* recorded compiler and build flags
* equivalent PGO/LTO status
* fixed CPU power mode and thread count
* one-thread and fixed four-thread results
* 0.5, 1, 2, and 4 bpp
* interleaved process order
* medians and dispersion
* wall time, CPU time, peak RSS, bytes, and metric tails
* independent decoding and conformance checks

The official `benchmark_xl` tooling supports codec comparisons, objective metrics, separate inner-thread controls, and repeated measurements; its documentation recommends ten encode/decode repetitions for more consistent results. ([GitHub][2])

The current Fast measurements should therefore be described as:

> Approximately 1.15–1.19× cjxl e7 wall time in one separate process window, with materially lower decoded quality and substantial host variance.

The current Balanced result should be described as:

> Approximately 4.4–5× faster than JPXL exhaustive Quality on two photographs, while globally reusing first-anchor cover/CfL; matched-quality cjxl comparison and broad-corpus validation remain outstanding.

That is still a strong result. It is simply the accurate result.

---

# Recommended implementation order

The most productive sequence is:

1. **Fix the Balanced/global-freeze contradiction and diagnostics semantics.**
2. **Separate rate, spatial, quantization, CfL, and entropy effort controls.**
3. **Take the byte-identical allocation and construction wins.**
4. **Implement raw event tapes on the default-entropy Fast path.**
5. **Implement Anchored Quality with a fresh spatial/CfL finalist.**
6. **Add fixed cover decision summaries and selective dirty refresh.**
7. **Add the analytical entropy ranker.**
8. **Expand the atlas and build an actual local RD-based AQ system.**
9. **Replace cover/CfL proxies with bounded model-informed refinement.**
10. **Add selectively gated standard transforms and adaptive restoration.**

The first six steps are primarily a speed program. The final four are what can move JPXL from “fast clean-room encoder with a quality fallback” toward genuine matched-quality parity or superiority.

I could not freshly compile or benchmark the archive because this container lacks `cargo` and `rustc`. The review is therefore static. The included AKR records report that the workspace build, tests, formatting, diff checks, and no-default-features policy build passed at the recorded Phase 10 checkpoint, but that is project evidence rather than a fresh verification performed here.

[1]: https://github.com/libjxl/libjxl/blob/main/doc/encode_effort.md "libjxl/doc/encode_effort.md at main · libjxl/libjxl · GitHub"
[2]: https://github.com/libjxl/libjxl/blob/main/doc/benchmarking.md "libjxl/doc/benchmarking.md at main · libjxl/libjxl · GitHub"
