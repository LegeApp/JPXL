# JPXL optimization questions for outside review

This is a non-authoritative advisor brief, not the project plan. Settled work
and acceptance remain in AKR. The questions below are kept current when
progress exposes an architectural limit that local profiling does not answer.

## Current checkpoint — 2026-08-15

The runtime `fast` lossy preset uses two exact navigation anchors, a fitted
log-rate prediction, one freshly rebuilt default-entropy finalist, and at most
one exact correction. Navigation retains the production hierarchical cover but
uses neutral CfL; the finalist restores the full CfL search. Its exact retained
Store emission is now also its size verification, removing a redundant Count
traversal. Normal builds include Fast, while `quality` remains the exhaustive
default. Fast permits up to 3% target undershoot but never exceeds the target.

The forward 2-D DCT now transposes around each separable pass so its 1-D
kernels operate on contiguous rows while preserving the original
column-then-row floating evaluation exactly. Fresh PGO A/B against Phase 11
improved the 2400x1800 median from 1.14 to 1.10 s (3.5%) and the 4000x3000
median from 1.93 to 1.91 s (1.0%). The seven canonical Fast codestreams, both
timing codestreams, and a Quality-preset codestream were byte-identical to
Phase 11.

The following quantization cleanup removes full-lane clears from scratch and
output arenas that are overwritten by the same call, and walks chroma HF row
spans directly instead of dividing every coefficient index to rediscover LLF.
Fresh PGO A/B against the contiguous-DCT checkpoint improved medians by 0.9%
at 2400x1800 and 1.5% at 4000x3000. Canonical Fast training outputs and a
Quality-preset output remain byte-identical.

Fresh four-thread, process-to-process matched-SSIMULACRA2 timing is 1.09 s
versus cjxl 0.40 s on 2400x1800 (2.73x), and 1.89 s versus 1.05 s on
4000x3000 (1.80x). The seven-scene 1 bpp screen still has zero fallbacks and
two corrections; its mean SSIMULACRA2 delta versus Phase 10 is -0.034 points
and the worst is -0.154. Quality remains available for callers that do not
want the cumulative speed/quality trade.

The newest 12 MP PGO profile, captured immediately before the small
overwrite-only quantization cleanup, contains 7,762 core-cycle samples with
zero lost.
Its largest self-costs are cover scoring 19.5% across two hot closures,
quantization 16.2% across the chunk closure and lane kernel, contiguous DCT
rows 6.5% (plus 1.3% in DCT16), forward-varblock preparation 6.5%, pass-group
writing 4.6%, entropy tables 3.8%, ANS 3.5%, CfL 3.3%, and census 3.2%. The
former column-DCT leaf fell from 11.5% to a 6.5% contiguous-row leaf.

### Open architectural questions

1. How can the finalist refresh only structurally unstable cover decisions?
   Reusing the initial cover/CfL for the finalist saved almost no thin-LTO
   wall time and regressed SSIMULACRA2 by as much as 3.95 points, so global
   freezing is rejected. Would winner/runner-up margins plus a local dirty
   frontier avoid most of the second cover/DCT pass without that quality loss?
2. Can the two far-apart anchor quantizers and finalist be quantized from one
   coefficient traversal without tripling result storage? Quantization is
   still 10.3% self time after probe collapse; the existing advice to batch
   adjacent rungs does not directly fit this wide two-anchor geometry.
3. What compact token representation could serve census, exact Count, and
   final Store without becoming another frame-sized allocation? Even the
   default-entropy Fast finalist still spends about 16% across census, table
   construction, ANS, and pass-group writing.
4. The mid-size frame remains 2.73x slower than cjxl while the 12 MP frame is
   1.80x slower. Cover construction currently takes a write lock for one
   LF-group-wide dense coefficient bank, even though its aligned 32x32 regions
   are logically independent. Which deterministic ownership model best exposes
   those regions to work stealing without adding per-region full-bank arenas:
   partitioned dense banks, a prefilled immutable candidate bank, or staged
   cover/DCT pipelining?
5. Can an inexpensive confidence signal identify the one corpus class where
   Fast loses about 0.8 SSIMULACRA2 points and route it to Quality, without
   first doing the exhaustive search that the preset exists to avoid?

## Original advisor verdict

The agent’s architectural diagnosis is **directionally correct, but too broad if interpreted as “rewrite the encoder.”**

You do **not** need to replace the VarDCT transforms, quantizer, validated plan types, exact writer, decoder, or conformance infrastructure. Those are valuable assets and are no longer the main reason the target-rate path is 16–41× slower.

You **do** need a targeted architectural refactor of two boundaries:

1. **The boundary between analysis and a rate probe**
2. **The boundary between entropy candidate generation and exact emission**

Put differently: **rewrite what a probe does, not how JPEG XL is encoded.**

The 3.3% diagnostics win was worth taking, but it also confirms that ordinary overhead removal will not close the remaining gap. The current target-rate path repeatedly performs work that should happen once per image, once per structural anchor, or only for the final candidate.

This conclusion is based on the supplied source and AKR measurements rather than a fresh runtime profile.

---

# Why the current rate architecture is still expensive

## 1. `Fast` is not actually fast

In `jpxl-encode-policy/src/rate.rs`, every Fast ladder rung still does:

```text
cover selection
CfL estimation
group quantization
entropy census
entropy training
exact serial count-only emission
```

The only major operation Fast avoids is trying the additional Full entropy alternatives.

That makes Fast an **exact encode with a reduced entropy search**, not a cheap rate estimate.

Your measured searches execute roughly:

* 23 probes at 12 MP: 11 Fast + 12 Full
* 25 probes at 1024×768: 13 Fast + 12 Full

Forward DCT caching successfully prevents those probes from recomputing most transforms—the observed hit rates are 96–97%—but all those probes still rescore the cover, estimate CfL, quantize, train entropy, and traverse the writer.

That is the main search-amplification problem.

## 2. One recorded Full probe can contain many hidden full passes

`RateProbeStats.full_prices` counts the outer Full rungs, but it does not expose all the work inside `plan_at_with_cfl`.

A single Full plan can currently perform:

* natural-order census and training;
* reordered census and training;
* exact natural-versus-reordered prices;
* another census/train/order sequence for a custom block-context candidate;
* exact best-versus-custom prices;
* another sequence for multi-preset assignment;
* exact best-versus-preset prices;
* the outer stored emission.

Based on the control flow in `lib.rs`, one sufficiently complex Full probe can reach **up to ten internal `price_codestream` traversals**, before its outer emission. Not every image triggers the maximum, but the actual aggregate count is currently hidden.

Therefore “12 Full probes” may represent far more than 12 writer-equivalent passes.

## 3. Exact count pricing is forced to serial execution

In `jpxl-encode/src/vardct/write.rs`:

```rust
price_codestream(plan)
    -> emit_codestream_mode(..., EmitMode::Count, EncodeResources::serial())
```

So:

* every Fast exact price is serial;
* entropy natural/reordered comparisons are serial;
* custom-context comparisons are serial;
* preset comparisons are serial.

Meanwhile, the stored Full emission can use the request’s group resources. That is an unnecessarily large distinction in a rate loop dominated by exact prices.

## 4. The planning side is largely serial

The existing coarse parallelism primarily applies to emitted sections. Important planning operations are still serial:

* LF-group cover selection;
* CfL preparation and search;
* selected-group quantization;
* parts of entropy census/training.

That means accelerating the writer alone cannot bring the fixed path—or the target-rate path—to multicore parity.

## 5. More cover-loop micro-optimization is now a secondary target

Your measurements already showed that only around 31–35% of cover time was in the exact scoring portion; most was forward/cache work. The lane SIMD improvement correctly gained around 14–15%, while the exact lower-bound prune lost 20–25%.

That is useful evidence. Another isolated scorer optimization might gain several percent. It cannot erase a 16–41× total gap while the encoder still performs 20-plus full-ish probes.

---

# What should remain intact

| Keep                                        | Refactor                              |
| ------------------------------------------- | ------------------------------------- |
| Exact quantizer behavior                    | What constitutes a rate probe         |
| DCT implementations and cached coefficients | Cache representation and ownership    |
| Validated spatial/quantized/emission plans  | Monolithic `plan_at_with_cfl` staging |
| Exact final writer                          | How often exact writing occurs        |
| Final target-size gate                      | Intermediate rate estimation          |
| Deterministic ordered reduction             | Planning-side parallel scheduling     |
| Current exhaustive rate loop                | Retain only as oracle/fallback        |

The current loop should survive as a slow reference path until the replacement has broad evidence. Do not delete it during the refactor.

---

# Recommended architecture

The useful split is approximately:

```text
PreparedAnalysis
    geometry
    preconditioned frame
    analysis atlas
    per-group candidate coefficient banks

SpatialAnchor
    chosen cover
    selected forward references
    exact CfL parameters
    cover decision margins

QuantizedProbe
    quantized selected coefficients for one quantizer
    LF data
    tokenizable group data

TokenTape / RateSketch
    actual coefficient tokens
    raw-bit accounting
    base context properties
    compact size estimate

EntropyFinalist
    trained distributions
    selected orders
    block-context plan
    preset assignment

ExactEmission
    exact count or retained codestream
```

This is not cosmetic type splitting. It gives the rate controller legal, testable stopping points at which it can reuse earlier work.

---

# Work plan

## Phase 0: expose the real multiplicity

This should be the next commit because it changes no decisions and prevents agents from optimizing the wrong layer.

Add a search-scoped aggregate diagnostic rather than relying on the per-`plan_at` diagnostic that gets reset for every probe.

At minimum count and time:

```text
fast_plans
full_plans
cover_passes
cfl_searches
quantize_group_passes

census_passes
entropy_trainings
order_candidates
block_context_candidates
preset_candidates

internal_count_emissions
outer_count_emissions
stored_emissions

lf_section_encodes
pass_group_section_encodes
executor_pool_builds

candidate_cache_entries
candidate_payload_bytes
candidate_allocations
```

Split the times by Fast and Full.

Two derived metrics should become standard benchmark output:

```text
search_amplification =
    total target-rate wall time / selected plan encoded once

writer_amplification =
    all count/store section traversals / one final stored emission
```

The first milestone is not yet “beat cjxl.” It is to make those amplification values sane.

Also instrument `ordered_map_rayon` pool construction. It currently constructs a local Rayon pool on each invocation, and an emission invokes ordered maps separately for LF and pass groups. Repeating this across many internal prices is structurally wasteful.

---

## Phase 1: output-preserving execution improvements

These changes should retain exact fingerprints.

### A. Introduce a request-scoped executor

Replace the resource description that reconstructs execution infrastructure with an object such as:

```text
EncodeSession
  PreparedAnalysis
  EncodeExecutor
  scratch/buffer pools
  diagnostics accumulator
```

The executor should own or reference a persistent worker pool. It should be reused across:

* planning groups;
* census groups;
* exact Count emissions;
* final Store emission.

Keep the existing fixed-index result collection and ordered reduction. That preserves deterministic output.

### B. Add parallel exact pricing

Add the equivalent of:

```rust
price_codestream_with(plan, executor_or_resources)
```

Count-only section bodies are just as independent as stored section bodies. They should use the same deterministic group-level parallelism.

This will not solve excessive emission count, but it makes every remaining exact candidate cheaper and is low risk.

### C. Replace the global candidate `HashMap` with per-group dense banks

Do **not** cap or evict the forward cache. The 96–97% cross-probe hit rate proves that retaining one frame’s complete candidate set is economically correct.

The problem is its representation:

* global hash lookup;
* `contains_key` followed by `get`;
* three `Vec<f32>` allocations per candidate;
* roughly 245,000 candidates on the measured 12 MP image;
* therefore potentially around 735,000 small coefficient allocations.

Use LF-group-local storage, separately indexed by transform family. For example:

```text
PreparedLfGroup
  dct8_coefficients
  dct16_coefficients
  dct32_coefficients
  candidate metadata
  reusable scratch
```

A dense offset can generally be computed from transform size and aligned position. One allocation per transform bank or group is preferable to one allocation per channel per candidate.

This retains the successful cross-probe cache while removing hashing, pointer chasing, allocator traffic, and global mutable ownership.

As an interim micro-change, use an entry-style lookup rather than `contains_key` plus `get`, but treat that as temporary hygiene rather than the result.

### D. Parallelize planning at LF-group granularity

After sharding the candidate banks, the principal ownership obstruction to planning parallelism is removed.

Parallelize:

1. candidate forward construction;
2. cover selection per LF group;
3. CfL sample extraction per group;
4. selected-group quantization;
5. group census/token generation.

For global CfL, compute group-local results in parallel and reduce them in original LF-group order. This should retain deterministic floating-point accumulation behavior more reliably than a general parallel reduction.

These Phase 1 changes may make fixed encoding materially faster, but they still do not address the 23–25-probe rate loop.

---

## Phase 2: make intermediate rate probes genuinely cheap

This is the main architectural work.

### A. Use one structural anchor

Run a complete cover and exact CfL search at an initial quantizer. Preserve:

* selected cover;
* selected coefficient references;
* exact CfL values;
* winner/runner-up cost margin at each cover decision;
* entropy topology information.

This becomes a `SpatialAnchor`.

### B. Probe nearby quantizers without rebuilding structure

For nearby quantizers:

* reuse the selected cover;
* reuse the forward coefficients;
* initially reuse exact anchor CfL;
* re-quantize only the selected coefficients;
* walk the actual tokenization path;
* estimate output size without full entropy alternative search or exact ANS serialization.

The existing `residual_bits` proxy is not sufficient. Your own experiment found it accounts for only approximately 22%, 34%, and 51% of emitted symbols at 0.5, 1, and 2 bpp. The new estimator must derive from the real token/context traversal.

A useful `RateSketch` would contain:

* symbol counts by base context;
* raw-bit counts;
* nonzero/run structure;
* histogram/model overhead estimate;
* per-section fixed overhead;
* optionally a frozen entropy model price.

It does not need to be perfect. It needs to predict the quantizer closely enough that only a few exact finalists remain.

### C. Use a bounded anchor–predict–verify loop

A normal search should look like this:

1. Build one full structural anchor.
2. Run two cheap sketch probes around the initial rung.
3. Fit the local log-size/quantizer slope.
4. Predict the target rung.
5. Run one or two nearby sketch corrections.
6. Fully re-plan and exactly price the predicted finalist.
7. If outside tolerance, perform one corrected full candidate.
8. Fall back to the current exhaustive loop when confidence or monotonicity fails.

Set explicit normal-path budgets:

```text
structural cover/CfL builds: <= 2
full entropy searches:       <= 2
exact candidate emissions:   <= 4
stored final emissions:      1
```

The old loop remains the fallback, not the default.

### D. Do not globally freeze cover decisions

Your winner-stability measurements show why:

* many adjacent probes have no cover churn;
* churn is concentrated around the wide initial bracket and Fast/Full transition;
* at least one very large local spike occurred.

So use **local structural reuse with a refresh guard**, not “select blocks once for the entire search.”

The simplest guard is exact final replanning: if the fully replanned candidate differs enough in size, refresh and correct.

A stronger later version can use the cover winner/runner-up margins. Nodes with a large decision margin can stay frozen; nodes near a crossover can be recomputed. That gives you selective structural refresh without returning to full-frame cover selection for every rung.

### E. Batch adjacent quantizer probes

Once `QuantizedProbe` and `RateSketch` exist, test quantizing four nearby rungs in one coefficient traversal.

For each coefficient, much of the input loading, transform lookup, context preparation, and magnitude analysis is shared. A four-rung batch can use either:

* SIMD lanes for adjacent quantizers;
* coefficient quantization breakpoints;
* incremental zero/nonzero and token-count updates.

This is more likely to produce a meaningful SIMD win than further vectorizing an isolated exact-choice loop.

---

## Phase 3: stop repeating entropy finalization

Even after reducing outer probes, Full entropy planning remains internally repetitive.

### A. Only perform expensive entropy alternatives on finalists

Do not try:

* natural versus reordered;
* default versus custom block contexts;
* single versus multi-preset;

for every Full rung.

During search, use either:

* the anchor’s entropy topology; or
* default entropy only.

Run the full alternative set for the best one or two exact finalists. If the chosen Full entropy plan shrinks the output enough to create excessive undershoot, perform one finer correction.

This is similar in principle to the current Fast/Full split, but with a genuinely cheap Fast stage and a finalist-sized Full stage.

### B. Introduce a token tape

`census_frame` and exact emission currently walk related frame data repeatedly. A typed token tape can be produced once per quantized candidate and consumed by both training and emission.

It might store, per independent section:

```text
symbol value
base context properties
raw-bit payload
block/order metadata
preset/group identity
```

Then:

* natural and reordered modes create or select the appropriate token order;
* custom block contexts remap base context properties;
* presets alter context offsets;
* histogram training consumes the tape;
* exact ANS Count and Store modes consume the same tape.

This keeps the “census and writer cannot disagree” property while avoiding repeated traversal of the quantized frame.

### C. Rank entropy alternatives before exact writing

Use trained histogram cost plus known model/header overhead to rank alternatives. Exact-price only the candidates capable of winning.

You do not need to trust the estimate for final output size. It is only an elimination and ranking mechanism. The selected entropy plan still passes through the exact writer.

### D. Reconsider Count-all-then-Store-one versus Store-every-finalist

The current Full loop stores every exact candidate so the winner is not re-encoded. That avoids one repeated winner encode but causes every losing candidate to allocate and populate a codestream.

Benchmark both policies after the exact-finalist count is reduced:

```text
Policy A: Store every finalist
Policy B: Count every finalist, then Store the winner once
```

With two candidates, A may remain better. With four or more, B may win. Instrument rather than assume.

---

# What not to do next

Do not spend the next optimization cycle on:

* another blanket SIMD pass;
* fine-grained Rayon inside quantizer loops;
* forward-cache eviction;
* returning to Full-only search;
* an analytic replacement for exact CfL;
* more per-cell prune logic;
* deleting quality tools to make a nominally fast default;
* a big-bang rewrite of the encoder crates.

The previous negative experiments already answer several of those questions:

* Full-only became 7%, 16%, and 105% slower at the tested rates.
* Cache reuse is extremely high, so eviction exchanges memory for large recomputation.
* The cheap cover prune costs more than it skips.
* CfL approximation attempts affected correctness or size.
* Local SIMD helped, but its ceiling is nowhere near the remaining total gap.

---

# How to structure agent-driven optimization

Agents will work better if each assignment names one multiplicity or ownership problem rather than saying “optimize the encoder.”

## Required workflow for every optimization branch

### 1. State one falsifiable hypothesis

Examples:

> Exact Count emissions are spending significant time serially encoding independent pass groups; request-scoped parallel Count pricing will reduce rate-loop wall time without altering sizes.

> Global hash-based candidate ownership is preventing LF-group planning parallelism and causing excessive allocation; a group-local dense bank will preserve coefficients while reducing fixed and rate-loop wall time.

### 2. Add the counter before changing the mechanism

For example, do not build a token tape until the agent has counted:

* census walks;
* quantized-frame traversals;
* internal exact price calls;
* section encodes.

### 3. Keep structural experiments feature-gated

Output-preserving refactors can land normally. Search-changing work should initially use a feature flag or explicit research policy so that:

* old and new paths coexist;
* the old path acts as an oracle;
* failures can fall back cleanly;
* negative results do not require a complicated revert.

### 4. Use the established interleaved A/B methodology

At minimum:

* approximately 0.8 MP, 4 MP, and 12 MP;
* photographic, low-detail, highly textured/noisy, and line-art/UI content;
* 0.5, 1, 2, and 4 bpp;
* one thread and a fixed multicore count;
* warm-process and stated cache regime;
* minimum, median, and dispersion;
* CPU time, wall time, and peak RSS.

Add scaling measurements at 1, 2, 4, 8, and the chosen host maximum when changing parallelism. A lower wall time accompanied by badly inflated CPU time may indicate oversubscription rather than good scaling.

### 5. Apply the right correctness contract

For output-preserving work:

* identical bytes;
* identical fingerprint;
* identical sizing;
* identity across thread counts;
* decoder/oracle suites unchanged.

For search-changing work:

* exact final output never exceeds the target;
* target undershoot remains within the accepted tolerance or is explicitly reported;
* matched-rate Butteraugli and SSIMULACRA2 gates;
* no significant corpus-class regression;
* fallback frequency and reason recorded;
* exact decoder/conformance contracts remain green.

Do not require identical fingerprints from a better rate search. That would prevent it from making legitimate decisions.

### 6. Measure end-to-end, not only the edited function

A dense cache that makes lookup 40% faster but increases target-rate time is a failure. A token estimator that is inaccurate but reduces exact candidates from 23 to 3 and retains quality is a success.

The promotion metric is total encode time at the required output quality.

---

# Benchmarking against libjxl

Do not use one undifferentiated “cjxl speed” number.

libjxl explicitly treats effort as a speed-versus-search/tool trade-off. Its documented VarDCT progression has only 8×8 blocks at e1, while variable blocks, adaptive quantization, Gaborish, CfL, and fuller variable-block heuristics arrive at higher efforts. Higher effort can also improve visual-quality consistency at a given file size. ([GitHub][1])

Use three benchmark lanes:

1. **Production-quality parity:** JPXL production mode against cjxl e6/e7 at matched bytes and perceptual quality.
2. **Fast-mode parity:** a deliberately bounded JPXL mode against cjxl e1/e3, with the tool differences disclosed.
3. **Absolute goal:** best JPXL production architecture versus the fastest cjxl mode that meets the same quality gate.

This does not weaken the goal of beating libjxl. It tells you whether a regression comes from inefficient implementation or from doing materially more analysis.

Pin:

* the exact cjxl commit/binary hash;
* compiler and architecture flags;
* thread count;
* input conversion path;
* whether CLI I/O is inside the measurement.

The official libjxl repository also provides `benchmark_xl`; using an in-memory or codec-oriented harness is preferable to comparing differently structured CLI runs. ([GitHub][2])

---

# Realistic performance outlook

The remaining gap is large, but the source contains a plausible route to closing it:

* **Probe collapse:** reducing 23–25 full-ish plans to one or two structural plans plus a few cheap sketches can plausibly remove a large majority of target-rate time.
* **Finalist-only entropy:** removing repeated internal exact prices can eliminate another substantial multiplier.
* **Planning parallelism:** group-level cover, coefficient preparation, quantization, and census can attack the fixed-plan cost.
* **Dense candidate storage:** better locality and dramatically fewer allocations should help both serial and parallel planning.
* **Persistent execution infrastructure:** avoids repeated pool construction and makes Count pricing parallel.
* **Later leaf optimization:** PGO, `target-cpu=native`, writer/ANS tuning, and additional SIMD become worthwhile after the repeated work has been removed.

These gains overlap, so multiplying optimistic percentages would be misleading. Nevertheless, this is the first route visible in the current code that could reasonably produce an **order-of-magnitude** improvement. Local loop tuning alone cannot.

After the structural work, establish a dedicated benchmark profile and test:

* PGO using the real image/rate corpus;
* native CPU feature selection;
* fat versus thin LTO;
* stripped debug data;
* allocator behavior;
* ANS and bit-writer hardware counters.

The current release profile’s debug information is not the cause of a 16–41× gap. Treat build-profile work as the final 5–20%, not the first 10×.

---

# Recommended immediate sequence

1. **Aggregate nested rate-search diagnostics**, including every census, entropy training, exact Count traversal, Store traversal, section encode, and pool construction.
2. **Introduce a persistent executor and parallel `price_codestream_with`**, preserving exact output.
3. **Replace the global candidate `HashMap` with LF-group-local dense coefficient banks**, then parallelize cover and quantization by group.
4. **Prototype the `SpatialAnchor` plus context-aware `RateSketch`**, with exact finalist replanning and the current loop as fallback.
5. **Confine Full entropy alternatives to finalists**, then introduce a shared token tape if entropy traversal remains significant.
6. Only after those changes, return to remaining SIMD, ANS, bit-writer, PGO, and compiler-level work.

The architectural change is therefore substantial but contained: **turn the target-rate controller from a loop that repeatedly encodes the image into a predictor that analyzes once, probes cheaply, and verifies exactly.**

[1]: https://github.com/libjxl/libjxl/blob/main/doc/encode_effort.md "libjxl/doc/encode_effort.md at main · libjxl/libjxl · GitHub"
[2]: https://github.com/libjxl/libjxl "GitHub - libjxl/libjxl: JPEG XL image format reference implementation · GitHub"
