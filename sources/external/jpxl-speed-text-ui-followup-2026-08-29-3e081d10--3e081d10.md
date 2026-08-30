# JPXL follow-up after the three-way encoder comparison

Status: advisory, non-authoritative. This memo recommends experiments; AKR remains the plan of record.

## Executive recommendation

Run two separate programs:

1. **General speed:** reduce the fixed cost of a successful, canonically verified quality encode. Do not build another surrogate navigator. First measure an accepted-one-probe floor against matched-quality `cjxl`, then remove duplicated source analysis, reconstruction, metric, allocation, and finalization work in descending measured order.
2. **Text/UI density:** use the reported reliable variance classifier to gate a second, independently encoded candidate: palette-oriented Modular over an edge-preserving, colour-quantized image. Canonically score both it and the existing VarDCT result, then emit the smallest candidate that meets the requested SSIMULACRA2 floor. Keep VarDCT as the fallback until the routed arm wins a locked real-world text/UI screen.

This order follows the evidence. JPXL is already competitive or better on photographic quality per byte, while the text/UI point is a class-specific failure large enough to justify a separate coding path. The speed gap, meanwhile, is dominated by work required by JPXL's verified-score contract, not by a single entropy or transform kernel.

## Evidence that governs the recommendation

The current five-image diagnostic is recorded at `@jpegxl-rs.observation.three-way-encoder-comparison-2026-08-29/2`:

- at the raw middle controls, JPXL Balanced q85 was 1.57x ZenJXL d1.5 and 2.23x `cjxl` d1.5 wall by geometric mean;
- on the 4.3 MP photograph, JPXL used essentially the same bytes as the d1.0 competitors and scored 2.09-2.32 SSIMULACRA2 points higher;
- on the 12 MP photograph, JPXL q85 used 23-28% fewer bytes than the first measured competitor points that reached at least its score;
- on the text screenshot, JPXL q85 used 7.21x ZenJXL d3 bytes and 9.23x `cjxl` d3 bytes while scoring lower.

The timing is diagnostic, not a promoted baseline: the encoders were batched, settings were not quality-equivalent, dispersion was not reported, and host load was non-trivial. Any promotion decision must follow `@jpegxl-rs.policy.performance-baseline-rules/1`.

The text result is not a one-image surprise. The earlier development split found photographs ahead of `cjxl` at matched score while synthetic text and line art trailed by as much as 3.4x; see `@jpegxl-rs.observation.pqc-pr4-development-split-2026-08-22/1`. The locked corpus already contains text-screenshot and line-art classes; see `@jpegxl-rs.evidence.pqc-pr0-corpus-manifest-2026-08-22/1`.

For speed, `@jpegxl-rs.observation.pqc-wall-quiet-host-attribution-2026-08-25/1` found that:

- three 12 MP pixel probes cost about 642-697 ms each;
- plan, entropy, emission, and exact pricing were smaller contributors;
- roughly 1.4 seconds sat outside the controller in source loading, metric-reference preparation, feature extraction, and output I/O;
- even a perfect one-probe navigator could not meet the old 2.0x quality/rate target without reducing fixed per-pixel work.

That record is stale after later optimizations, so its numbers are hypotheses to remeasure, not current truth. Its structural conclusion has survived three later surrogate experiments: 0.6x, 0.398x, and 0.316x surrogate probes all left engaged end-to-end wall near 1.0x. See `@jpegxl-rs.assessment.surrogate-navigation-does-not-pay-for-wall/1` and `@jpegxl-rs.evidence.s3-k1-economics-verdict/1`.

## Program A: close the general speed gap

### A0. Replace the diagnostic with matched-quality timing truth

Extend the three-way harness or reuse the fuller comparison harness so that it:

- interleaves encoders and settings;
- reports min, median, p90 or MAD, not only best-of-N;
- hashes binaries before each timed block;
- records per-run host load and rejects promotion runs under pressure;
- compares at matched achieved `ssimulacra2-jpxl-1`, using measured points or bounded interpolation;
- reports the five existing size/content classes separately rather than hiding the text outlier in a pooled mean.

Add four JPXL timing modes on the 4.3 and 12 MP photographs:

1. fixed quantizer at the finally selected rung;
2. target-rate encoding near the same bytes;
3. a forced single canonical quality probe at the known successful rung;
4. the normal Fast and Balanced quality controllers.

Record source decode/load, source analysis, reference-metric preparation, plan, quantize, render, metric, entropy, emission, and output I/O separately.

**Gate A0:** do not claim a `cjxl` speed ratio until at least five interleaved runs per cell pass the performance-baseline policy. Continue to A1 even if the ratio changes; the phase attribution is the important output.

### A1. Establish the accepted-one-probe floor

The hard quality contract requires canonical verification before successful emission; see `@jpegxl-rs.decision.perceptual-quality-contract/1`. Measure the fastest possible path that still obeys it:

1. use the known successful rung from the completed run;
2. build one production plan;
3. reconstruct once;
4. compute the canonical metric once;
5. attach entropy and emit once.

This isolates fixed work from navigation. It also prevents another predictor or crossing-rule change from being credited for work it cannot remove.

**Gate A1:**

- if the one-probe verified path is more than 1.25x the matched-quality `cjxl` wall, prioritize fixed-cost work exclusively;
- if it is at or below 1.25x, profile normal q70/q85/q90 probe counts and improve one-shot coverage under `@jpegxl-rs.work.pqc-one-shot-controller/3`;
- do not start another surrogate branch unless a precomputed cost model predicts at least a 10% whole-encode win and its measured candidate cost is below 0.20x canonical.

### A2. Remove duplicated source-derived work

Audit ownership and conversions before changing kernels. Source features, VarDCT preparation, the renderer, and SSIMULACRA2 reference setup may derive overlapping luma, linear-light, XYB-like, pyramid, or quantized-depth data. The desired boundary is one immutable source analysis object whose neutral products are borrowed by the policy and metric layers without coupling encoder and decoder implementations.

Screen these changes independently:

- share already-identical source colour/depth conversions;
- build metric reference state once per request and retain it across all probes;
- retain capacity in request-scoped scratch instead of releasing and reallocating frame-sized buffers between phases;
- keep CLI PPM I/O separate from an in-memory API benchmark so codec and process overhead are both visible;
- verify that the accepted candidate's plan, reconstruction, coefficient storage, token census, and emitted bytes are reused rather than regenerated.

Do not force two representations to share merely because they look similar. Require bit-identical scores and codestreams for any supposedly neutral reuse.

**Gate A2:** each slice must save at least 3% end-to-end wall on one anchor without losing more than 1% on the other. Batch smaller wins only when one profile shows they remove the same pass or allocation family.

### A3. Optimize the canonical accepted evaluation, not an approximation

The exact banded metric and fused render already removed substantial memory and wall. Re-profile current code before choosing among:

- better scale/channel scheduling in the exact streamed metric;
- fusing adjacent exact colour, downsample, or moment passes where rounding order can be preserved;
- persistent aligned scratch and fewer large zero-fill/release operations;
- eliminating repeated edge extension or restoration setup across scale bands;
- improving four-thread load balance on the 12 MP accepted candidate.

Every change in this phase must pass the existing score-bit and production-stream identity matrices in `@jpegxl-rs.work.pqc-usable-efforts-cost/9`. SIMD is a last step after a current profile names a scalar hot loop.

**Gate A3:** target at least 15% reduction in accepted-one-probe wall before changing navigation again. A kernel that improves its own timer but moves end-to-end wall by less than 2% is closed as a negative result.

### A4. Only then reduce remaining navigation

Once fixed work is lower, measure how many q70/q85/q90 cells still take more than one canonical probe. Improve the existing one-shot model only on those cells. Preserve these rules:

- canonical score is the only feasibility proof;
- a predicted rung may propose but never authorize emission;
- corrected/fallback navigation reuses the first plan, score, and features;
- the total probe cap remains explicit;
- photo gains are not purchased with a text/UI regression, because routed text/UI still needs a safe fallback.

**Promotion gate:** on the locked matched-quality photo/scene corpus, zero floor violations, no material byte regression, and geometric-mean wall no worse than 1.10x `cjxl` with no anchor above 1.25x. Treat 1.25x as the first milestone, not the final definition of parity.

## Program B: classifier-routed text/UI encoding

### B0. Verify what kind of win is available without reading competitor source

Use ZenJXL and libjxl only as black boxes under `@jpegxl-rs.policy.clean-room-boundary/1`. For the existing text and line-art corpus:

- sweep the same distance range used by the three-way harness;
- inspect emitted syntax using `jxlinfo`, JPXL's own parser/tracing, file sizes, and decoded pixels;
- record whether winning streams are Modular or VarDCT and which normative features are present;
- do not infer algorithms, constants, or architecture from implementation source.

This is an experiment about reachable JPEG XL representations, not a plan to copy another encoder.

**Gate B0:** if the winning black-box streams are not predominantly paletteable/Modular, still run B1 because JPXL's own Modular candidate is cheap to falsify, but do not assume it explains the competitor result.

### B1. Integrate the variance classifier as a routing signal

Introduce a typed content hint at the policy boundary, for example:

```text
PhotoLike | TextUiLineArt | Unknown
```

The classifier may be implemented in-tree or supplied through a codec-neutral input seam. If supplied externally, align the boundary with `@jpegxl-rs.work.optional-semantic-guidance-consumer/1`: version it, validate dimensions, account for producer cost separately, and make absence preserve current bytes.

Initially the hint must only enable an extra candidate. It must not suppress the existing VarDCT path. False positives therefore cost bounded wall time but cannot force a worse stream.

Shadow-log:

- class and confidence;
- classifier wall time;
- exact unique-colour count or capped census;
- edge/flat fractions already available to source analysis;
- whether an exact palette is possible;
- current VarDCT bytes and achieved score.

**Gate B1:** classifier plus census costs at most 2% of normal photo wall, false-positive photo output remains byte-identical, and every existing text/line-art family routes as intended. If the user's detector is outside the codec, report its wall separately as required by the optional-guidance work item.

### B2. Ship the safest candidate first: exact Modular competition

JPXL already has exact palette, squeeze, MA-tree, and tiered final pricing machinery. Full Modular search is wasteful on photographs but specifically retained for paletteable UI/line-art; see `@jpegxl-rs.assessment.modular-effort-lean-default/1`, `@jpegxl-rs.decision.modular-lean-default/1`, and `@jpegxl-rs.evidence.opt-m-tiered-planner/1`.

On classifier-positive inputs only:

1. encode the current VarDCT quality candidate;
2. encode an exact Modular candidate with the UI-relevant palette/squeeze search enabled;
3. score both decoded results canonically;
4. emit the smaller candidate that meets the requested score.

The Modular candidate is lossless and therefore automatically meets any q<100 floor. It may already win on flat screenshots and gives a no-quality-risk baseline before any lossy colour reduction exists.

**Gate B2:** promote the exact candidate competition if it wins at least 20% of locked text/UI images, never grows an emitted stream, and adds no more than 5% wall on classifier-positive losers. Otherwise keep the measurement and proceed directly to B3.

### B3. Add an edge-preserving near-palette ladder

The existing forward palette is exact, limited to 256 colours, and emits no delta palette. Anti-aliased text and UI gradients can exceed that exact-colour cap while remaining highly paletteable after careful reduction.

The shortest clean-room route is not new lossy wire syntax. It is:

1. derive a small deterministic ladder of colour-reduced RGB images from the original;
2. avoid dithering, which creates entropy around edges;
3. encode each reduced image losslessly with JPXL's existing Modular palette path;
4. decode and score against the original with canonical SSIMULACRA2;
5. keep the smallest candidate that meets the requested floor;
6. compete it against the unchanged VarDCT result.

Start with no more than three palette budgets chosen from corpus evidence. Preserve exact colours for dominant flat regions and allocate remaining entries to anti-aliased edge ramps. Use the variance detector only for routing; let measured score and bytes choose the winner.

This design has a strong safety property: the JPEG XL stream is a normal lossless Modular encoding of the reduced raster, while the overall API correctly reports a lossy result measured against the original. It also avoids coupling a new quantizer to JPEG XL's palette syntax before the value is proven.

**Gate B3:**

- zero SSIMULACRA2 floor violations;
- no output larger than the existing VarDCT candidate;
- on the locked text/UI subset, first milestone <=2.0x the smaller of ZenJXL/cjxl bytes at matched achieved score, then <=1.25x;
- text/UI wall <=1.25x ZenJXL at matched score;
- classifier-positive losers add <=10% wall;
- classifier-negative photo output is byte-identical and wall-neutral within 2%.

### B4. Optimize only the winning Modular search

If B3 wins density but misses wall:

- replace the current linear first-seen palette lookup with a deterministic bounded lookup structure;
- run a capped colour census before allocating full palette state;
- price palette plus a small predictor shortlist instead of invoking the entire effort-7 Modular search;
- use the existing cheap Shannon ranking and exact-price only finalists;
- reuse the reduced raster and palette index plane across candidate prices;
- stop palette-cardinality search when adjacent candidates cannot beat the current byte winner.

Do not globally raise Modular effort. The existing decision that lean Modular is correct for ordinary content remains valid; this is a classifier-gated exception.

### B5. Consider native delta/near-palette syntax only after B3/B4 plateau

If reduced-raster Modular remains materially behind while black-box syntax experiments show a representation gap, return to the standard and scope native delta-palette or other normative Modular features. This is a later codec feature with its own roundtrip, storage-order, multigroup, malformed-input, and oracle-parity tests. It should not block the simpler reduced-raster experiment.

## Corpus and acceptance design

The current corpus is enough for a first falsification, not for promotion of a text/UI default. Expand it by source family, keeping calibration/development/holdout separation:

- light/dark desktop UI;
- terminals and source editors;
- browser pages with text, icons, and photographs;
- diagrams, plots, maps, and presentation slides;
- anti-aliased and non-anti-aliased text;
- grayscale and coloured line art;
- mixed screenshots containing a photographic region.

Report SSIMULACRA2 as the promotion metric per `@jpegxl-rs.decision.ssimulacra2-is-the-primary-promotion-metric/1`. Continue reporting Butteraugli as a caveat. For diagnosis, also report exact bytes, colour count, maximum pixel error, and an edge-masked error summary, but do not silently replace the public quality contract with a text-specific metric.

Every routed test must include:

- classifier result and cost;
- both candidate sizes and scores;
- selected candidate and reason;
- `djxl` decode success;
- deterministic output across worker counts;
- at least one multi-group fixture;
- false-positive photo controls.

## Recommended order and stop conditions

1. **A0 + B0:** obtain matched-quality timing truth and black-box syntax facts.
2. **B1:** land classifier shadow routing with no output change.
3. **B2:** test exact Modular competition. Stop it if it almost never wins.
4. **B3:** test a three-point near-palette ladder. This is the highest expected quality/size payoff.
5. **B4:** optimize the routed Modular winner only if density passes and wall fails.
6. **A1-A3:** reduce the verified accepted-candidate floor on photos.
7. **A4:** revisit navigation only if multi-probe cells remain material after fixed-cost work.
8. **B5:** add native lossy Modular features only if the simpler approach plateaus above the competitor gate.

The largest likely mistake would be to blend these programs: weakening the verified quality contract to gain wall time, or retuning the successful photo VarDCT policy to repair a text/UI representation failure. Keep the routes independent, score both candidates exactly, and let the smaller valid stream win.

## AKR cross-reference

- Current result: `@jpegxl-rs.observation.three-way-encoder-comparison-2026-08-29/2`
- Benchmark work: `@jpegxl-rs.work.three-way-encoder-benchmark/2`
- Performance methodology: `@jpegxl-rs.policy.performance-baseline-rules/1`
- Clean-room boundary: `@jpegxl-rs.policy.clean-room-boundary/1`
- Public quality floor: `@jpegxl-rs.decision.perceptual-quality-contract/1`
- Primary metric: `@jpegxl-rs.decision.ssimulacra2-is-the-primary-promotion-metric/1`
- Current wall program: `@jpegxl-rs.work.pqc-usable-efforts-cost/9`
- Fixed-cost attribution: `@jpegxl-rs.observation.pqc-wall-quiet-host-attribution-2026-08-25/1`
- One-shot controller: `@jpegxl-rs.work.pqc-one-shot-controller/3`
- Surrogate verdict: `@jpegxl-rs.assessment.surrogate-navigation-does-not-pay-for-wall/1`
- Coefficient-surrogate kill evidence: `@jpegxl-rs.evidence.s3-k1-economics-verdict/1`
- Prior text/UI class gap: `@jpegxl-rs.observation.pqc-pr4-development-split-2026-08-22/1`
- Quality corpus: `@jpegxl-rs.evidence.pqc-pr0-corpus-manifest-2026-08-22/1`
- Modular selective-search rationale: `@jpegxl-rs.assessment.modular-effort-lean-default/1`
- Modular default decision: `@jpegxl-rs.decision.modular-lean-default/1`
- Tiered Modular pricing: `@jpegxl-rs.evidence.opt-m-tiered-planner/1`
- Optional external guidance seam: `@jpegxl-rs.work.optional-semantic-guidance-consumer/1`
