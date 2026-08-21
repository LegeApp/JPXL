# JPXL: Guide to Bridging the Remaining Gap to libjxl

**Repository snapshot:** `jpegXL-rs-agent-pack-2026-08-21_133315`  
**Scope:** JPXL VarDCT target-rate encoding, production latency, entropy/writer cost, and high-effort Modular lossless compression  
**Constraint:** clean-room implementation. Use the JPEG XL standard, JPXL's own measurements, published perceptual literature, and `cjxl`/`djxl` only as black-box interoperability and comparison tools. Do not copy libjxl source structure, heuristics, constants, tables, or tuning data.

---

## Executive diagnosis

JPXL no longer has one simple “performance gap.” The current repository has several different gaps, and treating them as one number will send optimization work in the wrong direction.

1. **The production `Balanced` path is already very fast in the narrow benchmark window that established Phase 42.** On the two pinned photo anchors and four P-cores, the historical Phase 42 measurements put `Balanced` ahead of `cjxl -e 7` in wall time and CPU use at approximately matched SSIMULACRA2.
2. **That Phase 42 density result is now stale.** It predates Q0b through Q9, including a 9.4% fixed-decision LF/control-image reduction, a dense upper rate ladder, cover-rate recalibration, chroma policy changes, and controller changes. Do not continue quoting the historical “about 13% larger at equal SSIMULACRA2” result as the current gap until the new encoder is remeasured on full curves.
3. **The best demonstrated current quality gap is localized, not global.** At matched bytes in the Q4 matrix, JPXL led SSIMULACRA2 in all nine photo/rate cells, but trailed PSNR in eight and Butteraugli max and 3-norm in seven. The largest failures occur in low-to-mid activity DCT8 regions containing a strong edge beside flat content. This is a tail-risk and coefficient-allocation problem, not evidence that the entire transform path is weak.
4. **The production controller is not fully bounded.** `Fast` and `Balanced` start with bounded anchor work, but can silently fall into the exhaustive `Quality` controller. That preserves quality but creates a misleading latency contract and severe p99 outliers.
5. **Lossless Modular is a separate problem.** JPXL's low-effort/default lane is competitive, but the historical gap to `cjxl -e 7` is much larger and comes from model-search strength rather than the same VarDCT hot paths.

The shortest credible route to parity is therefore:

> **repair the benchmark contract → measure native tail risk → add finalist-only run-aware quantization → replace hidden exhaustive fallback with one bounded fresh rescue → selectively improve cover and entropy pricing → optimize scaling only where a fresh profile proves it remains material.**

Do **not** begin another broad SIMD pass. SIMD, forward-transform caching, request-scoped execution, candidate banks, parallel writer work, PGO/LTO, scratch reuse, and a single token-tape traversal are already present. The remaining useful work is more selective.

---

# 1. Define what “parity” means

A codec cannot be declared faster or better from one fixed setting. `cjxl` distance and JPXL target bpp are not equivalent controls, and the two encoders optimize different perceptual tradeoffs. Maintain six separate parity lanes.

| Lane | Hold constant | Compare | What it answers |
|---|---|---|---|
| A. Equal bytes | Encoded size, within a tight interpolation band | SSIMULACRA2, Butteraugli 3-norm, Butteraugli max, PSNR | Which encoder spends the same budget better? |
| B. Equal SSIMULACRA2 | Interpolated SSIMULACRA2 | Bytes, wall time, CPU time | Density and speed at JPXL's strongest current metric |
| C. Equal Butteraugli 3-norm | Interpolated 3-norm | Bytes, wall time, SSIMULACRA2 | Whether the localized perceptual tail has been closed |
| D. Production target | Requested bpp and preset | Rate miss, p50/p95/p99 latency, work counts | Whether `Balanced` has a truthful production contract |
| E. Scaling | Same input, output target, and binary | 1/2/4/8-thread wall and CPU time | Whether more parallel work is still worth doing |
| F. Lossless | Identical source pixels and effort class | Bytes, wall, CPU, memory | Modular model-search parity |

A release claim should identify its lane. For example:

- “JPXL is 12% faster” is incomplete.
- “JPXL `Balanced` is 12% faster at equal SSIMULACRA2 on the 12 MP photo corpus, four pinned P-cores, with a 95% confidence interval of X–Y” is useful.
- “JPXL is smaller” is incomplete.
- “JPXL is 4% smaller at equal Butteraugli 3-norm, while retaining its SSIMULACRA2 lead” is useful.

## Proposed parity gates

These are initial engineering gates, not permanent marketing thresholds.

### VarDCT production gate

- Target-rate output remains at or below the requested hard cap.
- `Balanced` has a documented finite work budget and never opens an unbounded exhaustive search.
- Common-path p50 wall time does not regress by more than 2% without an offsetting density or perceptual improvement.
- p99 is reported across the corpus rather than inferred from two anchors.
- One-, four-, and forced-scalar outputs remain deterministic under the existing project contract.

### Quality/density gate

- Preserve JPXL's matched-byte SSIMULACRA2 advantage on the broad corpus.
- Reduce the aggregate Butteraugli 3-norm gap and the number/severity of edge-flat outliers.
- Do not accept an improvement that merely moves error from Butteraugli 3-norm into a large SSIMULACRA2 loss.
- Keep max-norm as a reported diagnostic and 3-norm as the stable promotion gate, consistent with the repository's current Contract B practice.

### Lossless gate

- Compare low, medium, and high effort separately.
- Require exact pixel reconstruction and both in-tree and external decoder acceptance.
- Every extra search tool must justify itself by exact final bytes, not a proxy alone.

---

# 2. Freeze the current facts before changing code

The guide assumes the following repository state.

## 2.1 VarDCT facts already established

- Phase 42's narrow, equal-resource timing window put `Balanced` at roughly 0.42–0.43 s on 2400×1800 and 0.89–1.13 s on 4000×3000, versus `cjxl -e 7` at roughly 0.46–0.51 s and 1.24–1.66 s. This is historical evidence that raw wall time is no longer the first problem.
- The same Phase 42 comparison found about a 13% byte disadvantage at approximately equal SSIMULACRA2 and a Butteraugli disadvantage. That figure predates the quality track and must be refreshed.
- Q0b entropy-coded the LF/control images, reducing fixed-decision bytes from 738,930 to 669,109 on the mid photo and improving the 27-cell ladder substantially, at about a 9% instruction cost.
- Q4's matched-byte matrix showed:
  - SSIMULACRA2 ahead in 9/9 cells by approximately 0.4 to 2.6 points.
  - PSNR behind in 8/9 cells by approximately 0.03 to 0.54 dB.
  - Butteraugli max behind in 7/9 cells, with the worst relative gap on the busy mid photo at 2 bpp.
  - Butteraugli 3-norm behind in 7/9 cells, up to roughly 14% in that matrix.
- Q3 localized the worst error to low-to-mid activity blocks, especially DCT8 blocks mixing a strong edge and a flat side. JPXL left approximately ±8–14 luma error on the flat side where the comparison stream was around ±2.
- A DCT8-only cover did not remove the hot spots. Transform size alone is therefore not the root cause.
- Broad activity AQ, fine-lattice AQ, adaptive EPF, global measured size penalties, one-cluster static entropy, blanket larger LZ search, global frozen cover/CfL, and an exact second-pass dirty frontier have already been tried or measured negatively.

## 2.2 Current production architecture

Relevant code points in this snapshot:

- Presets: `crates/jpxl-encode-policy/src/request.rs`, `RateSearchPreset` near line 137.
- Anchor controller and fallback: `crates/jpxl-encode-policy/src/rate.rs`, `search_frame_with_executor` near line 1396 and `MAX_ANCHOR_CORRECTIONS` near line 1599.
- Probe telemetry: `RateProbeStats` in `rate.rs` near line 375.
- Request-scoped analysis: `crates/jpxl-encode-policy/src/analysis.rs`; `AtomFeatures` currently contains only XYB mean and variance.
- Structural reuse: `StructuralAnchor` and `AnchorReuse` in `crates/jpxl-encode-policy/src/lib.rs` near lines 1432 and 1445.
- Raw forward-transform reuse: `CandidateForwardCache` near line 2395.
- Quantization scratch reuse: `QuantizationWorkspace` near line 3596.
- Cover objective: `block_cost_bounded` near line 4444.
- HF quantizer: `crates/jpxl-encode-policy/src/quantize.rs`, `HfQuantizer` near line 217 and trailing truncation near line 462.
- Coefficient event order and contexts: `crates/jpxl-encode/src/vardct/walk.rs`.
- Entropy training and clustering: `crates/jpxl-encode-policy/src/entropy.rs`.
- Token tape: `crates/jpxl-entropy/src/encode/tape.rs`, `TokenTape` near line 36.
- Lossless selection: `crates/jpxl-encode/src/lossless.rs`.

Keep the raw forward cache and sequential quantization arenas. They are valid request-scoped reuse. The architectural work below should be layered around them, not replace them.

---

# 3. Workstream 0: make the benchmark incapable of lying

This is the first change because every subsequent promotion depends on it.

## 3.1 Correct `tools/compare-libjxl.ps1`

The current script is a useful start but has four material problems.

1. Its description says timing alternates codec order, but the loop at line 78 always runs JPXL and then `cjxl`. Thermal drift and background load can therefore bias one side.
2. The TSV writes `$Threads` for both codecs at line 81, even when `$CjxlThreads` differs.
3. The TSV writes `$JpxlPreset` into the `cjxl` row as well.
4. It accepts manually paired JPXL bpp and `cjxl` distance arrays. Those points do not establish equal bytes or equal quality unless a separate matching process has already done so.

Fix these before collecting another headline number.

### Required timing schedule

Use balanced blocks rather than simple repetition:

```text
warm JPXL
warm cjxl
AB
BA
BA
AB
... randomized or counterbalanced with a fixed recorded seed
```

For every run record:

- actual execution order;
- start timestamp;
- affinity and thread count for each process;
- wall time;
- process CPU time;
- peak working set/RSS;
- output hash and bytes;
- whether the host-state guard accepted the run.

Do not silently discard noisy runs. Mark them invalid with a reason and retain the raw row.

## 3.2 Separate curve construction from timing

Distance bisection and perceptual scoring should happen outside the timed process window.

For each image and encoder:

1. Build a sufficiently dense rate-distortion curve.
2. Decode every point through the same declared decode/color path.
3. Compute PSNR, SSIMULACRA2, Butteraugli max, and Butteraugli 3-norm.
4. Interpolate to find:
   - `cjxl` distance matching each JPXL byte target;
   - each encoder's bytes at the same SSIMULACRA2;
   - each encoder's bytes at the same Butteraugli 3-norm.
5. Once the settings are frozen, run timing only.

Do not time the bisection, decoding, metric calculation, or result parsing as encoder work.

### Curve rules

- Refuse interpolation across a visibly non-monotone segment without adding samples.
- Report the bracketing points and interpolation fraction.
- Set a maximum interpolation span; add a point when the span is too wide.
- Preserve both raw points even when one is later excluded.
- Rebuild curves after any change that alters the stream, because the old matched settings are no longer valid.

## 3.3 Expand the corpus by failure mode

Two anchors are good profiler inputs, not a parity corpus. Build a manifest with strata rather than a random image pile.

Minimum useful strata:

- smooth natural photographs;
- high-detail foliage, grass, hair, and fabric;
- strong edge beside flat sky/wall/skin;
- low-light/noisy images;
- saturated chroma and colored lights;
- portraits and skin gradients;
- architecture and repeated edges;
- synthetic graphics, text, and screenshots;
- small, medium, and large pixel counts;
- 8-bit and any higher-bit-depth path JPXL claims to support.

Keep the seven synthetic scenes, but add synthetic edge-flat fixtures designed to vary:

- edge orientation;
- edge contrast;
- flat-side width;
- low-amplitude texture near the edge;
- chroma-only edges;
- noise level;
- DCT-grid phase.

These fixtures are not substitutes for photographs. They are unit tests for the diagnosed failure shape.

## 3.4 Emit one machine-readable record

Replace loosely coupled TSVs with a versioned record, while retaining a flat export for analysis.

```json
{
  "schema": "jpxl.codec-comparison/2",
  "binary": {
    "codec": "jpxl",
    "sha256": "...",
    "git_revision": "...",
    "features": ["perceptual", "avx2"]
  },
  "input": {
    "id": "mid-photo",
    "sha256": "...",
    "width": 2400,
    "height": 1800,
    "bit_depth": 8,
    "strata": ["photo", "busy-texture", "edge-flat"]
  },
  "setting": {
    "preset": "balanced",
    "target_bpp": 1.0,
    "threads": 4
  },
  "rate_outcome": {
    "requested_bytes": 540000,
    "actual_bytes": 539958,
    "status": "inside_band",
    "exact_prices": 3,
    "anchor_fallbacks": 0
  },
  "timing": {
    "order": "AB",
    "wall_ms": 401.977,
    "cpu_ms": 790.0,
    "peak_rss_bytes": 0
  },
  "metrics": {
    "ssimulacra2": 77.92,
    "butteraugli_max": 2.744,
    "butteraugli_pnorm3": 0.0,
    "psnr_db": 0.0
  }
}
```

Populate all existing `RateProbeStats` fields. The internal work counts are necessary to distinguish “same latency because the host was quiet” from “same latency despite twice the encoder work.”

## 3.5 Baseline outputs

Every benchmark revision should generate:

- per-image curves;
- corpus aggregate curves with confidence intervals;
- equal-byte table;
- equal-SSIMULACRA2 table;
- equal-Butteraugli-3-norm table;
- p50/p95/p99 production latency;
- 1/2/4/8-thread scaling;
- rate-controller work-count histogram;
- peak-memory distribution;
- list of the worst perceptual tiles and their coordinates.

**Promotion gate:** no performance or quality change lands on evidence from the old two-image comparison alone.

---

# 4. Workstream 1: make `Balanced` bounded and truthful

The existing preset semantics are internally inconsistent:

- `Balanced` is documented and used as the production path.
- Its normal anchored path is bounded: two anchors, an exact finalist, and at most one correction.
- When that misses, `search_frame_with_executor` can invoke the exhaustive path, aggregate the work, and report an anchor fallback.
- Q5 showed why the fallback exists: simply removing it can cost roughly four SSIMULACRA2 points on a hard scene.

The correct fix is not to disable fallback. It is to replace an unbounded fallback with one **bounded fresh-structure rescue**.

## 4.1 Preserve three explicit products

### `Fast`

- Lowest bounded work.
- Fixed or cheap structural policy where already defined.
- No exhaustive fallback.
- Returns the best legal stream and an explicit rate-status result when it cannot enter the target band.

### `Balanced`

- Production default.
- Two anchors.
- Exact finalist.
- At most one ordinary correction.
- At most one fresh-structure rescue sequence under a fixed total exact-price cap.
- Never silently enters `Quality`.

### `Quality`

- Exhaustive/reference path.
- Allowed to spend substantially more work.
- Called only when explicitly requested by the caller or by an application policy outside the encoder core.

An application can choose “retry with Quality” after seeing a `Balanced` status, but the production preset itself should not hide that decision.

## 4.2 Add a result status

Extend `RateOutcome` with a stable status rather than requiring callers to infer behavior from traces.

```rust
pub enum RateStatus {
    InsideBand,
    UnderTargetAdjacentRungs,
    UnderTargetWorkCap,
    SaturatedTop,
    RescuedFreshStructure,
    ExhaustiveReference,
}
```

Also expose:

- requested bytes;
- tolerance bytes;
- selected bytes;
- closest over-target candidate, if any;
- exact price count;
- fast price count;
- structural build count;
- whether cover and CfL were fresh or reused;
- rescue trigger bits;
- predicted slope and residual error.

## 4.3 Bounded rescue state machine

A suitable `Balanced` state machine is:

```text
A0: exact first anchor
A1: exact second anchor
F0: predict and exact-price finalist
C0: optional one correction if outside band
GATE:
    return if inside band or ordinary adjacent-rung limit explains miss
    otherwise decide whether one fresh rescue is justified
R0: rebuild cover + CfL once at a slope-predicted rescue rung
R1: optional one local correction from R0, only if total work cap permits
RETURN best legal candidate with explicit status
```

A practical hard cap is six exact prices: two anchors, finalist, ordinary correction, fresh rescue, rescue correction. `Fast` should remain lower. The exact number is less important than making it fixed, tested, and visible.

## 4.4 Rescue triggers

Do not trigger rescue from one weak heuristic. Use a small bitset assembled from evidence JPXL already owns.

Trigger candidates:

- finalist remains outside the target band after correction;
- anchor slope residual is unusually large;
- cover winner/runner-up margins predict structural instability;
- CfL residual confidence is low;
- the forthcoming edge-flat risk atlas reports substantial tail-risk mass;
- rate lies above the dense-ladder ceiling where prior fallback behavior clusters;
- the two exact anchors disagree strongly with the navigation estimate.

The trigger must be calibrated from JPXL's own corpus. It must not contain constants inferred from libjxl internals.

## 4.5 What “fresh rescue” means

The rescue should rebuild structure once at the best predicted rung. It must not globally freeze cover and CfL from an anchor; the repository already measured that global reuse as low-value and capable of a large SSIMULACRA2 loss.

Reuse only what is quantizer-independent:

- `AnalysisAtlas`/future `AnalysisAtlasV2`;
- raw forward coefficients in `CandidateForwardCache`;
- immutable candidate banks;
- request-scoped transformed/preconditioned frame;
- allocation arenas.

Rebuild what is potentially quantizer-dependent:

- low-margin cover choices;
- CfL where confidence is low;
- final quantized coefficients;
- exact entropy decisions and final price.

## 4.6 Acceptance tests

- A unit test enumerates every state transition and proves the exact-price cap.
- No `Balanced` trace contains an exhaustive-controller phase.
- Existing hard scenes remain within the current quality contract or return an explicit miss status.
- p99 latency falls materially on the fallback subset.
- Common-path streams remain byte-identical unless a separately reviewed quality policy changes them.
- `Quality` preserves its role as the exhaustive oracle.

This work improves production predictability even if it produces no mean-speed win. That is still a real performance improvement.

---

# 5. Workstream 2: expand analysis around the actual failure

`AnalysisAtlas` currently stores only per-atom XYB means and variances. That is too weak to distinguish “busy texture that masks error” from “one strong edge with a perceptually exposed flat side,” which is precisely the current Butteraugli failure.

Create `AnalysisAtlasV2` as a compact, request-scoped, quantizer-independent feature atlas.

## 5.1 Features to compute

At the native 8×8 atom scale, compute:

1. **Horizontal and vertical gradient energy.**
2. **Structure tensor terms** (`gx²`, `gy²`, `gx·gy`) and orientation coherence.
3. **Laplacian or high-pass energy** to distinguish a clean edge from texture.
4. **Plane-fit residual** or robust local smoothness on each side of the dominant edge.
5. **Robust noise estimate**, such as a median absolute high-pass residual.
6. **Flat-side asymmetry:** one half-plane is smooth while the other contains the edge/texture.
7. **XYB covariance terms** and a chroma-residual confidence estimate.
8. **Local dynamic range and clipping proximity.**
9. **Optional DCT-grid phase descriptors** for the synthetic edge fixtures.

Aggregate these to 16×16 and 32×32 candidates using sums/min/max where mathematically valid. Avoid rescanning source pixels inside every cover candidate.

## 5.2 Keep analysis cheap

- Compute features in one source-frame traversal or piggyback on an existing request-scoped traversal.
- Store structure-of-arrays, not an object per block.
- Begin with `f32` for correctness and profiling; quantize storage only after distributions are known.
- Expose `byte_size()` and per-feature timing.
- Do not add a neural model. The diagnosed failure is simple enough for explicit local signals, and a model would make clean-room attribution and deterministic behavior harder.

## 5.3 Define a native tail-risk score

The risk score should predict JPXL's own reconstruction failure, not imitate libjxl.

A first diagnostic label can be produced from source versus JPXL reconstruction:

```text
edge_flat_leak =
    robust_max_error_on_flat_side
    + alpha * ringing_energy_across_edge
    + beta  * low_frequency_bias_on_flat_side
```

Use this label only in development tooling. Fit a simple monotone or linear ranker from source-side features to the label, then freeze explicit coefficients only after cross-validation on held-out JPXL images.

The production score should answer:

- Is this block likely to create exposed flat-side error?
- Is the current cover decision low-margin?
- Is this block worth spending finalist-only search work on?

It should **not** directly assign a broad adaptive quantization field. Broad AQ has already failed because signaling and coarse allocation costs overwhelm its benefit.

## 5.4 First PR is diagnostics only

Before changing encoding decisions, emit:

- risk score per atom;
- selected transform;
- quantization level;
- trailing truncation count;
- local reconstruction error summaries;
- Butteraugli diffmap tile rank in the research harness;
- cover margin;
- CfL residual confidence.

Then answer:

- What fraction of the worst Butteraugli tiles fall in the top 1%, 5%, and 10% of the native risk score?
- How much image area would a selective repair pass visit at useful recall?
- Does the risk score remain predictive across rates and image classes?

**Promotion gate:** a risk mechanism is not allowed into production unless it substantially concentrates known failures into a small area. If it needs to touch half the image, it is not selective enough.

---

# 6. Workstream 3: finalist-only run-aware quantization

This is the highest-upside quality/density workstream.

JPXL's current quantizer is efficient, SIMD-friendly, and broadly well tuned. Its weak point is that final coefficient decisions do not see the actual entropy consequences of a run and cannot explicitly protect the edge-flat tail. `truncate_trailing` can remove tail coefficients using an estimated own bit length and a constant interior-zero price, but I.4 coding cost depends on more than that:

- one `non_zeros` symbol per channel;
- neighboring nonzero prediction;
- coefficient position;
- remaining nonzeros;
- whether the preceding coefficient was nonzero;
- hybrid-uint token and extra bits;
- early termination after the last nonzero.

Changing one coefficient can therefore alter the contexts and cost of later coefficients. A per-coefficient independent lambda test is structurally incomplete.

## 6.1 Do not replace the baseline quantizer

Keep the existing SIMD nearest-quantization path as the baseline for all blocks. Add a **finalist-only selective refinement** after the normal quantized result exists.

This preserves the common path and constrains complexity.

## 6.2 Stage A: build an `EntropyCostView`

Expose a compact read-only view of the finalist or first-anchor entropy model:

```rust
pub struct EntropyCostView {
    // Fixed-point -log2 costs or another deterministic monotone unit.
    token_cost: Box<[u16]>,
    cluster_offsets: Box<[u32]>,
    nonzero_symbol_cost: Box<[u16]>,
    hybrid_extra_cost: Box<[u8]>,
}
```

Requirements:

- deterministic fixed-point cost;
- no entropy crate dependency leaking into the core quantizer API;
- exact table layout identified by the current plan/model revision;
- ability to price a coefficient walk under a declared fixed block context;
- cheap enough to use on a small selected block set.

The first implementation can use the first anchor's trained model to rank refinements. The exact writer remains the acceptance oracle.

## 6.3 Stage B: replace constant trailing cost as an experiment, not a presumed win

Implement context-aware trailing truncation that includes:

- change in the leading `non_zeros` symbol;
- token and extra-bit cost of retained coefficients;
- zero-token costs up to the new last nonzero;
- early-termination savings.

However, Q1 showed that sweeping the old constant `zero_token_bits` over a wide range barely moved quality. Treat this stage as validation of plumbing and attribution. Do not promote it merely because the cost estimate is more exact. Promote only if it changes useful decisions and improves exact final results.

## 6.4 Stage C: small beam search over joint coefficient decisions

The real lever is joint run-aware selection.

For selected high-risk DCT8 blocks, allow a small candidate set around the baseline quantized coefficient:

```text
{ baseline, 0, baseline - sign, baseline + sign }
```

Restrict candidates to:

- currently nonzero coefficients;
- coefficients near a zero threshold;
- low/mid frequencies capable of causing visible ringing or flat-side bias;
- a small top-N ranked by estimated distortion/risk effect.

A useful beam state is:

```text
(k, used_or_remaining_nonzeros, previous_was_nonzero, last_nonzero, cost, edits)
```

For each candidate total nonzero count, the I.4 coefficient context can be evaluated under the current fixed block/neighborhood context. Keep a beam width around 4–8 initially and profile it. The objective is:

```text
sample-domain distortion
+ tail-risk penalty
+ lambda * entropy_cost
+ edit regularization
```

The tail-risk term should be derived from source/reconstruction geometry, for example error leaking into the smooth side normal to a coherent edge. It should not embed Butteraugli code or libjxl-derived weights in the codec core.

### Important context caveat

The resulting nonzero count affects prediction for later blocks. There are two safe implementation choices:

1. Process selected blocks in raster order and update the local nonzero grid as changes are accepted.
2. Rank under frozen neighbor contexts, then exact-walk and reject changes whose real cost or downstream effect fails the gate.

Start with the second for simplicity, but record the mismatch. Do not claim the local optimization is exact when neighbor contexts are frozen.

## 6.5 Respect the Y → CfL → chroma dependency

Current code quantizes Y first, refreshes its reconstruction for CfL, and then quantizes X/B. A Y refinement may invalidate chroma residual decisions.

Therefore:

- Any accepted Y change must update the local Y reconstruction.
- Recompute or validate CfL for the affected tile when the change exceeds a small declared threshold.
- Re-run selected chroma refinement afterward.
- Never independently “repair” Y while leaving stale chroma residual assumptions.

## 6.6 Add a byte-neutral repair/donor mode

The current gap is a small number of severe local errors, while JPXL already has strong average structural quality. Exploit that asymmetry.

### Repair candidates

For top-risk blocks, exact-price alternatives such as:

- undoing an aggressive trailing truncation;
- retaining one or two low/mid-frequency coefficients normal to the dominant edge;
- one-step finer quantization for a tightly selected coefficient set;
- fresh CfL in a high chroma-residual tile;
- testing a nearby cover split only when the cover margin is low.

### Donor candidates

Find low-risk blocks where one small coarsening or extra truncation has low measured source/reconstruction cost. Rank donors by bytes saved per risk increase.

### Selection

- Build a bounded repair list and donor list.
- Use an exact or close entropy price for each local delta.
- Select repairs under the donor byte budget.
- Run one exact final writer count.
- If over target, discard the lowest-value repairs or the entire pass; do not open a new global rate search.
- Permit one iteration only.

This targets the actual metric divergence: spend a few bytes on the exposed outliers without abandoning the global SSIMULACRA2 advantage.

## 6.7 Acceptance gates

- Feature-gated and off by default until corpus evidence exists.
- Visits a bounded fraction of blocks and reports that fraction.
- Common-path time cost stays small because only the exact finalist is refined.
- Reduces Butteraugli 3-norm aggregate and edge-flat outlier count.
- Does not reduce corpus SSIMULACRA2 beyond the project's declared Contract B bounds.
- Exact output remains under the rate cap.
- Both independent decoders accept every changed stream.
- Scalar/AVX2 and thread-count determinism continue to pass where required.

A run-aware selective quantizer is a more credible bridge than another global QM, lambda, dead-zone, or EPF sweep because those global controls have already shown the wrong tradeoff.

---

# 7. Workstream 4: improve cover pricing only where it is uncertain

Q4 already corrected the cover proxy with transform-size calibration. The residual opportunity is not to replace the whole fast scorer with the exact writer. It is to make a small number of low-margin choices more plan-specific.

## 7.1 Preserve the calibrated fast pass

Keep `CoverRateModel::Calibrated` as the default broad scorer. It is fast, neutral in wall time in the measured screen, and improved both SSIMULACRA2 and Butteraugli 3-norm slightly.

## 7.2 Generate stability evidence during the existing pass

For each hierarchical region, retain a compact summary:

```rust
pub struct CoverDecisionEvidence {
    winner: CandidateId,
    runner_up: CandidateId,
    margin_q: u16,
    lower_bound_gap_q: u16,
    flags: u8,
}
```

The margin should be generated while the candidate costs are already being evaluated. Do not run a second full cover pass merely to recover margins; the exact dirty-frontier prototype already showed that buying a second pass plus fresh CfL can make the encoder 1.6–1.9× slower.

Use fixed-size storage or one packed record per merge node. The existing `stability.rs` and `regret.rs` scaffolding should validate:

- false-stable rate;
- regret of frozen decisions;
- dirty-area fraction;
- correlation with final perceptual outliers.

## 7.3 Add candidate summaries for selective repricing

For the winner and runner-up of only low-margin nodes, retain or cheaply derive:

- nonzero count per channel;
- last nonzero position;
- interior zero count;
- magnitude/token-class histogram;
- predicted `non_zeros` symbol class;
- DctSelect/meta signaling;
- empty/nearly-empty class;
- edge-flat risk mass covered by the candidate.

The Q4 audit showed that writer/proxy residuals are stable enough by transform size for a broad fit, but DCT8 residuals improve when zero runs are considered and DCT32 has a mixed empty/dense population. These summaries address that residual without putting a coding-order scan in every hot SIMD candidate.

## 7.4 Selective plan-specific reprice

At finalist construction:

1. Build a token-cost lookup from the anchor/finalist entropy model.
2. Select only nodes below a margin threshold or above a tail-risk threshold.
3. Reprice winner and runner-up using their summaries and the current model.
4. Re-open the decision only when the refined cost overcomes a hysteresis margin.
5. Recompute affected ancestors/descendants locally.
6. Exact-price the finalist as usual.

This is not a fully exact cover objective. It is a second-stage ranker. The writer remains authoritative.

## 7.5 Fresh CfL is a separate gate

`AnchorReuse::CoverOnly` exists but is not the current production path. Use it only after measuring a separate CfL confidence signal. Cover stability and CfL stability are related but not identical.

A reasonable CfL gate can use:

- chroma/luma covariance stability across anchors;
- residual-energy increase under reused CfL;
- chroma edge misalignment;
- tail-risk atlas flags;
- rate distance from the structural anchor.

Do not rebuild CfL everywhere merely because one cover node changed.

## 7.6 Acceptance gates

- Low-margin evidence is generated with negligible extra scoring work.
- Repriced area is small and explicitly reported.
- No second global cover traversal.
- Exact bytes and quality improve on the chosen corpus, not just proxy residuals.
- Common-path wall cost remains low single digit at most; otherwise the mechanism must be narrowed.
- The old calibrated path remains available as a bit-identical control.

Expected payoff is likely modest—small single-digit density or tail-risk improvement—not a new 2× speedup. That is appropriate at the current maturity level.

---

# 8. Workstream 5: reduce entropy/writer memory traffic and search waste

The entropy/writer stack remains a meaningful portion of production profiles, but much of the obvious work has already landed. Focus on representation and bounded candidate selection.

## 8.1 Pack the token tape

The current `TokenTape` uses four structure-of-array vectors:

- cluster: `u8`;
- token: `u16`;
- extra-bit count: `u8`;
- extra value: `u32`.

That is eight payload bytes per symbol before allocator capacity overhead, even though many symbols have no extra bits. The repository notes roughly 1.5 million tokens per mid-image plan and around 35 MB on a 12 MP plan.

Use a packed base record plus sparse extras:

```rust
#[repr(transparent)]
pub struct PackedToken(u32);

// Suggested logical fields, not fixed wire bits:
// [ cluster:8 | token:16 | extra_bits:8 ]

pub struct PackedTokenTape {
    base: Vec<PackedToken>,
    extras: Vec<u32>,
}
```

Replay keeps an `extra_cursor`; when `extra_bits != 0`, it consumes the next sidecar value. No per-token sidecar index is needed because replay order is stable.

Payload cost becomes approximately:

```text
4 + 4p bytes/token
```

where `p` is the fraction of tokens carrying an extra value. The theoretical no-extra limit is 50% of the current payload. Measure `p` by cluster and stream before implementation, because actual savings depend on it.

Requirements:

- preserve token order exactly;
- preserve count/store replay determinism;
- add debug assertions that the extra cursor ends exactly at `extras.len()`;
- benchmark construction and replay separately;
- compare cache misses and peak RSS, not only wall time;
- keep the old representation behind a test feature until bit identity is proven.

## 8.2 Reuse LF/control-image models when valid

Q0b's tree learner currently runs once per priced/stored plan per LF group. The project's own estimate puts hoisting at roughly a 1% wall opportunity.

Implement a plan-owned control-image model cache keyed by all inputs that affect the token stream. Reuse only when the control image and model inputs are identical. Every reuse must be checked by exact count in development mode.

This is a small win. Treat it as cleanup after the higher-value quality/controller work, not the headline project.

## 8.3 Rank entropy alternatives before exact training

`Full` searches multiple hybrid-uint configurations; production fast entropy uses a narrower model. To improve density without bringing exhaustive cost into `Balanced`:

1. Collect cheap statistics from the first anchor's tape.
2. Rank a very small candidate set—usually one or two configurations.
3. Fully train and exact-price only that set on the finalist.
4. Retain the current fast model when predicted gain is below a minimum threshold.

The ranker can use:

- symbol alphabet and tail distribution;
- fraction of values needing extra bits;
- zero/nonzero mixture;
- per-cluster sample count;
- estimated table signaling cost;
- prior exact regret collected by instrumentation.

Do not add all nine `Full` alternatives to `Balanced`. The goal is to capture high-confidence density gains at bounded cost.

## 8.4 Improve clustering signaling estimates

Current clustering uses fixed approximations such as histogram fixed bits, per-symbol bits, and cluster overhead. Instrument actual serialized table cost versus estimate by:

- cluster count;
- alphabet size;
- sparsity;
- merge stage;
- image/rate class.

Then replace fixed estimates with a compact calibrated table or formula if it improves exact finalist decisions. As with cover pricing, keep exact adoption as the final gate.

## 8.5 Keep the greedy merge deterministic

Parallel candidate-cost construction is already present. The serial greedy merge may be a scalability limit, but changing it risks non-determinism and altered tie behavior. Only parallelize or batch it after a current profile shows it is material, and preserve a total-order key for every decision.

---

# 9. Workstream 6: optimize scaling only after a new matrix

Historical profiles showed weak core utilization and limited 1→4 scaling. Subsequent phases parallelized substantial writer, table, and candidate work. The old scaling conclusion may no longer be current.

Run the new 1/2/4/8-thread matrix first, separately for:

- fixed VarDCT;
- one predetermined VarDCT probe;
- production `Balanced` with no rescue;
- `Balanced` with rescue;
- `Quality`;
- lossless Modular low and high effort.

Record wall time, CPU time, instructions, cache misses, context switches, and peak memory.

## 9.1 Interpret CPU and wall together

- Lower wall with proportional CPU increase can be good parallelism.
- Flat wall with rising CPU is scheduling or memory contention.
- Lower CPU with flat wall often means a serial barrier remains.
- Large p95 spread can mean task granularity or allocator contention rather than missing arithmetic optimization.

## 9.2 Likely areas only if profile-confirmed

### Quantization task grain

HF quantization remains material in recent profiles. Tune group ranges so tasks are large enough to amortize scheduling but small enough to balance busy and smooth regions. Use deterministic contiguous ranges, not work stealing that changes reduction order unless the output contract permits it.

### Final writer phase barriers

Look for barriers where count, table construction, and store could pipeline by independent section without changing the final deterministic ordering. Do not create a second frame-sized result solely to overlap phases.

### Request-scoped arena pressure

Audit remaining full-frame or per-plan allocations after the token tape is packed:

- candidate descriptors;
- cover evidence;
- CfL scratch;
- pass-group descriptors;
- temporary ANS/tables;
- copied coefficient arrays.

The goal is fewer bytes moved, not merely fewer allocator calls.

### NUMA and hybrid-core control

The benchmark already prefers pinned P-cores on hybrid Intel hosts. Production execution should either expose affinity control to the application or avoid making claims that assume homogeneous cores. Do not hard-code machine-specific affinity in the codec library.

## 9.3 Stop condition

Stop leaf optimization when all of the following are true:

- no single self-cost above roughly 5–8% has a credible output-preserving improvement;
- 1→4 scaling is reasonable for the dominant image sizes;
- p99 is controlled by bounded algorithmic work rather than scheduler noise;
- the codec's remaining loss is density/perceptual rather than CPU.

At that point, another micro-optimization round is less valuable than the selective quantizer and controller work.

---

# 10. Separate roadmap for lossless Modular

Do not mix lossless work into the VarDCT parity headline.

The repository's historical evidence says JPXL's default/low-effort output is competitive with `cjxl -e 1`, while the gap to `cjxl -e 7` is much larger and grows with image size. That is expected to require stronger model selection, not another copy-loop optimization.

Several obvious items are already done:

- sampled property gathering is promoted and materially faster;
- global MA-tree selection exists and is exact-size gated against local;
- exact-tier LZ77 alignment exists;
- a row-sized blanket lookback expansion was rejected;
- palette and squeeze alternatives are exact-priced;
- the current RCT choice is effectively fixed YCoCg versus no RCT.

The next lossless sequence should be:

## 10.1 Rebuild the current effort matrix

Measure current code after all Phase 4B–4G changes:

- JPXL effort 1/default versus `cjxl -e 1`;
- a medium JPXL effort versus `cjxl -e 3` or `e4`;
- JPXL effort 7 versus `cjxl -e 7`;
- bytes, wall, CPU, peak memory;
- photographic, synthetic, screenshot/text, palette-heavy, noisy, and high-bit-depth inputs.

Old 16–42% high-effort gaps are planning evidence, not a current release claim.

## 10.2 Expand bounded RCT search

`lossless.rs` currently represents RCT as a boolean and emits the standard YCoCg type when enabled. Add a bounded candidate set of standard-defined reversible color transforms.

Design:

- cheap sample-based residual ranking;
- retain top K candidates by predicted entropy;
- build MA/predictor model only for finalists;
- exact-price complete streams;
- effort-gated K;
- always include current YCoCg and none as controls.

This is a likely high-value density lever because channel decorrelation changes every later residual model.

## 10.3 Make effort levels structurally meaningful

Each effort should add an explicit bounded search capability, not merely increase a loop constant without changing model strength.

Example hierarchy:

- **Low:** fixed predictor family, none/YCoCg, sampled properties, simple clustering.
- **Medium:** several RCTs, several predictor/property sets, local/global tree finalists.
- **High:** broader RCT/predictor interaction, more MA properties, several exact entropy finalists, repetition-gated LZ alternatives.

Record work counts so a regression cannot accidentally make default effort perform high-effort search.

## 10.4 Improve MA-tree candidate ranking

Use sampled exact residual statistics to rank:

- property sets;
- split thresholds;
- predictor families;
- local versus global topology;
- channel-conditioned choices allowed by the standard.

Do not fully train every cross-product. Use a staged tournament:

```text
cheap sampled rank → medium exact residual pass → exact complete-stream finalists
```

The complete stream remains the authority because tree and histogram signaling can reverse a residual-only win.

## 10.5 Improve entropy clustering with exact finalist pricing

As in VarDCT, calibrate clustering estimates against serialized table cost. High effort may evaluate more merge/topology alternatives, but only a bounded finalist set should run the full writer.

## 10.6 Gate larger LZ search by repetition evidence

The blanket row-sized lookback experiment was negative. A new LZ attempt needs a different mechanism:

- compute a cheap repetition score by channel/tile;
- identify long horizontal/vertical repeats, repeated rows, sprites, or metadata-like streams;
- expand search only for those regions or streams;
- enforce a comparison budget;
- exact-price against the current LZ result.

Do not increase lookback globally.

---

# 11. Recommended implementation order

## Milestone G0 — comparison truth

**Changes**

- Replace/fix `compare-libjxl.ps1`.
- Add curve builder and automatic equal-byte/equal-quality interpolation.
- Add corpus manifest and versioned JSON output.
- Add production p50/p95/p99 and 1/2/4/8 scaling reports.

**Exit evidence**

- Current Q9 `Balanced` curves against the pinned `cjxl -e 7` binary.
- Current density gap at equal SSIMULACRA2 and equal Butteraugli 3-norm.
- Current scaling and p99 fallback distribution.

No encoder policy change should precede this baseline.

## Milestone G1 — native risk atlas

**Changes**

- `AnalysisAtlasV2` source-side features.
- Edge-flat synthetic fixtures.
- Reconstruction/error diagnostics and risk-recall report.
- No production decision change.

**Exit evidence**

- Top-risk 5–10% of atoms captures a useful majority of the known worst edge-flat errors.
- Atlas construction cost and memory are bounded.

## Milestone G2 — selective run-aware quantizer

**Changes**

- `EntropyCostView`.
- Context-aware trailing-cost experiment.
- Finalist-only DCT8 beam refinement behind a research flag.
- Exact writer and decoder gates.

**Exit evidence**

- Butteraugli 3-norm/outlier improvement at matched bytes.
- SSIMULACRA2 preserved.
- Small visited area and bounded finalist overhead.

## Milestone G3 — bounded production controller

**Changes**

- Explicit `RateStatus`.
- Hard exact-price caps.
- One fresh-structure rescue.
- Remove hidden `Balanced` → exhaustive transition.

**Exit evidence**

- No unbounded production trace.
- Hard scenes retain quality or return an explicit rate status.
- p99 improves substantially on former fallback cases.

## Milestone G4 — selective cover/CfL refresh

**Changes**

- Winner/runner-up evidence emitted in the existing cover pass.
- Low-margin plan-specific reprice.
- Separate CfL confidence gate.

**Exit evidence**

- Small repriced area.
- Exact density or tail-risk win.
- No second global cover pass.

## Milestone G5 — entropy representation and bounded alternatives

**Changes**

- Packed token tape with sparse extras.
- Control-image model reuse where identical.
- Top-K entropy alternative ranker.
- Better clustering signaling estimates.

**Exit evidence**

- Reduced peak RSS/cache misses.
- No bitstream or determinism regression for representation-only changes.
- Density gains pay for any extra training work.

## Milestone G6 — profile-directed scaling and lossless effort

Run only after the preceding work changes the profile. Then address the current serial region, not the Phase 42 serial region.

---

# 12. First three pull requests

These are the highest-confidence opening sequence.

## PR 1: comparison harness v2

### Files

- `tools/compare-libjxl.ps1`
- new `tools/codec-curve.ps1` or a small Rust/Python harness under `tools/`
- corpus manifest under research tooling
- summarizer and schema tests

### Required changes

- Real AB/BA counterbalancing.
- Correct per-codec thread and preset metadata.
- Binary/input hashes checked before every timed block.
- Automatic byte/quality matching.
- Process CPU and peak memory.
- Current internal work counters.
- Bootstrap confidence intervals or at minimum median plus robust dispersion.

### No codec changes

This PR exists to freeze the target.

## PR 2: `AnalysisAtlasV2` diagnostics

### Files

- `crates/jpxl-encode-policy/src/analysis.rs`
- request statistics/diagnostics structures
- research-only visualization/export tool
- edge-flat fixture tests

### Required changes

- Add gradient, structure tensor, Laplacian, plane residual, noise, and flat-side asymmetry.
- Aggregate without rescanning.
- Emit risk maps and correlation reports.
- Keep all production streams byte-identical.

## PR 3: run-aware finalist quantizer prototype

### Files

- `crates/jpxl-encode-policy/src/quantize.rs`
- `crates/jpxl-encode-policy/src/entropy.rs`
- `crates/jpxl-encode/src/vardct/walk.rs` or a read-only cost adapter
- rate/quality experiment tests

### Required changes

- Add `EntropyCostView`.
- Reproduce the current coefficient walk cost under frozen contexts.
- Add research-only DCT8 beam refinement for top-risk blocks.
- Exact-price and exact-decode every candidate.
- Report visited blocks, candidate states, estimated/exact rate regret, and quality deltas.

Do not mix the bounded-controller refactor into PR 3. First prove that the new rescue would have a better-quality finalist worth rescuing to.

---

# 13. Experiments not worth repeating without a new mechanism

The repository has already paid for these answers. Do not restart them under new names.

- Broad variance/activity AQ fields.
- Fine-lattice AQ or an `HfMul` signaling plane without a radically cheaper representation.
- Adaptive EPF sharpness; uniform sharpness 7 currently has evidence behind it.
- Global dead-zone or lambda sweeps.
- A trailing-truncation price tweak with the same independent coefficient model.
- Global fixed DCT8 cover.
- Global cover/CfL freezing for the finalist.
- A second full cover pass solely to construct an exact dirty frontier.
- The safe-but-slower S8/cheap cover prune.
- Blanket larger LZ lookback.
- One-cluster static entropy.
- Full entropy alternatives on every `Balanced` finalist.
- Disabling fallback without a bounded fresh rescue.
- More generic SIMD work without a current profile and a specific self-cost.
- Training production constants from libjxl source or trying to reproduce its internal heuristic architecture.

A rejected mechanism may be revisited only when the new proposal explains why the old cost/quality failure no longer applies.

---

# 14. Instrumentation that should become permanent

The following counters should survive optimization rounds because they make regressions attributable.

## Controller

- anchor count;
- exact finalist count;
- correction count;
- rescue count;
- exhaustive count;
- target miss reason;
- slope prediction error;
- time and bytes per priced rung.

## Analysis/structure

- atlas construction time and bytes;
- cover candidate count;
- low-margin node count;
- dirty/reopened node count;
- false-stable regret sample;
- CfL reused/refreshed tiles;
- risk-selected atom count.

## Quantizer

- SIMD baseline time;
- refined blocks by transform/channel;
- beam states visited/pruned;
- coefficients changed to zero/from zero/by ±1;
- trailing truncations undone/applied;
- estimated versus exact byte delta;
- repair and donor counts.

## Entropy/writer

- token count;
- extra-value token fraction;
- token tape payload/capacity bytes;
- entropy candidate count;
- exact trained alternatives;
- clustering predicted versus serialized bits;
- table build, ANS count, and store times;
- model-cache hits.

## Output quality diagnostics

- top edge-flat error tiles;
- aggregate flat-side leakage;
- per-transform perceptual outlier counts;
- SSIMULACRA2, Butteraugli max/3-norm, and PSNR.

All diagnostic work must remain opt-in or outside timed builds. The project has already correctly made expensive diagnostics optional; preserve that boundary.

---

# 15. Decision matrix

| Proposal | Main target | Upside | Risk | Priority |
|---|---|---:|---:|---:|
| Harness v2 | Truthfulness | Essential, no direct codec gain | Low | Immediate |
| Explicit bounded rescue | p99/production contract | High on fallback cells | Medium | Very high |
| `AnalysisAtlasV2` | Tail-risk selection | Enables all selective work | Low if diagnostic-only | Very high |
| Run-aware DCT8 finalist refinement | Butteraugli/density | Highest plausible quality upside | Medium-high | Very high |
| Repair/donor allocation | Tail risk at fixed bytes | High if risk is sparse | Medium-high | High after prototype |
| Cover margins in existing pass | Structural reuse/quality | Enables selective refresh cheaply | Medium | High |
| Selective cover reprice | Density/tail risk | Likely modest but broad | Medium | Medium-high |
| Packed token tape | RSS/cache/time | Up to nearly 50% payload reduction when extras are rare | Low-medium | Medium-high |
| LF model hoist | Wall time | Around 1% by current estimate | Low | Medium |
| Top-K entropy alternatives | Density | Moderate, bounded | Medium | Medium |
| More generic SIMD | Wall time | Probably small now | Medium opportunity cost | Low |
| Expanded lossless RCT search | Lossless density | Potentially high | Medium | High, separate lane |
| Broader lossless MA search | Lossless density | Potentially high at high effort | High CPU cost | High, effort-gated |

---

# 16. What success looks like

The bridge is complete when the repository can make all of these statements from the same current benchmark system:

1. **Production latency:** `Balanced` has a fixed work cap, no hidden exhaustive path, and stable p99 behavior across the corpus.
2. **Equal-byte quality:** JPXL retains its SSIMULACRA2 lead and no longer has a material aggregate Butteraugli 3-norm disadvantage.
3. **Tail behavior:** edge-flat DCT8 outliers are substantially reduced without a broad signaling field or global smoothing change.
4. **Equal-quality density:** at both equal SSIMULACRA2 and equal Butteraugli 3-norm, JPXL is at parity or better in bytes over the declared corpus and rates.
5. **Speed:** at the chosen equal-quality lane, JPXL remains at least competitive in wall and CPU time under equal resources.
6. **Scaling:** the 1/2/4/8-thread curve and peak memory are known, reproducible, and free of a large unexplained serial cliff.
7. **Lossless:** low-effort parity and high-effort density are reported separately, with bounded model-search tools explaining the difference.

The main correction to the current optimization strategy is simple: **stop treating average quality, worst-case perceptual quality, density, latency, and lossless model strength as one gap.** JPXL has already solved much of the raw VarDCT speed problem. The remaining path is selective: improve the few decisions that create exposed local error, make production work bounded, and make the benchmark compare equivalent outputs.

---

# Appendix A: implementation sketch for selective refinement

```rust
fn refine_finalist(
    plan: &mut QuantizedPlan,
    source: &Frame,
    atlas: &AnalysisAtlasV2,
    entropy: &EntropyCostView,
    budget: RefinementBudget,
) -> RefinementStats {
    let mut stats = RefinementStats::default();

    let mut candidates = atlas
        .risk_candidates(plan)
        .filter(|c| c.transform == Transform::Dct8)
        .take(budget.max_blocks)
        .collect::<Vec<_>>();

    candidates.sort_by_key(|c| Reverse(c.risk_score));

    for candidate in candidates {
        if stats.states_visited >= budget.max_states {
            break;
        }

        let baseline = plan.block(candidate.block_id).clone();
        let proposal = beam_refine_block(
            source,
            &baseline,
            candidate,
            entropy,
            budget.beam_width,
            &mut stats,
        );

        let Some(proposal) = proposal else { continue };
        if proposal.estimated_objective >= baseline.estimated_objective {
            continue;
        }

        // Y edits require local reconstruction and CfL/chroma validation.
        let checkpoint = plan.local_checkpoint(candidate.tile_id);
        plan.apply(proposal);
        plan.refresh_local_reconstruction(candidate.tile_id);

        if !plan.local_constraints_hold(candidate.tile_id) {
            plan.restore(checkpoint);
            stats.rejected_local += 1;
            continue;
        }

        stats.accepted_local += 1;
    }

    stats
}
```

The exact stream price is deliberately outside this function. Local search ranks candidates; the existing exact writer decides whether the complete finalist is legal and worthwhile.

# Appendix B: implementation sketch for bounded `Balanced`

```rust
fn search_balanced_bounded(...) -> RateOutcome {
    let mut budget = ExactBudget::new(6);
    let a0 = price_anchor(..., &mut budget);
    let a1 = price_anchor(..., &mut budget);

    let f0_rung = predict_from_two_anchors(&a0, &a1, target);
    let f0 = price_finalist(f0_rung, Reuse::SafeRequestScoped, &mut budget);
    if inside_band(&f0, target) {
        return outcome(f0, RateStatus::InsideBand, budget);
    }

    let c0 = maybe_price_one_correction(&a0, &a1, &f0, target, &mut budget);
    let incumbent = best_legal([a0, a1, f0, c0]);
    if inside_band(&incumbent, target) || !rescue_gate(...) {
        return outcome(incumbent, miss_status(...), budget);
    }

    let r0_rung = predict_fresh_rescue(...);
    let r0 = price_with_fresh_structure(r0_rung, &mut budget);
    let r1 = maybe_price_rescue_correction(&r0, target, &mut budget);
    let best = best_legal([incumbent, r0, r1]);

    outcome(best, RateStatus::RescuedFreshStructure, budget)
}
```

Tests should prove that every path consumes no more than the declared budget and that `Quality` is never called from this function.

# Appendix C: external behavioral baseline

The official libjxl effort documentation describes higher effort as enabling more coding tools, more expensive heuristics, and more exhaustive search. It also distinguishes lossy consistency/quality-at-size from lossless size reduction and notes that higher effort is not guaranteed to win on every individual image. JPXL should mirror that **product-level meaning**—clear effort/latency/quality tiers—without copying libjxl's internal implementation choices.

External references consulted as behavioral/context baselines:

- libjxl `doc/encode_effort.md`
- libjxl `doc/benchmarking.md`
- JPEG XL normative standard materials already retained by the repository
