# JPXL optimization questions for outside review

This is a non-authoritative advisor brief, not the project plan. Settled work
and acceptance remain in AKR. The questions below are kept current when
progress exposes an architectural limit that local profiling does not answer.

## Current checkpoint — 2026-08-15

The runtime `fast` lossy preset uses two exact navigation anchors, a fitted
log-rate prediction, an anchor-reused default-entropy finalist, and at most one
exact correction. Fast probes now use only the legacy hybrid-uint
configuration, fixed DCT8x8 cover selection, nearest HF quantization, and the
captured anchor's neutral CfL factors; this is an intentional speed tier. The
prediction is biased by one eighth of Fast's 3% undershoot band, so the common
near-crossing case pays one finalist instead of a correction while the exact
never-over check remains in force. Its exact retained Store emission is still
its size verification, removing a redundant Count traversal. Normal builds
include Fast, while `quality` remains the exhaustive default with hierarchical
cover, full hybrid-uint search, and trailing quantization. Fast permits up to
3% target undershoot but never exceeds the target.

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

Frames with fewer LF groups than worker threads now prefill each group's
complete square-transform candidate bank once and score independent aligned
32x32 cover regions through immutable reads on the existing executor. Results
are reduced in raster order; frames whose LF groups already saturate the
workers retain the lazy per-group path. The latest fresh PGO A/B against the
overwrite-only quantization checkpoint improved the 2400x1800 median from
1.27 to 1.10 s (13.4%) and held the 4000x3000 median at 2.22 s. Fast and
Quality outputs remain byte-identical between one and four threads and to the
prior checkpoint. An earlier lower-load run measured the same direction at
1.10 to 0.94 s (14.5%) and 1.88 to 1.87 s (0.5%).

Fresh four-thread, process-to-process matched-SSIMULACRA2 timing is 1.09 s
versus cjxl 0.47 s on 2400x1800 (2.32x), and 2.21 s versus 1.21 s on
4000x3000 (1.83x). The seven-scene 1 bpp screen still has zero fallbacks and
two corrections; its mean SSIMULACRA2 delta versus Phase 10 is -0.034 points
and the worst is -0.154. Quality remains available for callers that do not
want the cumulative speed/quality trade.

The newest 2400x1800 four-thread PGO profile contains 3,156 core-cycle samples
with zero lost and reflects the new scheduling topology. Its largest
self-costs are cover scoring 18.0% across the luma and chroma closures,
quantization 12.6% across the chunk closure and lane kernel, contiguous DCT
rows 6.0%, entropy best-config search 5.5%, forward-varblock preparation 5.4%,
pass-group writing 4.7%, entropy tables 4.1%, ANS 3.9%, census 3.5%, and CfL
3.0%. Region parallelism closes the mid-size worker-underfill gap without
reducing the total cover work.

Phase 15 then made the Fast tier structurally cheaper while leaving Quality
byte-identical. Against the retained Phase 14 PGO binary, the final interleaved
screen improved the 2400x1800 median from 1.24 to 0.75 s (39.5%) and the
4000x3000 median from 2.35 to 1.83 s (22.1%). The Fast outputs are deterministic
between one and four threads (mid hash
`9022bd5482e9c9bfb6ace0ce6211c2f1e83e79a82a2849e5a026e94fa7b344b9`) and the
Quality output remains hash
`d4b03810d0bcb73981bb559815c8952eda0fbe2b27040abf8c99d4792c98c2fa`.

The matched process window measured Fast at 0.69 s versus cjxl 0.50 s on
2400x1800 (1.38x), and 1.80 s versus 1.30 s on 4000x3000 (1.38x). This is not
a matched-quality comparison: the relaxed Fast tier scored SSIMULACRA2
69.8750/80.3353 and Butteraugli 3.6753/2.8892 at 1 bpp, while Quality remains
available when that loss is unacceptable. A fresh 2400x1800 Fast profile has
4,029 core-cycle samples with zero lost; its largest self-costs are SIMD lane
quantization 13.3%, pass-group writing 9.5%, ANS 6.3%, entropy-table building
6.2%, census 5.7%, quantization-group orchestration 5.5%, CfL 4.6%, and forward
DCT 2.7%. A one-cluster static entropy experiment was rejected after it lowered
Fast SSIMULACRA2 to 64.7/77.0 despite a further speed gain.

Phase 16 then reused the captured neutral-CfL anchor for the Fast finalist and
reserved 1/8 of the allowed undershoot band in its predicted crossing. Against
the Phase 15 Fast PGO binary, interleaved four-thread A/B medians improved from
0.72 to 0.59 s (18.1%) at 2400x1800 and from 1.88 to 1.41 s (25.0%) at
4000x3000. The candidate emits 537,999 and 1,495,016 bytes at the 1 bpp
targets; repeated one/four-thread outputs are byte-identical and decode. In a
separate interleaved process window against cjxl e7, Fast medians were 0.60 s
versus 0.52 s (1.15x) and 2.85 s versus 2.39 s (1.19x); host variance is high,
so report the raw log rather than treating one window as a universal baseline.
The relaxed Fast metrics were SSIMULACRA2 69.4584/80.4677, Butteraugli
3.6963/2.8693, and pnorm3 1.2352/0.7882. Quality remains unchanged and is the
high-quality fallback. The neutral-CfL reuse is deliberately not applied to
Quality/Full.

Phase 17 adds a retained `balanced` preset rather than changing either
existing tier. It uses the two-anchor controller, but keeps the request's
hierarchical cover and trailing HF quantizer, captures CfL once, and runs the
the fast entropy model only on the anchored finalist (plus one possible
correction). Its tolerance is 2% undershoot, between Quality's requested band
and Fast's 3% band; an over-target stream still rejects and falls back to
Quality. On the 2400x1800 and 4000x3000 photos at 1 bpp, isolated release
processes measured `balanced` at 0.89 s and 1.65 s versus Quality at 8.91 s and
13.58 s (10.0x and 8.2x faster). It emitted 538,600 and 1,495,941 bytes,
compared with Quality's 539,315 and 1,490,235. Decoded metrics were
SSIMULACRA2 72.5443/83.6424 and Butteraugli 3.1308/2.0041 for Balanced,
versus Quality's 72.2863/83.5429 and 3.2742/1.9378; the larger image's
Butteraugli is slightly worse, but the SSIMULACRA2 result remains within the
observed Quality/cjxl parity band. The Quality hashes remain
`d4b03810d0bcb73981bb559815c8952eda0fbe2b27040abf8c99d4792c98c2fa` and
`4baefbd0b8a055bdb01813f1284411cfd06a5d6013b305c392a78eaec486d775`.
In a pinned, interleaved four-thread window, Balanced medians were 1.15 s and
2.49 s versus cjxl e7 at 0.59 s and 1.51 s (1.95x and 1.65x); the raw window
also records CPU time and RSS, and is the more reproducible comparison than a
single unrestricted process run.
Diagnostics show the intended collapse on both photos: two Fast plans, one
anchored finalist, one structural build, and no fallback. On the other three
mid-size corpus classes (line, low-detail, and noise), the anchored result
misses the requested band or saturates and falls back to Quality; the emitted
bytes are identical to Quality in each case. Raw outputs and metrics are under
`.agent/scratch/phase17-balanced/`; this is a preset experiment, not a claim
that JPXL has reached the overall libjxl speed goal. A fresh Balanced
fast-finalist flamegraph captured 416 samples with zero lost: lane-4 HF
quantization (24.2%), entropy-table construction (13.2%), quantization-group
orchestration (13.0%), and pass-group writing (5.3%) dominate. The earlier
Full-finalist profile is retained beside it for comparison; the next bounded
optimization target is shared quantization or entropy-table work across the
two anchors and finalist.

## Phase 18 — entropy-model reuse and review answers (2026-08-15)

The outside review in `optimize-answers.md` confirms that the next useful
boundary is inside one mostly fixed plan, not another broad rate-loop rewrite.
Its first priorities are now adopted as constraints: expose real multiplicity,
separate rate/spatial/quantization/CfL/entropy effort, keep exhaustive Quality
as the oracle, and make structural reuse explicit. The review also identifies
the missing middle to investigate next: Anchored Quality should use cheap
navigation anchors but rebuild a fresh hierarchical cover and fresh CfL at the
predicted finalist, with one exact correction before exhaustive fallback.

Phase 18 removes one measured repeated pass without changing Fast or Quality.
Balanced trains its fast default entropy model on the first anchor, then uses a
validated copy of that model for the second anchor, finalist, and correction;
the exact writer still prices every candidate and enforces the never-over
target. The policy now names structural reuse explicitly (`None`, `CoverOnly`,
or `CoverAndCfl`) so a globally frozen cover/CfL is visible as a deliberate
trade rather than an optional `Option` detail. Attempted anchored work is also
merged into fallback telemetry instead of being overwritten by the exhaustive
result.

The reuse-only probe preserved the Phase 17 Fast hash and the exhaustive
Quality hashes. On the canonical 1 bpp photos, decoded reuse metrics were
mid: SSIMULACRA2 72.3798, Butteraugli 3.1908, pnorm3 1.0932, PSNR 34.1744;
large: SSIMULACRA2 83.6276, Butteraugli 1.9912, pnorm3 0.6393, PSNR 37.9729.
The corresponding streams were 538,646 and 1,496,087 bytes. A pinned,
interleaved four-thread window measured reuse medians of 0.895 s and 1.87 s
versus cjxl e7 at 0.49 s and 1.20 s (1.83x and 1.56x); raw hashes, timings, and
emissions are retained under `.agent/scratch/phase18-entropy-reuse/`.

Two deliberately temporary probes are recorded as negative guidance. Making
Balanced use fixed DCT8 cover reduced the photos to about 0.61/1.40 s but
lowered SSIMULACRA2 to 68.2743/80.4937, so it remains a Fast-tier trade rather
than a Balanced change. Disabling CfL saved time but cost about 0.60/0.36
SSIMULACRA2 points; Balanced keeps CfL until a confidence/refresh policy is
measured. The next implementation work is therefore output-preserving
construction cleanup, raw default-entropy event tapes, and an Anchored Quality
fresh-spatial finalist—not global cover freezing.

## Phase 19 — dequantization-matrix construction reuse (2026-08-15)

The first output-preserving item from the review is now implemented. The
`HfQuantizers` builder derives each transform's three default dequantization
matrices once and shares them across all of that transform's `HfMul` lanes;
`HfQuantizer::new` still owns the single-quantizer API and follows the same
step-table code. This removes repeated matrix construction without changing a
rate decision, quantizer rule, cover choice, entropy model, or wire integer.

The release candidate is byte-identical to Phase 18 for Balanced, Fast, and
Quality on both canonical photos, and to the representative Balanced fallback
streams. A wider pinned A/B screen against the Phase 18 commit measured
Balanced mid medians of 847.557 ms baseline versus 842.154 ms candidate and
large medians of 1897.536 ms versus 1676.179 ms (three interleaved processes,
five timed iterations, four threads). The earlier three-iteration window was
mixed, so this is retained as a bounded construction win with host-noise
caveats, not as a new libjxl-parity claim. Raw candidate hashes, streams, and
timings are under `.agent/scratch/phase19-dequant-matrix-reuse/`.

No new flamegraph was needed: the change only removes setup reconstruction and
does not alter the dominant target-rate stages. The next low-risk items remain
fixed-DCT8-only quantizer construction, completed-cache scratch removal, and
immutable structural geometry before the higher-risk raw event tape and
Anchored Quality spatial refresh.

## Phase 20 — Fast-only DCT8 quantizer construction (2026-08-15)

Fast already forces a fixed DCT8x8 cover, so its planner now builds only the
DCT8x8 HF quantizer tables. Balanced and Quality still receive the complete
DCT8/DCT16/DCT32 square vocabulary. The branch is selected from the same
Fast-plus-fast-entropy condition that selects fixed cover and nearest
quantization; no transform decision or emitted value changes.

Fast mid and large streams remain byte-identical to Phase 19, as do Balanced
and Quality. A Fast masking-AQ encode also compares byte-for-byte, covering the
multiple-`HfMul` construction. In a pinned interleaved five-iteration screen,
Fast large improved from a 1177.699 ms baseline median to 1134.649 ms (3.7%),
while mid measured 485.844 ms versus 497.810 ms; the mixed result is retained
with raw logs rather than treated as a universal speed claim. This is still a
useful output-preserving Fast-tier cleanup because the unreachable transform
tables are removed without narrowing Balanced or Quality. Raw hashes, streams,
and timings are under `.agent/scratch/phase20-fast-dct8-quantizers/`.

No new flamegraph was warranted for this small construction-only change. Next
items remain completed-cache scratch removal, immutable structural geometry,
and then the raw default-entropy event tape before the higher-risk Anchored
Quality spatial refresh.

## Phase 21 — completed-cover scratch elimination (2026-08-15)

The parallel completed-cover scorer now uses a zero-capacity `ForwardScratch`
marker. Its immutable `CoverForwardBank::Complete` access never enters the
forward-transform insertion path, so constructing the lazy path's DCT32-sized
transform, sample, and coefficient buffers for every 4×4-atom region was
unreachable work. The lazy hierarchical path still uses `ForwardScratch::new`
with full capacity; the luma reconstruction buffer remains unchanged because
the scorer writes it while pricing chroma residuals.

The completed path is exercised by the one-group parallel case. A pinned,
interleaved six-run screen on the 256×256 synthetic input (four workers,
Balanced, 1 bpp) had medians 816.811 ms for the Phase 20 baseline and
785.957 ms for the candidate (about 3.8% faster), with high host variance.
The larger Fast screen is retained only as a diagnostic because Fast's fixed
cover does not enter the completed-cache scorer. A 512×512 gradient that does
use the completed path produced byte-identical baseline and candidate streams
(22,866 bytes, SHA-256
`3de70270ebd9e6ef53421f30e8fa9ae5ca95f6f2e8157e2df17a9f2a3f4d5b40`) and
self-decoded successfully. Canonical Fast, Balanced, Quality, and masking-AQ
streams also remain byte-identical to Phase 20. Raw hashes, output checks, and
both timing windows are under `.agent/scratch/phase21-cover-complete-scratch/`.

No new flamegraph was warranted: this is a path-local allocation removal with
unchanged scoring and wire decisions. The next structural target remains
immutable geometry plus a small `HfMul` overlay; the raw event tape and fresh
Anchored Quality spatial refresh stay behind it.

## Phase 22 — shared immutable CfL estimate (2026-08-15)

Anchored probes now share the captured `CflEstimate` through `Arc` instead of
cloning the frame-wide correlation decision and per-group CfL grids for every
probe. Fresh plans allocate the estimate once, `CoverAndCfl` reuses the pointer,
and the validated `LfGroupPlan` still receives its own wire-facing `CflGrid`
clone. This is intentionally an ownership cleanup: `VarblockDecision`, cover
selection, quantization, entropy, and emitted syntax are unchanged.

Fast, Balanced, Quality, and masking-AQ canonical streams remain byte-identical
to Phase 21, as do the representative fallback streams. Candidate Balanced,
Quality, and Fast streams independently decoded to valid PPM images. The
candidate policy suite passed all 98 focused tests.

A pinned interleaved four-thread, five-iteration A/B at Balanced 1 bpp measured
triplet medians of 966.945 ms baseline versus 932.874 ms candidate on the
2400×1800 input (3.524% faster), and 1863.457 ms versus 1887.425 ms on
4000×3000 (1.286% slower). The larger result is within host noise, so this is
recorded as a bounded setup cleanup rather than a general speed or libjxl-parity
claim. Both arms produced identical short fingerprints and output sizes. Raw
binary/input hashes, timing order, output comparisons, and decode checks are in
`.agent/scratch/phase22-cfl-arc/`.

No new flamegraph was warranted (`paranoid=0`): this removes an immutable clone
and does not move the dominant target-rate stages. The next structural target
is true immutable geometry with a quantizer-dependent `HfMul` overlay; the raw
default-entropy event tape and Anchored Quality spatial refresh remain gated by
their counters and quality checks.

## Phase 23 — copy-on-write anchored geometry (2026-08-15)

Anchored probes no longer deep-clone every `VarblockDecision` before checking
whether its `HfMul` changes. Fresh plans retain owned vectors; captured groups
convert once to immutable `Arc<[VarblockDecision]>` storage. A reused probe
shallow-clones the group map, compares the desired multipliers, and materializes
an owned vector only for a group with an actual retarget. The writer-facing
`LfGroupPlan` remains an owned box, so this changes allocation ownership without
changing the cover, CfL, quantization, entropy, or wire decisions.

Fast, Balanced, Quality, and masking-AQ canonical streams remain byte-identical
to Phase 22, and candidate Balanced, Quality, and large Fast streams independently
decoded to valid PPM images. The focused policy suite passed all 98 tests.

A pinned interleaved four-thread, five-iteration A/B at Balanced 1 bpp measured
triplet medians of 900.053 ms baseline versus 841.658 ms candidate on the
2400×1800 input (6.488% faster), and 1707.062 ms versus 1711.443 ms on
4000×3000 (0.257% slower). The large result is neutral within host noise, while
the mid-size result is the useful anchored-copy signal; neither is a libjxl
parity claim. Output sizes and short fingerprints were identical. Raw hashes,
timings, output comparisons, and decode checks are in
`.agent/scratch/phase23-cow-geometry/`.

No new flamegraph was warranted (`paranoid=0`): this is a narrow ownership and
allocation change with unchanged work selection. The next structural step is a
true immutable transform-geometry representation plus a compact per-probe
`HfMul` overlay; selective dirty cover refresh, raw event tapes, and Anchored
Quality remain later phases.

## Phase 24 — compact per-probe `HfMul` overlay (2026-08-16)

Anchored probes no longer materialize an owned `VarblockDecision` vector when
a group's multipliers move: `PlannedVarblocks` gained a `Retargeted` form that
layers a dense `Box<[HfMul]>` over the shared immutable decisions. Geometry
reads keep using the base decisions; multiplier reads resolve through
`hf_mul_at` / `PlannedVarblockRange`. Re-retargeting keeps the one shared base
and replaces only the overlay; capturing an anchor applies the overlay before
freezing the base; length mismatches are rejected rather than truncated. This
commit also carries the workspace version bump to `0.3.0`.

All five canonical streams (Fast/Balanced/Quality mid, Fast large, Fast
masking-AQ mid) remain byte-identical to Phase 23, and the candidate
Balanced/Quality/large decodes are bit-identical to the Phase 23 PPMs. The
focused policy suite passed all 98 tests and the workspace gates match HEAD
(fmt clean, policy clippy clean, `jpxl-core` clippy blockers pre-existing).

**No timing claim.** The timed Balanced path never exercises the overlay —
under the neutral production AQ the desired multipliers already match, so
Phase 23's CoW path is taken — and both pinned A/B attempts on 2026-08-16 ran
against heavy competing host load, so the recorded runs are noise. The
definitive pinned pass (Balanced plus a masking-AQ stream, which is what
actually exercises the overlay) is deferred to a quiet host; the work record's
speed check stays open until then. Raw hashes, identity logs, decode checks,
and the contaminated timing logs are in `.agent/scratch/phase24-hfmul-overlay/`.

### Open architectural questions

1. How can the finalist refresh only structurally unstable cover decisions?
   Reusing the initial cover/CfL for the finalist saved almost no thin-LTO
   wall time and regressed SSIMULACRA2 by as much as 3.95 points, so global
   freezing is rejected. Would winner/runner-up margins plus a local dirty
   frontier avoid most of the second cover/DCT pass without that quality loss?
2. Can the two far-apart anchor quantizers and finalist be quantized from one
   coefficient traversal without tripling result storage? Quantization is
   still 12.6% self time in the new mid-size profile; the existing advice to
   batch adjacent rungs does not directly fit this wide two-anchor geometry.
3. What compact token representation could serve census, exact Count, and
   final Store without becoming another frame-sized allocation? Even the
   default-entropy Fast finalist still spends about 22% across best-config
   search, census, table construction, ANS, and pass-group writing.
4. Immutable candidate banks and deterministic region work stealing reduce
   mid-size wall time by 13-15%, but cover scoring still consumes about 18% of
   core cycles. Can winner/runner-up summaries or another compact intermediate
   share cover scoring between the two navigation anchors and finalist while
   still allowing the finalist's structurally unstable regions to change?
5. Can an inexpensive confidence signal identify the one corpus class where
   Fast loses about 0.8 SSIMULACRA2 points and route it to Quality, without
   first doing the exhaustive search that the preset exists to avoid?
6. The fixed-cover Fast tier is now about 1.4x cjxl on the measured window but
   intentionally gives up roughly two SSIMULACRA2 points on the mid image.
   Can a cheap variance or cover-margin signal selectively restore larger
   transforms without bringing back the full hierarchical scorer?
7. The Fast anchor now reaches the relaxed parity band by skipping a finalist
   CfL rebuild and biasing its predicted rung. Can the same safe margin be
   learned from a bounded rate-slope confidence signal, so pathological scenes
   fall back before paying a correction without narrowing the documented band?
8. Balanced's anchored fast-entropy finalist improves SSIMULACRA2 over exhaustive
   Quality on both audit photos while slightly worsening large-image
   Butteraugli. Is that a stable metric trade across the corpus, or should
   Balanced carry a small distortion guard that routes a Butteraugli outlier
   to Quality without re-running the full search for every image?
9. Balanced's fast-finalist path now avoids Full entropy alternatives, but
   lane-4 HF quantization is still 24% of its mid-image profile. Can two anchor
   quantizers and the finalist share one coefficient traversal or a compact
   multi-quantizer workspace without changing wire integers or retaining
   three frame-sized coefficient copies?

The review answers are retained verbatim in `optimize-answers.md`; the
questions above remain open until the proposed evidence exists. In particular,
the two-photo Balanced result is not broad quality validation, and the current
global `CoverAndCfl` path must not be described as a fresh spatial finalist.

## Superseded historical advisor verdict

The material below predates the Phase 15–18 bounded-anchor work. It remains as
history and rationale, but its statements that Fast still performs a full
cover/CfL loop per rung are no longer current. New work should follow the
current checkpoint and the reviewed answers above; do not use this section as
an implementation checklist without revalidating it against current
diagnostics.

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
