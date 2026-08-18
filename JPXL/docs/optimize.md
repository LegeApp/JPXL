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

**Timing result.** The pinned interleaved image pass used the hashed candidate
and pre-Phase-24 binaries, the 2400x1800 and 4000x3000 PPMs, four workers on
CPUs 0,2,4,6, and five timed iterations per `bench` invocation. Balanced
medians were 821.823 ms candidate vs 801.560 ms baseline at 2400x1800, and
1689.535 ms candidate vs 1710.202 ms baseline at 4000x3000. The masking-AQ
encode path, which exercises the overlay, measured 527 ms vs 560 ms at
2400x1800 and 1233 ms vs 1235 ms at 4000x3000. Every pair kept the same output
size and SHA-256/fingerprint. The mixed small positive and negative deltas are
within the observed run-to-run spread, so this closes the speed check as a
neutral result rather than a promoted performance baseline. Raw logs remain in
`.agent/scratch/phase24-hfmul-overlay/`.

## Phase 25 — reusable multi-quantizer coefficient workspace (2026-08-16)

The anchored rate controller now owns a request-scoped HF coefficient-arena
workspace. Each quantized plan keeps immutable `Arc` views into the arena; the
first and second exact-price plans are explicitly released before the next
probe, and the finalist is released before a correction. If a plan is still
live, the workspace allocates a separate arena rather than mutating data that
the earlier plan can observe. This removes repeated HF-arena allocation and
keeps the normal anchored path from retaining three frame-sized coefficient
copies without changing the quantizer arithmetic, entropy walk, or wire
integers.

Fast, Balanced, Quality, large Fast, and masking-AQ canonical streams remained
byte-identical to the pre-Phase-25 binary. Candidate and baseline decodes were
also byte-identical on the same mid/large checks. The focused policy suite
passed 100 unit tests plus the feature-gated rate, truncation, oracle, and
roundtrip suites.

The pinned interleaved image screen used the same four-worker affinity and
hashed inputs as Phase 24. Balanced medians were 807.379 ms candidate versus
792.170 ms baseline on 2400x1800 (within the observed host spread), and
1706.301 ms versus 1767.633 ms on 4000x3000 (3.5% faster). Masking-AQ direct
encodes measured 530 ms versus 559 ms on 2400x1800 (5.2% faster), and 1204 ms
versus 1296 ms on 4000x3000 (7.1% faster). The targeted AQ gain is useful, but
the mixed Balanced result is not promoted as a general baseline. Raw hashes,
identity checks, timings, and decode comparisons are under
`.agent/scratch/phase25-multi-quantizer-workspace/`.

This is the safe workspace half of open question 9. A single coefficient
traversal for far-apart anchors and an unknown finalist still needs a compact
batched representation; retaining all candidate outputs would violate the
storage constraint, so that higher-risk step remains open.

## Phase 26 — nested rate-search multiplicity screen (2026-08-16)

The existing opt-in `bench vardct-rate --diag` counters were rerun on the
hashed 2400x1800 and 4000x3000 inputs with four workers at 1 bpp. This was a
measurement-only pass; it did not change the codestream path. It establishes
that the persistent executor is already doing its job: every search built one
pool, so pool construction is not the next pre-SIMD target.

Fast and Balanced stayed on the shallow path: two outer Count emissions and
one Store emission for Fast, with Balanced using one or two planning passes
depending on the image, and no nested internal entropy Count alternatives.
Quality is different. On both images its six Full plans generated 36 census
passes, 36 entropy trainings, 18 order candidates, six block-context
candidates, six preset candidates, and 36 internal exact Count emissions.
The resulting writer amplification was about 50x on the mid image and 49x on
the large image; search amplification was 250.5x and 218.2x respectively,
relative to the selected final Store emission. The forward DCT cache was
already about 93% hit-rate on these Quality runs, so cache reuse is not the
dominant missing piece either.

This closes the aggregate-multiplicity step. The next pre-SIMD mechanism
should reduce the Full-path entropy/candidate work— finalist-only entropy,
RateSketch, or a similarly bounded token/census representation—before any
leaf SIMD or ANS/bit-writer tuning. Raw command output and hashes are under
`.agent/scratch/phase26-rate-multiplicity/`.

## Phase 27 — finalist-only entropy refinement (2026-08-16)

Quality refinement now navigates with exact Counts from the trained default
entropy model (`FinalFast`) and pays for the slice-18 alternative search only
on the selected finalist and a bounded exact correction window when its
undershoot exceeds tolerance. The final codestream always comes from an
exact Full-entropy Store; the change is a bounded search-policy change, so
large-image byte identity with the former Quality path is not assumed.

On the same four-worker screen, Quality fell from 9.654 s to 5.545 s on the
2400x1800 image and from 18.229 s to 11.213 s on the 4000x3000 image. Internal
Full Counts fell from 36 to six on both images; writer amplification fell from
50x to 21x and from 49x to 20x. Mid remained byte-identical at 539,315 bytes.
Large changed from 1,490,235 to 1,490,211 bytes. The in-tree decoder accepted
both streams; under the current decoder the large candidate moved RMSE from
3.206970 to 3.206832, SSIMULACRA2 from 83.5266 to 83.5149, and Butteraugli
from 1.9526 to 1.9513. That small SSIMULACRA2 trade is recorded rather than
hidden and needs a wider corpus screen before treating the change as a final
Quality promotion.

Raw diagnostics, encoded streams, decoded pixels, hashes, and metric output
are under `.agent/scratch/phase27-finalist-only-entropy/`.

## Phase 28 — reusable CfL LF-sample scratch buffers (2026-08-16)

`perf`/`cargo flamegraph` are blocked in this session's sandbox
(`perf_event_paranoid=4`, no `sudo`), so this phase used the existing
`bench vardct-rate --diag` stage counters instead of a fresh flamegraph. That
diagnostic shows CfL search is now the largest or second-largest single cost
bucket post-Phase-27: about 19-20% of Balanced's wall time and, for Quality,
the single largest bucket at roughly 37-40% of total wall — ahead of both
cover scoring and entropy work, which prior phases already addressed.
Structural cover/CfL reuse across anchors was already tried and rejected
(regressed SSIMULACRA2 by up to 3.95 points; open question 1), so this phase
does not touch the search itself, only its allocation pattern.

`estimate_cfl` allocated three fresh `vec![0.0f32; n*n]` buffers
(`y_lf`/`x_lf`/`b_lf`) per varblock — thousands of heap allocations per CfL
search on a photo-sized image. These are now three per-LF-group scratch
buffers, reused across every varblock in the group via `Vec::resize` the same
way the group's `TransformScratch` was already reused. `lf_samples_of`
unconditionally overwrites exactly the first `n * n` cells it is given, so
this is pure construction reuse with no arithmetic change.

All six canonical Fast/Balanced/Quality mid/large streams and both Balanced
masking-AQ mid/large streams stayed byte-identical to the pre-change binary;
every candidate output decoded. The full workspace test suite passed
unchanged, and strict Clippy on the changed crate isolated to the same
pre-existing `jpxl-core` debt as prior phases.

The pinned five-iteration timing screen showed small, mostly positive deltas:
Balanced mid 942.267 to 928.328 ms (1.5% faster), Balanced large 1787.939 to
1741.568 ms (2.6% faster), Quality mid 6849.583 to 6644.058 ms (3.0% faster).
Quality large's median rose slightly (13592.348 to 13889.233 ms) while its
five-iteration total fell (73961.143 to 68760.513 ms, 7.0% less) and its
minimum fell (13457.713 to 12841.807 ms, 4.6% faster), which reads as one
noisy iteration rather than a regression — every pair kept identical output
bytes and fingerprints. Raw logs are under
`.agent/scratch/phase28-cfl-lf-scratch-reuse/`.

This closes the easy allocation-only win in CfL search. The remaining cost is
believed to be the search's own arithmetic (forward DCT plus regression), so
further reduction needs the local dirty-frontier idea in open question 1, not
more construction reuse.

## Phase 29 — sRGB-to-linear lookup table (2026-08-16)

This session's host had `perf_event_paranoid` relaxed to `0`, unblocking real
instruction-level profiling for the first time in this pass (prior phases
relied on the built-in `bench vardct-rate --diag` stage timers because
`perf record` was refused). A DWARF-unwound `perf`/`inferno` flamegraph on
Balanced immediately found `<f32>::powf`, called from
`jpxl_core::color::srgb_to_linear` on its non-linear segment, as a top
self-time leaf — entirely from one call site, `PreparedFrame::from_srgb8`'s
per-pixel EOTF loop. `__powf_fma` and `cbrtf` (reached from the same libm
implementation) together were about 4.4% of sampled self-time in an earlier
shallow (frame-pointer) profile of the same binary.

An 8-bit sample has only 256 distinct byte values, so `from_srgb8` now builds
a 256-entry lookup table once per encode — by calling the same
`srgb_to_linear` function on each possible `byte / 255.0` input — and indexes
it per pixel instead of calling `srgb_to_linear` directly. Every lookup
returns the exact `f32` a direct call would have, so this is construction
reuse, not an approximation.

All six canonical Fast/Balanced/Quality mid/large streams and both Balanced
masking-AQ mid/large streams stayed byte-identical to the pre-change binary;
every candidate output decoded. The full workspace test suite passed, and
Clippy reported nothing on the changed file. A post-change profile shows
`cbrtf` gone entirely and `__powf_fma` reduced to a small residual from
elsewhere in the encoder.

The pinned five-iteration timing screen was faster on every one of six
cases, with identical output bytes and fingerprints throughout: Fast mid
496.134 to 454.801 ms (8.3% faster), Fast large 1086.481 to 1048.599 ms (3.5%
faster), Balanced mid 877.992 to 852.995 ms (2.8% faster), Balanced large
1814.148 to 1712.627 ms (5.6% faster), Quality mid 6444.107 to 6102.028 ms
(5.3% faster), Quality large 13825.880 to 12427.433 ms (10.1% faster). Unlike
prior phases' probe-count reductions, this gain applies once per encode
regardless of search multiplicity, which is why even Fast (the cheapest,
lowest-multiplicity preset) shows the largest relative improvement. Raw
folded profiles and the timing log are under
`.agent/scratch/phase29-srgb-lut/`.

This closes the color-conversion hotspot the profile pointed at. With real
profiling now available, the next phase should get a fresh flamegraph on
Quality (still CfL-dominated per Phase 28) before picking the next target,
rather than continuing to reason from the pre-Phase-28 flamegraph.

## Phase 30 — zero-copy natural coefficient order (2026-08-16)

A DWARF-unwound profile on Quality mid (33,580 samples) — the first in this
pass to look at the write/entropy stage rather than color conversion or CfL —
found `jpxl_core::varblock::natural_coeff_order` cloning its cached I.3.2
table on every call, even though the cache is a per-Order-ID `OnceLock` that
never changes after the first call for a given shape (13 possible Order IDs
total) and one call site was inside a loop over every varblock in every LF
group. The clone chain was about 1.24% of sampled self-time.

`natural_coeff_order_ref` (plus `TransformType::natural_coeff_order_ref`) now
returns a `'static` borrow into the same cache instead of an owned `Vec`.
Every read-only call site this profile found — `entropy.rs`'s two
order-frequency loops (one of them per-varblock), `walk.rs`'s
`OrderTables::from_order_set`, and `write.rs`'s `write_hf_coeff_orders` —
switched to it. The three call sites that mutate their own copy (two
order-search table builders and the order-length validator) keep calling the
owned `natural_coeff_order`, unchanged.

All six canonical Fast/Balanced/Quality mid/large streams and both Balanced
masking-AQ mid/large streams stayed byte-identical to the pre-change binary;
every candidate output decoded. The full workspace test suite passed, and
Clippy reported nothing on any changed file. A post-change profile of the
same Quality-mid scenario shows the clone chain reduced to about 0.03% of
sampled cycles — roughly a 40x reduction in that specific cost.

The pinned wall-clock screen was mixed rather than a clean win: Fast large
(-2.2%), Balanced large (-11.6%), and Quality mid (-3.1%) moved in the
profile-predicted direction, but Fast mid (+8.8%) and Quality large (+4.2%)
were slower, both with markedly higher run-to-run spread in exactly those
two candidate runs than their baselines — read as host-noise contamination
of specific samples rather than a real regression, since a redundant-clone
removal has no mechanism to slow anything down and every output stayed
byte-identical. Unlike Phases 24-29, the profile delta (not the wall-clock
screen) is this phase's load-bearing evidence; the acceptance record omits a
formal speed check for that reason rather than mis-stating a "pass". Raw
folded profiles and the timing log are under
`.agent/scratch/phase30-natural-order-ref/`.

## Phase 31 — lane-parallel candidate scoring in `choose_lane4` (2026-08-16)

A fresh DWARF-unwound profile of the same Quality-mid scenario, taken after
Phase 30 shifted the write/entropy cost, moved the leaf cost to
`HfQuantizer::choose_lane4` itself: 12.7% self time, plus another ~10% spread
across the bounds-check helpers its candidate loop de-vectorized into
(`SliceIndex::get` for `[i32]`/`[f32]`, `Option::copied`, `wrapping_abs`,
range-iterator and comparison glue — a "quantize-loop family" at 34.8% of
samples). The cause was structural: the SIMD `choose_lane4` built its
reconstructions as `f32x4` but then spilled every candidate lane back to
scalar arrays and re-selected the winner with sixteen bounds-checked `.get()`
updates per call, so the function paid for the vector arithmetic *and* a
scalar selection pass.

The candidate loop is now lane-parallel throughout: `best_q`/`best_err`/
`best_recon` stay in registers as `f32x4`, and the scalar update rule —
first legal candidate with strictly smaller error, else equal error and
strictly smaller magnitude, in `choose`'s `[0, estimate-1, estimate,
estimate+1]` order — is expressed as comparison masks and `blend`s. The
`|q| > MAX_QUANT` legality skip becomes `q.abs().cmp_le(splat(MAX_QUANT))`
(the bound is exact at 2^20 in f32), and the zero-threshold lanes are forced
to zero after the search exactly as the scalar shortcut would have returned
early. Two smaller hoists rode along: the per-channel step/threshold row
lookups leave the lane loop, and the remaining scalar tails iterate by
`zip` instead of indexed `.get()`. Results stay bit-identical to four scalar
`choose` calls by construction — the same comparisons on the same values in
the same order, just lane-parallel.

Verified: all eight canonical/masking-AQ streams byte-identical to the
pre-change binary (Quality-mid SHA-256 `d4b03810…`, unchanged); every
candidate output decodes; the 99-test `jpxl-encode-policy` suite and the full
workspace suite pass; `cargo fmt --all --check` clean. A post-change profile
of the same scenario puts the quantize-loop family at 27.2% of samples
(choose_lane4 self 10.5%), a ~22% relative reduction of the family, and the
pinned wall-clock screen moved in the profile-predicted direction on four of
six cases (Quality mid −6.4% median with tight spread, Fast large −31%,
Balanced large −2.5% total, Balanced mid flat) while Fast mid and Quality
large were slower/noisier — the same host-contamination signature Phase 30
hit, so the profile delta again carries the cost claim. One pre-existing
condition surfaced by the gate run and *not* caused by this change: workspace
`cargo clippy --all-targets -- -D warnings` fails at HEAD on 70
`indexing_slicing` hits in `jpxl-core/src/color.rs` (introduced with the
f3be8b8 leaf-SIMD commit; present on the clean tree, verified by stashing
this change). `color.rs` is outside this phase's brief, so it is reported
here rather than fixed opportunistically. Raw folded profiles and the timing
log are under `.agent/scratch/phase31/`.

## Phase 35 — lane-batched separable DCT and cached `ScaleF` (2026-08-17)

Phases 32–34 are closed in AKR (dirty-frontier screen and honest-negative
prototype; indexed lane-4 CfL quantization). A fresh P-core profile of the
Balanced 1-bpp mid encode taken on the Phase 34 head binary attributed about
19% of self time to the separable DCT family — `dct_2d_in_place` 4.1%,
`dct_1d` 3.8%, `forward_dct_rc` 3.4%, `dct_iv_16` 2.5%, `dct_iv_8` 1.9%,
`dct_ii_32` 1.7%, `lf_from_llf_into` 1.4% — plus 0.9% to `libm` `cos`
reached through I.8's `ScaleF`, which evaluated three `f64` cosines per LLF
cell on every call. The 1-D kernels were per-vector scalar (the `simd`
feature only shuffled one 8-vector through `f32x4`), and both 2-D drivers
transposed around each pass so a scalar kernel could run on contiguous rows.

The kernels are now written once, generically over a `Lane` — a single
`f32`, or with `simd` a `wide::f32x4`/`f32x8` holding one value from each of
several adjacent columns — and every lane executes exactly the scalar
operation sequence (no FMA, no reassociation). `DCT_2D` and `IDCT_2D` are
driven by lane-batched *column* passes: the column pass runs straight down
the row-major matrix, and the row pass is a column pass over one transpose,
which also drops one transpose from the forward direction. Lengths above 32
keep the dense-matrix scalar path through a per-column gather. `ScaleF` is
tabled once per power-of-two LF count from the same closed form, and
`forward_dct_rc` copies varblock rows in bulk instead of cell by cell.

Verified: all eight canonical/masking-AQ streams plus the lossless modular
stream are byte-identical to the Phase 34 head binary (Balanced-mid
`7f70ae00…`, Quality-mid `d4b03810…`); one-thread and four-thread outputs
are identical; a `--no-default-features` (no-SIMD) release CLI reproduces
the same Balanced and Quality hashes; the old and new decoders agree pixel
for pixel on every candidate stream; vendored `djxl` 0.13.0 and `jxl-oxide`
0.12.6 accept the candidates. New unit tests pin the invariant directly: the
lane-batched inverse equals the scalar column-first reference at tolerance
zero for every Table I.1 shape, `f32x4`/`f32x8`/dispatched column passes are
`to_bits`-identical to the `f32` lane pass for every butterfly length and
both directions, and the cached `ScaleF` table is bit-identical to the closed
form. Workspace fmt/build/strict-Clippy/tests, oracle, roundtrip,
determinism, and no-default-feature suites all pass. Pinned four-core,
five-iteration Balanced 1-bpp medians (two interleaved rounds each): mid
890/910 → 727/727 ms (−19%), large 1730/1679 → 1506/1492 ms (−12%); Fast is
flat to slightly faster (its fixed DCT8 cover does little transform work).
The post-change profile puts the DCT family at about 7.9% (`column_pass_lanes
<f32x8>` 4.2%, transposes in `dct_2d_in_place` 1.6%, `lf_from_llf_into`
1.0%, `forward_dct_rc` 0.6%, `column_pass` 0.6%) with `cos` gone from the
listing. Raw profiles, timing log, and identity hashes are under
`.agent/scratch/phase35-lane-dct/`.

Not done here: the crate is still built for baseline x86-64, so `f32x8` is
two SSE registers; a `target-cpu`/AVX2 build decision would roughly double
lane throughput without changing any result (rustc never contracts
`a * b + c`, and the tree's `f32::mul_add` calls are fused on every target),
but it is a build-configuration decision for the plan, not a code change for
this phase.

## Phase 36 — runtime AVX2 dispatch and the row-oriented lane quantizer (2026-08-17)

Requested outcome: AVX2 is the default whenever the host supports it, without
giving up a portable baseline binary. The workspace compiles for baseline
x86-64, so `wide::f32x8` is two SSE registers, and simply wrapping the Phase 35
column pass in a `#[target_feature(enable = "avx2")]` function did *not* widen
it (disassembly of that wrapper: 2,418 `xmm` and zero `ymm` arithmetic
instructions). Real 256-bit lanes need a `__m256`-backed lane type.

`jpxl_core::cpu::has_avx2` caches one `std::arch` detection (with a
`JPXL_DISABLE_AVX2` opt-out for A/B and for exercising the fallback on an AVX2
host). `jpxl_core::simd::F32Vec` is one lane trait — splat/load/store,
arithmetic, abs, truncation, ordered comparisons as all-bits masks, bit ops,
blend, all/any — with a scalar `f32` implementation, `wide` `f32x4`/`f32x8`
implementations, and `simd::avx2::F32x8` on `core::arch` intrinsics. Every
implementation performs the same IEEE-754 operation per lane, so a kernel
written once against the trait is bit-identical whichever lane type it is
instantiated with; `simd::tests` pins that against the scalar implementation
for every op. `avx2::F32x8` is the only `unsafe` in the tree: its contract is
documented at the type (every method executes AVX instructions; the type is
only reached inside a target-feature entry point behind `has_avx2`), and each
entry-point call site carries the same one-line contract.

Kernels: the Phase 35 DCT column pass gains an AVX2 entry point (now 1,269
`ymm` float ops, zero `xmm`, zero FMA). The bigger change is the HF quantizer.
`HfQuantizer::choose_run` quantizes a contiguous run of one channel's cells
(`q` and optional reconstruction), hoisting the per-channel rows, bias and
bounds once and processing eight cells per AVX2 chunk with a zero-padded final
chunk (target 0 / step 1 / threshold 1 pad lanes take the zero shortcut and are
dropped) instead of the old four-cell batches plus scalar tails; the indexed
sibling `choose_cells` gathers steps/thresholds per explicit cell. Each chunk
mirrors `choose_cells4`'s per-lane rule exactly — zero-threshold shortcut,
`clamp_round`'s ties-away rounding through truncation with the sign bit carried
by a copysign, the `[0, e-1, e, e+1]` candidate order and tie rule, the
`|q| <= MAX_QUANT` legality skip — and a chunk with any failing lane replays the
scalar loop, so the returned error is the first scalar one. `quantize_lane`,
cover scoring (`score_channel_lanes`), the CfL luma walk (per row) and the CfL
factor pricing (`hf_residual_cost_bounded`, 64-sample blocks) all use it; the
two cutoff-interleaved callers fall back to their scalar per-cell loops when a
run reports failure, which reproduces the exact cutoff-before-error order.

Verified: all eight canonical/masking-AQ streams and the lossless stream are
byte-identical to the Phase 35 binary; one- vs four-thread, `JPXL_DISABLE_AVX2`
vs AVX2, and the no-SIMD build reproduce the same hashes; new property tests
pin `choose_run`/`choose_cells` against `choose`/`reconstruct` (q, recon bits,
and error identity) at every lane width for runs of length 1–33 at random
offsets and scrambled index lists. Fresh mid-Balanced P-core profile: the
choose family fell from 17.6% (`choose_cells4`) + 3.8% (`quantize_lane`) to
6.8% (`choose_run_avx2`) + 4.3% (`choose_cells_avx2`) + 0.8%; the DCT family
is about 8%. Pinned wall time is reported in the AKR evidence with its noise
caveat: this host ran concurrent builds (load average up to 90) during the
phase.

Next leaf: `cbrtf` (compiler_builtins' portable implementation, 4.3%) in the
XYB conversion is now the largest scalar leaf.

## Phase 37 — exact AVX2 cube root for the XYB conversion (2026-08-17)

`f32::cbrt` resolves to `compiler_builtins`' portable FreeBSD/musl `cbrtf`
(a 5-bit integer estimate, two Newton steps in `f64`, one rounding to `f32`)
and was the largest scalar leaf left (4.3%, plus 1.7% in the planar
conversion around it). `simd::avx2::F32x8::cbrt` reproduces that algorithm
lane for lane — the integer estimate through an exact `floor(hx / 3.0)` in
`f64`, the two Newton steps in the scalar operation order on `f64x4` halves,
one `cvtpd2ps` — and declines any chunk holding the scalar special cases
(zero, subnormal, infinite, NaN) so the caller can use `f32::cbrt` there;
`simd::tests::avx2_cbrt_matches_std` pins it bit for bit on 28k inputs.
`color::avx2::linear_srgb_to_xyb_planes_avx2` runs the mixing matrix in
[`linear_srgb_to_xyb`]'s operation order on `F32x8` and uses that cube root,
selected by `has_avx2`; `color::tests::planes_match_single_pixel_bitwise` pins
the planar path (whichever the host selects) against the single-pixel
function, special cases and ragged tail included.

One trap worth recording: the first cut wrote the Newton steps in a closure
inside the `#[inline(always)]` method. A closure does not inherit the caller's
target feature, so every `_mm256_*_pd` inside it became an out-of-line call
(`core::core_arch::x86::avx::_mm256_add_pd` at 9% self time) and the encode
was 60% *slower*. Intrinsics used from AVX2 entry points must sit directly in
the target-feature function or in `#[inline(always)]` bodies that inline into
it — never in closures.

Verified: canonical streams, thread counts, `JPXL_DISABLE_AVX2` and the
no-SIMD suites byte-identical / passing as before; workspace gates green.
Profile: color conversion 6.0% -> 2.2%. Pinned medians p36 -> p37: mid
544/532/551 -> 535/518/527 ms, large 1169/1148/1218 -> 1040/1085/1077 ms.

## Phase 38 — filling the workers: band-parallel CfL and source prep, narrowed walks (2026-08-17)

With the leaf kernels lane-batched, the `--diag` phase clock on the mid photo
showed where wall time hid: `cfl_ms=103` of a 470 ms encode, although CfL's
CPU share was ~12%. Two serial stretches: the per-tile HF factor refinement
ran on the calling thread, and sample collection was parallel only over LF
groups — of which a 2400×1800 frame has two, so two of four workers idled.
The same shape sat in front of the search: `PreparedFrame::from_srgb8` (LUT
linearisation + XYB) ran serially before any executor existed.

Changes, all Contract A (output-preserving by construction):

- CfL sample collection is split into 64-pixel bands (one tile row of one LF
  group each; a varblock is ≤ 32 px and 8-aligned, so it never crosses a tile
  boundary, and raster-ordered varblocks make band concatenation reproduce the
  original order). LF samples are appended band by band in that order, so the
  frame-wide regression replays the exact same scalar additions; HF tile
  vectors are concatenated into the group's tile raster.
- HF factor refinement runs over the executor per tile (X tiles then B tiles),
  reduced in tile order.
- `encode_srgb8_to_target` builds the worker pool once and passes it through
  `search_frame_with_executor`; `PreparedFrame::from_srgb8_with` converts
  ~64-row bands into disjoint plane slices on that pool (per-pixel arithmetic,
  so bit-identical to the serial path; grayscale is the conjunction of band
  flags).
- `pass_group_walk` no longer scans the whole LF group's varblock list per
  pass group: it starts at the group's first tile-row via `partition_point`
  and stops after its last (validated raster order; asserted in debug).
- `gather_square` copies interior squares row-wise from the resident plane
  instead of clamping every sample.
- Not kept: an ANS encoder rewrite with per-symbol reciprocals and a
  branchless renormalisation. `perf annotate` showed 60% of
  `encode_symbols`' time on the `slots[start + offset]` load — the serial
  state chain is bound by that dependent lookup, not by the division — and
  the rewrite measured no gain, so it was reverted rather than kept as
  complexity.

Also in this phase: the `fast-debug` Cargo profile (release optimisation, no
LTO, 16 codegen units, incremental, debug assertions on) for the edit/test
loop — the full workspace suite runs in ~30 s — with `release` reserved for
benchmarks, profiles and shipped binaries (AGENTS.md §5).

Verified: canonical streams, thread counts, AVX2 on/off, no-SIMD suites and
external decoders as before; workspace gates green in both `fast-debug` and
`release`. Mid `--diag`: `cfl_ms` 103 → 37, wall 472 → 405 ms (quiet host);
interleaved pinned medians in a shared-host window: mid 568/574/559 → 428/425/423
ms (−25%), large 1127/1070/1140 → 911/954/877 ms (−18%). Remaining wall
budget on mid (~405 ms): cover 123, writer counts+store ~97, plan-full 37,
CfL 37, quantize 36, entropy training 31.

## Phase 39 — reconstruction from the lane pass; two honest negatives (2026-08-17)

An instruction-count profile (`perf record -e instructions`) put the
quantization closure second (10.7%) behind the run kernel itself; its hot
lines were the scalar per-cell `reconstruct(q)` sweep that filled `d_y_hf`
after Y quantization and the per-cell `coeff - k*d_y` chroma-target loop
with `.get()` on every index. `HfQuantizer::quantize_lane_with_recon` now
returns each cell's reconstruction from the run kernel (already computed
there; LLF cells receive `reconstruct(0)`), truncation refreshes only the
cells it zeroed with `reconstruct(0)`, and the chroma-target loop is a zip
over row slices that the compiler vectorises (same per-cell operations and
rounding). Interleaved pinned mid medians: 479/477/475 → 473/466/461 ms
(≈ −2%), all streams identical.

Two experiments measured nothing and were reverted rather than kept:

- ANS backward pass with per-symbol reciprocals and a select instead of the
  renormalisation branch (Phase 38 notes): `perf annotate` put ~60% of
  `encode_symbols` on the `slots[start + offset]` load — the serial state
  chain waits on that dependent lookup, not on the division.
- The pass-group walk's per-coefficient divisions replaced by shifts, and its
  non-zero count turned into a branch-free sweep with a backward scan for the
  last non-zero: walk shares 14.3% → 13.7%, inside noise. The overall encode
  runs at IPC ≈ 3.4, so the walks are not stall-bound either; their cost is
  instruction volume spread over three walks per written plan (census,
  entropy-table census, emission) — the token-tape question (open question 3)
  is the lever, not micro-work inside one walk.

Remaining wall budget on mid at ~400 ms (quiet host): cover 123, writer
counts + store ~97, plan-full 37, CfL 37, quantize ~34, entropy training 31.

## Phase 40 — cheapening the single cover-scoring pass (2026-08-17)

Planning review (two independent code-cited analyses, recorded in the plan for
this round): no module needs an overhaul. Cover, CfL and entropy training
already run once per encode; the three quantize replays are kept sequential by
the controller's data dependency (second rung from the first exact size,
finalist from both); the writer walks each written plan twice. What remained
on the planning side was instruction volume in the one 123 ms cover-scoring
pass, whose hot loop is `score_channel_lanes` inside `block_cost_bounded`.

`score_channel_lanes` now takes its targets as a `LaneTargets` (the Y plane
directly, or the `plane - k * d_y` chroma residual computed per row as a
vectorisable zip), writes the luma reconstruction straight into the caller's
`d_y_hf` through the run kernel's reconstruction output instead of a per-cell
closure, folds `lambda * to_sample_domain` once under the flat policy (the
per-cell weight is exactly 1.0 there, so `(lambda*tsd) * err²` is the same
product), and tests the cutoff once per row instead of after every cell. The
per-row cutoff is decision-preserving because both accumulators only grow (bits
by saturating adds, `weighted_sse` by non-negative finite terms) and a pruned
candidate is discarded whole — its partial sums and its `d_y_hf` scratch are
never read — so anything pruned at cell k is still pruned at the end of that
row and every survivor performs exactly the same additions in the same order.
The scalar fallback (run kernel error) keeps the per-cell order.

Measured with `perf stat` instruction counts, which are load-independent (this
host was carrying other jobs): 39.08 G → 37.60 G instructions per two-iteration
mid Balanced bench (−3.8%); the two `block_cost_bounded` scoring closures
(4.9% + 3.7% of instructions) became one `score_channel_lanes` symbol at 6.2%.
All canonical streams identical. `HfQuantizers::get` was left alone: with AQ off
its linear scan is over three entries.

## Phase 41 — the HF token tape: one walk per written plan (2026-08-17)

Each written plan (two anchor Counts and the finalist Store) walked its
coefficients twice: `build_entropy_tables` drove `walk_pass_group` into a raw
`TokenCensus` (two dense `vec![RawHistogram; 7425]` per worker, ~4 MB), built
the tables, and `write_pass_group` walked again into `SymbolEncoder` (re-
tokenizing every value, then collecting a `Vec<AnsSymbol>` for the backward
pass). `jpxl_entropy::encode::tape` now provides `TokenTapeRecorder` and
`TokenTape`: the writer's one walk per pass group tokenizes each event under
its cluster's configuration as it arrives, counts the token per cluster (the
same per-cluster token counts `EntropyTables::build` derives from a census —
`EntropyTables::build_from_token_counts` is that second half, split out), and
appends `(cluster u8, token u16, extra_bits u8, extra u32)` to a per-group tape.
After the tables are built the group's section replays the tape:
`encode_symbols_with` reads the tape's columns directly (no `Vec<AnsSymbol>`),
then the seed, renormalisation words and extra bits are written exactly as
`SymbolEncoder::write_stream` writes them. Behind the `hf-token-tape` feature
(default on); off, the two-walk path compiles as the oracle.

Bit-identity by construction (same triples in the same order, same integer
counts), pinned by `stream::tests::token_tape_matches_symbol_encoder` (tables
and bytes equal for clustered ANS and prefix plans across merged sections) and
by every canonical stream hash. Diagnostics: `tape_symbols` in the writer
phase lines — the mid photo records ~1.5 M tokens per plan (~12 MB of tape
resident per in-flight plan; ~35 MB on the 12 MP photo). Instruction count per
two-iteration mid Balanced bench: 37.66 G → 34.57 G (−8.2%; −11.5% since
Phase 39). Wall time on this shared host was too noisy to assert (load 7–12);
the writer's `count_ms`/`store_ms` moved down in most pairs and will be
re-measured on a quiet host. Walks per encode: 7 → 4 (policy census, and one
per written plan).

Not done: the policy-side training census (`CensusSink`, 7425 raw histograms)
still exists once per encode; sharing the finalist's tape with it is only
valid when the finalist plan shape equals the trained one and is deferred.

## Phase 41b — tape recorder fast path; Phase 42 (fixed-bracket anchors) deferred by measurement (2026-08-17)

Follow-through on the tape: `TokenTapeRecorder` now flattens `cluster_of` per
context and pairs each cluster's configuration with its `split` so the common
"value below the split is its own token" case is taken inline (what
`tokenize` returns there), and both `TokenTape::write_stream` backends skip
the zero-width extra-bit write. Instructions per two-iteration mid Balanced
bench 34.5 G → 33.3 G → 32.8 G (−5% on top of Phase 41; −16% since Phase 39).
Pinned wall in a moderately loaded window (load 2.5–6): mid 421/412/424 →
398/409/409 ms, large 869/961 → 840/899 ms; writer `store_ms` 35–38 → 28–30
ms on mid, `count_ms` 73–81 → 56–83 ms.

The plan's Phase 42 (a gated Contract-B experiment: choose both anchor rungs
before pricing so the two anchors quantize from one traversal) was checked
against the phase clock before building anything, per this document's own
workflow rule ("add the counter before changing the mechanism"). The second
anchor's whole plan is ~18 ms of the ~400 ms mid encode (`rate_plan_full`
plan_ms 37 for the second anchor plus the finalist, ~17 ms of it quantize),
and its Count is unaffected by batching. Even a perfect two-rung traversal at
1.3× the cost of one saves ≈ 12 ms (≈ 3%), while a bracket chosen without the
first anchor's exact size is a less informed predictor than today's
`second_anchor_rung` (which uses that size) and so risks extra corrections at
≥ 60 ms each and a changed finalist — a quality-screen phase for a ≤ 3%
ceiling with negative-expectation tails. It is deferred, not attempted; the
numbers are recorded here and in AKR so it is not re-derived.

## Phase 42 — filling the workers, second round: writer section map, cluster tables, training, bit writer (2026-08-17)

(The plan's Phase 42, a Contract-B anchor experiment, was deferred by
measurement — see Phase 41b; this phase reuses the number.) A per-thread
profile showed only ~1.7 of 4 pinned cores busy on average (jpxl-balanced
mid: 0.74 s CPU in 0.43 s wall). The 1- vs 4-thread phase clock located the
non-scaling phases: writer count/store (1.9×), entropy training (1.4×),
quantize (2.2×), and a temporary per-stage trace inside `write_frame_body`
found the causes: per written plan the LF-group sections (two on a 4 MP
frame, 8–14 ms each, on two threads), a serial `HfGlobal` (context-map form
probing, ~4 ms) and a serial per-cluster ANS table build (~7 ms for 173
clusters) each ran as their own barrier.

Changes, all Contract A:

- `write_frame_body`: LF-group sections, `HfGlobal` and every pass group go
  through **one** ordered map (results reduced in TOC order), so the heavy LF
  sections and the serial `HfGlobal` overlap the pass groups.
- `ordered_map_rayon` uses `with_max_len(1)`: one job per item, so a heavy
  prefix (the LF sections) is stolen item by item instead of staying on the
  worker that received the contiguous range — measured on the 12 MP frame,
  where the first combined map made `store_ms` worse before this.
- `EntropyTables::build_from_token_counts_with`: the per-cluster ANS
  histogram/alias-table construction runs through a caller-supplied ordered
  runner (`jpxl-entropy` takes no threading dependency); the writer passes the
  executor.
- `entropy::train_with_executor` / `train_fast_with_executor`: the
  per-context configuration search and the initial window-edge costs are
  ordered maps; the greedy merge is unchanged, and the queue is a total order
  on its tuples so push order cannot change the pop sequence.
- `BitWriter::write_bits`: fast path that ORs into the partial last byte and
  pushes whole bytes (the buffer always holds `ceil(bit_len/8)` bytes), with
  the byte-loop kept for any state off that invariant; pinned bit-for-bit
  against a `write_bit` reference.
- `FlatCode::write_token`: one `write_bits` of the bit-reversed token instead
  of `token_bits` single-bit writes; `write_modular_stream` computes the H.3
  gradient residuals row-wise (per-cell form kept as the debug oracle).

Verified byte-identical (all canonical streams incl. lossless, thread counts,
AVX2 on/off, tape on/off) and gates green. Effect on the mid photo (pinned,
moderately loaded host): writer `count_ms` 59–69 → 42–58, `store_ms` 27–31 →
17–24, `entropy_ms` 30–36 → 22–25; wall −4% to −6% per step in the interleaved
pairs (≈ 380–440 → 357–373 ms across the round). The 12 MP frame's wall was
too noisy in this window to assert (load spikes), its `count_ms` moved 118–141
→ 100–117.

**libjxl comparison after this pass** (process-to-process, pinned four
P-cores, cjxl v0.13.0 `-e 7` at the Phase 5G equal-SSIMULACRA2 distances
d = 2.25 mid / 1.25 large; three interleaved reps; a shared host so treat
absolutes as ±5%): mid 2400×1800 — jpxl Balanced 0.42–0.43 s (0.80 s CPU),
jpxl Fast 0.31–0.33 s, cjxl 0.46–0.51 s (1.25 s CPU); large 4000×3000 — jpxl
Balanced 0.89–1.13 s (quiet: 0.89–1.06), jpxl Fast 0.76–0.97 s, cjxl 1.24–1.66 s
(quiet: 1.24–1.44). Quality at those settings: SSIMULACRA2 72.38 (Balanced) /
69.46 (Fast) vs cjxl 72.20 on mid, 83.63 / 80.47 vs 83.70 on large;
Butteraugli 3.19 / 3.70 vs 3.08 (mid), 1.99 / 2.87 vs 1.45 (large). Density:
at that matched-SSIMULACRA2 point cjxl's files are ~13% smaller (467 KB vs
539 KB; 1,389 KB vs 1,496 KB) — jpxl is now faster in wall and uses ~40% less
CPU, but still trails libjxl in bytes at equal SSIMULACRA2 and on Butteraugli.


## Phase Q0 — quality track opened: harness, `--sections`, and the first attribution (2026-08-17)

Speed being where the plan wanted it, the track turned to density/quality
under a standing speed budget (Balanced pinned wall ≤ +15% cumulative vs the
Phase 42 binary). Q0 added the research controls `--dead-zone-scale` (a
multiplier on every HF cell's zero threshold; 1.0 is the exact nearest rule
and byte-identical), `--tolerance`, and `--sections` (per-`SectionKind` bytes
from `CodestreamSizing` after a lossy encode), and a standing harness under
`.agent/scratch/quality-track/` (`ladder.sh`: three photos × 0.5/1/2 bpp ×
Fast/Balanced/Quality, base vs candidate, djxl + jxl-oxide decodes, `jpxl
compare`; `scenes.sh`: the seven 1024×768 scenes at 1 bpp; `cjxl-match.sh`:
cjxl `-e 7` bisected to matched bytes; `summarise.py`: deltas, BD-SSIM2 /
BD-rate, verdict against the recorded bounds).

The very first `--sections` run answered the attribution question before any
ablation: on the mid photo at 1 bpp the **LF-group sections were 38% of the
stream** (205 of 539 KB; 567 of 1,496 KB on the 12 MP frame). A temporary
trace inside `write_modular_stream` measured the HF-metadata stream (CfL
tiles, `BlockInfo`, `Sharpness`) at 44.8 KB of flat-code tokens against a
6.4 KB order-0 entropy in the larger LF group — a 57,600-block `Sharpness`
plane of constant zeros was costing 3 bits a sample — and the three LF planes
at ~150 KB against ~114 KB order-0. Every control image went out under one
leaf, one flat prefix code sized by the largest residual in the stream. That
is Phase Q0b; the rest of Q0's ablation matrix (dead zone, quant_lf, QM
scales, EPF, tolerance) is superseded as attribution by that finding and moves
into Q1/Q2 as their own sweeps.

## Phase Q0b — entropy-coded LF-group control images (2026-08-17)

`vardct::modular_out::write_modular_stream` now writes a real modular coder for
VarDCT's control images:

- MA tree: a chain on property 0 (channel index) so each channel owns its
  contexts, then a bounded greedy learner per channel over the static Table
  H.4 neighbourhood properties (`|N|`, `|W|`, `W − property9(x−1,y)`, `W−NW`,
  `NW−N`, `N−NE`, `N−NN`, `W−WW`; never property 15) against a fixed
  threshold grid, priced by token entropy on a ~8k-sample row subsample,
  depth ≤ 3, one split penalty per new context. Gradient at every leaf.
- Entropy: one context per leaf, ANS through the token tape (`TokenCensus`
  → searched hybrid-uint configuration → `TokenTapeRecorder` →
  `EntropyTables::build_from_token_counts`), prefix codes tried on streams
  under 4,096 symbols.
- Property evaluation is this file's own, checked sample-exactly against
  `jpxl-decode` (`tests/control_image_roundtrip.rs`: LF triple, HF metadata
  set with a two-row `BlockInfo` and a constant `Sharpness`, a zero-size
  channel in the middle, wide residuals). H.4.1's previous-channel properties
  are implemented and round-trip through `jpxl-decode` and djxl, but are
  **off in production** (`USE_PREVIOUS_CHANNEL_PROPERTIES`): jxl-oxide 0.12.6
  reads them differently and fails its ANS final-state check on every stream
  whose tree uses one, and they were worth ~0.3% of the file.
- The rate loop's exact correction window grew from 4 to 6 slots: on the
  300×260 test frame the Full alternatives now shrink the FinalFast navigator
  price by 17–20% (the flat LF code used to pad every total), and the bracket
  needed one more interpolation to land inside 1%. Two `rate_loop` tests were
  restated: the trace test's "winner == best feasible Final" was only ever
  true because navigation and exact prices coincided (it now asserts ≤), and
  the LF/HF-ratio test's 8,000-byte target fell into the unreachable gap
  between the finest `global_scale` rung and the first `HfMul` rung on the
  cheaper frame (now 6,000).

Same decisions, same pixels, fewer bytes; under the target-rate presets the
loop hands the saved bytes to HF. At fixed decisions on the mid photo:
738,930 → 669,109 B (−9.4%; per-channel histograms alone reach 678,115, the
tree learner adds the rest). At 1 bpp the LF-group share dropped from 38% to
23% (mid: 205 → 123 KB, HF 328 → 411 KB).

Quality screen (`out/summary-q0b*.md`; base = Phase 42 binary): ladder
SSIMULACRA2 mean +7.2 (mid 1 bpp Balanced 72.4 → 76.7, large 83.6 → 86.1;
0.5 bpp cells +14 to +35), Butteraugli mean −19%; seven scenes mean +6.8 /
−13.5%, worst cell +2.9 / −1.2%; BD-rate −19% to −25% per image/preset;
djxl and jxl-oxide accept every stream. cjxl `-e 7` at matched bytes:
jpxl now leads on SSIMULACRA2 in all nine cells (mid 1 bpp 76.7 vs 75.7;
large 86.1 vs 84.9; mid2 86.1 vs 85.4) while cjxl keeps the lead on
Butteraugli (mid 1 bpp 2.76 vs 2.21; large 1.71 vs 1.42; mid2 1.56 vs 1.58) —
the red/green HF deficit Q2 targets. Cost: +9% instructions per Balanced
encode (writer ≈ 0.7 G of the 1.4 G delta on mid, the rest is the finer HF the
loop now buys), i.e. most of the track's speed budget is still available but
not all of it; the learner runs once per priced/stored plan per LF group
(three writes on Balanced) and could be hoisted into the plan later (~1%).

Follow-ups this opens: `quant_lf` 8 was promoted under the flat code's
mispriced LF residuals and must be re-screened (LF is still 23% of a 1 bpp
stream); LZ77 for the near-constant metadata channels; a per-channel
predictor choice for the LF planes.

## Phase Q1 — the quantizer under the harness: quant_lf 4 promoted; dead zone, λ and truncation price are flat (2026-08-17)

With the LF sections entropy-coded, the ledger's quantizer pointers were
swept on the harness (`sweep-q1*.sh`, `out/sweep-q1-summary.txt`; same
binary both arms, ladder at Fast + Balanced, three photos × three rates):

| knob | values | result |
|---|---|---|
| `quant_lf` (held, no LF fill) | 2 3 **4** 5 6 12 16 24 vs 8 | finer is worse everywhere (12: SSIM2 −0.43, 24: −1.5); coarser helps until it doesn't: 4 = SSIM2 +0.41 / worst −0.12, 3-norm Butteraugli −3.0%, 6 = +0.26, 3 = +0.28, 2 = −0.55 (0.5 bpp cells −1.4). Seven scenes: 4 = +0.67 / worst +0.34, Butteraugli −5.9%, 3-norm −3.3%. |
| `--dead-zone-scale` | 0.85 1.15 1.3 1.5 | < 1 is exactly a no-op (the nearest rule's candidate set already contains zero); > 1 trades SSIM2 down (−0.03 … −0.23) for a Butteraugli mean gain and a worst cell of −3.4 SSIM2 at 1.5. Not a lever. |
| `--lambda-scale` | 2 3 6 8 vs 4 | ±0.1–0.26 SSIM2, Butteraugli mean up with λ; the analytic ×4 sits at the flat optimum. |
| `--zero-token-bits` (new; Phase 7.1's interior-zero price) | 0.5 1.5 2 3 vs 1 | ±0.05 SSIM2 across a 6× range: the trailing-truncation pass is insensitive to its price, so 7.1a's real-cost model would be built on a dead lever and is not funded. |

Promoted: **`quant_lf` 4** as the target-rate default (`--quant-lf 8` reaches
Phase 5G's value; LF fill remains off under `for_target`). Its promotion
screen against the Q0b binary is `out/summary-q1.md`. The Butteraugli max-norm
turned out to swing +8% to +18% in single cells on settings whose SSIMULACRA2,
RMSE and 3-norm were neutral or better, so the contract's worst-cell
Butteraugli bound now reads on the 3-norm (`butteraugli_pnorm3` ≤ +5%), the
max-norm mean stays bounded and its worst cell is reported. quant_lf 4's one
max-norm outlier (mid 2 bpp Fast, +7.6%) has SSIM2 +0.26, RMSE and 3-norm
better.

Speed: neutral within noise (LF planes cost fewer bytes at the coarser
quantizer; the writer's learner sees the same sample counts).

## Phase Q2 — selective low-rate B-channel HF allocation (2026-08-18)

The Q1 harness was extended to sweep X/B quantization-matrix scales, EPF, and
effort boundaries. A global B scale of 5 improved the 27-cell photo ladder by
+0.605 mean SSIMULACRA2 but failed the seven-scene screen; adding EPF 2 did
not repair it. X=3/B=3 passed scenes narrowly but exceeded the photo
Butteraugli mean bound. Restricting B=5 to Quality improved both corpora but
missed one 2 bpp Butteraugli 3-norm cell (+6.06% against the +5% bound).

Promoted: **B scale 5 only for the Quality preset at targets no greater than
1 bpp**. Fast, Balanced, and Quality above 1 bpp remain byte-identical to Q1;
an explicit `--b-qm-scale` always bypasses the automatic policy. The exact
48-cell composite screen (three photos and seven scenes, with both `djxl` and
`jxl-oxide` decoding every changed stream) passed:

| corpus | SSIMULACRA2 mean / worst | Butteraugli max mean / worst | Butteraugli 3-norm mean / worst |
|---|---:|---:|---:|
| photos (27 cells) | +0.149 / +0.000 | -0.22% / +4.17% | +0.33% / +2.88% |
| scenes (21 cells) | +0.022 / -0.269 | +0.40% / +15.19% | +0.50% / +2.78% |

Quality BD-rate moved -4.2%, -3.7%, and -1.3% on the three photos. Alternating
native Windows A/B timing on the mid photo measured the candidate about 1.5%
faster (noise-level), so Q2 adds no measurable encode cost. The Windows
PowerShell harness produced the same stream hashes as the earlier Linux/WSL
commands while avoiding WSL path-translation overhead; perceptual metric
calculation and the two independent decodes remain the dominant runtime.

## Phase Q3 — the rate-ladder ceiling, and where the Butteraugli deficit actually sits (2026-08-18)

Q3 opened by asking the standing harness the pass's own question: how far is
the Q2 encoder from `cjxl -e 7` at matched bytes on *every* metric, not just
SSIMULACRA2? Answering it first exposed a rate-control defect. Above the
`global_scale` ceiling the ladder stepped by whole `HfMul` multiples, and
`HfMul` 2 at the ceiling is a full octave finer than `HfMul` 1 — a 65% byte
jump on the smooth mid2 photo — so every 2 bpp target between the two rungs
landed 35% under ("budget spent, not at the ladder's limit"); large at 2 bpp
landed 27% under. The `JPXL_RATE_TRACE=1` dump added to the CLI shows it:

```
Bisect rung=73726 scale=73727 hf_mul=1  bytes=703405  feasible=true
Bisect rung=73728 scale=73728 hf_mul=2  bytes=1163964 feasible=false   (target 1,080,000)
```

**Promoted: the dense upper ladder.** Segment `k` (2..=65) walks
`global_scale` from `floor((k-1)*MAX/k)+1` to `MAX` with `HfMul = k`, so
consecutive rungs differ by `k` units of effective scale and the top stays
`65 * MAX`. `quant_lf` is coupled to the segment (`base * j`, `j <= k`, while
that stays at or below 16) so the LF/HF balance carries through the first
segments; the bound is the `MAX * 16` product the fixed-quantizer defaults have
always reached, because jxl-oxide 0.12.6 narrows `LfQuant` to signed 16 bits
and a synthetic high-contrast oracle fixture wraps at twice that. Search paths
that recover a `quant_lf` from a `QuantizerChoice` now carry the request's base
value (feeding the coupled wire value back coupled it twice; the Phase 7
truncation test, which had been pinned to one rung only by the old cliff, now
pins its quantizer explicitly).

Result against the Phase Q2 outputs (27 photo cells, three presets): 17 cells
byte-identical; the ten ceiling-bound cells all improve — mid2 2 bpp
698 KB → 1,080 KB (+2.84 SSIMULACRA2, −31% Butteraugli), large 2 bpp
2.18 MB → 3.00 MB (+1.6…+2.1, −18…−22%), mid 2 bpp and mid2 1 bpp within
tolerance of target with +0.07…+0.12; ladder mean +0.528, worst −0.008,
Butteraugli max −5.9% mean / worst +0.03%; the 14 scene cells are identical.
One-thread and four-thread outputs are byte-identical on every changed cell;
`djxl` and `jxl-oxide` accept all of them. Timing on the standing mid 1 bpp
Balanced cell: 423 ms vs 428 ms for the frozen Phase Q1 binary (alternating
4×3), so the cumulative quality-track cost stays at Q0b's ~+9%. Balanced at
mid2 2 bpp is 5.9 s (was 5.2 s and undershooting) because the two-anchor
controller's second anchor extrapolates with exponent 2 where the measured
exponent is 0.95–1.8 and falls back to the exhaustive controller — recorded
as `jpegxl-rs.observation.q3-balanced-second-anchor-overshoots-above-the-ceiling-2026-08-18`
for the next pass.

**Where JPXL stands against `cjxl -e 7` at matched bytes (Balanced, Q3 head):**

| image | bpp | bytes JPXL / cjxl | PSNR JPXL / cjxl (Δ dB) | SSIMULACRA2 JPXL / cjxl (Δ) | Butteraugli max JPXL / cjxl (Δ%) | 3-norm JPXL / cjxl (Δ%) |
|---|---:|---:|---:|---:|---:|---:|
| mid-photo | 0.5 | 266,642 / 266,650 | 31.58 / 31.62 (-0.04) | 56.89 / 54.51 (+2.38) | 5.188 / 5.121 (+1.3%) | 1.7172 / 1.7258 (-0.5%) |
| mid-photo | 1 | 539,694 / 539,684 | 35.38 / 35.51 (-0.13) | 77.18 / 75.68 (+1.51) | 2.783 / 2.228 (+24.9%) | 0.8716 / 0.8387 (+3.9%) |
| mid-photo | 2 | 1,077,076 / 1,077,129 | 39.31 / 39.43 (-0.12) | 88.77 / 87.53 (+1.24) | 1.292 / 0.897 (+44.1%) | 0.4069 / 0.3562 (+14.2%) |
| large-photo | 0.5 | 748,380 / 748,369 | 35.57 / 35.81 (-0.23) | 72.87 / 71.48 (+1.39) | 3.436 / 3.116 (+10.3%) | 1.0539 / 1.0129 (+4.0%) |
| large-photo | 1 | 1,495,861 / 1,495,703 | 38.95 / 39.15 (-0.20) | 86.38 / 84.88 (+1.50) | 1.632 / 1.419 (+15.0%) | 0.5194 / 0.4888 (+6.3%) |
| large-photo | 2 | 2,998,909 / 2,999,060 | 42.53 / 42.93 (-0.40) | 92.54 / 92.14 (+0.40) | 0.823 / 0.669 (+22.9%) | 0.2721 / 0.2456 (+10.8%) |
| mid2-photo | 0.5 | 269,873 / 269,949 | 40.64 / 40.78 (-0.15) | 80.06 / 78.66 (+1.40) | 2.259 / 2.161 (+4.5%) | 0.9959 / 0.9974 (-0.2%) |
| mid2-photo | 1 | 533,843 / 533,687 | 42.71 / 43.24 (-0.53) | 86.01 / 85.44 (+0.56) | 1.551 / 1.495 (+3.7%) | 0.7378 / 0.7027 (+5.0%) |
| mid2-photo | 2 | 1,079,753 / 1,080,066 | 46.85 / 46.81 (+0.04) | 91.13 / 90.46 (+0.67) | 0.923 / 0.916 (+0.7%) | 0.4231 / 0.4098 (+3.2%) |

SSIMULACRA2 parity is met and exceeded in 9/9 cells; PSNR trails in 8/9
(0.04–0.53 dB), Butteraugli max-norm in 9/9 (0.7–44%), 3-norm in 7/9 (up to
14%). "Matched libjxl across the board" is therefore **not** met: the
remaining gap is Butteraugli (and a small PSNR gap), largest on the busy mid
photo at 1–2 bpp, smallest on the smooth mid2 photo.

**Localisation.** A per-tile Butteraugli diffmap comparison
(`.agent/scratch/q3/bdiff`) shows the deficit is not in flat regions and not
in the busiest texture but in low-to-mid activity blocks (JPXL/cjxl 3-norm
ratio by variance quintile 1.00 / 1.06 / 1.05 / 1.02 / 0.96), and its worst
cases are DCT8×8 blocks that mix a strong edge with flat content, where JPXL
leaves ±8…14 luma error on the flat side against cjxl's ±2. A DCT8-only cover
still shows the same hot spots (max 2.45 vs 2.78) while gaining 0.03 dB PSNR
and losing 0.37 SSIMULACRA2 at equal bytes: the merge decisions are the one
lever that moves PSNR and Butteraugli together.

**Honest negatives** (all retained as research controls, byte-identical when
off; full tables in
`docs/experiments/2026-08-18-q3-fine-lattice-aq-and-adaptive-epf.md`):

* Fine-lattice adaptive quantization (`--aq-mode fine-masking|fine-uniform|
  edge-refine`, baseline `HfMul` 16, 3×3 erosion, edge dead zone): every
  strength of every direction loses SSIMULACRA2 and Butteraugli 3-norm on the
  mid photo at 1 bpp; the `mul` plane alone costs 1–3% of the file. This is a
  stronger negative than Phases 4J/5A, whose lattice rounded any coarsening
  to a full octave.
* Activity-adaptive EPF sharpness (`--epf-sharpness adaptive`): worse
  everywhere; uniform sharpness 7 is doing real work in busy content.
* `--cover-size-penalty measured` (max-norm −5% on the mid cell) and
  `--x-qm-scale 3` are neutral-to-negative on the corpus (SSIMULACRA2 −0.037 /
  +0.021 photos, −0.134 / −0.007 scenes; 3-norm +0.4…+0.7%). Not promoted.

**Next.** The cover objective's rate proxy (bit length of nonzeros, zeros
free) is the natural target for the merge-decision lever; the two-anchor
second-anchor exponent is the speed follow-up at high rates.

## Phase Q4 — pricing the cover objective's rate the way the writer spends it (2026-08-18)

Q3 left one lever that moved PSNR and Butteraugli in the same direction as
SSIMULACRA2 did not: the cover's merge decisions. Q4 asked the writer what a
varblock actually costs. `tests/rate_proxy_audit.rs` (an ignored measurement
harness, `JPXL_RATE_AUDIT_PPM=<ppm>`) walks the chosen plan of a real
target-rate encode with a costing `HfEventSink` — the new no-op
`varblock(transform, hf_mul)` hook on the trait attributes each I.4 event to
its transform — and prices every `non_zeros` symbol and coefficient token at
`-log2 p(token | cluster)` plus its hybrid-uint extra bits under the plan's
own trained histograms. Against the shipped proxy (`bitlen(|q|) + 1` per
nonzero, zeros free, 2 bits per varblock, 32 per merged transform), 1 bpp
Balanced:

| photo | side | varblocks | actual / proxy | `non_zeros` bits/sym | nonzero bits/tok | zero bits/tok | zeros per vb |
|---|---:|---:|---:|---:|---:|---:|---:|
| mid | 8 | 28,268 | 1.63 | 1.84 | 3.54 | 1.07 | 8.8 |
| mid | 16 | 6,432 | 1.37 | 2.81 | 3.76 | 0.78 | 49.0 |
| mid | 32 | 844 | 1.27 | 2.17 | 3.57 | 0.55 | 45.2 |
| large | 8 | 38,976 | 1.60 | 2.35 | 3.84 | 1.11 | 8.8 |
| large | 16 | 18,767 | 1.35 | 3.17 | 4.10 | 0.81 | 42.3 |
| large | 32 | 4,591 | 1.53 | 3.27 | 3.75 | 0.60 | 105.9 |
| mid2 | 8 | 12,356 | 1.89 | 1.72 | 3.32 | 0.87 | 18.9 |
| mid2 | 16 | 6,126 | 1.60 | 2.58 | 3.42 | 0.68 | 106.2 |
| mid2 | 32 | 1,915 | 1.78 | 3.69 | 3.26 | 0.40 | 289.3 |

Interior zeros are not free (0.4–1.1 bits each), a nonzero token costs
3.3–4.1 bits rather than `bitlen + 1`, the three `non_zeros` symbols cost
5–11 bits per varblock rather than 2, and the `DctSelect` signalling of a
merged block is a few bits, not 32. Net of the per-varblock constants the
residual proxy under-prices DCT8x8 / DCT16x16 / DCT32x32 by 1.56× / 1.63× /
1.88× on the mid photo (1.52 / 1.58 / 1.76 large, 1.84 / 1.89 / 2.0 mid2):
the ordering is stable, so the legacy proxy over-charges merging through its
32-bit constant and under-charges large transforms per coefficient.

**Promoted: `CoverRateModel::Calibrated`** for target-rate requests
(`--cover-rate-model legacy` reproduces the Phase Q3 streams; the
fixed-quantizer defaults keep the legacy proxy). It scales a candidate's
residual bits by 1.56 / 1.63 / 1.88 and charges 5.5 / 12.4 / 10.5 fixed bits
per varblock; the pruning arithmetic scales with it, so `Legacy` is
bit-identical. On the mid photo at 1 bpp the cover moves from 42/38/20% of
the area in DCT8/16/32 to 33/52/16% — more 16×16 merges, fewer 32×32 — and
the standing gate against Phase Q3:

| corpus | SSIMULACRA2 mean / worst | Butteraugli max mean / worst | 3-norm mean / worst | bytes |
|---|---:|---:|---:|---:|
| photos (27 cells) | +0.048 / −0.19 | +0.04% / +6.26% | −0.15% / +3.22% | −0.06% |
| scenes (14 cells) | +0.059 / −0.05 | +0.61% / +2.52% | −0.13% / +0.08% | −0.13% |

Every stream decodes in `djxl` and `jxl-oxide`; one- and four-thread outputs
are byte-identical; alternating A/B timing is neutral (mid 1 bpp 470 vs
478 ms, large 1 bpp 889 vs 887 ms). Fast cells are unchanged (fixed cover).
A "relative" variant that keeps only the size ordering (1.0 / 1.045 / 1.2 with
scaled constants) was SSIMULACRA2 +0.036 / +0.083 but Butteraugli max +0.50% /
+1.22% — the absolute scale matters, because it also lowers the cover's
effective λ by 1.56× against the residual proxy Phase 7.2's λ×4 was tuned on.
This is a small win on every metric at once, not a seesaw: the first change
since Q1 that improves SSIMULACRA2 and Butteraugli 3-norm together.

**Standing against `cjxl -e 7` at matched bytes (Balanced, Q4 head; cjxl
points from the Q3 match, JPXL bytes within 0.2%):**

| image | bpp | bytes JPXL / cjxl | PSNR JPXL / cjxl (Δ dB) | SSIMULACRA2 JPXL / cjxl (Δ) | Butteraugli max JPXL / cjxl (Δ%) | 3-norm JPXL / cjxl (Δ%) |
|---|---:|---:|---:|---:|---:|---:|
| mid-photo | 0.5 | 266,625 / 266,650 | 31.58 / 31.62 (-0.03) | 57.05 / 54.51 (+2.55) | 4.833 / 5.121 (-5.6%) | 1.7123 / 1.7258 (-0.8%) |
| mid-photo | 1 | 539,680 / 539,684 | 35.38 / 35.51 (-0.13) | 77.31 / 75.68 (+1.63) | 2.739 / 2.228 (+22.9%) | 0.8701 / 0.8387 (+3.7%) |
| mid-photo | 2 | 1,077,350 / 1,077,129 | 39.27 / 39.43 (-0.16) | 88.77 / 87.53 (+1.24) | 1.286 / 0.897 (+43.4%) | 0.4070 / 0.3562 (+14.3%) |
| large-photo | 0.5 | 749,731 / 748,369 | 35.58 / 35.81 (-0.23) | 73.09 / 71.48 (+1.61) | 3.440 / 3.116 (+10.4%) | 1.0498 / 1.0129 (+3.6%) |
| large-photo | 1 | 1,495,833 / 1,495,703 | 38.91 / 39.15 (-0.24) | 86.35 / 84.88 (+1.47) | 1.643 / 1.419 (+15.8%) | 0.5195 / 0.4888 (+6.3%) |
| large-photo | 2 | 2,998,384 / 2,999,060 | 42.49 / 42.93 (-0.44) | 92.59 / 92.14 (+0.44) | 0.874 / 0.669 (+30.6%) | 0.2673 / 0.2456 (+8.8%) |
| mid2-photo | 0.5 | 269,767 / 269,949 | 40.62 / 40.78 (-0.16) | 80.14 / 78.66 (+1.49) | 2.157 / 2.161 (-0.2%) | 0.9925 / 0.9974 (-0.5%) |
| mid2-photo | 1 | 533,688 / 533,687 | 42.70 / 43.24 (-0.54) | 86.02 / 85.44 (+0.58) | 1.611 / 1.495 (+7.7%) | 0.7347 / 0.7027 (+4.6%) |
| mid2-photo | 2 | 1,079,897 / 1,080,066 | 46.85 / 46.81 (+0.04) | 91.14 / 90.46 (+0.68) | 0.922 / 0.916 (+0.6%) | 0.4222 / 0.4098 (+3.0%) |

SSIMULACRA2 ahead in 9/9 (+0.4 … +2.6); PSNR behind in 8/9 (0.03–0.54 dB);
Butteraugli max-norm behind in 7/9 (mid 0.5 and mid2 0.5 now ahead), 3-norm
behind in 7/9. The picture is the same as after Q3, a notch better on the
low-rate mid cells; the busy mid photo at 1–2 bpp remains the largest gap.

**Why the metrics diverge this much.** SSIMULACRA2 is a multi-scale
structural-similarity score: it averages, over the whole image and over
scales, how well local means, contrasts and structural correlations are
preserved, so it rewards an encoder that keeps texture and edges
statistically right everywhere and it forgives a moderate, localised error.
Butteraugli is a psychovisual difference model reported as a *max-norm*
(and, in the 3-norm, a high-power mean): a single 16×16 patch with a visible
error sets the score, and its masking model punishes error next to flat
content far more than error inside texture. PSNR is plain mean-square error
and rewards nothing perceptual. Two encoders can therefore rank oppositely
without either being "wrong": JPXL's target-rate policy was promoted
SSIMULACRA2-first (Phases 5G–Q2: quant_lf 4, the quant-donor frequency
weight, trailing truncation at λ×4, uniform EPF 7, hierarchical merges), each
of which improves the average structural score while accepting occasional
localised errors — exactly what Q3 found at edge/flat DCT8×8 blocks — and
several of which (truncation, the perceptual frequency weight, merges) trade
MSE for structure, which is the PSNR gap. `cjxl` targets Butteraugli distance
directly, so at matched bytes it holds the local worst case down at the price
of the average structural score. The two encoders sit at different points of
the same tradeoff surface; "matching libjxl across the board" would mean
finding changes that move the local worst case without giving back the
average — Q4's rate calibration is one such change, small; the per-block
allocation levers of Q3 were not.

**Next.** (1) The two-anchor controller above the ceiling (screened in Phase
Q5 below). (2) A zero-run-aware proxy inside the scoring kernel (interior
zeros priced at their coding-order position) would replace the per-size
averages with a per-candidate count — the audit's per-varblock residual
spread (p10…p90 of −0.5…+0.3 around the size fit for DCT8x8) is the size of
the prize; the audit harness is the tool to check whether it is worth the
kernel cost.

## Phase Q5 — the anchored controller above the ceiling: a speed fix that exposes a hidden quality tier (2026-08-18)

The Q3 follow-up: Balanced at mid2 2 bpp took 5.9 s (0.42 s at 1 bpp) because
the two-anchor controller's finalist and its one correction missed the band
and the search fell back to the exhaustive controller. Three changes were
screened together, each byte-identical when set to its legacy value:

* second-anchor exponent 1.5 instead of 2.0 (measured `1/alpha` is 1.57–1.80
  on the busy photos, 0.95–1.05 on the smooth one);
* rebuilding the structure (cover, CfL, entropy model) at the second anchor
  when the first anchor priced more than 2× from the target;
* a second exact correction aimed with the *local* slope between the two
  exact points already priced.

Only the second correction fixed the fallback: mid2 2 bpp Balanced went from
5.9 s to 0.78 s (2 fast + 3 full prices) and the 20240503_105759 scene from
23–25 prices to 4. A fresh structural finalist was also tried (`AnchorReuse::
None` at the predicted rung): +0.10 / −0.03 SSIMULACRA2 on the standing 1 bpp
cells for +6–22% time, and it made the anchors' curve a worse predictor of
the finalist's bytes (mid2 at 1 bpp fell back), so it was dropped.

The corpus arm against Q4: photos SSIMULACRA2 +0.021 (worst −0.25), 3-norm
+0.03%; scenes **−0.272 mean, worst −4.04**. The worst cells are exactly the
ones that used to fall back. The exhaustive fallback re-plans cover, CfL and
entropy at every probe, so those cells had been receiving Quality-tier output
under a Fast/Balanced label (the 20240503_105759 scene at Fast: 44.5 with the
fallback, 40.5 on the genuine Fast tier at the same bytes; mid2 2 bpp
Balanced: 91.14 vs 90.89). Removing the fallback does not degrade the encoder,
it reveals the tier — but it fails the Contract B bound on the scene, and the
5.9 s cell predates this pass. Under the quality-first rule the defaults stay
at the legacy behaviour; the mechanism (bounded correction loop, rebuild
threshold, exponent) is kept as constants next to the screen's numbers.

The clean fix is a cheaper *fresh-structure* fallback seeded from the anchored
points (the exhaustive controller currently restarts its geometric bracket
from scratch, ~20 prices), which would keep the hidden quality on those cells
at a fraction of the cost. Recorded as the next controller item.

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
