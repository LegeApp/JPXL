# Rust JPEG XL Encoder: Function-by-Function Parity Roadmap

Status: revised execution plan, 2026-07-31  
Applies to: `jxl-encoder/`  
Reference: `libjxl/` and its `cjxl` encoder  
Primary objective: meet or exceed libjxl's compression quality and encoding
speed by profiling, comparing, and optimizing equivalent functions one at a
time.

## 1. Corrected direction

This is not primarily a perceptual-tuning project. The encoder already contains
substantial perceptual work, content dispatch, experimental metrics, and a large
body of parameter sweeps. More tuning may produce isolated wins, but prior work
here and in `bpg-rs` shows that it is difficult, expensive to validate, and
unlikely to close more than a modest final gap.

The main program is therefore:

1. Build matched Rust and libjxl reference binaries.
2. Flamegraph the same encode in both binaries.
3. Create a paired ledger of equivalent Rust and C++ functions.
4. Separate excess call count from excess cost per call.
5. Optimize the largest measured discrepancy while preserving decisions and
   output whenever possible.
6. Run interleaved A/B measurements and correctness gates.
7. Promote a win or revert a miss.
8. Re-profile and repeat.

This is the method that brought `bpg-rs` from a large performance gap to roughly
speed parity. It is also the method behind the largest already-documented JPEG
XL speed corrections.

Perceptual tuning is a bounded secondary workstream. It should consume no more
than roughly 10% of optimization effort until speed parity and libjxl
decision-path parity are established.

## 2. What the existing documents actually establish

The existing JPEG XL documents contain the right evidence, but their top-level
direction is inconsistent.

- `GOAL_BEAT_CJXL.md` correctly calls wall time the long pole and says to port
  what cjxl does first.
- `LIBJXL_DIVERGENCES.md` and `CODE-HISTORY.md` record decisive profile-driven
  wins:
  - a wrong tree-learning effort gate consumed 78.6% of CPU on one e5 cell;
  - using 14 predictors where libjxl used 2 consumed about 47% of CPU on an e8
    cell;
  - correcting that predictor path reduced wall time from 5.46 s to 2.08 s;
  - a proposed strip-stage optimization could save at most 3.3% even with
    infinite local speedup, an Amdahl-bound reason not to prioritize it.
- The same documents also contain a very large number of content gates,
  parameter sweeps, metric experiments, speculative research branches, and
  retained opt-in scaffolds. Those are not a substitute for a complete
  Rust-versus-C++ cost ledger.
- `JXL_ENCODER_LEARNINGS.md` is explicitly an open research addendum. Its
  proposals should not drive the parity program unless a current profile first
  identifies the relevant function as important.

The BPG work provides the clearer governing lesson:

- compare calls per unit of input and time per call;
- do not assume fewer calls means faster;
- do not cut search when the real gap is implementation cost;
- verify assumptions against reference source;
- prefer byte-identical changes;
- interleave measurements because machine load can reverse an apparent result;
- retain negative results in a concise ledger and revert their code;
- optimize the dominant leaf function, then re-profile because the bottleneck
  moves.

## 3. Definition of success

### 3.1 Core correctness

Every promoted production change must:

- produce valid JPEG XL decoded by `djxl`, jxl-rs, and jxl-oxide;
- remain pixel-exact in lossless modes;
- preserve JPEG reconstruction exactly where advertised;
- remain deterministic for a fixed configuration and thread count;
- pass all relevant hash locks, round-trip tests, and conformance tests;
- reject unsupported input combinations without corrupt output;
- account for alpha and extra-channel behavior in the tested cell.

### 3.2 Speed targets

Measure warm-process and cold-process results separately. The main optimization
target is warm encoding wall time; startup and CLI overhead get a separate
ledger.

Staged targets at matched effort, threads, input, and build quality:

| Stage | 1-thread geometric mean | 8-thread geometric mean | Worst core cell |
|---|---:|---:|---:|
| Floor | <= 1.30x libjxl | <= 1.75x | <= 2.00x |
| Competitive | <= 1.10x | <= 1.25x | <= 1.35x |
| Parity | <= 1.00x | <= 1.05x | <= 1.15x |
| Exceed | <= 0.95x | <= 0.95x | <= 1.05x |

Single-thread parity comes before a major parallel redesign. Parallel work can
hide per-call inefficiency, complicate profiles, and increase memory use.

### 3.3 Compression-quality targets

At every production effort:

- no core cell may lose both size and decoded quality outside declared noise
  bands;
- lossless is judged first by exact reconstruction and then by bytes;
- lossy comparisons use matched decoded pixels and at least Butteraugli plus
  SSIMULACRA2;
- metric disagreement remains a mixed result, not a win;
- comparisons are per cell and per quality band, not only corpus averages.

Parity target:

- zero calibrated `CJXL_DOMINATES` cells in the core matrix;
- geometric-mean size at matched quality no worse than libjxl;
- no content family with a material regression.

Exceed target:

- at least 60% of core cells are strict Pareto wins;
- at least 2% lower geometric-mean bytes at matched quality;
- no material speed regression used to purchase that size improvement.

Quality parity is first pursued through function and decision parity with
libjxl: the same inputs to a decision, the same candidates, the same cost terms,
and the same selected result. New perceptual heuristics come later.

## 4. Measurement contract

No speed claim is accepted without the following.

### 4.1 Matched builds

Build both encoders with:

- release optimization and LTO;
- the same native CPU feature policy;
- symbols sufficient for call-stack attribution;
- assertions and diagnostic instrumentation either disabled in both timed
  builds or accounted for;
- recorded compiler, linker, allocator, source revision, and binary hash;
- equivalent thread count and effort semantics.

Do not compare a native, LTO Rust binary to a generic or debug-instrumented
libjxl binary, or vice versa.

### 4.2 Stable wall measurements

- Use an idle host with the power and frequency policy recorded.
- Pin the one-thread tests to an appropriate physical core.
- Interleave A/B runs: Rust, C++, Rust, C++, rather than two separate batches.
- Use at least five runs for short cells and three for long cells.
- Report minimum, median, and dispersion.
- Record temperature, load, worker count, peak RSS, output bytes, and output
  hash.
- Keep cold-start, warm-start, and library-only timings separate.

### 4.3 Profile artifacts

For every canonical cell, retain:

- a Rust flamegraph;
- a libjxl flamegraph;
- folded stacks or raw `perf.data` needed to regenerate each graph;
- inclusive and exclusive sample tables;
- internal phase and function counters;
- output hashes and correctness results;
- benchmark metadata sufficient to repeat the run.

Use `perf record --call-graph dwarf`, `samply`, or an equivalent sampling
profiler. If system profiling is unavailable, use feature-gated internal timers,
call counters, and hardware counters where accessible. Internal timing must be
calibrated for its own overhead and disabled in normal releases.

### 4.4 Canonical profile cells

Start with a small set that represents distinct pipelines:

1. 4 MP SDR photo, lossy, effort 5, 1 thread.
2. 4 MP SDR photo, lossy, effort 7, 1 thread.
3. Screenshot or line art, lossy, effort 8, 1 thread.
4. 4 MP 8-bit photo, lossless, effort 5, 1 thread.
5. Structured document, lossless, effort 7, 1 thread.
6. 16-bit image, lossless, effort 5, 1 thread.
7. HDR gradient/photo, lossy, effort 7, 1 thread.
8. The same large lossy and lossless cells at 8 threads.
9. A 64x64 and a 256x256 cell for fixed overhead.

Do not begin with the full quality matrix. First obtain deep, repeatable profiles
on representative cells; use the larger matrix to validate promoted changes.

## 5. The paired function ledger

The ledger is the central artifact of this project. Each row represents a Rust
function or tightly coupled function group and its closest libjxl counterpart.

Required columns:

| Field | Meaning |
|---|---|
| Cell and effort | Exact benchmark where the row was measured |
| Pipeline stage | Modular, VarDCT, shared, container, or scheduling |
| Rust symbol | File and function |
| libjxl symbol | File and function |
| Inclusive time | Total cost including callees |
| Exclusive time | Cost in the function body |
| Calls | Invocations per image, group, block, token, or pixel |
| Work units | Pixels, blocks, candidates, coefficients, or tokens processed |
| ns/call and ns/unit | Implementation cost independent of call count |
| Rust/C++ call ratio | Detects excess or missing work |
| Rust/C++ unit-cost ratio | Detects implementation inefficiency |
| Allocations and bytes | Allocation pressure and traffic |
| Branch/cache/vector data | When hardware counters are available |
| Decision class | Byte-identical, decision-neutral, or decision-affecting |
| Output result | Hash, bytes, quality, decode status |
| Status | Open, testing, promoted, ruled out, or superseded |

Every hot row must be diagnosed as one or more of:

1. **Call-count gap**: Rust performs more candidates, passes, or reconstructions.
2. **Per-call gap**: equivalent work is more expensive in Rust.
3. **Different algorithm or effort gate**: the functions are not actually
   equivalent.
4. **Parallel/scheduling gap**: equivalent single-thread work, poor scaling.
5. **Fixed overhead**: setup, allocation, headers, or process cost dominates.

Do not optimize until this classification is supported by measurements and
reference-source inspection.

## 6. Promotion classes

### 6.1 Byte-identical optimization

Preferred class. The encoded bytes match the baseline exactly.

Examples:

- eliminate duplicate calculation;
- cache and reuse an already-final value;
- fuse loops without changing arithmetic order where output depends on it;
- specialize a common type or size;
- remove allocation and copying;
- improve lookup layout;
- add bit-exact SIMD;
- correct an effort gate to match libjxl.

Promotion gate:

- output hashes match on the focused cell and the full lock matrix;
- at least 0.5% end-to-end wall improvement, or at least 3% in a stage that
  accounts for 10% or more of wall time;
- no measurable regression on another core cell;
- code and maintenance cost are proportionate to the win.

### 6.2 Decision-neutral optimization

Output bytes may differ, but selected codec decisions and decoded pixels or
quality are equivalent within a very tight declared band.

Promotion requires the full rate/distortion matrix and an explanation for the
byte difference.

### 6.3 Decision-affecting change

Changes search, quantization, predictor choice, entropy decisions, or perceptual
allocation.

This is quality work, not a free speed optimization. It requires:

- decision-diff evidence showing what changed;
- matched-quality size results;
- full multi-content and multi-effort validation;
- wall cost included in the result;
- a separate commit and ledger entry from mechanical optimizations.

Do not combine these three classes in one benchmark patch.

## 7. The optimization loop

Use this exact loop for each performance change.

1. Select the hottest remaining ledger row by weighted end-to-end opportunity.
2. Inspect the Rust and libjxl call paths and source side by side.
3. Confirm that both functions receive comparable work.
4. Measure calls, work units, and unit cost.
5. State one falsifiable hypothesis.
6. Build the smallest gated implementation that tests it.
7. Verify local function output before timing.
8. Run interleaved single-thread A/B.
9. Run correctness and output-class gates.
10. Validate on at least one cell from each affected content family.
11. Promote or revert.
12. Update the negative-results ledger.
13. Re-profile the whole encode.

A microbenchmark win is insufficient. The final promotion metric is end-to-end
wall time because inlining, cache behavior, allocator effects, and call-site
frequency can erase a kernel win.

## 8. Phase 0: Minimal build and benchmark repair

This phase is deliberately narrow. It exists only to make profiling trustworthy.

Deliverables:

- make `cargo metadata`, release build, test, clippy, and package work from this
  project without undisclosed sibling repositories;
- pin the exact libjxl revision and build recipe;
- produce a one-command paired benchmark for one Rust and one cjxl encode;
- record mandatory decoder paths and fail rather than silently skipping them;
- correct only documentation claims that would invalidate benchmark selection;
- fix the premultiplied-alpha `Auto` bug before alpha cells are measured.

Exit criteria:

- a clean clone can reproduce both binaries and one paired row;
- the benchmark emits machine-readable timing, bytes, hash, RSS, and revisions;
- the same output passes all required decoders.

This phase must not expand into general API cleanup or feature completion.

## 9. Phase 1: Flamegraphs and the first complete ledger

### 9.1 Generate profiles before more optimization

Produce Rust and libjxl flamegraphs for every canonical profile cell. Start at
one thread. For the two large cells, also profile 8 threads with per-thread
stacks and blocked/runnable time.

### 9.2 Instrument work counts

Sampling profiles show where time lands but not why. Add feature-gated counters
for:

- image, group, block, and transform visits;
- candidate evaluations per search;
- predictor trials;
- tree split and cost evaluations;
- reconstruction passes;
- perceptual-loop iterations;
- coefficients quantized and tokens emitted;
- histogram and clustering operations;
- ANS/prefix symbols encoded;
- allocations, reallocations, and copied bytes;
- task counts, queue waits, steals, and ordered-commit waits.

Counters must use low-overhead thread-local accumulation and merge after the
timed region.

### 9.3 Pair equivalent functions

Build the first ledger by tracing each major Rust stack into the corresponding
libjxl source. Source inspection is mandatory; names alone do not establish
equivalence.

The first pass should cover at least 90% of sampled one-thread CPU time. Unknown
samples remain explicit rows rather than being hidden in “other.”

Exit criteria:

- paired rows account for at least 90% of one-thread CPU on every canonical
  cell;
- the top ten Rust/C++ discrepancies have a call-count or unit-cost diagnosis;
- each discrepancy has an Amdahl upper bound;
- no speculative rewrite is scheduled ahead of a larger measured row.

## 10. Phase 2: Single-thread function parity

Work down the ledger in weighted order. The following are inspection domains,
not a presumed priority order.

### 10.1 VarDCT

- color conversion and XYB/opsin transforms;
- adaptive quantization setup;
- AC strategy candidate generation and scoring;
- CfL fitting and refinement;
- forward transforms and coefficient layout;
- quantization and encoder-side reconstruction;
- perceptual refinement loop;
- DC and AC token generation;
- histogram building, clustering, and code selection;
- ANS and prefix writing.

For each search function, compare:

- candidate count;
- shortlist size;
- reconstructions per candidate;
- cost-function calls;
- coefficients and tokens processed;
- time per candidate.

### 10.2 Modular/lossless

- transform selection and application;
- sample gathering;
- predictor evaluation;
- tree construction and split-cost estimation;
- residual generation;
- palette and patch search;
- LZ77 matching;
- tokenization;
- histogram clustering;
- entropy writing.

The existing tree-learning incidents make effort-gate and predictor-set parity a
first-class audit item. Do not assume the remaining tree path is correct merely
because two large mistakes were already fixed.

### 10.3 Shared runtime costs

- repeated planar/interleaved conversion;
- full-image copies;
- temporary zeroing;
- small-vector growth;
- hash-map or tree-map use in inner loops;
- bounds checks and iterator abstractions visible in exclusive samples;
- missed inlining or code-size-driven de-optimization;
- scalar fallbacks on hot target CPUs;
- allocator and deallocator cost;
- serialization and bit-buffer flushes.

### 10.4 SIMD policy

Do not perform a broad “SIMD completeness” project. SIMD work is admitted only
when:

- the function is a measured hot row;
- the function has enough independent work to vectorize;
- arithmetic and rounding requirements are pinned by tests;
- generated assembly or counters show that auto-vectorization is insufficient;
- the end-to-end result survives A/B measurement.

Exit criteria:

- one-thread geometric mean is at most 1.10x libjxl;
- no one-thread core cell exceeds 1.35x;
- the top remaining discrepancy is understood at function level;
- promoted mechanical changes are byte-identical wherever technically possible.

## 11. Phase 3: Parallel scaling

Begin only after the comparable single-thread pipeline is near parity.

### 11.1 Measure, do not infer

For 1, 2, 4, 8, and all physical cores, record:

- useful work per thread;
- runnable versus blocked time;
- queue wait;
- task size distribution;
- critical-path length;
- ordered-commit wait;
- synchronization and allocation contention;
- memory bandwidth and peak RSS.

Compare these values to libjxl's scaling on the same cells.

### 11.2 Diagnose the scaling gap

Classify it as:

- insufficient independent groups;
- tasks too coarse;
- tasks too fine;
- a serial phase on the critical path;
- centralized queue or allocator contention;
- excess cloning/copying between tasks;
- deterministic ordering barrier;
- full-image ownership preventing pipeline overlap;
- memory-bandwidth saturation.

Only then alter the work graph.

### 11.3 Promotion rules

- preserve deterministic output unless a separately documented mode opts out;
- include RSS and bytes in every scaling result;
- do not accept an 8-thread win that slows 1-thread or small images materially;
- require at least a 2% end-to-end wall win for non-trivial scheduler
  complexity;
- delete abandoned scheduler scaffolding rather than leaving inactive paths.

Exit criteria:

- at least 4x speedup from 1 to 8 threads on large eligible images;
- 8-thread geometric mean at most 1.05x libjxl;
- small-image parallel overhead below 3%;
- no unexplained serial plateau.

## 12. Phase 4: Quality and size parity by decision diff

Quality work uses the same fine-tooth-comb method.

For each remaining `CJXL_DOMINATES` cell:

1. Find the first bitstream section or decoded intermediate that diverges.
2. Dump equivalent Rust and libjxl inputs and outputs at that boundary.
3. Compare candidates, cost components, effort gates, and the winner.
4. Trace the first differing decision to one function.
5. Port or correct the smallest missing behavior.
6. Validate the focused cell and the full affected matrix.

Preferred order:

1. wrong or missing libjxl behavior;
2. extra fixed overhead;
3. inaccurate cost estimate;
4. duplicate or prematurely finalized work;
5. an ours-only improvement with a provable keep-best rule;
6. new heuristic or perceptual tuning.

Useful decision-diff artifacts include:

- AC strategy map and per-candidate costs;
- quant fields and quantized coefficients;
- CfL parameters and residual costs;
- Modular tree samples, split candidates, selected predictors, and actual
  clustered entropy cost;
- token counts, histogram assignments, and section sizes;
- perceptual-loop input, per-iteration score, and accepted quant field.

Keep-best designs are favored when both alternatives can be scored by the real
downstream cost and the cheaper result selected without a regression. Their wall
cost must still be paid or eliminated through reuse.

Exit criteria:

- zero calibrated dominated cells in the core matrix;
- no content family loses on geometric-mean size at matched quality;
- one-thread and eight-thread speed targets remain green.

## 13. Phase 5: Exceed libjxl

Beating libjxl is the same loop continued past parity, not a separate speculative
architecture program.

Priority order:

1. functions where Rust remains slower per work unit;
2. duplicate work libjxl also performs and both implementations can avoid;
3. better data layout or ownership proven by cache and allocation evidence;
4. safe specialization for common bit depth, channel count, transform size, or
   entropy case;
5. keep-best decisions that improve bytes without reducing quality;
6. higher-value search funded by measured speed savings;
7. narrowly targeted perceptual tuning.

Perceptual and content-aware work remains capped near 10% of engineering effort
until:

- speed parity is achieved;
- function and decision ledgers cover the core pipelines;
- the remaining quality loss cannot be explained by a libjxl divergence;
- the expected win is large enough to survive a full-corpus gate.

Alternative metric backends, broad content classifiers, and large parameter
sweeps are not default roadmap items. They require a specific remaining cell
cluster, a falsifiable mechanism, and a measured upper bound worth pursuing.

Exit criteria:

- geometric-mean wall time at most 0.95x libjxl at 1 and 8 threads;
- at least 60% strict per-cell Pareto wins;
- geometric-mean bytes at matched quality at least 2% below libjxl;
- no content family or supported target with a material regression.

## 14. Work deliberately deferred

These must not displace the profiling loop unless a profile or correctness
failure calls for them:

- a new perceptual model or another metric fork;
- more global parameter sweeps;
- a new content-classification layer;
- speculative GPU offload;
- wholesale pipeline or scheduler rewrite;
- generalized streaming architecture;
- broad SIMD coverage for cold code;
- decode-speed wire-format experiments;
- feature-completeness work unrelated to the core encoder target;
- papers-derived algorithms without a measured matching bottleneck.

The two supplied PDFs remain useful reference material when an exact algorithm,
numeric convention, or JPEG XL design detail is unclear. They are not evidence
that a full port or research detour is required.

## 15. First 24 tasks

1. Make the crate independently resolve and build in release mode.
2. Pin and build the exact libjxl reference with matched optimization.
3. Create one paired, interleaved benchmark command.
4. Select and freeze the canonical profile inputs.
5. Capture 1-thread Rust flamegraphs for all canonical cells.
6. Capture matching 1-thread libjxl flamegraphs.
7. Add low-overhead phase, call, and work-unit counters.
8. Pair symbols covering at least 90% of CPU time.
9. Publish the initial calls-versus-cost-per-call ledger.
10. Verify effort-gate and candidate-count parity for every top-ten row.
11. Select the largest Amdahl opportunity.
12. Implement one byte-identical function-level optimization.
13. Run interleaved A/B and the focused correctness matrix.
14. Promote or revert it and record the result.
15. Re-profile; do not assume the old hotspot remains dominant.
16. Repeat tasks 11-15 until 1-thread geometric mean is <= 1.10x.
17. Capture 1/2/4/8-thread scheduler profiles for the two large cells.
18. Build the scaling ledger and identify the critical path.
19. Fix the largest measured scheduler or ownership cost.
20. Reach the parallel competitive gate.
21. Build decision-diff tooling for the worst remaining lossless cell.
22. Build decision-diff tooling for the worst remaining VarDCT cell.
23. Close quality/size wedges one decision function at a time.
24. Continue the same loop past parity until the exceed gates hold.

## 16. Required living documents

Keep these small and operational:

1. `PERF_BASELINE.md`
   - revisions, build commands, machine, canonical cells, current walls.
2. `FUNCTION_PARITY_LEDGER.tsv`
   - the paired function data defined above.
3. `OPTIMIZATION_RESULTS.md`
   - promoted and reverted hypotheses with measured outcomes.
4. `QUALITY_DECISION_LEDGER.tsv`
   - first differing decision and status for every dominated cell.
5. `CAPABILITIES.md`
   - truthful supported-input and feature matrix.

Large historical developer notes can move under the project's docs archive. The
living ledgers should link to old evidence without inheriting its narrative
sprawl.

## 17. Stop rules

Stop or revert an optimization when:

- the end-to-end win disappears under interleaved measurement;
- the optimized function is not material in the current whole-program profile;
- a supposed reference mismatch is disproved by source inspection;
- a byte-identical change alters bytes;
- a decision-affecting speed win worsens matched-quality rate/distortion;
- three attempts at one row fail without new profile evidence;
- maintenance complexity is larger than the measured opportunity;
- the work depends on an unverified content proxy when real downstream cost can
  be measured instead.

An honest negative result is progress. Retain the measurement and conclusion,
revert the production code, and move to the next ledger row.

## 18. Governing principle

Do not try to outguess libjxl at the top of the pipeline while equivalent inner
functions remain slower or make different decisions for accidental reasons.

First make every important Rust function explainable against its libjxl
counterpart: how often it runs, how much work it performs, how much each unit
costs, and whether it chooses the same result. Then make the expensive functions
cheaper, one at a time, and keep only measured wins.

That fine-tooth-comb process is the credible path to both parity and a durable
lead.
