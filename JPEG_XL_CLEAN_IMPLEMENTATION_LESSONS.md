# Lessons for a clean MIT/Apache JPEG XL implementation

## Purpose and scope

This document preserves the durable lessons from the abandoned
`jxl-encoder` project without treating its implementation as the design for a
new codebase. The old tree declares
`AGPL-3.0-only OR LicenseRef-Imazen-Commercial`; do not copy its source,
tests, comments, or documentation into a project intended to be
`MIT OR Apache-2.0`.

No on-disk `AGENTS.md` was present. The verbose developer record referred to
in this review was `jxl-encoder/CLAUDE.md`, together with `CHANGELOG.md` and
the archived code history.

The useful inheritance is knowledge: failure modes, test strategy,
measurement discipline, and architectural cautions. Re-express each lesson
from the applicable standard and independently written design notes.

This is an engineering summary, not legal advice. Before publishing a
clean-room implementation, have the source-provenance and patent plan
reviewed by someone qualified to do so.

## Licensing and clean-room boundary

1. Start the new encoder/decoder in a new repository with no copied history.
2. Record the intended license (`MIT OR Apache-2.0`, if dual licensing is the
   goal) before adding code.
3. Keep a provenance ledger for every normative document, test vector,
   constant table, and third-party dependency.
4. Do not copy from this AGPL project, including “small” helpers, tests,
   comments, tables, or generated code whose origin is uncertain.
5. Write design notes in original language with clause references to the
   standard. Do not reproduce long passages from the standard.
6. Separate specification study from implementation when practical:
   specification notes describe observable requirements and test cases;
   implementation authors write code from those notes and licensed inputs.
7. Use other implementations as black-box interoperability oracles unless
   their licenses and the desired derivation model have been reviewed.

The local `libjxl` checkout is BSD-3-Clause, not AGPL. BSD-3-Clause is
permissive, but a direct translation remains derived from BSD-licensed code
and must preserve its notices. More importantly, libjxl is not a substitute
for the normative standard: it mixes required decoding semantics with one
encoder's heuristics, tuning, platform abstractions, and historical choices.
For a genuinely independent implementation, use the standard as the source
of normative behavior and libjxl only as an interoperability and performance
oracle.

The software license and patent position are separate questions. Review the
JPEG XL declarations under the ISO/IEC/ITU common patent policy rather than
assuming that a permissive code license settles patent rights.

## Obtaining the JPEG XL specifications legitimately

The current core specification is
[ISO/IEC 18181-1:2024, edition 2](https://www.iso.org/standard/85066.html).
ISO currently lists the 91-page PDF/ePub at CHF 227. The official sources
checked while preparing this document did **not** offer a verified free,
complete copy of the current Part 1.

For a complete implementation, obtain legitimate access to:

- Part 1, core coding system: normative codestream and decoding processes.
- Part 2, file format: container, boxes, and file-level signaling.
- Part 3, conformance testing: conformance requirements and test methodology.
- Part 4, reference software: a short publication pointing to reference
  software; it is not a replacement for Parts 1 and 2.

The [JPEG XL workplan](https://jpeg.org/jpegxl/workplan.html) tracks the
current editions. Practical legitimate routes are an employer or university
standards subscription, a national standards-body library, inter-library
access, or purchase from ISO/IEC or a member body.

Useful free and legitimate companions are:

- The [2019 JPEG XL Committee Draft](https://arxiv.org/abs/1908.03565). It is
  valuable background but predates the final standard and must not be treated
  as current normative text.
- The official
  [JPEG XL documentation page](https://jpeg.org/jpegxl/documentation.html),
  including the JPEG white paper and open publications.
- The
  [JPEG XL white paper](https://ds.jpeg.org/whitepapers/jpeg-xl-whitepaper.pdf).
- The open-access paper
  [“The JPEG XL Image Coding System: History, Features, Coding Tools, Design
  Rationale, and Future”](https://arxiv.org/abs/2506.05987).
- The official
  [JPEG XL conformance repository](https://github.com/libjxl/conformance).
- ISO/IEC 18181-4:2022 is listed as a
  [zero-cost download by VDE](https://www.vde-verlag.de/iec-standards/251068/iso-iec-18181-4-2022.html),
  but it is only three pages and points to reference software.

Do not begin a production decoder from the old committee draft alone. Use it
to learn terminology while arranging access to the current Part 1 and Part 2.

## Why VarDCT was unusually difficult in the old project

VarDCT was attempted too broadly and too early. On 2026-01-01 the project
implemented XYB conversion, AC strategies, quantization, DCT8, context
modeling, frame assembly, adaptive quantization, and multi-group support in
one pass. Tests proved only that headers parsed; they did not render decoded
pixels. Bitstream tracing arrived after the encoder was already broken.

The recurring failures were not one hard algorithm. They were many exact
semantic mismatches whose symptoms appeared far downstream:

### Transform layout was easy to misread

- libjxl's square forward DCT left coefficients in a transposed layout; the
  Rust port transposed them back.
- Rectangular transforms had shape-dependent transpose rules.
- DCT16x16 LLF extraction confused contiguous indices with a two-dimensional
  grid.
- DCT16x8 used the reciprocal resampling scale.
- AC group traversal interleaved channels and blocks in the wrong order.

A transform can round-trip in isolation and still be wrong for the wire
layout. Every transform shape needs tests for mathematical values,
coefficient storage order, LLF/DC extraction, quantized symbol order, and
decoder reconstruction.

### A single syntax error shifted unrelated fields

Examples included:

- writing LF correlation fields as signed variable-length integers instead
  of fixed unsigned bytes;
- omitting `num_hf_presets` when its conditional bit width became nonzero;
- writing two LZ77 header bits where the configured integer used zero bits;
- selecting a different ANS `omit_pos` tie than the decoder;
- omitting extra-channel frame fields.

The decoder then reported errors such as an invalid transform ID even though
the transform ID itself was not the cause. Build bit-position tracing before
the first encoder field is written, and retain it as zero-cost,
feature-gated instrumentation.

### Quantization contains reciprocal and unit traps

The project repeatedly confused:

- a quantization weight with its inverse;
- forward and inverse resampling scales;
- raw and shifted nonzero counts;
- per-channel and shared quantization weights;
- generated default weights and explicitly signaled weights;
- floating encoder coefficients and the decoder's modular integer width.

At very fine VarDCT distances, DC values exceeded `i16`. The old encoder
initially widened its own DC storage but still signaled that a decoder's
16-bit modular buffer was sufficient. Strict decoders then diverged during
prediction and eventually reported an ANS failure. This is exactly the kind
of normative signaling rule that source-porting can miss and specification
work should expose.

Use unit-bearing names and types such as `ForwardScale`, `InverseScale`,
`QuantWeight`, `InvQuantWeight`, `BlockCoord`, and `PixelCoord`. Avoid bare
`f32`, `usize`, and ambiguous names at transform boundaries.

### VarDCT combines several stateful subsystems

VarDCT is not “just a DCT”:

- linearization, color encoding, intensity target, and XYB conversion;
- adaptive quantization and masking;
- variable transform selection and LLF/DC construction;
- coefficient prediction and context modeling;
- histogram clustering and ANS/prefix serialization;
- restoration filters and optional perceptual refinement;
- groups, passes, extra channels, and progressive structure.

Each subsystem can produce a decodable file while still damaging quality,
rate, color, or cross-decoder conformance. Implementing all of them together
made failures difficult to localize.

### Encoder heuristics were mistaken for normative requirements

The standard primarily specifies codestream syntax and decoding; encoding
processes have substantial freedom. libjxl's candidate searches, cost
constants, perceptual loops, and effort policy are not all required to make a
valid JPEG XL encoder. The old project tried to match a mature encoder's
quality machinery before it had a small, fully proven normative core.

A fresh implementation should first emit a deliberately simple, valid
DCT8-only VarDCT stream. Add transform families, adaptive quantization,
strategy search, and perceptual refinement one independently measured layer
at a time.

### Color and metrics repeatedly produced false conclusions

The old project at different times signaled the wrong transfer function,
computed metrics on gamma-encoded values as if they were linear, and treated
PNG/ICC metadata differences as codec quality differences. During the final
profiling work, five lossless decoder/output combinations matched source
pixels exactly while jxl-oxide differed only on libjxl output containing a
DCI-P3 ICC profile. All tested libjxl lossless option variants produced the
same codestream, pointing to decoder-specific color-management behavior
rather than lossy coding.

Always distinguish:

- stored sample equality;
- decoded samples in the codestream's native color encoding;
- color-converted output in a requested target space;
- displayed appearance;
- perceptual metric input requirements.

Record ICC/CICP/transfer/primaries and the requested decoder output encoding
with every quality result.

## Durable lessons from the final optimization passes

### Measure the whole program with immutable provenance

The final harness improvements were worth retaining as design principles:

- pin exact source revisions and build recipes;
- hash both encoder binaries and recheck them before every timed/profiled
  invocation;
- interleave A/B samples instead of running separate batches;
- record minimum, median, dispersion, output bytes, hashes, RSS, CPU
  affinity, thread count, and decoder results;
- keep cold-cache, warm-cache, and warm-process claims distinct;
- reject incomplete captures and failed provenance rather than explaining
  them away.

Instrumentation must be calibrated. Feature-gated thread-local counters were
accepted only after paired builds showed less than 1% overhead on a stable
host. A result whose binary had changed at a live path was rejected even
though its timing looked plausible.

### Decoder success is not lossless correctness

The original profiling harness only checked that each decoder emitted a PNM.
The corrected gate compared native-depth PNM dimensions, max value, and
sample bytes against a source reference, while also verifying that retained
decoder artifacts had not changed since the timed run.

For lossless claims require:

1. successful decode by independent decoders;
2. exact native sample equality;
3. correct color/metadata interpretation;
4. exact reconstruction of any advertised JPEG recompression path;
5. deterministic output for a fixed configuration.

### Flamegraphs can be valid-looking and wrong

On the hybrid Intel host, generic `cycles` split into `cpu_core` and
`cpu_atom`. The first generated SVGs contained only one or two Atom startup
samples while the useful P-core samples were absent. Valid captures used the
explicit userspace event `cpu_core/cycles/u` and required:

- folded period total equal to the selected `perf` event total;
- almost all periods rooted in the intended encoder process;
- retained `perf.data`, scripts, folded stacks, reports, symbols, and hashes;
- a fresh artifact directory for every capture.

Do not trust an SVG merely because it renders.

### Size classes expose different algorithms

An initial tile-map optimization was correctly reverted after a 4 MP,
one-thread test measured it 0.28% slower. Later profiles on a 49.9 MP image
revealed the real pathology: every parallel 64x64-pixel AC-search tile
retained an `AcStrategyMap` backed by the entire image.

The corrected map kept global logical coordinates but used tile-local
storage and deterministic row-major merging. In a frozen, counterbalanced
49.9 MP effort-7/eight-thread A/B:

- wall median improved from 9.971 s to 9.118 s: **8.55% faster**;
- peak RSS fell from 10.89 GiB to 2.74 GiB: **74.82% lower**;
- all six outputs were byte-identical;
- all three decoders produced identical pixel hashes.

The 4 MP improvement was only about 1.27% and noisy. The lesson is not “never
revert a small regression”; it is to model allocation complexity and test
threshold sizes. A data structure sized by the whole image inside work
multiplied by the number of tiles is a structural warning even when small
fixtures hide it.

### Host state can overwhelm codec measurements

One large Rust sample took about 148 seconds and was later killed after
reaching roughly 10.4 GiB RSS. Task-generated build caches occupied tmpfs,
swap was exhausted, and another workload was active. After those caches were
removed and the host became idle, the same old binary took about 9.8–10.1
seconds.

Record available memory, swap, tmpfs use, CPU policy, load, and competing
jobs. Resource-pressure measurements are diagnostics, not performance
baselines.

### Unknown profile coverage must stay unknown

The final source-backed one-thread function ledger explained only about 48%
of sampled Rust CPU at effort 5 and 34% at effort 7. The remainder was kept
as explicit `UNKNOWN`. That is more useful than a confident but invented
mapping between vaguely similar Rust and C++ functions.

## What to retain from the previous developer's process

The old `CLAUDE.md`, changelog, and code history were far too large, but they
contained several sound rules:

- Trace bitstream writes from the beginning.
- Prove layers in order: component math, serialization, full decode, pixel
  correctness, then real-image quality.
- Test test-infrastructure against known-good and known-bad artifacts.
- Use at least three independent decoders.
- Include multi-group images in every relevant path; single-group fixtures
  hide conditional fields and region bugs.
- Use synthetic fixtures for unit/layout tests and real, licensed,
  stratified images for quality conclusions.
- Preserve measured negative results so failed ideas are not rediscovered.
- Keep scalar-versus-SIMD parity tests across vector widths, tails, and edge
  values.
- Pad working buffers deliberately instead of spreading inconsistent edge
  behavior through hot loops.
- Bound dimensions, allocations, CPU work, and cancellation points before
  accepting hostile input.

These should become short contribution rules and executable tests, not a
1,600-line agent prompt.

## What not to carry into the new project

### Do not turn the changelog into a research database

The abandoned changelog grew to 6,965 lines and mixed user-visible changes,
experiments, hypotheses, benchmark reports, and internal process. The agent
instructions became another historical database, then required an archive
to explain the first database.

Use four small records instead:

- `CHANGELOG.md`: released user-visible changes only.
- `CONFORMANCE.md`: supported clauses/profiles and test status.
- `PERFORMANCE.md`: current reproducible baselines.
- `experiments/`: immutable reports, including negative results.

Delete stale hypotheses instead of making every future contributor read
them. Git already preserves history.

### Keep research metrics outside the codec core

The last three commits (`20ea00e4`, `f195c8c0`, `d17cf7ce`) added
Zensim/model-attribution steering, experiment arms, environment switches,
and roughly 1,600 lines of implementation/tests/reports. The research was
more honest than earlier work—it used preregistered gates and recorded
failures—but the architectural priority was wrong for an encoder still far
from function and performance parity.

The surviving H3 result was narrow: it helped one MLP “bake” mainly at
low/mid targets, failed to beat baseline with the shipped linear bake, and
added measurable per-comparison cost. Such work belongs in an external
research crate or experiment driver until the normative codec, quality
matrix, and function ledger are stable.

Avoid:

- environment-variable experiment dispatch in production algorithms;
- optional metric backends pulling unpublished or sibling dependencies into
  the core crate;
- tuning constants before decision and work-count parity is understood;
- aggregate “wins” that hide losing content families or quality bands;
- optimizing a proxy while a larger measured normative stage is unexplained.

### Do not confuse number of tests with confidence

The project repeatedly had hundreds of passing tests while VarDCT either did
not render or rendered badly. Confidence comes from what invariants are
proved, not test count.

## Recommended architecture for the new pair

Keep normative mechanics independent from encoder policy:

```text
jxl-bitstream       integer coding, bit I/O, syntax types, tracing
jxl-color           color metadata and explicitly typed transforms
jxl-decoder         normative Part 1 decoding
jxl-container       Part 2 boxes and codestream/container boundary
jxl-encoder-core    valid, simple modular and VarDCT emitters
jxl-encoder-policy  optional strategy search and rate/distortion policy
jxl-conformance     official vectors, malformed-input tests, cross-decoder tools
jxl-cli             file I/O and user interface
experiments/        perceptual metrics, tuning, research backends
```

The decoder should come first. It turns the normative inverse process into
executable understanding, provides a trustworthy round-trip oracle, and
forces precise handling of conditional syntax before encoder heuristics
obscure the problem.

The encoder and decoder may share syntax types, but they should not share
enough implementation that an encoder bug is automatically accepted by its
paired decoder. Always retain external conformance vectors and independent
decoder checks.

## Suggested implementation order

### Phase 0: legal and normative foundation

1. Obtain current Parts 1 and 2 legitimately; obtain Part 3 if possible.
2. Freeze licenses, provenance policy, patent review, and contribution rules.
3. Import official conformance assets with exact revisions and hashes.
4. Build bit-level tracing and a strict limits/allocation model.

### Phase 1: decoder first

1. Signature/container detection and size/header parsing.
2. Bit reader and all normative integer encodings.
3. Modular decoding for a deliberately narrow profile.
4. Entropy decoding with isolated serialization vectors.
5. Multi-group, extra-channel, color metadata, and malformed-input coverage.
6. VarDCT inverse path: DCT8 first, with coefficient-layout snapshots.
7. Add transform shapes one at a time, each with layout and reconstruction
   tests.

Do not call the decoder complete until official conformance streams render,
not merely parse.

### Phase 2: minimal encoder

1. Modular lossless with one simple predictor and entropy mode.
2. Exact round-trip through the new decoder plus independent decoders.
3. DCT8-only VarDCT with fixed, explicitly signaled choices.
4. One group before multi-group, but add multi-group before claiming support.
5. No adaptive strategy search, perceptual loop, SIMD, or learned metric yet.

### Phase 3: complete normative breadth

Add transform families, extra channels, animation/progressive structure,
ICC/CICP/EXIF/XMP, JPEG reconstruction, high bit depth, HDR, and profiles in
small clause-driven increments. Every addition gets:

- a bitstream trace fixture;
- positive and malformed test vectors;
- multi-group/edge-size coverage where relevant;
- external decoder or conformance evidence;
- native-depth pixel validation.

### Phase 4: encoder quality and performance

Only after validity and breadth are stable:

1. Profile representative real cells.
2. Count calls and work units, not only time.
3. Add AC strategies and decision-diff them against a reference.
4. Add adaptive quantization and rate control.
5. Add SIMD after scalar reference paths are locked.
6. Keep perceptual metrics outside the normative core.
7. Promote changes with interleaved whole-program A/B evidence.

## Minimal permanent test matrix

Every supported feature should be checked at the applicable layers:

1. **Syntax:** exact bit counts and parse/serialize round-trip.
2. **Math:** transform values, layout, scaling, and inverse reconstruction.
3. **Entropy:** histogram and symbol round-trip, including degenerate and
   tie cases.
4. **Decode:** official conformance streams and malformed-input rejection.
5. **Lossless:** native-depth pixel equality, not successful decode.
6. **Lossy:** decoded pixels evaluated in a declared color space with at
   least two metrics and visual spot checks.
7. **Structure:** single/multi-group, odd dimensions, tiny and very large
   images, alpha/extra channels, 8/16-bit, SDR/HDR, and animation.
8. **Security:** size caps, checked arithmetic, fallible allocation,
   cancellation, fuzzing, and decompression amplification.
9. **Performance:** one/eight threads, fixed overhead, large-image memory,
   immutable binary hashes, and regeneration-ready profiles.

## Forensic sources summarized

- `jxl-encoder/CLAUDE.md`
- `jxl-encoder/CHANGELOG.md`
- `jxl-encoder/docs/llm-docs/CODE-HISTORY.md`
- `jxl-encoder/docs/llm-docs/JXL_ENCODER_LEARNINGS.md`
- `jxl-encoder/docs/OPTIMIZATION_RESULTS.md`
- `jxl-encoder/docs/PERF_BASELINE.md`
- Git commits `20ea00e4`, `f195c8c0`, `d17cf7ce`, and their benchmark notes
- The final frozen-binary flamegraph and AC-strategy allocation A/B artifacts

## Immediate next actions

1. Copy this file outside the old tree before deleting anything.
2. Create a new empty repository and add only license/provenance/process
   files.
3. Arrange legitimate access to ISO/IEC 18181-1:2024 and the current Part 2.
4. Download and hash the official conformance corpus.
5. Write a clause-indexed decoder plan.
6. Implement tracing, limits, and the bit reader before codec tools.
7. Build a minimal decoder before beginning another VarDCT encoder.

The most important change of tack is this: implement the normative inverse
process first, then build the simplest valid encoder against it. Do not begin
by recreating libjxl's mature encoder heuristics.
