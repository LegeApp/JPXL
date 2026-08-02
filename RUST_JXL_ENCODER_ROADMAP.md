# Recommended direction

The `jp2lam` process is a good template, but its **engineering method** should be reused, not its codec decomposition.

JPEG 2000 naturally decomposes around transforms, subbands, code-blocks, coding passes, packets, and tile-parts. JPEG XL has a different center of gravity:

```text
bit-level syntax and headers
        ↓
shared entropy coding
        ↓
Modular sub-bitstreams
        ↓
├── full Modular frames
└── VarDCT LF/control information
        ↓
frame groups, sections, TOC, passes
        ↓
frame rendering and composition
```

The diagram on page 14 of the attached JPEG XL paper is the right mental model. Modular and VarDCT are not two isolated codecs: VarDCT uses Modular sub-bitstreams for the LF image, adaptive-quantization maps, block selection, chroma-from-luma data, filter controls, and extra channels. JPEG XL also uses the same HybridUint/prefix-or-ANS entropy system for almost everything.   

The current published specifications to target are **ISO/IEC 18181-1:2024** for the core codec, **18181-2:2026** for the file format, and **18181-3:2025** for conformance. A fourth edition of Part 2 is under development, so the implementation should skip unknown boxes and extension fields safely rather than hard-coding the assumption that the current set is permanent. ([JPEG][1])

There are already several Rust implementations: `jxl-oxide` is a pure-Rust conforming decoder, the official `libjxl/jxl-rs` decoder is under active development, `zune-jpegxl` implements a narrower Modular encoder, and a separate pure-Rust VarDCT/Modular encoder now exists. Therefore, a new project should have a clear identity beyond “JPEG XL in Rust”: **standard-first architecture, permissive licensing, bounded-memory streaming, conservative dependencies, and an integrated encoder/decoder whose structure does not imitate libjxl**. ([GitHub][2])

## What to carry over from `jp2lam`

The most successful parts of the JPEG 2000 work transfer directly:

1. **Typed stage boundaries instead of a giant mutable codec state.** The attached Rust guidelines explicitly frame the codec as transformations between increasingly encoded representations. Apply exactly that principle here. 

2. **Standard-first implementation.** Keep “what ISO/IEC 18181 says” separate from “what libjxl or another decoder does.” Existing codecs are interoperability oracles and diagnostic references, not the architecture. That is the same rule your `AGENTS.md` established for OpenJPEG. 

3. **Borrowed input, validated plans, resource budgets, and streaming writers.** The final `jp2lam` architecture—borrowed image view, resource planner, bounded working set, encoded payload store, final plan, and direct writer—is highly applicable. 

4. **Vertical slices with recorded evidence.** Your agent trace worked because every slice ended with tests, measurements, and a precise statement of what remained. That should become part of the new repository from the first session.

What should **not** be carried over is the JPEG 2000 terminology and Tier-1/Tier-2 structure. JPEG XL’s closest rough mapping is frames → groups → sections → passes, but even that is only conceptual. 

# High-level architecture

A mature implementation should look approximately like this:

```text
ENCODER

Borrowed ImageView
        │
        ▼
Validated EncodeRequest
  image metadata
  color interpretation
  frame intent
  quality / lossless
  resource limits
        │
        ▼
Encoder Analysis
  mode choice
  transforms
  predictor/context choices
  block partition
  quantization
        │
        ▼
Validated CodestreamPlan
  image header
  frame headers
  group/section topology
  entropy configurations
        │
        ▼
Section Encoders
  Modular or VarDCT
        │
        ▼
SectionStore
  bounded RAM
  optional temporary-file spill
  exact section lengths
        │
        ▼
TOC + Codestream Writer
        │
        ▼
Optional Container Writer
        │
        ▼
Write sink


DECODER

Input source
        │
        ▼
Container / raw codestream reader
        │
        ▼
Image and frame header parser
        │
        ▼
SectionIndex from TOC
        │
        ▼
Shared entropy decoder
        │
        ├── Modular decoder
        └── VarDCT decoder
        │
        ▼
Frame reconstruction
  inverse transforms
  filters
  features
  blending
  orientation
        │
        ▼
Normalized output image
```

The `SectionStore` is the JPEG XL counterpart to your encoded block store. JPEG XL places the TOC before the actual sections so that a decoder can locate groups early and decode them in parallel or by region. A streaming encoder therefore needs section lengths before final emission. The clean solution is to encode each section once into bounded RAM or temporary storage, generate the TOC, and then stream the stored sections without assembling a second complete codestream. This is an architectural inference from JPEG XL’s TOC ordering and your proven `jp2lam` ownership model.  

## Suggested module organization

Start with one library crate rather than a workspace of many tiny crates:

```text
src/
    bitstream/
        reader.rs
        writer.rs
        integers.rs
        half.rs

    container/
        boxes.rs
        reader.rs
        writer.rs

    headers/
        image.rs
        metadata.rs
        color.rs
        frame.rs
        extensions.rs

    entropy/
        hybrid_uint.rs
        prefix.rs
        ans.rs
        histogram.rs
        context_map.rs
        lz77.rs

    modular/
        channel.rs
        transform.rs
        rct.rs
        palette.rs
        squeeze.rs
        properties.rs
        predictor.rs
        ma_tree.rs
        codec.rs

    vardct/
        xyb.rs
        lf.rs
        block_grid.rs
        transform.rs
        quant.rs
        hf_metadata.rs
        coefficients.rs
        filters.rs

    frame/
        groups.rs
        sections.rs
        toc.rs
        passes.rs
        blending.rs
        render.rs

    encode/
        request.rs
        plan.rs
        analysis.rs
        section_store.rs
        pipeline.rs

    decode/
        request.rs
        limits.rs
        pipeline.rs

    diagnostics/
        dump.rs
        counters.rs

    image.rs
    error.rs
    lib.rs
```

The important dependency direction is:

```text
bitstream
    ↓
headers + entropy
    ↓
modular
    ↓
vardct
    ↓
frame
    ↓
encode / decode orchestration
```

`vardct` may depend on `modular`; `modular` must never depend on `vardct`.

# Development plan

## Phase 0 — Establish the project constitution

Define the project as:

> A native, standard-first, safe Rust implementation of JPEG XL with a broad decoder, a progressively improving encoder, bounded resource usage, no runtime dependency on libjxl, and no architectural port of libjxl.

Create immediately:

```text
AGENTS.md
docs/architecture.md
docs/iso-18181-crosswalk.md
docs/conformance-matrix.md
docs/unsupported.md
llm-docs/most-recent-agent-trace.md
THIRD_PARTY_NOTICES.md
```

The crosswalk should map every implemented type and function to the relevant standard clause. The 2025 overview paper is useful for rationale, but it explicitly says that it is not a substitute for the standard. 

The older 2021 comparison paper should be treated only as historical compression background. It used pre-final reference software and is not suitable as an implementation guide. 

## Phase 1 — Build the validation system before the codec

Set up three independent validation paths:

* Current `djxl`/`cjxl` from libjxl 0.12 or newer.
* `jxl-oxide`.
* `jxl-rs` where it accepts the relevant feature set.

The libjxl repository currently advises updating to version 0.12 because of security fixes, so older oracle binaries should not be used as the principal reference. ([GitHub][3])

Clone or download the official conformance corpus and maintain a machine-readable support matrix. The current corpus covers Modular, VarDCT, alpha, animation, Palette, Squeeze, LZ77, patches, progressive passes, ICC, JPEG reconstruction, blending, and other features, so it can become the project roadmap rather than merely an end-stage test. ([GitHub][4])

Tests should compare:

```text
parse success
dimensions
bit depth
color interpretation
frame count
pixel hash for exact paths
peak error and RMSE for conforming lossy paths
```

Do not use byte-for-byte parity with libjxl as a general target.

## Phase 2 — Implement bitstream syntax and headers

Implement and exhaustively test:

* Bit reader and writer with JPEG XL bit ordering.
* `Bool`, `U32`, `U64`, enum coding, signed packing, half-floats.
* Size headers.
* Extension-bit handling.
* Image metadata.
* Color encoding.
* Extra-channel descriptors.
* Image and frame headers.
* Crop, passes, blend information, and restoration-filter signaling as typed models, even if initially unsupported by pixel decoding.

Every parser operation must use checked arithmetic and resource limits. Invalid or unsupported syntax must return a typed error, never panic or silently substitute defaults.

## Phase 3 — Implement shared entropy coding as an independent subsystem

This is the foundation of the whole codec and deserves its own test corpus.

Implement in this order:

1. HybridUint tokenization and reconstruction.
2. One legal entropy backend selected from the current standard—the simplest is likely prefix coding with one context cluster, but the implementation agent must confirm this against the normative syntax.
3. Context-map representation.
4. ANS distributions and coding.
5. Histogram signaling.
6. LZ77.
7. Encoder-side context clustering and histogram optimization.

The first encoder may deliberately use:

```text
one cluster
one histogram
no LZ77
a conservative HybridUint configuration
```

The decoder must later handle the full legal range.

## Phase 4 — Produce the first interoperable vertical slice

The first real milestone should be:

```text
naked JPEG XL codestream
single still image
single frame
8-bit unsigned grayscale
non-XYB color interpretation
Modular mode
single group
no extra channels
no transforms
one-leaf MA tree
one simple predictor
one entropy cluster
no LZ77
mathematically lossless
```

The exact predictor and entropy backend should be whichever legal combination produces the smallest correctly specified implementation. Compression efficiency is irrelevant at this milestone.

Completion means:

* The library encodes several deterministic images.
* Its own decoder reconstructs them exactly.
* `djxl` and `jxl-oxide` reconstruct them exactly.
* Corruptions produce errors rather than panics.
* The code already follows the final stage boundaries.

After that, add in this order:

```text
16-bit grayscale
Gradient predictor
RGB with reversible color transform
multiple groups
minimal jxlc container
```

Do not begin VarDCT before this milestone is real.

## Phase 5 — Complete Modular mode

Expand the decoder before making the encoder clever:

* All predictors.
* Self-correcting predictor and its row state.
* MA-tree parsing and traversal.
* RCTs and channel permutations.
* Palette and delta-palette.
* Squeeze.
* Global and local trees.
* Multiple channels and extra channels.
* Prefix and ANS streams.
* LZ77.
* Modular group partitioning and progressive Squeeze passes.

Then improve encoder decisions independently:

```text
predictor evaluation
MA-tree construction
context clustering
palette detection
RCT selection
Squeeze selection
LZ77 matching
```

Keep encoder analysis separate from the syntax model. A decoder-visible `MaTree` is normative data; an encoder’s algorithm for discovering that tree is not.

## Phase 6 — Implement group, section, and TOC orchestration

Introduce:

```text
GroupGrid
SectionKind
SectionId
SectionIndex
SectionDependency
SectionStore
PassPlan
```

The decoder should be able to:

* Parse the TOC.
* Validate all ranges before launching workers.
* Decode independent groups in parallel.
* Enforce dependencies between global, LF, and HF data.
* Skip unneeded groups for cropped or reduced decoding later.

The encoder should:

* Encode each section once.
* Store it in bounded RAM or spill storage.
* Determine exact section sizes.
* Write the header and TOC.
* Stream each section to the destination.

This is where memory limits and thread limits become part of correctness rather than optional tuning.

## Phase 7 — Implement the VarDCT decoder

Do this in increasingly capable slices:

1. XYB and inverse color conversion.
2. LF image decoding.
3. Fixed DCT8x8 only.
4. Quantization tables.
5. HF coefficient entropy decoding.
6. Block grid and all transform sizes.
7. HF metadata.
8. Chroma from luma.
9. Adaptive quantization.
10. Gaborish.
11. EPF.
12. Upsampling.
13. Progressive HF passes.

VarDCT should not be designed as a separate top-level codec. It consumes the already working Modular and entropy layers.

## Phase 8 — Build a baseline VarDCT encoder

The first lossy encoder should be intentionally unsophisticated:

```text
XYB
fixed DCT8x8
fixed coefficient order
simple global quantization
single HF pass
no patches, splines, or noise
basic filter settings
```

Its target is a valid codestream with monotonic quality, not libjxl-level compression.

Then add:

* Variable block-size search.
* Adaptive quantization.
* Chroma-from-luma estimation.
* Quantization-table choices.
* Coefficient ordering.
* Filter parameter selection.
* Multiple progressive passes.
* Target-byte and target-bpp rate control.

Use external perceptual metrics for encoder development. Do not embed Butteraugli into the core decoder.

## Phase 9 — Add full frame semantics and optional tools

Only after both core modes work:

* Layers and animation.
* Frame references and blend modes.
* Crop and orientation.
* Alpha and other extra channels.
* Patches.
* Splines.
* Noise.
* ICC profile handling.
* Exif, XMP, JUMBF, and compressed boxes.
* Partial codestream boxes.
* Frame index.
* Profiles and levels.
* Gain-map boxes.
* JPEG recompression and `jbrd`.

JPEG recompression should be one of the last major branches. It is not required to prove the general codec architecture.

## Phase 10 — Hardening and optimization

Correctness order:

```text
conformance
resource safety
memory ownership
scalar performance
parallelism
SIMD
encoder quality search
```

Use group-level parallelism with a resource planner. Avoid unconstrained nesting across frames, groups, channels, and transforms.

Start with `#![forbid(unsafe_code)]`. Add unsafe SIMD only after profiles show a concrete bottleneck and the safety contract is local.

The decoder must have explicit limits for:

```text
pixels
frames
channels
extra channels
section count
section bytes
ICC bytes
MA-tree nodes
histograms
LZ77 window
temporary memory
worker count
```

# Correct scope for the one-session agent

A complete Main Profile encoder and decoder is not the right one-session acceptance gate. The useful result from one long session is:

> A clean repository, normative crosswalk, conformance harness, core bitstream machinery, and one standards-valid lossless Modular vertical slice that external decoders accept.

That proves the difficult integration points:

```text
header syntax
entropy coding
Modular prediction
group/section layout
TOC
codestream writing
external interoperability
```

Once that works, additional codec features can be added in controlled slices. A broad repository containing unfinished VarDCT, animation, JPEG reconstruction, and ten empty abstractions would be less valuable.

# Copyable `/goal` prompt

```text
/goal

Build the first working vertical slice of a native Rust JPEG XL encoder/decoder library in the current repository.

The project must be an original, standard-first Rust implementation. It must not be a mechanical port of libjxl, jxl-rs, jxl-oxide, zune-jpegxl, jxl-encoder, or any other codec. Existing implementations may be used as executable interoperability oracles and, only after a standard-derived implementation exists, as diagnostic references. Do not transpose their class structures, ownership models, module names, or giant codec contexts.

Use Rust 1.95+ and edition 2024. Begin with safe Rust and add:

#![forbid(unsafe_code)]

unless the existing repository already has a documented alternative policy.

Read these local documents before changing code:

- rust-idiomatic-guidelines.md
- AGENTS.md
- jp2lam-hd-encode-plan.md
- most-recent-agent-trace.md
- 2506.05987v2.pdf

Treat mandeel2021.pdf as historical benchmarking background only, not as a codec specification.

Target the latest published standards:

- ISO/IEC 18181-1:2024 — core coding system
- ISO/IEC 18181-2:2026 — file format
- ISO/IEC 18181-3:2025 — conformance

Use the normative standard text as the primary source whenever it is available. The attached JPEG XL paper is a design-rationale companion, not a substitute for the standard. If exact normative text is unavailable for a field, consult official libjxl documentation or source only to resolve that specific syntax, record the uncertainty and evidence in docs/SPEC_GAPS.md, and do not invent behavior.

PROJECT IDENTITY

The long-term project is:

- a native Rust JPEG XL encoder and decoder;
- broad decoder support, initially targeting Main Profile Level 5;
- a progressively improving encoder that may emit a narrower legal subset;
- no runtime dependency on libjxl or another JPEG XL implementation;
- bounded-memory encoding and decoding;
- streaming writer support;
- explicit resource limits for untrusted files;
- typed stage boundaries rather than a giant mutable codec state;
- clear separation between normative codestream structures and non-normative encoder search heuristics;
- explicit unsupported-feature errors;
- no false claims of full conformance.

SOURCE-OF-TRUTH ORDER

Use sources in this order:

1. ISO/IEC 18181.
2. Official conformance vectors and reference decoded output.
3. The attached 2025 JPEG XL architecture paper.
4. Official libjxl format documentation.
5. libjxl, jxl-oxide, and jxl-rs as interoperability and diagnostic oracles.

Keep “what the standard requires” separate from “what another implementation happens to do” in comments, tests, and documentation.

Do not pursue whole-file byte parity with libjxl. Success is standards-valid syntax, exact lossless pixels, conformance bounds for lossy data, and acceptance by independent decoders.

REPOSITORY SETUP

If the directory is empty, initialize one library crate with an optional CLI binary. Do not create a workspace containing many tiny crates yet.

Create or update:

- Cargo.toml
- README.md
- AGENTS.md
- docs/architecture.md
- docs/iso-18181-crosswalk.md
- docs/conformance-matrix.md
- docs/unsupported.md
- docs/SPEC_GAPS.md
- llm-docs/most-recent-agent-trace.md
- THIRD_PARTY_NOTICES.md

Use MIT OR Apache-2.0 licensing only if all implementation code is compatible with that choice. Do not copy code from AGPL or other incompatible projects. If any BSD implementation material is adapted rather than independently derived, preserve required notices and document the exact source in THIRD_PARTY_NOTICES.md.

ARCHITECTURAL RULES

Use this conceptual encoder pipeline:

Borrowed ImageView
    -> validated EncodeRequest
    -> encoder analysis
    -> validated CodestreamPlan
    -> independently encoded sections
    -> bounded SectionStore
    -> TOC construction
    -> codestream writer
    -> optional container writer
    -> output sink

Use this conceptual decoder pipeline:

input
    -> raw/container reader
    -> image and frame headers
    -> SectionIndex from TOC
    -> shared entropy decoder
    -> Modular or VarDCT decoder
    -> inverse transforms and frame rendering
    -> normalized output image

The foundational dependency direction is:

bitstream
    -> headers and entropy
    -> Modular
    -> VarDCT
    -> frame rendering
    -> encode/decode orchestration

VarDCT may depend on Modular. Modular must not depend on VarDCT.

Do not create a giant EncoderState or DecoderState that owns unrelated data for the entire operation. Use immutable validated configuration, narrow mutable scratch, and typed outputs passed between stages.

Keep encoder analysis separate from syntax serialization. For example:

- MaTree is a normative codestream model.
- MaTreeBuilder is an encoder heuristic.
- FrameHeader is normative.
- FrameAnalysis is not.
- EntropyDistribution is normative.
- HistogramClusterer is encoder-side analysis.

The writer must serialize already validated structures. It must not also decide codec semantics.

PUBLIC API FOUNDATION

Create practical initial types similar to:

- ImageView<'a>
- ComponentView<'a>
- SampleStorage<'a> for borrowed u8 and u16
- BitDepth
- ColorEncoding
- EncodeOptions
- EncodeLimits
- DecodeRequest
- DecodeLimits
- DecodedImage
- DecoderInfo
- Error
- UnsupportedFeature

Provide:

- encode(...)
- encode_to_writer(...)
- decode(...)
- inspect(...)

encode may return Vec<u8> as a convenience wrapper. encode_to_writer must be shaped so it can later use a bounded RAM/spill SectionStore without constructing a second complete output buffer.

The first implemented pixel type may be gray8, but the image model must not hard-code 8-bit or grayscale assumptions into all internal APIs.

MODULE LAYOUT

Use a layout close to:

src/
    bitstream/
        reader.rs
        writer.rs
        integers.rs
        half.rs
    container/
        boxes.rs
        reader.rs
        writer.rs
    headers/
        image.rs
        metadata.rs
        color.rs
        frame.rs
        extensions.rs
    entropy/
        hybrid_uint.rs
        prefix.rs
        ans.rs
        histogram.rs
        context_map.rs
    modular/
        channel.rs
        transform.rs
        rct.rs
        palette.rs
        squeeze.rs
        properties.rs
        predictor.rs
        ma_tree.rs
        codec.rs
    frame/
        groups.rs
        sections.rs
        toc.rs
        passes.rs
    encode/
        request.rs
        plan.rs
        analysis.rs
        section_store.rs
        pipeline.rs
    decode/
        request.rs
        limits.rs
        pipeline.rs
    diagnostics/
        dump.rs
        counters.rs
    image.rs
    error.rs
    lib.rs
    bin/jxltool.rs

Adapt this only where a clearer one-way dependency structure results. Do not create hollow VarDCT modules merely to make the tree look complete.

MANDATORY ONE-SESSION IMPLEMENTATION TARGET

Implement one real standards-valid vertical slice:

- naked JPEG XL codestream;
- one still image;
- one frame;
- unsigned 8-bit grayscale;
- non-XYB color interpretation;
- Modular mode;
- no animation;
- no crop;
- no extra channels;
- no upsampling;
- no patches, splines, noise, or restoration filters;
- no Modular transforms initially;
- a single group;
- a one-leaf MA tree;
- one simple legal predictor;
- one context cluster;
- no LZ77;
- the simplest fully conforming entropy backend confirmed from the current standard;
- mathematically lossless reconstruction.

Do not optimize compression at this stage. A Zero predictor and simple entropy model are acceptable if legal. A poor but conforming file is better than a sophisticated invalid file.

Implement both the encoder and decoder for this subset. The decoder must reject unsupported features explicitly rather than silently interpreting them as the supported subset.

CORE BITSTREAM WORK

Implement and test all primitive encodings required by the vertical slice, including:

- bit-level reading and writing;
- conditional fields;
- U32 forms;
- U64 forms if reached by the subset;
- enum forms;
- signed integer packing;
- size header;
- image metadata fields;
- color encoding fields needed for grayscale;
- frame header fields needed to select Modular mode;
- extension handling needed by the subset;
- TOC and section sizes needed by the subset.

Use checked arithmetic for every dimension, offset, length, and allocation. Do not cast untrusted u64 values directly to usize without validation.

ENTROPY WORK

Implement HybridUint encode/decode and the simplest legal entropy stream needed for the first file.

Prefer a deliberately restricted encoder configuration such as:

- one context;
- one context cluster;
- no LZ77;
- one histogram;
- fixed or conservatively selected HybridUint configuration.

The decoder may initially accept only that entropy configuration, but the limitation must be explicit in docs/unsupported.md and represented by typed errors.

Add unit and property tests proving entropy roundtrips for:

- zero;
- small integers;
- boundary tokens;
- large supported values;
- deterministic random sequences;
- malformed and truncated streams.

MODULAR WORK

Implement:

- channel geometry for the supported group;
- neighbor handling at image edges;
- the chosen predictor;
- signed residual mapping;
- one-leaf MA-tree serialization and parsing;
- Modular sample encode/decode;
- exact reconstruction.

Use explicit row scratch for predictor state. Do not make each sample a heap object.

SECTION STORAGE AND WRITING

Even though the first file has only a small number of sections, introduce a SectionStore abstraction now.

The initial implementation may keep sections in memory, but its API must support:

- append section;
- stable SectionId;
- retrieve length;
- stream section to a Write sink;
- future RAM threshold and temporary-file spill.

The final writer should:

1. encode section payloads;
2. obtain exact lengths;
3. write image and frame headers;
4. write the TOC;
5. stream the stored section payloads.

Do not concatenate multiple copies of the final codestream.

CLI

Add a small diagnostic CLI, preferably using PGM initially to avoid making the codec core depend on a general image library:

- jxltool encode input.pgm output.jxl
- jxltool decode input.jxl output.pgm
- jxltool inspect input.jxl

inspect should print at least:

- raw codestream versus container;
- dimensions;
- bit depth;
- color encoding;
- frame count encountered;
- frame mode;
- group and section counts;
- supported versus unsupported feature flags.

VALIDATION

Install or locate current oracle tools where possible:

- djxl/cjxl from libjxl 0.12 or newer;
- jxl-oxide CLI;
- optionally jxl-rs tooling.

Create a script under scripts/verify_external.sh that:

1. creates deterministic supported input images;
2. encodes them with this crate;
3. decodes with this crate;
4. decodes with djxl;
5. decodes with jxl-oxide;
6. compares exact grayscale samples;
7. records tool versions and commands.

External tests may skip gracefully when a tool is unavailable, but the agent should make a serious attempt to install or build the tools in this environment and actually run them.

Test images must include:

- 2x2 all zero;
- 2x2 all 255;
- 3x5 ramp;
- 17x9 checkerboard;
- odd-dimension gradient;
- deterministic random samples;
- long constant runs;
- abrupt edges.

Add malformed-input tests for:

- truncated signature;
- truncated image header;
- impossible dimensions;
- section length beyond input;
- invalid enum;
- invalid entropy data;
- allocation-limit violation.

No malformed input may panic.

Add a fuzz target or at minimum a decoder fuzz harness that can be used by cargo-fuzz later.

TEST COMMANDS

Keep these green:

cargo fmt --check
cargo check --all-targets --all-features
cargo test
cargo test --all-features
cargo clippy --all-targets --all-features

Use property tests where useful for bitstream and entropy primitives.

DOCUMENTATION AND TRACE

Maintain docs/iso-18181-crosswalk.md while implementing. For every implemented field or algorithm, record:

- standard part and clause;
- local module/type/function;
- tests exercising it;
- any implementation latitude;
- oracle used for interoperability.

After every meaningful slice, append to llm-docs/most-recent-agent-trace.md:

- work completed;
- files changed;
- exact validation commands;
- pass/fail result;
- interoperability result;
- measured output where useful;
- known remaining gaps;
- next recommended slice.

Do not wait until the end to reconstruct the trace from memory.

NON-GOALS FOR THE MANDATORY SLICE

Do not spend mandatory-scope time on:

- VarDCT;
- animation;
- layers;
- JPEG recompression;
- JPEG reconstruction data;
- Palette;
- Squeeze;
- self-correcting prediction;
- LZ77;
- patches;
- splines;
- noise;
- ICC compression;
- gain maps;
- SIMD;
- encoder quality optimization;
- byte parity with libjxl.

Do not add fake support flags or empty implementations for these. Leave a precise roadmap and explicit UnsupportedFeature variants.

STRETCH GOALS, ONLY AFTER THE MANDATORY GATE PASSES

Proceed in this order:

1. 9–16-bit grayscale using u16 storage with meaningful precision distinct from storage width.
2. Gradient predictor.
3. RGB8 and RGB16 with a reversible color transform.
4. Multiple Modular groups.
5. Minimal ISOBMFF-style container with jxlc.
6. Prefix and ANS support if only one was needed initially.
7. Decode a deliberately simple external Modular file produced by cjxl or another encoder.
8. Palette transform.
9. Squeeze transform.
10. Start the VarDCT decoder only if all earlier work remains green.

DEFINITION OF DONE FOR THIS SESSION

The mandatory goal is complete only when:

- the crate builds cleanly;
- the architecture and standard crosswalk documents exist;
- a deterministic gray8 image is encoded as a real JPEG XL codestream;
- the crate decodes it exactly;
- at least djxl or jxl-oxide independently decodes it exactly;
- preferably both independent decoders accept it;
- unsupported syntax returns explicit errors;
- malformed test cases do not panic;
- encode_to_writer does not require a second complete final output buffer;
- all mandatory tests and clippy commands have been run;
- most-recent-agent-trace.md accurately records the result;
- README.md states the exact supported subset without exaggeration.

Do not stop after producing a plan or repository skeleton. Implement the vertical slice and validate it externally. If blocked on one exact syntax issue, isolate it, document the evidence in SPEC_GAPS.md, continue all independent work, and return to the blocker. Do not replace the target with a toy non-JPEG-XL format.

At the end, provide a concise implementation report containing:

- files and modules added;
- supported subset;
- external decoder results;
- commands run;
- known failures;
- next highest-leverage milestone.
```

This scope gives the agent a real interoperability result and a durable architecture, rather than a large repository whose individual parts have never formed a valid JPEG XL file.

[1]: https://jpeg.org/jpegxl/workplan.html?utm_source=chatgpt.com "Workplan & Specs of JPEG XL"
[2]: https://github.com/tirr-c/jxl-oxide "GitHub - tirr-c/jxl-oxide: Pure Rust implementation of JPEG XL decoder · GitHub"
[3]: https://github.com/libjxl/libjxl "GitHub - libjxl/libjxl: JPEG XL image format reference implementation · GitHub"
[4]: https://github.com/libjxl/conformance "GitHub - libjxl/conformance: Test bitstreams and reference decoded images for conformance testing · GitHub"
