# Core judgment

I inspected the [current JPXL workspace](sandbox:/mnt/data/JPXL.zip) and the attached [Part 1 transcription](sandbox:/mnt/data/part1.tex). The decoder is now mature enough to define the encoder’s **exact algebraic contract**, but the encoder should not be written as the decoder in reverse.

The right architecture is an **optimizing compiler**:

```text
source image
    ↓
canonical internal image
    ↓
analysis IR
    ↓
encoder-policy search
    ↓
exact VarDCT plan
    ↓
deterministic coefficient materialization
    ↓
entropy-model optimization
    ↓
validated codestream IR
    ↓
section encoder and writer
```

There should be feedback between analysis, spatial decisions, quantization, and entropy pricing. There should not be feedback from the byte writer back into those systems.

This follows the freedom the standard explicitly gives the encoder. Annex O says the encoding procedure is not normative and sketches only possible operations: XYB conversion, optional sharpening, DCT block search, LF construction from LLF coefficients, division-based quantization, and searches for adaptive quantization and chroma-from-luma factors. The JPEG XL overview likewise says that any process producing a conforming codestream is valid and that substantial room remains for encoder improvements. 

The architecture below is therefore an original encoder design, constrained by what the decoder must reconstruct, not by libjxl’s encoder organization.

---

# 1. Split the normative encoder from the policy engine now

Your `PLAN.md` already anticipates `jpxl-encode-policy` when heuristic search appears. That trigger has arrived.

Use this dependency direction:

```text
                     jpxl-core
                    ↗         ↖
          jpxl-entropy       jpxl-bitstream
                    ↖         ↗
                     jpxl-encode
                          ↑
                 jpxl-encode-policy

jpxl-decode remains a peer and test oracle, not an encoder dependency.
jpxl-conformance tests both sides and invokes external tools.
```

## `jpxl-core`

This contains only neutral codec mathematics and vocabulary:

* `TransformType`
* `SampleBlock` and `CoeffMatrix`
* forward and inverse transforms
* coefficient layouts
* LF-to-LLF and LLF-to-LF relations
* XYB forward/inverse primitives
* quantization/dequantization arithmetic
* CfL arithmetic
* restoration-filter kernels where they are genuinely neutral
* group geometry and typed positions

Your current separation between `SampleBlock` and landscape-oriented `CoeffMatrix` is correct and should remain. It prevents the row/column and orientation confusion that has already been a major decoder failure class.

## `jpxl-encode`

This is the **normative lowering and emission crate**. It should:

* define exact, validated encoder-plan types;
* deterministically transform and quantize according to those plans;
* generate the exact LF and HF symbol sequence;
* serialize Modular auxiliary streams;
* serialize entropy models;
* encode sections;
* generate the TOC and container;
* reject impossible plans.

It must not contain:

* activity detection;
* block-size heuristics;
* quality presets;
* perceptual search;
* rate targeting policy;
* histogram-clustering search;
* effort-level conditionals scattered through the writer.

## `jpxl-encode-policy`

This is the optimizing compiler front end. It may change aggressively without destabilizing syntax code. It chooses:

* preprocessing and filter policy;
* block tiling;
* adaptive quantization;
* global scale;
* LF quantization;
* CfL;
* Sharpness;
* block-context model;
* coefficient orders;
* HF presets;
* entropy clusters;
* progressive pass allocation;
* effort/speed trade-offs.

It consumes source pixels and cost models and produces a plan accepted by `jpxl-encode`.

The original JPEG 2000 Rust guidance attached to the project is directly applicable here: use typed stage inputs and outputs rather than one global mutable encoder context, and keep normative structures separate from temporary search state.

---

# 2. Use five distinct encoder IRs

Do not try to represent the whole encoder with one `VardctPlan`. There are several materially different representations, and conflating them will make both optimization and memory ownership harder.

```text
PreparedFrame
    ↓
AnalysisAtlas
    ↓
SpatialPlan
    ↓
QuantizedFrameIr
    ↓
EntropyPlan
    ↓
EmissionPlan
```

## 2.1 `PreparedFrame`

This is the normalized encoder input:

```rust
pub struct PreparedFrame<S: PlaneStore> {
    pub width: u32,
    pub height: u32,
    pub xyb: XybPlanes<S>,
    pub original_color: OriginalColorEncoding,
    pub intensity_target: f32,
}
```

It must not imply that the entire image is resident in RAM. `PlaneStore` should support:

```rust
pub enum PlaneStoreKind {
    Resident,
    TiledSpill,
}
```

A 50-megapixel three-channel `f32` XYB image is roughly 600 MB before any candidate transforms or coefficient storage. Full-frame resident XYB should therefore be an optimization selected by the resource planner, not the only implementation.

## 2.2 `AnalysisAtlas`

This is a compact, reusable description of the image, primarily on the 8×8 atom grid:

```rust
pub struct AnalysisAtlas {
    pub grid: AtomGrid,
    pub atoms: Box<[AtomFeatures]>,
    pub integral: IntegralFeatures,
    pub perceptual: PerceptualField,
}
```

Each `AtomFeatures` should contain data such as:

```rust
pub struct AtomFeatures {
    pub mean_xyb: [f32; 3],
    pub variance_xyb: [f32; 3],
    pub gradient_h: f32,
    pub gradient_v: f32,
    pub gradient_diag_a: f32,
    pub gradient_diag_b: f32,
    pub laplacian_energy: f32,
    pub anisotropy: f32,
    pub noise_estimate: f32,
    pub x_y_covariance: f32,
    pub b_y_covariance: f32,
    pub masking: f32,
    pub saliency: f32,
}
```

Integral images over energy, variance, covariance, and masking allow candidate rectangles to obtain first-pass statistics in constant time. This avoids running a full DCT merely to reject obviously unsuitable 64×64 or 128×128 blocks.

## 2.3 `SpatialPlan`

This contains decisions but no encoded symbols:

```rust
pub struct SpatialPlan {
    pub frame: FrameDecision,
    pub quantizer: QuantizerDecision,
    pub lf: LfDecision,
    pub restoration: RestorationDecision,
    pub lf_groups: Box<[LfGroupDecision]>,
}

pub struct LfGroupDecision {
    pub id: LfGroupId,
    pub blocks: Box<[VarblockDecision]>,
    pub cfl: CflGrid,
    pub sharpness: SharpnessGrid,
}

pub struct VarblockDecision {
    pub id: BlockId,
    pub origin: LfBlockPos,
    pub transform: TransformType,
    pub hf_mul: HfMul,
}
```

Use newtypes for every easily confused integer:

```rust
pub struct HfMul(NonZeroU32);
pub struct GlobalScale(NonZeroU32);
pub struct CflFactor(i32);
pub struct BlockId(u32);
pub struct HfGroupId(u32);
pub struct LfGroupId(u32);
pub struct PassId(u8);
pub struct OrderId(u8);
pub struct PreContextId(u16);
pub struct ClusterId(u8);
```

Do not pass raw `usize` values through block, group, channel, pass, context, and order APIs.

## 2.4 `QuantizedFrameIr`

This is the first representation that contains exact integers expected by the decoder:

```rust
pub struct QuantizedFrameIr<C: CoeffStore> {
    pub lf_global: QuantizedLfGlobal,
    pub lf_groups: Box<[QuantizedLfGroup]>,
    pub hf_groups: C,
}

pub struct QuantizedLfGroup {
    pub quantized_lf: LfQuantPlanes,
    pub reconstructed_lf: ReconstructedLfPlanes,
    pub block_info: BlockInfo,
    pub cfl: CflGrid,
    pub sharpness: SharpnessGrid,
}
```

The split between `quantized_lf` and `reconstructed_lf` is important. Your decoder already preserves both because the block-context calculation consumes quantized `qdc`, while LLF reconstruction consumes dequantized, CfL-corrected, optionally smoothed LF samples. The encoder needs the same distinction.

`CoeffStore` should support resident and spill-backed storage. It exists so that:

* transforms need not be repeated while tuning global scale or entropy models;
* entropy census and final ANS emission can replay the same quantized coefficients;
* peak memory is not proportional to the full uncompressed coefficient image.

## 2.5 `EntropyPlan` and `EmissionPlan`

```rust
pub struct EntropyPlan {
    pub block_context: HfBlockContextPlan,
    pub passes: Box<[HfPassEntropyPlan]>,
}

pub struct HfPassEntropyPlan {
    pub coefficient_orders: OrderSet,
    pub presets: Box<[HfPresetPlan]>,
}

pub struct HfPresetPlan {
    pub context_map: Box<[ClusterId]>,
    pub histograms: Box<[AnsDistribution]>,
    pub hybrid_uint: Box<[HybridUintConfig]>,
}
```

After this is complete, `jpxl-encode` lowers it into:

```rust
pub struct EmissionPlan {
    pub image_header: ImageHeaderSyntax,
    pub frame_header: FrameHeaderSyntax,
    pub sections: SectionLayout,
}
```

The writer only accepts a validated form:

```rust
pub struct ValidatedEmissionPlan(EmissionPlan);

pub fn validate(plan: EmissionPlan) -> Result<ValidatedEmissionPlan>;

pub fn write_to<W: Write>(
    plan: ValidatedEmissionPlan,
    sections: SectionStore,
    out: &mut W,
) -> Result<()>;
```

---

# 3. Preserve the JPEG XL section structure instead of inventing a monolithic VarDCT payload

VarDCT is not just DCT coefficients. Its control images are Modular sub-bitstreams. The LF image, adaptive quantization information, block selection, CfL factors, Sharpness, and extra channels all enter through the same section architecture.

The encoder should lower to the decoder’s natural section dependency graph:

```text
LfGlobal
    ├── LF dequantization
    ├── Quantizer
    ├── HF block context
    ├── LF correlation
    └── GlobalModular

LfGroup[n]
    ├── LfQuant
    ├── ModularLfGroup
    └── HfMetadata
          ├── XFromY
          ├── BFromY
          ├── BlockInfo
          └── Sharpness

HfGlobal
    ├── dequantization matrices
    ├── number of HF presets
    └── orders and entropy distributions per pass

PassGroup[pass][group]
    ├── HF coefficients
    └── Modular group data
```

VarDCT HF groups are fixed at 256×256, while each LF group corresponds to a 2048×2048 frame region. Sections are independently entropy coded, and an HF group depends only on global HF data and its associated LF data. That is both the correct serialization model and the natural parallel execution model.

Extend the existing `SectionStore` rather than replacing it:

```rust
pub enum StoredSection {
    Memory(Vec<u8>),
    Spill {
        file: Arc<SpillFile>,
        offset: u64,
        len: u64,
    },
}
```

Required behavior:

1. Encode each section exactly once.
2. Store its payload and exact length.
3. Emit the TOC.
4. Stream sections in prescribed order.
5. Never assemble another full codestream copy.

---

# 4. Block selection should be an exact-cover search aligned with `BlockInfo`

This is the most important original structural decision.

An HF group contains at most a 32×32 grid of 8×8 atoms. Varblocks must cover that grid exactly, cannot overlap, and cannot cross HF-group boundaries. `BlockInfo` is decoded by repeatedly placing its next transform at the earliest uncovered atom in raster order. 

Make the encoder search use exactly that representation.

## 4.1 Frontier state

```rust
struct CoverState {
    covered_rows: [u32; 32],
    next_atom: u16,
    objective: Objective,
    parent: SearchNodeId,
    chosen: CandidateId,
}
```

For clipped edge groups, atoms beyond the actual image are initialized as covered.

At each step:

1. Find the earliest uncovered atom.
2. Enumerate legal transform candidates beginning there.
3. Reject candidates whose footprint crosses the group or overlaps coverage.
4. Add the candidate’s cost.
5. Advance to the next uncovered atom.

The completed search path is already the exact raster-order `BlockInfo` sequence. There is no later conversion from a spatial partition tree into the codestream’s greedy representation.

## 4.2 Two block solvers

Use two interchangeable solvers over the same candidate API.

### Fast hierarchical solver

This is the normal low-effort path:

* naturally aligned DCT8×8;
* DCT16×16;
* DCT32×32;
* optionally common 2:1 rectangles;
* recursive split decisions;
* exact dynamic programming within the restricted hierarchy.

This corresponds closely to the starting approach suggested by Annex O without copying any implementation. The standard overview shows the full transform vocabulary and explains how larger transforms contribute multiple low-frequency coefficients to the LF image. 

### Frontier beam solver

This is the high-effort path:

* all enabled rectangular transforms;
* special 8×8-footprint transforms;
* arbitrary legal combinations;
* bounded beam width;
* occupancy-state deduplication;
* deterministic tie-breaking.

A 32-row bitset makes overlap tests and placement cheap. Beam width, candidate set, and retained final plans become effort controls.

Do not call this an exact optimum over every possible tiling. It is a bounded search. Its advantage is that it covers a much broader space without requiring a C-style recursive object graph or an infeasible full-width dynamic program.

## 4.3 Candidate R-D envelopes

Do not assign one cost to a transform candidate. Give it a short local rate-distortion curve:

```rust
struct CandidateRdCurve {
    candidate: CandidateKey,
    points: InlineRdPoints,
}

struct RdPoint {
    hf_mul: HfMul,
    estimated_rate: BitCost,
    distortion: Distortion,
    encode_work: WorkCost,
    decode_work: WorkCost,
}
```

For each transform candidate, test a small set of nearby `HfMul` values and prune dominated points. For a given Lagrange multiplier, the cover solver chooses both:

* the transform;
* its adaptive quantization point.

The objective should be explicit:

```text
J =
    R_bits
  + λ · D_perceptual
  + α · encode_work
  + β · decode_work
  + metadata_bits
  + boundary_penalty
```

At normal effort, `α` and `β` may be near zero. At a fast preset, they become meaningful. This gives effort levels an actual objective rather than a collection of unrelated `if effort >= 7` branches.

## 4.4 Multi-tier candidate evaluation

Candidate evaluation should be progressively more expensive:

```text
feature-atlas rejection
    ↓
approximate transform energy and rate
    ↓
actual forward transform
    ↓
actual quantization
    ↓
exact local symbol simulation
    ↓
local reconstruction metric with halo
```

Most large transforms should die at the first or second stage.

Retain several complete group plans, not only the first winner. Exact entropy models are unavailable during initial block search, so the top few spatial plans should survive to the exact symbol-cost pass.

---

# 5. Build an allocation-free transform kernel boundary

The semantic `SampleBlock` and `CoeffMatrix` types should remain in `jpxl-core`, but they should not be allocated for every search candidate.

Add borrowed hot-path views:

```rust
pub struct SampleView<'a> {
    data: &'a [f32],
    rows: usize,
    cols: usize,
    stride: usize,
}

pub struct CoeffViewMut<'a> {
    data: &'a mut [f32],
    rows: usize,
    cols: usize,
    stride: usize,
}
```

Then provide:

```rust
pub fn forward_varblock_into(
    transform: TransformType,
    samples: SampleView<'_>,
    coeffs: CoeffViewMut<'_>,
    scratch: &mut TransformScratch,
);

pub fn inverse_varblock_into(
    transform: TransformType,
    coeffs: CoeffView<'_>,
    samples: SampleViewMut<'_>,
    scratch: &mut TransformScratch,
);
```

Do not use a dynamic trait call inside row or coefficient loops. Dispatch once at the varblock boundary:

```rust
type ForwardKernel = fn(
    SampleView<'_>,
    CoeffViewMut<'_>,
    &mut TransformScratch,
);

static FORWARD_KERNELS: [ForwardKernel; 27] = [
    forward_dct8x8,
    forward_hornuss,
    forward_dct2x2,
    // ...
];
```

## Kernel strategy

The performance path should have:

* dedicated DCT8×8;
* dedicated 16-point and 32-point one-dimensional kernels;
* allocation-free separable 2-D transforms;
* generic power-of-two fallback for large transforms;
* exact forward forms for Hornuss, DCT2×2, DCT4×4, DCT4×8, DCT8×4, and AFV;
* per-worker scratch pools;
* no temporary `Vec` from a DCT call.

The current `jpxl-core::dct_2d_raw`/`dct_2d` APIs are suitable as scalar correctness references, not as the final search-loop interface.

## Transform candidate caching

For the fast hierarchical solver, build an aligned transform pyramid:

```text
8×8 candidates
16×16 candidates
32×32 candidates
64×64 candidates when enabled
```

Cache transformed results or compact statistics for the common sizes. For uncommon large candidates, compute lazily and discard unless they enter a retained plan.

Do not retain full coefficients for every possible candidate. A single 256×256 transform is 256 KiB per channel in `f32`; speculative storage across many candidates would become the dominant memory cost.

---

# 6. Treat LF planning as a separate compression problem

The LF image is not merely “DC coefficients.” For larger transforms it contains the information needed to reconstruct the lowest-frequency rectangle of coefficients. The transform selection therefore changes both HF behavior and the LF image. 

Add the encoder-side counterpart to the decoder’s current `llf_from_lf`:

```rust
pub fn lf_from_llf(
    transform: TransformType,
    llf: &CoeffMatrix,
) -> SampleBlock;
```

The two functions need direct inverse-pair tests for every transform type.

## `LfPlanner`

```rust
pub struct LfDecision {
    pub extra_precision: u8,
    pub quant_lf: u32,
    pub channel_dequant: LfChannelDequantDecision,
    pub correlation: LfCorrelationDecision,
    pub adaptive_smoothing: bool,
}
```

The planner should:

1. Derive the exact LF image from the selected varblocks.
2. Search a small set of `quant_lf` and `extra_precision` choices.
3. Quantize.
4. Apply exact decoder-side dequantization.
5. Apply LF CfL.
6. Apply adaptive LF smoothing when selected.
7. Measure the actual reconstructed LF image.

Give LF its own minimum-quality floor. A global rate controller that sacrifices LF too aggressively can produce banding and coarse color transitions even while preserving many HF coefficients. The overview specifically identifies adaptive LF smoothing as a defense against low-rate gradient banding. 

Block search can initially use a source-derived 1:8 proxy. Once a candidate tiling is chosen, exact LF construction must be part of final plan evaluation.

---

# 7. Couple adaptive quantization and block choice without tangling them

JPEG XL’s `HfMul` locally adjusts quantization and also influences the baseline EPF strength; `Sharpness` provides additional filter modulation. CfL uses per-64×64 factors for HF data.

The architecture should make those couplings explicit.

## 7.1 Perceptual field first

Create a desired local log-step field on the 8×8 grid:

```rust
pub struct DesiredQuantField {
    pub log_step: Box<[f32]>,
}
```

Use:

* local XYB masking;
* edge strength;
* texture;
* noise;
* saliency;
* channel sensitivity;
* a minimum quality floor.

Work in log space because quantization scales multiply.

## 7.2 Factor local steps into `global_scale × HfMul`

The desired effective step field needs to be represented by:

```text
global_scale × per-varblock HfMul × matrix/channel factors
```

This is a small factorization problem, not just a direct conversion.

Search a narrow range of candidate `global_scale` values. For each candidate:

1. Derive the best integer `HfMul` for each candidate block.
2. Measure approximation error to the desired field.
3. Measure `BlockInfo` signaling cost.
4. Measure actual coefficient rate and distortion.

Choose a factorization that keeps `HfMul` values in a compact, predictable distribution. A slightly less accurate local step field may win if its metadata compresses much better.

## 7.3 Quantize against the actual decoder function

The decoder’s quantization bias makes the inverse mapping nonlinear near zero. A high-quality encoder should not always use simple rounding.

For each coefficient:

1. Compute an approximate integer `q₀`.
2. Evaluate a tiny candidate set such as:
   `0`, `±1`, `q₀−2..q₀+2`.
3. Reconstruct each candidate using the exact decoder-side bias adjustment and dequantization.
4. Choose the minimum local:
   `distortion + λ × symbol_price`.

At fast effort, use direct rounding. At high effort, use this small scalar R-D search. It is branchable by effort at the quantizer boundary rather than inside the transform engine.

---

# 8. Estimate CfL from selected HF coefficients, then optimize the signaled integer

The first CfL estimate should use weighted regression in the unquantized HF domain.

For each 64×64 CfL tile:

```text
kX ≈ Σ w·Y·X / Σ w·Y²
kB ≈ Σ w·Y·B / Σ w·Y²
```

Weights should include:

* perceptual frequency weight;
* local masking;
* coefficient significance;
* exclusion or reduction of unstable near-zero Y coefficients.

Then convert the continuous value to the exact signaled factor and search a few neighboring integers. The winning factor should minimize the **post-quantization** objective, not merely the unquantized least-squares error:

```text
residual entropy
+ λ · reconstructed XYB distortion
+ factor signaling cost
```

The block map affects the coefficient samples used in regression. CfL affects the coefficient entropy used by the block map. Resolve this with bounded alternation:

```text
initial CfL from a fixed DCT8×8 grid
    ↓
block and HfMul search
    ↓
re-estimate CfL from selected transforms
    ↓
local block refinement in materially changed regions
```

One or two refinements should be enough. Do not create an unconstrained convergence loop.

LF correlation should be trained separately from HF CfL because it has different signaling and operates on a different representation.

---

# 9. Use a two-pass entropy compiler, not direct ANS calls from coefficient loops

The HF context model can expose thousands of pre-clustering contexts. The final context map reduces those to at most 255 distributions per pass. The context depends on block context, predicted nonzero count, coefficient position, remaining nonzeros, and whether the preceding coefficient was nonzero.

Do not entropy-encode while traversing coefficients for the first time.

## 9.1 Exact event generator

Implement one deterministic event walk:

```rust
pub trait HfEventSink {
    fn nonzeros(&mut self, context: PreContextId, value: u32);
    fn coefficient(&mut self, context: PreContextId, value: u32);
}
```

The event generator owns:

* Y, X, B traversal order;
* nonzero prediction;
* `prev` state;
* coefficient order;
* pass accumulation;
* block-context calculation.

It does not know whether the sink is:

* a census collector;
* an exact bit-cost estimator;
* an ANS writer;
* a diagnostic trace.

This makes the hardest normative loop testable independently of histogram training.

## 9.2 Census raw integers, not prematurely tokenized symbols

HybridUint configuration belongs to an entropy cluster, so collect raw unsigned-value distributions per pre-context:

```rust
pub struct RawHistogram {
    small: [u32; 32],
    tail: Vec<(u32, u32)>,
}
```

Most values will stay in the inline small range. Keep the tail sorted and sparse.

After census:

1. Remove or alias empty contexts.
2. Estimate optimal HybridUint configurations.
3. Cluster similar contexts.
4. Optimize the distribution for each cluster.
5. Build the context map.
6. Replay the coefficient IR and emit ANS symbols in reverse order.

This replay is far cheaper than recomputing DCTs because it reads `QuantizedFrameIr`.

## 9.3 Histogram clustering

Avoid an all-pairs merge across thousands of contexts.

A practical clusterer should:

1. Generate a small log-binned fingerprint for each raw histogram.
2. Place histograms into nearby fingerprint buckets.
3. Create candidate merge edges only within nearby buckets.
4. Compute the exact merged data cost and signaling cost for those candidates.
5. Use a priority queue with generation counters.
6. Stop when there is no positive saving or 255 clusters remain.

The merge criterion must include:

```text
encoded data cost
+ context-map cost
+ histogram signaling cost
+ HybridUint configuration cost
```

Not just divergence between normalized histograms.

## 9.4 Coefficient-order optimizer

Keep natural orders for the first encoder.

Later, collect per-position data for each Order ID and channel:

* probability of nonzero;
* magnitude distribution;
* probability that coding stops before the position;
* effect on `prev`;
* effect on remaining-nonzero contexts.

Generate one or two candidate permutations and evaluate them by exact event replay. Do not assume that simply sorting by average energy gives the best entropy result.

## 9.5 HF presets

Start with one preset.

Later, cluster groups by their entropy-census fingerprints. A split is worthwhile only when the reduction in coefficient data exceeds the cost of:

* another preset;
* extra histograms;
* preset signaling.

This is a natural second-level agglomerative clustering problem.

---

# 10. Make the encoder a set of three explicit bounded feedback loops

A competitive encoder needs feedback, but it must not become one uninspectable loop.

## Spatial loop

```text
features
  ↔ block types
  ↔ adaptive quantization
  ↔ CfL
  ↔ filter policy
```

Bound it to a small number of iterations.

## Entropy loop

```text
spatial plan
  → coefficient census
  → entropy prices
  → optional spatial-plan refinement
```

The first iteration can use a generic coefficient price model. After exact histograms exist, build an `EntropyPriceBook` and optionally rerun only the groups where the predicted decision changes are large.

## Rate loop

```text
λ / global_scale
  → plan and quantization
  → exact section sizes
  → update λ / global_scale
```

Use bracketed search. Do not assume the entire encoder is perfectly monotonic because block decisions and histograms can change. Near the target:

1. Freeze block map, CfL, filters, and entropy structure.
2. Adjust quantization.
3. Perform a discrete final fill using the best next local R-D improvements that fit the remaining budget.

The outer loop should not repeatedly recompute source conversion or candidate DCTs.

---

# 11. Use a resource-aware execution graph

The natural parallel work unit is the 256×256 HF group, but some products are global:

```text
source conversion / tile store
        ↓
feature analysis by HF group
        ↓
candidate generation by HF group
        ↓
spatial search by HF group
        ↓
LF-group metadata and LF image merge
        ↓
quantized coefficient materialization by HF group
        ↓
global entropy census merge
        ↓
orders / contexts / presets
        ↓
section encoding in parallel
        ↓
TOC and ordered write
```

Do not use nested parallel iterators over:

```text
groups × candidates × channels × DCT rows
```

Use one fixed worker pool with explicit jobs.

## Memory permits

Every job should declare its expected temporary memory:

```rust
pub struct WorkRequest {
    pub class: WorkClass,
    pub transient_bytes: usize,
}
```

The scheduler acquires a memory permit before dispatch. A 256×256 transform candidate and a DCT8×8 candidate must not be charged the same amount.

## Per-worker scratch

```rust
pub struct WorkerScratch {
    pub transform: TransformScratch,
    pub quantized: Vec<i32>,
    pub reconstruction: Vec<f32>,
    pub histogram_temp: HistogramScratch,
    pub ans: AnsScratch,
}
```

Allocate once per worker and reuse it. Large-transform buffers should use size classes so that an occasional 256×256 candidate does not force every worker to permanently reserve maximum scratch.

## Determinism

Parallelism must not alter output:

* collect results by typed group ID;
* deterministic histogram merge ordering;
* deterministic tie-breaks;
* no hash-map iteration in emitted decisions;
* stable floating-point operation order where decisions depend on close scores.

---

# 12. Keep the fast path structurally cheap

Do not make every encode pay for the high-effort architecture.

All effort levels should produce the same IR types, but use different policy components:

| Mode     | Spatial policy                           | Quant/CfL                                 | Entropy policy                                |
| -------- | ---------------------------------------- | ----------------------------------------- | --------------------------------------------- |
| Fast     | Fixed DCT8×8                             | Constant HfMul, simple CfL                | Natural order, one preset, simple clustering  |
| Balanced | Hierarchical 8/16/32 search              | Activity AQ, refined CfL                  | Real clustering, one preset                   |
| High     | Rectangles + selected special transforms | Joint block/AQ search                     | Order optimization, optional multiple presets |
| Research | Frontier beam portfolio                  | Local R-D quantization, filter refinement | Repeated exact replay and model refinement    |

Represent this as a budget rather than an integer examined everywhere:

```rust
pub struct SearchBudget {
    pub transform_set: TransformSet,
    pub cover_mode: CoverMode,
    pub beam_width: usize,
    pub retained_group_plans: usize,
    pub quant_points_per_candidate: u8,
    pub spatial_iterations: u8,
    pub cfl_refinements: u8,
    pub entropy_refinements: u8,
    pub exact_metric_candidates: u8,
}
```

A quality target and an effort target are different things. Keep them separate.

---

# 13. Add an encoder-local reconstruction model, not calls into the public decoder

Candidate evaluation needs to predict what the decoder will display. Calling the full decoder for every candidate would be far too expensive and would couple the crates incorrectly.

Move neutral inverse primitives into `jpxl-core` where necessary:

* dequantize one coefficient block;
* apply CfL;
* reconstruct LLF;
* inverse transform;
* Gaborish;
* EPF building blocks;
* inverse XYB.

Then `jpxl-encode-policy` can reconstruct:

* one candidate block;
* one group plus a halo;
* a full final image at plan checkpoints.

Do not move:

* bit readers;
* section parsing;
* decoder state machines;
* decoder allocation logic.

Sharing exact mathematical primitives is acceptable. Sharing the bitstream control flow would defeat the clean separation and increase the risk of paired bugs.

External decoders remain the conformance gate.

---

# 14. Use a two-tier distortion model

PSNR alone will steer the encoder incorrectly. The JPEG XL overview specifically illustrates why average-error metrics can be deceived by large easy regions and explains the use of local perceptual heatmaps and higher-norm aggregation. 

## Fast search metric

Implement an internal metric that is cheap enough for block search:

```text
multiscale XYB error
× local masking
× frequency sensitivity
× edge preservation weight
```

It should return a local distortion sum and a local worst-error contribution.

## Final plan metric

At global checkpoints:

1. Reconstruct the full candidate image through the exact local reconstruction model.
2. Apply selected filters.
3. Convert to the requested output color space.
4. Evaluate a stronger perceptual metric.
5. Optionally invoke an external metric binary in the benchmark harness.

Do not place external metric dependencies inside the codec crates.

For aggregation, avoid a simple mean. A mixture of average and high-norm/worst-region terms is better suited to preventing a small damaged subject from being hidden by a large smooth background.

---

# 15. Suggested module layout

```text
crates/jpxl-core/src/
    plane.rs
    dct.rs
    varblock.rs
    vardct_forward.rs
    vardct_quant.rs
    vardct_lf.rs
    vardct_cfl.rs
    restoration.rs

crates/jpxl-encode/src/
    vardct/
        mod.rs
        plan.rs
        validate.rs
        materialize.rs
        lf_global.rs
        lf_group.rs
        hf_global.rs
        pass_group.rs
        events.rs
        section.rs
    section_store.rs
    writer.rs

crates/jpxl-encode-policy/src/
    lib.rs
    request.rs
    resources.rs
    source/
        image_view.rs
        color.rs
        plane_store.rs
        tile_cache.rs
    analysis/
        atlas.rs
        integral.rs
        perceptual.rs
        noise.rs
    block/
        candidate.rs
        transform_bank.rs
        rd_curve.rs
        hierarchical.rs
        frontier_beam.rs
    quant/
        field.rs
        scalar.rs
        factorization.rs
        rate_loop.rs
    lf.rs
    cfl.rs
    filter.rs
    entropy/
        price_book.rs
        census.rs
        cluster.rs
        order.rs
        preset.rs
    metric/
        fast_xyb.rs
        full_frame.rs
    schedule.rs
    scratch.rs
    diagnostics.rs
```

Do not split every submodule into a crate. The important crate split is the one-way boundary between policy and normative emission.

---

# 16. Top-level orchestration

The top-level encoder should read approximately like this:

```rust
pub fn encode_vardct_to_writer<W: std::io::Write>(
    source: ImageView<'_>,
    request: &EncodeRequest,
    out: &mut W,
) -> Result<EncodeReport> {
    let resources = ResourcePlanner::new(request.resources)?;
    let prepared = prepare_source(source, request, &resources)?;

    let atlas = analyze(&prepared, request, &resources)?;

    let initial_prices = EntropyPriceBook::generic();
    let spatial = plan_spatial(
        &prepared,
        &atlas,
        request,
        &initial_prices,
        &resources,
    )?;

    let mut quantized = materialize_quantized(
        &prepared,
        &spatial,
        request,
        &resources,
    )?;

    let census = collect_entropy_census(&quantized, &spatial)?;
    let entropy = plan_entropy(&census, request)?;

    let refined_spatial = maybe_refine_spatial(
        spatial,
        &prepared,
        &atlas,
        &entropy.price_book(),
        request,
        &resources,
    )?;

    if refined_spatial.changed() {
        quantized = materialize_quantized(
            &prepared,
            &refined_spatial,
            request,
            &resources,
        )?;
    }

    let final_entropy = finalize_entropy(&quantized, &refined_spatial, request)?;
    let emission = lower_to_emission_plan(
        &quantized,
        &refined_spatial,
        final_entropy,
    )?;
    let validated = jpxl_encode::vardct::validate(emission)?;

    let sections = jpxl_encode::vardct::encode_sections(
        &validated,
        &quantized,
        &resources,
    )?;
    jpxl_encode::write_to(validated, sections, out)?;

    Ok(build_report())
}
```

This is intentionally explicit. It is easier to profile, test, replace, and reason about than an `EncoderState::run()` that mutates itself for several thousand lines.

---

# 17. Profiling architecture must be built in before optimization

Keep the current `PERFORMANCE.md` discipline. In particular, preserve immutable binaries, pinned inputs, interleaved A/B tests, correctness gates, cache-regime declarations, and small/large image classes.

Add zero-cost-disabled stage spans:

```rust
profile_span!("vardct.prepare.rgb_to_linear");
profile_span!("vardct.prepare.linear_to_xyb");
profile_span!("vardct.analysis.atlas");
profile_span!("vardct.block.candidate_dct8");
profile_span!("vardct.block.candidate_large");
profile_span!("vardct.block.cover_search");
profile_span!("vardct.quant.materialize");
profile_span!("vardct.entropy.census");
profile_span!("vardct.entropy.cluster");
profile_span!("vardct.entropy.ans");
profile_span!("vardct.section.write");
```

Add counters that explain the flamegraph:

```text
candidate transforms considered by type
candidate transforms actually computed
candidate-cache hit rate
cover states expanded and pruned
beam-width high-water
DCT calls by size
coefficients quantized
zero coefficient percentage
CfL tiles refined
rate-loop iterations
pre-contexts used
histogram merge candidates
final cluster count
ANS symbols
section bytes by section kind
scratch high-water
resident and spilled coefficient bytes
```

A flamegraph saying “DCT is 42%” is not sufficient. You need to know whether it is:

* DCT8×8 final materialization;
* speculative DCT32×32 candidates;
* repeated rate-loop transforms;
* heap allocation around DCTs;
* transposes;
* source-tile cache misses.

## Benchmark claims

Only compare against libjxl as a black box, with:

* identical source pixels and color interpretation;
* matched output size or matched perceptual score;
* recorded effort settings;
* independent decode verification;
* median encode time;
* peak RSS;
* output size;
* metric score;
* binary hashes.

“Faster at a different quality” is not a performance result.

---

# 18. Implementation order

## Milestone 1 — Structural split

Create `jpxl-encode-policy` and exact VarDCT plan types. Add no heuristic sophistication yet.

Exit gate:

* a hand-built exact plan can be validated and dumped;
* malformed plans cannot reach the writer.

## Milestone 2 — Fixed DCT8×8 VarDCT vertical slice

Target:

```text
RGB8 sRGB
forward XYB
one frame
fixed DCT8×8
default dequant matrices
constant HfMul
natural coefficient order
one HF pass
one HF preset
simple or disabled CfL
filters disabled
ANS entropy
```

Exit gate:

* JPXL decodes it;
* `djxl` decodes it;
* `jxl-oxide` decodes it;
* dimensions and pixels are within the expected lossy tolerance;
* section trace and ANS terminal state are valid.

Do not start block search before this works.

## Milestone 3 — Complete forward transform algebra

Implement allocation-free forward forms for every transform your decoder already supports, plus `lf_from_llf`.

Exit gate:

* transform-without-quantization forward/inverse tests;
* orientation and coefficient-order tests;
* full VarDCT encode/decode using several forced transform maps.

## Milestone 4 — Exact rate loop with fixed blocks

Add:

* `global_scale`;
* HfMul;
* LF quantization;
* target bytes/bpp;
* exact section-size accounting;
* discrete budget fill.

This proves that rate control works independently of block heuristics.

## Milestone 5 — CfL

Add global LF correlation and per-64×64 HF factors, first by regression and then by local integer refinement.

Exit gate:

* size improvement at equal reconstruction quality on correlated color images;
* no regression on grayscale or weakly correlated material.

## Milestone 6 — Hierarchical block selector

Enable DCT8×8, DCT16×16, DCT32×32, then common rectangles.

Exit gate:

* selected maps are legal;
* forced fixed-block baselines remain available;
* matched-quality size improves on a mixed photo corpus.

## Milestone 7 — Adaptive quantization and block R-D curves

Introduce the perceptual field and joint transform/HfMul selection.

Exit gate:

* spatial quality becomes more uniform;
* hard regions no longer dictate global quality;
* target size remains stable.

## Milestone 8 — Entropy optimization

Add:

* trained block context;
* context clustering;
* custom orders;
* optional multiple presets.

Run one exact entropy refinement of the spatial plan, not an unlimited loop.

## Milestone 9 — Filter planning

Add:

* inverse Gaborish preconditioning;
* EPF iteration selection;
* Sharpness map;
* exact group-with-halo reconstruction.

Filters should enter only after the unfiltered encoder has a trustworthy R-D baseline.

## Milestone 10 — High-effort frontier search and kernel optimization

Then add:

* frontier beam search;
* special 8×8 transforms;
* transform pyramid;
* SIMD;
* resource-aware threading;
* tiled XYB/coeff spill;
* scratch reuse;
* tuned effort profiles.

---

# 19. Structural traps to avoid

**Do not mirror `decode_vardct_frame`.** Its section order is useful, but its ownership and control flow are decoder-specific.

**Do not place heuristics in `jpxl-encode`.** The writer must remain able to encode a hand-authored legal plan.

**Do not keep all speculative coefficients.** Keep compact statistics; recompute or retain coefficients only for finalist plans.

**Do not recompute DCTs when tuning histograms.** Replay quantized coefficient IR.

**Do not run a whole-frame perceptual metric for every candidate.** Use local search costs and full-frame checkpoints.

**Do not optimize block size, AQ, CfL, filters, orders, and histograms simultaneously from the beginning.** Bring up each lever against fixed values for the others.

**Do not scatter effort checks through kernels.** Select strategy and search budgets at stage boundaries.

**Do not allow nested unconstrained parallelism.** One scheduler must own both CPU and memory budgets.

**Do not optimize only output size.** A smaller file with worse reconstruction is not an encoder improvement.

**Do not optimize only encode speed.** A fast fixed-DCT path and a high-density path should coexist over the same exact emitter.

---

# Assessment

This architecture does not guarantee that the first encoder will match libjxl. Nothing architectural can guarantee that; rate-distortion quality is ultimately empirical. It does remove the main structural ceilings that would otherwise prevent catching it:

* policy is replaceable without risking syntax;
* transform work is reusable;
* block search has a broad but bounded representation;
* quantization and block selection share local R-D curves;
* entropy pricing can feed back into spatial decisions;
* coefficient replay avoids repeated transforms;
* memory is group-bounded and spillable;
* the fast path does not pay for the high-effort path;
* every major stage can be isolated in a flamegraph.

The most important first move is therefore **not** implementing a sophisticated varblock heuristic. It is creating `jpxl-encode-policy`, adding the exact VarDCT IR boundary, and producing a valid fixed-DCT8×8 VarDCT codestream through that final architecture. Once that works, every later improvement becomes a measurable policy substitution rather than a rewrite.
